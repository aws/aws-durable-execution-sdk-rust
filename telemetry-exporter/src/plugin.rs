//! The plugin factory and the per-invocation plugin.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use aws_durable_execution_sdk::BoxError;
use aws_durable_execution_sdk::plugin::{
    HookFuture, HookWork, InvocationEndInfo, InvocationInfo, OperationChangeInfo, OperationInfo,
    Plugin, PluginFactory,
};
use serde_json::Value;

use crate::arn::{ParsedArn, resolve_sampling_rate, should_sample};
use crate::config::{ContentConfig, ContentSetting, EmitMode, OperationDetail};
use crate::exporter::TelemetryExporter;
use crate::exporters::LambdaLogExporter;
use crate::record::{
    ErrorRecord, ExecutionStatus, OperationContentOptions, RECORD_TYPE, SCHEMA_VERSION,
    TelemetryRecord, operation_records, parse_payload,
};
use crate::scheduler::{Scheduler, Slot, lock};
use crate::timestamp::{epoch_ms, format_iso_ms, now_ms};

/// What every invocation's plugin shares: the exporters, the scheduler
/// that serializes them, and the resolved configuration.
#[derive(Debug)]
struct Environment {
    scheduler: Arc<Scheduler>,
    emit_mode: EmitMode,
    sampling_rate: f64,
    input: ContentSetting,
    output: ContentSetting,
    operations: OperationContentOptions,
}

/// The Durable Execution Telemetry plugin factory.
///
/// Register it with `OptionsBuilder::plugin`. The SDK then asks it for one
/// plugin per invocation. The exporters and the export scheduler belong
/// to the factory, so every invocation shares them. Each invocation's
/// state belongs to its own plugin, which the SDK drops when the
/// invocation returns.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use aws_durable_execution_sdk_telemetry_exporter::{
///     DurableExecutionTelemetry, EmitMode, LambdaLogExporter,
/// };
///
/// # fn options() -> Result<durable::Options, durable::BoxError> {
/// let telemetry = DurableExecutionTelemetry::builder()
///     .exporter(LambdaLogExporter::new())
///     .emit_mode(EmitMode::OnComplete)
///     .build();
/// let options = durable::Options::builder().plugin(telemetry).build()?;
/// # Ok(options)
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DurableExecutionTelemetry {
    env: Arc<Environment>,
}

/// Builds a [`DurableExecutionTelemetry`].
#[must_use]
pub struct DurableExecutionTelemetryBuilder {
    exporters: Vec<Arc<dyn TelemetryExporter>>,
    emit_mode: EmitMode,
    sampling_rate: f64,
    operation_detail: OperationDetail,
    content: ContentConfig,
}

impl std::fmt::Debug for DurableExecutionTelemetryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableExecutionTelemetryBuilder")
            .field("exporters", &self.exporters.len())
            .field("emit_mode", &self.emit_mode)
            .field("sampling_rate", &self.sampling_rate)
            .field("operation_detail", &self.operation_detail)
            .field("content", &self.content)
            .finish()
    }
}

impl DurableExecutionTelemetry {
    /// A builder with the defaults: the [`LambdaLogExporter`],
    /// [`EmitMode::OnComplete`], a sampling rate of 1.0,
    /// [`OperationDetail::TopLevel`] and the default [`ContentConfig`].
    pub fn builder() -> DurableExecutionTelemetryBuilder {
        DurableExecutionTelemetryBuilder {
            exporters: Vec::new(),
            emit_mode: EmitMode::default(),
            sampling_rate: 1.0,
            operation_detail: OperationDetail::default(),
            content: ContentConfig::default(),
        }
    }
}

impl DurableExecutionTelemetryBuilder {
    /// Adds an exporter. Without any, the plugin uses the
    /// [`LambdaLogExporter`].
    pub fn exporter(mut self, exporter: impl TelemetryExporter) -> Self {
        self.exporters.push(Arc::new(exporter));
        self
    }

    /// Sets when the plugin emits a record.
    pub fn emit_mode(mut self, emit_mode: EmitMode) -> Self {
        self.emit_mode = emit_mode;
        self
    }

    /// Sets the fraction of executions that emit records, from 0.0 to 1.0.
    ///
    /// The decision hashes the execution ARN, so every invocation of an
    /// execution makes the same decision. A value outside `[0, 1]` is
    /// clamped and a NaN becomes 1.0; each logs a warning.
    pub fn sampling_rate(mut self, rate: f64) -> Self {
        self.sampling_rate = rate;
        self
    }

    /// Sets which operations a record lists.
    pub fn operation_detail(mut self, detail: OperationDetail) -> Self {
        self.operation_detail = detail;
        self
    }

    /// Sets which payloads a record carries.
    pub fn content(mut self, content: ContentConfig) -> Self {
        self.content = content;
        self
    }

    /// Builds the factory.
    #[must_use]
    pub fn build(self) -> DurableExecutionTelemetry {
        let exporters = if self.exporters.is_empty() {
            vec![Arc::new(LambdaLogExporter::new()) as Arc<dyn TelemetryExporter>]
        } else {
            self.exporters
        };
        let ContentConfig {
            input,
            output,
            overrides,
            include_errors,
        } = self.content;
        let overrides_by_name = overrides
            .into_iter()
            .map(|o| (o.operation_name().to_owned(), o))
            .collect();
        DurableExecutionTelemetry {
            env: Arc::new(Environment {
                scheduler: Arc::new(Scheduler::new(exporters)),
                emit_mode: self.emit_mode,
                sampling_rate: resolve_sampling_rate(self.sampling_rate),
                input,
                output,
                operations: OperationContentOptions {
                    overrides_by_name,
                    include_errors,
                    top_level_only: self.operation_detail == OperationDetail::TopLevel,
                },
            }),
        }
    }
}

impl PluginFactory for DurableExecutionTelemetry {
    /// Returns the invocation's plugin, or `None` when the execution is
    /// sampled out. A sampled-out execution then costs no hook calls.
    fn create_plugin(&self, info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        if !should_sample(info.execution_arn(), self.env.sampling_rate) {
            return Ok(None);
        }
        Ok(Some(Box::new(TelemetryInvocation::new(
            Arc::clone(&self.env),
            info,
        ))))
    }
}

/// The start of the whole execution, in epoch milliseconds.
///
/// The SDK reports the execution start when the service sent it. Without
/// it, the oldest operation start is the earliest instant every invocation
/// agrees on. The current time is the last resort, and it makes the
/// duration cover only this invocation.
fn execution_start_ms(info: &InvocationInfo) -> i64 {
    first_known_start_ms(
        info.execution_start_timestamp(),
        info.operations()
            .values()
            .filter_map(OperationInfo::start_timestamp),
        now_ms,
    )
}

/// The fallback order of [`execution_start_ms`]: `reported`, then the
/// oldest of `operation_starts`, then `now`.
///
/// The SDK builds an `InvocationInfo` only inside its own crate, and its
/// local runner always reports the execution start. So this crate's tests
/// reach the two fallbacks only through this function.
fn first_known_start_ms(
    reported: Option<SystemTime>,
    operation_starts: impl IntoIterator<Item = SystemTime>,
    now: impl FnOnce() -> i64,
) -> i64 {
    reported
        .map(epoch_ms)
        .or_else(|| operation_starts.into_iter().map(epoch_ms).min())
        .unwrap_or_else(now)
}

/// Returns `now`, or 1 ms after `last` when `now` is not later, and stores
/// the result in `last`.
fn next_emitted_ms(last: &mut Option<i64>, now: i64) -> i64 {
    let ms = match *last {
        Some(previous) if now <= previous => previous.saturating_add(1),
        _ => now,
    };
    *last = Some(ms);
    ms
}

/// The plugin of one invocation.
#[derive(Debug)]
struct TelemetryInvocation {
    env: Arc<Environment>,
    slot: Arc<Slot>,
    execution_arn: String,
    arn: ParsedArn,
    start_ms: i64,
    input: Option<Value>,
    /// The `emittedAt` of the last record, in epoch milliseconds.
    last_emitted_ms: Mutex<Option<i64>>,
}

impl TelemetryInvocation {
    fn new(env: Arc<Environment>, info: &InvocationInfo) -> Self {
        Self {
            env,
            slot: Arc::new(Slot::default()),
            execution_arn: info.execution_arn().to_owned(),
            arn: ParsedArn::parse(info.execution_arn()),
            start_ms: execution_start_ms(info),
            input: Some(parse_payload(info.execution_input())),
            last_emitted_ms: Mutex::new(None),
        }
    }

    /// The `emittedAt` of the next record: now, or 1 ms after the previous
    /// record when now is not later.
    ///
    /// Exporters key records on the ARN and `emittedAt`, such as the SQS
    /// FIFO deduplication ID of the JavaScript exporter. Two records built
    /// in the same millisecond would collide. So the value strictly
    /// increases within one invocation.
    fn next_emitted_at(&self) -> String {
        format_iso_ms(next_emitted_ms(&mut lock(&self.last_emitted_ms), now_ms()))
    }

    fn build_record(
        &self,
        status: ExecutionStatus,
        operations: &BTreeMap<String, OperationInfo>,
        end_ms: Option<i64>,
        output: Option<Value>,
        error: Option<ErrorRecord>,
    ) -> TelemetryRecord {
        let arn = &self.arn;
        TelemetryRecord {
            record_type: RECORD_TYPE,
            schema_version: SCHEMA_VERSION,
            emitted_at: self.next_emitted_at(),
            execution_arn: self.execution_arn.clone(),
            execution_name: Some(arn.execution_name.clone()).filter(|name| !name.is_empty()),
            function_name: arn.function_name.clone(),
            function_qualifier: arn.qualifier.clone(),
            region: arn.region.clone(),
            account_id: arn.account_id.clone(),
            status,
            start_time: format_iso_ms(self.start_ms),
            end_time: end_ms.map(format_iso_ms),
            duration_ms: end_ms.map(|end| end.saturating_sub(self.start_ms)),
            input: self.env.input.apply(self.input.clone()),
            output: self.env.output.apply(output),
            error,
            operations: operation_records(operations, &self.env.operations),
            truncated: false,
            dropped_operations: 0,
            dropped_input: false,
            dropped_output: false,
        }
    }

    fn running_record(&self, operations: &BTreeMap<String, OperationInfo>) -> TelemetryRecord {
        self.build_record(ExecutionStatus::Running, operations, None, None, None)
    }
}

impl Plugin for TelemetryInvocation {
    fn name(&self) -> &'static str {
        "DurableExecutionTelemetry"
    }

    /// In on-change mode, schedules a `RUNNING` record and starts a task
    /// that exports it.
    ///
    /// Every later hook schedules a newer record before it exports. A
    /// record that only waited in the slot would always be replaced. So
    /// this hook starts the export itself. The SDK awaits this hook before
    /// the handler runs. So the hook does not wait for the export, and the
    /// handler starts at once. A newer record that is scheduled before the
    /// task runs replaces this one.
    fn on_invocation_start<'a>(&'a self, info: &'a InvocationInfo) -> HookFuture<'a> {
        if self.env.emit_mode == EmitMode::OnChange {
            self.slot.schedule(self.running_record(info.operations()));
            self.env.scheduler.spawn_export(&self.slot);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    /// In on-change mode, schedules a `RUNNING` record and returns the work
    /// that exports it.
    fn on_operation_change(&self, info: &OperationChangeInfo) -> HookWork {
        if self.env.emit_mode != EmitMode::OnChange {
            return Ok(None);
        }
        self.slot.schedule(self.running_record(info.operations()));
        let work = self.env.scheduler.export_pending_owned(&self.slot);
        Ok(Some(Box::pin(async move {
            work.await;
            Ok(())
        })))
    }

    /// Schedules the invocation's final record when the emit mode asks for
    /// it, then waits until every pending record is exported and flushed.
    ///
    /// | Emit mode | Emits on |
    /// | --- | --- |
    /// | on-change | every invocation end |
    /// | on-complete | `SUCCEEDED` and `FAILED` |
    /// | on-failure | `FAILED` |
    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let status = ExecutionStatus::from_invocation(info.status());
        let emit = match self.env.emit_mode {
            EmitMode::OnChange => true,
            EmitMode::OnFailure => status == ExecutionStatus::Failed,
            EmitMode::OnComplete => status.is_terminal(),
        };
        if emit {
            let record = self.build_record(
                status,
                info.operations(),
                Some(now_ms()),
                info.execution_result().map(parse_payload),
                info.execution_error().map(ErrorRecord::from_wire),
            );
            self.slot.schedule(record);
        }
        Box::pin(async move {
            self.env.scheduler.drain(&self.slot).await;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestamp::system_time;

    fn never_called() -> i64 {
        unreachable!("the current time is the last resort")
    }

    #[test]
    fn the_reported_start_wins_over_an_older_operation_start() {
        let start =
            first_known_start_ms(Some(system_time(5_000)), [system_time(1_000)], never_called);
        assert_eq!(start, 5_000);
    }

    #[test]
    fn without_a_reported_start_the_oldest_operation_start_is_used() {
        let starts = [system_time(3_000), system_time(1_000), system_time(2_000)];
        assert_eq!(first_known_start_ms(None, starts, never_called), 1_000);
    }

    #[test]
    fn without_any_start_the_current_time_is_used() {
        assert_eq!(first_known_start_ms(None, [], || 7), 7);
    }

    #[test]
    fn emitted_at_strictly_increases() {
        let mut last = None;
        assert_eq!(next_emitted_ms(&mut last, 100), 100);
        assert_eq!(next_emitted_ms(&mut last, 100), 101);
        assert_eq!(next_emitted_ms(&mut last, 99), 102);
        assert_eq!(next_emitted_ms(&mut last, 200), 200);
    }

    #[test]
    fn the_builder_defaults_to_the_lambda_log_exporter() {
        let telemetry = DurableExecutionTelemetry::builder().build();
        assert!(format!("{:?}", telemetry.env.scheduler).contains("exporters: 1"));
        assert_eq!(telemetry.env.emit_mode, EmitMode::OnComplete);
        assert!(telemetry.env.operations.top_level_only);
        assert!(telemetry.env.operations.include_errors);
    }

    #[test]
    fn the_builder_applies_every_setting() {
        let builder = DurableExecutionTelemetry::builder()
            .exporter(LambdaLogExporter::new())
            .exporter(LambdaLogExporter::new())
            .emit_mode(EmitMode::OnFailure)
            .sampling_rate(2.0)
            .operation_detail(OperationDetail::FullTree)
            .content(ContentConfig::new().include_errors(false));
        assert!(format!("{builder:?}").contains("exporters: 2"));
        let telemetry = builder.build();
        assert_eq!(telemetry.env.emit_mode, EmitMode::OnFailure);
        assert!((telemetry.env.sampling_rate - 1.0).abs() < f64::EPSILON);
        assert!(!telemetry.env.operations.top_level_only);
        assert!(!telemetry.env.operations.include_errors);
    }
}
