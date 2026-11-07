//! Conformance requirement 10-21: the attempt start and end hooks of a
//! retried step carry every attempt info field, in a canonical dump.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::plugin::{
    AttemptEndInfo, AttemptInfo, HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory,
};
use durable::{RetryDecision, WireError};

/// Records the hooks of one invocation as `CONFPLUGIN`.
struct Recorder {
    execution_arn: String,
}

impl Recorder {
    /// Emits `fields` after `plugin` `CONFPLUGIN` and `hook` `hook`.
    fn emit(&self, hook: &str, fields: &[(&str, serde_json::Value)]) {
        let mut record = vec![("plugin", "CONFPLUGIN".into()), ("hook", hook.into())];
        record.extend(fields.iter().cloned());
        conformance::emit_record(&self.execution_arn, &record);
    }
}

/// Whether `info` is a step's: the step filter of the plugin requirements.
fn is_step(info: &OperationInfo) -> bool {
    info.sub_type() == Some("Step")
}

/// The dump of an attempt's operation: its `attempt` is `attempt`, added
/// when the operation info has none, and its `error` is removed.
fn attempt_dump(operation: &OperationInfo, attempt: u32) -> Vec<(&'static str, serde_json::Value)> {
    let mut dump = conformance::operation_dump(operation);
    dump.retain(|(key, _)| *key != "error");
    if let Some(entry) = dump.iter_mut().find(|(key, _)| *key == "attempt") {
        entry.1 = attempt.into();
    } else {
        dump.push(("attempt", attempt.into()));
    }
    dump
}

impl Plugin for Recorder {
    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        if is_step(info.operation()) {
            self.emit(
                "attempt-start",
                &attempt_dump(info.operation(), info.attempt()),
            );
        }
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        if is_step(info.operation()) {
            let mut dump = attempt_dump(info.operation(), info.attempt());
            dump.push(("outcome", info.outcome().wire_str().into()));
            if let Some(message) = info.error().and_then(WireError::error_message) {
                dump.push(("error", message.into()));
            }
            self.emit("attempt-end", &dump);
        }
        Ok(None)
    }
}

/// Creates the plugin of each invocation.
struct Factory;

impl PluginFactory for Factory {
    fn create_plugin(
        &self,
        info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        Ok(Some(Box::new(Recorder {
            execution_arn: info.execution_arn().to_owned(),
        })))
    }
}

/// Retries an attempt below 3 after 1 s, and stops at attempt 3.
fn retry_below_3(_err: &durable::StepError, attempt: u32) -> RetryDecision {
    if attempt < 3 {
        RetryDecision::retry(Duration::from_secs(1))
    } else {
        RetryDecision::Stop
    }
}

/// Handler: one step named `flaky` that fails its first attempt and returns
/// `ok` on its second.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .step(|step_ctx| async move {
            if step_ctx.attempt() == 1 {
                return Err("attempt 1 failed".into());
            }
            Ok("ok".to_owned())
        })
        .name("flaky")
        .retry_strategy(retry_below_3)
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
