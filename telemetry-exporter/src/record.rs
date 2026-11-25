//! The telemetry record and the operation records it lists.

use std::collections::{BTreeMap, HashMap};
use std::time::SystemTime;

use aws_durable_execution_sdk::plugin::OperationInfo;
use aws_durable_execution_sdk::{InvocationStatus, OperationKind, WireError};
use serde::Serialize;
use serde_json::Value;

use crate::arn::ParsedArn;
use crate::config::{OperationOverride, apply_transform};
use crate::timestamp::{epoch_ms, format_iso_ms};

/// The `recordType` of every record.
pub const RECORD_TYPE: &str = "DurableExecutionTelemetry";

/// The `schemaVersion` of every record this crate writes.
pub const SCHEMA_VERSION: &str = "1.0";

/// The longest error `name` or `message` a record carries, in Unicode
/// code points. A longer value keeps this many code points followed by
/// `…`. The JavaScript, Python and Java exporters cut at the same point.
const MAX_TEXT_LENGTH: usize = 8192;

/// The execution status a record reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum ExecutionStatus {
    /// The execution has not reached a terminal status.
    Running,
    /// The execution completed successfully.
    Succeeded,
    /// The execution failed.
    Failed,
}

impl ExecutionStatus {
    /// The record status of an invocation's end status.
    ///
    /// A suspended invocation (`Pending`) and an invocation the service
    /// retries (`Retrying`) leave the execution in flight. So both map to
    /// `Running`.
    pub(crate) fn from_invocation(status: InvocationStatus) -> Self {
        match status {
            InvocationStatus::Succeeded => Self::Succeeded,
            InvocationStatus::Failed => Self::Failed,
            _ => Self::Running,
        }
    }

    /// Whether the status is terminal: `Succeeded` or `Failed`.
    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed)
    }
}

/// An error as a record carries it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ErrorRecord {
    /// The error type, or `Error` when the error has none. At most 8,192
    /// code points plus `…`.
    pub name: String,
    /// The error message, or the empty string when the error has none. At
    /// most 8,192 code points plus `…`.
    pub message: String,
}

impl ErrorRecord {
    /// An error record with the given name and message.
    ///
    /// The strings are stored as given. The plugin shortens the errors it
    /// reports to 8,192 code points. This constructor does not.
    pub fn new(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
        }
    }

    /// The record of an error the SDK reports.
    ///
    /// The SDK's error carries no cause chain. So the record never has a
    /// `cause`, unlike a JavaScript record of a JavaScript error.
    pub(crate) fn from_wire(error: &WireError) -> Self {
        Self::from_parts(error.error_type(), error.error_message())
    }

    /// The record of an error type and message. A missing type is `Error`
    /// and a missing message is the empty string.
    pub(crate) fn from_parts(error_type: Option<&str>, message: Option<&str>) -> Self {
        Self {
            name: cap_text(error_type.unwrap_or("Error")),
            message: cap_text(message.unwrap_or_default()),
        }
    }
}

/// Caps `text` at [`MAX_TEXT_LENGTH`] code points, then appends `…`.
pub(crate) fn cap_text(text: &str) -> String {
    let mut chars = text.chars();
    let mut capped: String = chars.by_ref().take(MAX_TEXT_LENGTH).collect();
    if chars.next().is_some() {
        capped.push('…');
    }
    capped
}

/// One operation of the execution.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct OperationRecord {
    /// The operation's wire ID.
    pub id: String,
    /// The operation's name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The operation type, such as `STEP` or `CONTEXT`.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub operation_type: Option<String>,
    /// The operation sub-type, such as `Step`, `Map` or `MapIteration`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_type: Option<String>,
    /// The wire ID of the parent operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// The operation status, such as `SUCCEEDED`, or `UNKNOWN`.
    pub status: String,
    /// When the operation started, `YYYY-MM-DDTHH:MM:SS.sssZ`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    /// When the operation ended, `YYYY-MM-DDTHH:MM:SS.sssZ`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    /// The end time minus the start time, when the operation has both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    /// The step's 1-based attempt number. The SDK reports it for steps
    /// only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attempt: Option<u32>,
    /// The operation's error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorRecord>,
    /// The operation's result, when an override transforms it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// `true` when truncation dropped the result.
    #[serde(skip_serializing_if = "is_false")]
    pub truncated: bool,
    /// The start time in epoch milliseconds, for the oldest-first order of
    /// truncation.
    #[serde(skip)]
    pub(crate) start_ms: Option<i64>,
    /// The operation's [`OperationInfo::sequence`]. It orders operations
    /// that have the same start time.
    #[serde(skip)]
    pub(crate) sequence: Option<u64>,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's skip_serializing_if passes a reference"
)]
pub(crate) fn is_false(value: &bool) -> bool {
    !*value
}

impl OperationRecord {
    /// An operation record with the given wire ID and status.
    ///
    /// Every other field is unset. The fields are public, so the caller
    /// assigns the ones it needs. The start time is the exception: set it
    /// with [`with_start_time`](Self::with_start_time).
    ///
    /// The plugin builds the records it exports. This constructor is for
    /// code that needs a record without a plugin run, such as the unit
    /// test of a [`TelemetryExporter`](crate::TelemetryExporter).
    pub fn new(id: impl Into<String>, status: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            operation_type: None,
            sub_type: None,
            parent_id: None,
            status: status.into(),
            start_time: None,
            end_time: None,
            duration_ms: None,
            attempt: None,
            error: None,
            result: None,
            truncated: false,
            start_ms: None,
            sequence: None,
        }
    }

    /// Sets when the operation started.
    ///
    /// [`truncate_record`](crate::truncate_record) drops the operations
    /// with the oldest start first. It reads the start from a private
    /// field, not from the `start_time` string. So assigning `start_time`
    /// alone leaves the operation without a start for truncation, and it
    /// is dropped last. This method sets both.
    #[must_use]
    pub fn with_start_time(mut self, start: SystemTime) -> Self {
        let start_ms = epoch_ms(start);
        self.start_time = Some(format_iso_ms(start_ms));
        self.start_ms = Some(start_ms);
        self
    }

    /// The record of one operation the SDK reports.
    pub(crate) fn from_info(op: &OperationInfo) -> Self {
        let start_ms = op.start_timestamp().map(epoch_ms);
        let end_ms = op.end_timestamp().map(epoch_ms);
        let duration_ms = match (start_ms, end_ms) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        };
        Self {
            id: op.id().to_owned(),
            name: op.name().map(str::to_owned),
            operation_type: op.type_str().map(str::to_owned),
            sub_type: op.sub_type().map(str::to_owned),
            parent_id: op.parent_id().map(str::to_owned),
            status: op.status_str().unwrap_or("UNKNOWN").to_owned(),
            start_time: start_ms.map(format_iso_ms),
            end_time: end_ms.map(format_iso_ms),
            duration_ms,
            attempt: op.attempt(),
            error: op.error().map(ErrorRecord::from_wire),
            result: None,
            truncated: false,
            start_ms,
            sequence: op.sequence(),
        }
    }
}

/// The operation settings that the plugin derives from its configuration.
#[derive(Debug, Clone, Default)]
pub(crate) struct OperationContentOptions {
    pub(crate) overrides_by_name: HashMap<String, OperationOverride>,
    pub(crate) include_errors: bool,
    pub(crate) top_level_only: bool,
}

/// Parses a serialized payload. A payload that is not valid JSON becomes a
/// JSON string, as in the JavaScript exporter.
pub(crate) fn parse_payload(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

/// Builds the operation records of a record.
///
/// 1. The `EXECUTION` operation is skipped. The record's top level already
///    carries the execution's status and times.
/// 2. An unnamed operation is skipped, because it cannot be keyed or
///    targeted by an override.
/// 3. In top-level mode, an operation with a parent is skipped.
/// 4. An operation whose override excludes it is skipped.
///
/// The SDK reports operations keyed by wire ID. A wire ID is a hash, so
/// the wire ID order is unrelated to the order the operations ran in. The
/// records are therefore sorted by start time, oldest first, and an
/// operation without a start time comes last. Two operations can have the
/// same start time. Those are sorted by [`OperationInfo::sequence`], which
/// follows the order the service reported them in. So the last record of a
/// repeated name is its latest occurrence, and the name's summary reports
/// the status of that occurrence.
pub(crate) fn operation_records(
    operations: &BTreeMap<String, OperationInfo>,
    opts: &OperationContentOptions,
) -> Vec<OperationRecord> {
    let mut records: Vec<OperationRecord> = operations
        .values()
        .filter_map(|op| operation_record(op, opts))
        .collect();
    records.sort_by_key(|record| (record.start_ms.is_none(), record.start_ms, record.sequence));
    records
}

fn operation_record(op: &OperationInfo, opts: &OperationContentOptions) -> Option<OperationRecord> {
    if op.kind() == Some(OperationKind::Execution) {
        return None;
    }
    let name = op.name().filter(|name| !name.is_empty())?;
    if opts.top_level_only && op.parent_id().is_some() {
        return None;
    }
    let override_ = opts.overrides_by_name.get(name);
    if override_.is_some_and(OperationOverride::is_excluded) {
        return None;
    }

    let mut record = OperationRecord::from_info(op);
    if !opts.include_errors {
        record.error = None;
    }
    if let Some(transform) = override_.and_then(OperationOverride::result_transform) {
        record.result = op
            .result()
            .and_then(|raw| apply_transform(transform, parse_payload(raw)));
    }
    Some(record)
}

/// One snapshot of a durable execution.
///
/// A record is complete: a newer record of the same execution replaces
/// every older one. Its JSON form is the Durable Execution Telemetry
/// schema 1.0 with the `operations` array. An exporter that writes the
/// `operationsByName` form renders the record with
/// [`render_json`](crate::render_json).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TelemetryRecord {
    /// Always [`RECORD_TYPE`].
    pub record_type: &'static str,
    /// Always [`SCHEMA_VERSION`].
    pub schema_version: &'static str,
    /// When the plugin built the record. It strictly increases within one
    /// invocation.
    pub emitted_at: String,
    /// The durable execution ARN.
    pub execution_arn: String,
    /// The execution name from the ARN, when the ARN has one.
    pub execution_name: Option<String>,
    /// The function name from the ARN.
    pub function_name: String,
    /// The function qualifier from the ARN.
    pub function_qualifier: String,
    /// The region from the ARN.
    pub region: String,
    /// The account ID from the ARN.
    pub account_id: String,
    /// The execution status.
    pub status: ExecutionStatus,
    /// When the execution started.
    pub start_time: String,
    /// When the invocation that built the record ended. Absent from a
    /// `RUNNING` record built during an invocation.
    pub end_time: Option<String>,
    /// The end time minus the start time.
    pub duration_ms: Option<i64>,
    /// The execution input, as the content settings allow.
    pub input: Option<Value>,
    /// The execution output, as the content settings allow.
    pub output: Option<Value>,
    /// The execution error.
    pub error: Option<ErrorRecord>,
    /// The operations, oldest first.
    pub operations: Vec<OperationRecord>,
    /// `true` when truncation dropped anything.
    pub truncated: bool,
    /// How many operations truncation dropped.
    pub dropped_operations: usize,
    /// `true` when truncation dropped the input.
    pub dropped_input: bool,
    /// `true` when truncation dropped the output.
    pub dropped_output: bool,
}

impl TelemetryRecord {
    /// A record of the execution `execution_arn`.
    ///
    /// The function name, qualifier, region, account ID and execution
    /// name come from the ARN, as in a record the plugin builds. A part
    /// the ARN lacks is the empty string, and a missing execution name is
    /// `None`. Both times are written as `YYYY-MM-DDTHH:MM:SS.sssZ`.
    ///
    /// Every other field is unset: no end time, duration, input, output,
    /// error or operation, and nothing truncated. The fields are public,
    /// so the caller assigns the ones it needs.
    ///
    /// The plugin builds the records it exports. This constructor is for
    /// code that needs a record without a plugin run, such as the unit
    /// test of a [`TelemetryExporter`](crate::TelemetryExporter).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::{Duration, SystemTime};
    ///
    /// use aws_durable_execution_sdk_telemetry_exporter::{
    ///     ErrorRecord, ExecutionStatus, OperationRecord, TelemetryRecord,
    /// };
    ///
    /// let start = SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000);
    /// let mut step = OperationRecord::new("1", "FAILED").with_start_time(start);
    /// step.name = Some("charge".to_owned());
    /// step.error = Some(ErrorRecord::new("CardDeclined", "insufficient funds"));
    ///
    /// let mut record = TelemetryRecord::new(
    ///     "arn:aws:lambda:us-west-2:123456789012:function:orders:$LATEST/durable-execution/order-1/inv-1",
    ///     ExecutionStatus::Failed,
    ///     start,
    ///     start + Duration::from_secs(5),
    /// );
    /// record.operations.push(step);
    ///
    /// assert_eq!(record.function_name, "orders");
    /// assert_eq!(record.execution_name.as_deref(), Some("order-1"));
    /// assert_eq!(record.start_time, "2027-01-15T08:00:00.000Z");
    /// ```
    pub fn new(
        execution_arn: impl Into<String>,
        status: ExecutionStatus,
        start_time: SystemTime,
        emitted_at: SystemTime,
    ) -> Self {
        let execution_arn = execution_arn.into();
        let arn = ParsedArn::parse(&execution_arn);
        Self {
            record_type: RECORD_TYPE,
            schema_version: SCHEMA_VERSION,
            emitted_at: format_iso_ms(epoch_ms(emitted_at)),
            execution_arn,
            execution_name: Some(arn.execution_name).filter(|name| !name.is_empty()),
            function_name: arn.function_name,
            function_qualifier: arn.qualifier,
            region: arn.region,
            account_id: arn.account_id,
            status,
            start_time: format_iso_ms(epoch_ms(start_time)),
            end_time: None,
            duration_ms: None,
            input: None,
            output: None,
            error: None,
            operations: Vec::new(),
            truncated: false,
            dropped_operations: 0,
            dropped_input: false,
            dropped_output: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{OperationsFormat, render_json, render_value};
    use crate::timestamp::system_time;
    use crate::truncation::truncate_record;

    #[test]
    fn status_maps_every_invocation_status() {
        assert_eq!(
            ExecutionStatus::from_invocation(InvocationStatus::Succeeded),
            ExecutionStatus::Succeeded
        );
        assert_eq!(
            ExecutionStatus::from_invocation(InvocationStatus::Failed),
            ExecutionStatus::Failed
        );
        assert_eq!(
            ExecutionStatus::from_invocation(InvocationStatus::Pending),
            ExecutionStatus::Running
        );
        assert_eq!(
            ExecutionStatus::from_invocation(InvocationStatus::Retrying),
            ExecutionStatus::Running
        );
    }

    #[test]
    fn only_succeeded_and_failed_are_terminal() {
        assert!(ExecutionStatus::Succeeded.is_terminal());
        assert!(ExecutionStatus::Failed.is_terminal());
        assert!(!ExecutionStatus::Running.is_terminal());
    }

    #[test]
    fn status_serializes_in_upper_case() {
        assert_eq!(
            serde_json::to_string(&ExecutionStatus::Running)
                .ok()
                .as_deref(),
            Some("\"RUNNING\"")
        );
    }

    #[test]
    fn cap_text_keeps_short_text() {
        assert_eq!(cap_text("short"), "short");
        let exact = "a".repeat(MAX_TEXT_LENGTH);
        assert_eq!(cap_text(&exact), exact);
    }

    #[test]
    fn cap_text_cuts_at_code_points() {
        let long = "é".repeat(MAX_TEXT_LENGTH + 5);
        let capped = cap_text(&long);
        assert_eq!(capped.chars().count(), MAX_TEXT_LENGTH + 1);
        assert!(capped.ends_with('…'));
        assert!(capped.starts_with("éé"));
    }

    #[test]
    fn an_error_without_type_or_message_gets_the_defaults() {
        let record = ErrorRecord::from_parts(None, None);
        assert_eq!(record.name, "Error");
        assert_eq!(record.message, "");
    }

    #[test]
    fn an_error_keeps_its_type_and_message() {
        let record = ErrorRecord::from_parts(Some("ValidationError"), Some("bad input"));
        assert_eq!(record.name, "ValidationError");
        assert_eq!(record.message, "bad input");
    }

    #[test]
    fn a_new_error_stores_its_strings_as_given() {
        let long = "a".repeat(MAX_TEXT_LENGTH + 5);
        let record = ErrorRecord::new("ValidationError", long.clone());
        assert_eq!(record.name, "ValidationError");
        assert_eq!(record.message, long);
    }

    #[test]
    fn a_new_operation_has_only_its_id_and_status() {
        let op = OperationRecord::new("1", "SUCCEEDED");
        assert_eq!(
            serde_json::to_value(&op).ok(),
            Some(serde_json::json!({"id": "1", "status": "SUCCEEDED"}))
        );
        assert_eq!(op.start_ms, None);
    }

    #[test]
    fn with_start_time_sets_the_text_and_the_truncation_start() {
        let op = OperationRecord::new("1", "STARTED").with_start_time(system_time(1_500));
        assert_eq!(op.start_time.as_deref(), Some("1970-01-01T00:00:01.500Z"));
        assert_eq!(op.start_ms, Some(1_500));
    }

    const ARN: &str = "arn:aws:lambda:us-west-2:123456789012:function:orders:$LATEST/durable-execution/exec-1/inv-1";

    #[test]
    fn a_new_record_takes_its_identity_from_the_arn() {
        let record = TelemetryRecord::new(
            ARN,
            ExecutionStatus::Running,
            system_time(1_000),
            system_time(2_500),
        );
        assert_eq!(
            render_value(&record, OperationsFormat::Array).ok(),
            Some(serde_json::json!({
                "recordType": "DurableExecutionTelemetry",
                "schemaVersion": "1.0",
                "emittedAt": "1970-01-01T00:00:02.500Z",
                "executionArn": ARN,
                "executionName": "exec-1",
                "functionName": "orders",
                "functionQualifier": "$LATEST",
                "region": "us-west-2",
                "accountId": "123456789012",
                "status": "RUNNING",
                "startTime": "1970-01-01T00:00:01.000Z",
                "operations": [],
            }))
        );
    }

    #[test]
    fn a_new_record_of_an_arn_without_an_execution_name_has_none() {
        let record = TelemetryRecord::new(
            "arn:aws:lambda:us-west-2:123456789012:function:orders",
            ExecutionStatus::Succeeded,
            system_time(0),
            system_time(0),
        );
        assert_eq!(record.execution_name, None);
        assert_eq!(record.function_name, "orders");
        assert_eq!(record.function_qualifier, "");
    }

    #[test]
    fn truncation_drops_the_result_of_the_oldest_constructed_operation() {
        let result = serde_json::json!("r".repeat(200));
        let mut late =
            OperationRecord::new("late", "SUCCEEDED").with_start_time(system_time(2_000));
        late.name = Some("late".to_owned());
        late.result = Some(result.clone());
        let mut early =
            OperationRecord::new("early", "SUCCEEDED").with_start_time(system_time(1_000));
        early.name = Some("early".to_owned());
        early.result = Some(result.clone());
        let mut record = TelemetryRecord::new(
            ARN,
            ExecutionStatus::Succeeded,
            system_time(0),
            system_time(3_000),
        );
        record.operations = vec![late, early];

        let format = OperationsFormat::Array;
        let full = render_json(&record, format).map_or(0, |json| json.len());
        let truncated = truncate_record(&record, Some(full.saturating_sub(100)), format);

        let results: Vec<(&str, bool)> = truncated
            .operations
            .iter()
            .map(|op| (op.id.as_str(), op.result.is_some()))
            .collect();
        assert_eq!(results, [("late", true), ("early", false)]);
    }

    #[test]
    fn parse_payload_falls_back_to_a_string() {
        assert_eq!(parse_payload("{\"a\":1}"), serde_json::json!({"a": 1}));
        assert_eq!(parse_payload("not json"), serde_json::json!("not json"));
    }
}
