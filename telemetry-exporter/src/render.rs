//! The JSON forms of a record: the `operations` array, the
//! `operationsByName` index, or both.

use std::collections::HashMap;

use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use serde_json::Value;

use crate::record::{ErrorRecord, OperationRecord, TelemetryRecord};

/// How a rendered record lists its operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationsFormat {
    /// The `operations` array. This is the default.
    #[default]
    Array,
    /// The `operationsByName` index instead of the array. A store that
    /// can address a map key but not "the array element named X", such as
    /// `CloudWatch` Logs Insights, uses this form.
    ByName,
    /// The array and the index.
    Both,
}

/// The summary of every operation with one name.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub struct OperationSummary {
    /// The type of the latest occurrence.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub operation_type: Option<String>,
    /// How many operations have the name.
    pub count: u64,
    /// How many of them failed.
    pub failed_count: u64,
    /// The status of the latest occurrence.
    pub status: String,
    /// The sub-type of the latest occurrence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_type: Option<String>,
    /// The shortest duration. Absent when no occurrence has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_duration_ms: Option<i64>,
    /// The longest duration. Absent when no occurrence has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_duration_ms: Option<i64>,
    /// The sum of the durations. Absent when no occurrence has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration_ms: Option<i64>,
    /// The highest attempt number. Absent when no occurrence has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_attempt: Option<u32>,
    /// The result, when exactly one operation has the name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// The error, when exactly one operation has the name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorRecord>,
}

/// The operation summaries of a record, keyed by name, in the order each
/// name first occurs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OperationsByName(Vec<(String, OperationSummary)>);

impl OperationsByName {
    /// The summary of the operations named `name`.
    pub fn get(&self, name: &str) -> Option<&OperationSummary> {
        self.0
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, summary)| summary)
    }

    /// The names and summaries, in the order each name first occurs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &OperationSummary)> {
        self.0
            .iter()
            .map(|(name, summary)| (name.as_str(), summary))
    }

    /// The number of distinct names.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no operation has a name.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for OperationsByName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(name, summary)| (name, summary)))
    }
}

/// Builds the name-keyed summaries of `operations`.
///
/// An unnamed operation is skipped. The first occurrence of a name inserts
/// a summary with its result and error. Each later occurrence adds to the
/// counts, durations and attempts, takes over the type, sub-type and
/// status, and removes the result and error: a name that ran more than
/// once has no single representative value.
pub fn build_operations_by_name<'a, I>(operations: I) -> OperationsByName
where
    I: IntoIterator<Item = &'a OperationRecord>,
{
    let mut groups: Vec<(String, OperationSummary)> = Vec::new();
    // Truncation builds this index once per size check. A scan of `groups`
    // for each operation would cost the operation count times the number
    // of distinct names. So this map gives each name's position in
    // `groups`, and `groups` alone keeps the first-occurrence order.
    let mut positions: HashMap<&'a str, usize> = HashMap::new();
    for op in operations {
        let Some(name) = op.name.as_deref().filter(|name| !name.is_empty()) else {
            continue;
        };
        let failed = u64::from(op.status == "FAILED");
        let existing = positions
            .get(name)
            .and_then(|&position| groups.get_mut(position));
        match existing {
            None => {
                positions.insert(name, groups.len());
                groups.push((
                    name.to_owned(),
                    OperationSummary {
                        operation_type: op.operation_type.clone(),
                        count: 1,
                        failed_count: failed,
                        status: op.status.clone(),
                        sub_type: op.sub_type.clone(),
                        min_duration_ms: op.duration_ms,
                        max_duration_ms: op.duration_ms,
                        total_duration_ms: op.duration_ms,
                        max_attempt: op.attempt,
                        result: op.result.clone(),
                        error: op.error.clone(),
                    },
                ));
            }
            Some((_, summary)) => {
                summary.count += 1;
                summary.failed_count += failed;
                summary.operation_type.clone_from(&op.operation_type);
                summary.status.clone_from(&op.status);
                summary.sub_type.clone_from(&op.sub_type);
                if let Some(duration) = op.duration_ms {
                    summary.min_duration_ms = Some(
                        summary
                            .min_duration_ms
                            .map_or(duration, |m| m.min(duration)),
                    );
                    summary.max_duration_ms = Some(
                        summary
                            .max_duration_ms
                            .map_or(duration, |m| m.max(duration)),
                    );
                    summary.total_duration_ms = Some(
                        summary
                            .total_duration_ms
                            .unwrap_or(0)
                            .saturating_add(duration),
                    );
                }
                if let Some(attempt) = op.attempt {
                    summary.max_attempt =
                        Some(summary.max_attempt.map_or(attempt, |m| m.max(attempt)));
                }
                summary.result = None;
                summary.error = None;
            }
        }
    }
    OperationsByName(groups)
}

/// A record and the operations it lists, in one format.
///
/// Truncation measures candidate records that keep a subset of the
/// operations. A view over a slice of references serializes such a
/// candidate without copying the record.
pub(crate) struct RecordView<'a> {
    pub(crate) record: &'a TelemetryRecord,
    pub(crate) operations: &'a [&'a OperationRecord],
    pub(crate) format: OperationsFormat,
}

impl Serialize for RecordView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let r = self.record;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("recordType", r.record_type)?;
        map.serialize_entry("schemaVersion", r.schema_version)?;
        map.serialize_entry("emittedAt", &r.emitted_at)?;
        map.serialize_entry("executionArn", &r.execution_arn)?;
        if let Some(name) = &r.execution_name {
            map.serialize_entry("executionName", name)?;
        }
        map.serialize_entry("functionName", &r.function_name)?;
        map.serialize_entry("functionQualifier", &r.function_qualifier)?;
        map.serialize_entry("region", &r.region)?;
        map.serialize_entry("accountId", &r.account_id)?;
        map.serialize_entry("status", &r.status)?;
        map.serialize_entry("startTime", &r.start_time)?;
        if let Some(end) = &r.end_time {
            map.serialize_entry("endTime", end)?;
        }
        if let Some(duration) = r.duration_ms {
            map.serialize_entry("durationMs", &duration)?;
        }
        if let Some(input) = &r.input {
            map.serialize_entry("input", input)?;
        }
        if let Some(output) = &r.output {
            map.serialize_entry("output", output)?;
        }
        if let Some(error) = &r.error {
            map.serialize_entry("error", error)?;
        }
        if self.format != OperationsFormat::ByName {
            map.serialize_entry("operations", self.operations)?;
        }
        if r.truncated {
            map.serialize_entry("truncated", &true)?;
        }
        if r.dropped_operations > 0 {
            map.serialize_entry("droppedOperations", &r.dropped_operations)?;
        }
        if r.dropped_input {
            map.serialize_entry("droppedInput", &true)?;
        }
        if r.dropped_output {
            map.serialize_entry("droppedOutput", &true)?;
        }
        if self.format != OperationsFormat::Array {
            let by_name = build_operations_by_name(self.operations.iter().copied());
            map.serialize_entry("operationsByName", &by_name)?;
        }
        map.end()
    }
}

/// Serializes the record in the `operations` array form.
impl Serialize for TelemetryRecord {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let operations: Vec<&OperationRecord> = self.operations.iter().collect();
        RecordView {
            record: self,
            operations: &operations,
            format: OperationsFormat::Array,
        }
        .serialize(serializer)
    }
}

/// Renders a record as JSON in `format`.
///
/// # Errors
///
/// Returns the serializer's error. A record holds only JSON values and
/// strings, so serialization does not fail in practice.
pub fn render_json(
    record: &TelemetryRecord,
    format: OperationsFormat,
) -> Result<String, serde_json::Error> {
    let operations: Vec<&OperationRecord> = record.operations.iter().collect();
    serde_json::to_string(&RecordView {
        record,
        operations: &operations,
        format,
    })
}

/// Renders a record as a JSON value in `format`.
///
/// # Errors
///
/// Returns the serializer's error, as [`render_json`] does.
pub fn render_value(
    record: &TelemetryRecord,
    format: OperationsFormat,
) -> Result<Value, serde_json::Error> {
    let operations: Vec<&OperationRecord> = record.operations.iter().collect();
    serde_json::to_value(RecordView {
        record,
        operations: &operations,
        format,
    })
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "test assertions on known JSON keys"
)]
pub(crate) mod tests {
    use super::*;
    use crate::record::{ExecutionStatus, RECORD_TYPE, SCHEMA_VERSION};
    use serde_json::json;

    pub(crate) fn op(name: &str, status: &str) -> OperationRecord {
        OperationRecord {
            id: format!("id-{name}"),
            name: Some(name.to_owned()),
            operation_type: Some("STEP".to_owned()),
            sub_type: Some("Step".to_owned()),
            parent_id: None,
            status: status.to_owned(),
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

    pub(crate) fn record(operations: Vec<OperationRecord>) -> TelemetryRecord {
        TelemetryRecord {
            record_type: RECORD_TYPE,
            schema_version: SCHEMA_VERSION,
            emitted_at: "2026-01-01T00:00:01.000Z".to_owned(),
            execution_arn: "arn".to_owned(),
            execution_name: Some("exec".to_owned()),
            function_name: "fn".to_owned(),
            function_qualifier: "$LATEST".to_owned(),
            region: "us-west-2".to_owned(),
            account_id: "123".to_owned(),
            status: ExecutionStatus::Succeeded,
            start_time: "2026-01-01T00:00:00.000Z".to_owned(),
            end_time: Some("2026-01-01T00:00:01.000Z".to_owned()),
            duration_ms: Some(1000),
            input: Some(json!({"in": 1})),
            output: Some(json!({"out": 2})),
            error: None,
            operations,
            truncated: false,
            dropped_operations: 0,
            dropped_input: false,
            dropped_output: false,
        }
    }

    #[test]
    fn a_first_occurrence_keeps_every_field() {
        let mut first = op("a", "SUCCEEDED");
        first.duration_ms = Some(5);
        first.attempt = Some(2);
        first.result = Some(json!("r"));
        let by_name = build_operations_by_name(&[first]);
        let value = serde_json::to_value(&by_name).ok();
        assert_eq!(
            value,
            Some(json!({"a": {
                "type": "STEP", "count": 1, "failedCount": 0, "status": "SUCCEEDED",
                "subType": "Step", "minDurationMs": 5, "maxDurationMs": 5,
                "totalDurationMs": 5, "maxAttempt": 2, "result": "r"
            }}))
        );
    }

    #[test]
    fn a_repeated_name_aggregates_and_drops_result_and_error() {
        let mut first = op("a", "FAILED");
        first.duration_ms = Some(5);
        first.attempt = Some(3);
        first.result = Some(json!("r"));
        first.error = Some(ErrorRecord::from_parts(Some("E"), Some("m")));
        let mut second = op("a", "SUCCEEDED");
        second.duration_ms = Some(2);
        second.attempt = Some(1);
        second.sub_type = None;
        let mut third = op("a", "SUCCEEDED");
        third.duration_ms = Some(9);
        let by_name = build_operations_by_name(&[first, second, third]);
        let summary = by_name.get("a");
        assert_eq!(
            summary.cloned(),
            Some(OperationSummary {
                operation_type: Some("STEP".to_owned()),
                count: 3,
                failed_count: 1,
                status: "SUCCEEDED".to_owned(),
                sub_type: Some("Step".to_owned()),
                min_duration_ms: Some(2),
                max_duration_ms: Some(9),
                total_duration_ms: Some(16),
                max_attempt: Some(3),
                result: None,
                error: None,
            })
        );
    }

    #[test]
    fn a_later_occurrence_without_a_sub_type_clears_it() {
        let mut second = op("a", "SUCCEEDED");
        second.sub_type = None;
        let by_name = build_operations_by_name(&[op("a", "SUCCEEDED"), second]);
        assert_eq!(by_name.get("a").and_then(|s| s.sub_type.clone()), None);
    }

    #[test]
    fn a_later_duration_or_attempt_starts_the_aggregate() {
        let mut second = op("a", "SUCCEEDED");
        second.duration_ms = Some(4);
        second.attempt = Some(2);
        let by_name = build_operations_by_name(&[op("a", "SUCCEEDED"), second]);
        let summary = by_name.get("a");
        assert_eq!(summary.and_then(|s| s.min_duration_ms), Some(4));
        assert_eq!(summary.and_then(|s| s.total_duration_ms), Some(4));
        assert_eq!(summary.and_then(|s| s.max_attempt), Some(2));
    }

    #[test]
    fn unnamed_operations_are_skipped() {
        let mut unnamed = op("x", "SUCCEEDED");
        unnamed.name = None;
        let mut empty = op("y", "SUCCEEDED");
        empty.name = Some(String::new());
        let by_name = build_operations_by_name(&[unnamed, empty]);
        assert!(by_name.is_empty());
        assert_eq!(by_name.len(), 0);
    }

    #[test]
    fn names_keep_their_first_occurrence_order() {
        let by_name = build_operations_by_name(&[
            op("b", "SUCCEEDED"),
            op("a", "SUCCEEDED"),
            op("b", "FAILED"),
        ]);
        let names: Vec<&str> = by_name.iter().map(|(name, _)| name).collect();
        assert_eq!(names, ["b", "a"]);
        assert_eq!(by_name.len(), 2);
        assert!(by_name.get("missing").is_none());
        let json = serde_json::to_string(&by_name).unwrap_or_default();
        assert!(json.starts_with("{\"b\":"), "{json}");
    }

    #[test]
    fn the_array_form_lists_operations() {
        let value = render_value(&record(vec![op("a", "SUCCEEDED")]), OperationsFormat::Array).ok();
        let value = value.unwrap_or_default();
        assert_eq!(value["operations"][0]["name"], json!("a"));
        assert!(value.get("operationsByName").is_none());
    }

    #[test]
    fn the_by_name_form_replaces_the_array() {
        let value = render_value(
            &record(vec![op("a", "SUCCEEDED")]),
            OperationsFormat::ByName,
        )
        .ok();
        let value = value.unwrap_or_default();
        assert!(value.get("operations").is_none());
        assert_eq!(value["operationsByName"]["a"]["count"], json!(1));
    }

    #[test]
    fn the_both_form_has_the_array_and_the_index() {
        let value = render_value(&record(vec![op("a", "SUCCEEDED")]), OperationsFormat::Both).ok();
        let value = value.unwrap_or_default();
        assert!(value.get("operations").is_some());
        assert!(value.get("operationsByName").is_some());
    }

    #[test]
    fn a_full_record_has_the_schema_keys() {
        let mut full = record(Vec::new());
        full.error = Some(ErrorRecord::from_parts(Some("E"), Some("m")));
        full.truncated = true;
        full.dropped_operations = 2;
        full.dropped_input = true;
        full.dropped_output = true;
        let value = serde_json::to_value(&full).unwrap_or_default();
        assert_eq!(
            value,
            json!({
                "recordType": "DurableExecutionTelemetry",
                "schemaVersion": "1.0",
                "emittedAt": "2026-01-01T00:00:01.000Z",
                "executionArn": "arn",
                "executionName": "exec",
                "functionName": "fn",
                "functionQualifier": "$LATEST",
                "region": "us-west-2",
                "accountId": "123",
                "status": "SUCCEEDED",
                "startTime": "2026-01-01T00:00:00.000Z",
                "endTime": "2026-01-01T00:00:01.000Z",
                "durationMs": 1000,
                "input": {"in": 1},
                "output": {"out": 2},
                "error": {"name": "E", "message": "m"},
                "operations": [],
                "truncated": true,
                "droppedOperations": 2,
                "droppedInput": true,
                "droppedOutput": true
            })
        );
    }

    #[test]
    fn optional_keys_are_omitted_when_absent() {
        let mut bare = record(Vec::new());
        bare.execution_name = None;
        bare.end_time = None;
        bare.duration_ms = None;
        bare.input = None;
        bare.output = None;
        let value = serde_json::to_value(&bare).unwrap_or_default();
        for key in [
            "executionName",
            "endTime",
            "durationMs",
            "input",
            "output",
            "error",
            "truncated",
        ] {
            assert!(value.get(key).is_none(), "{key}");
        }
    }

    #[test]
    fn render_json_matches_render_value() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let text = render_json(&r, OperationsFormat::ByName).unwrap_or_default();
        let parsed: Value = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(
            Some(parsed),
            render_value(&r, OperationsFormat::ByName).ok()
        );
    }
}
