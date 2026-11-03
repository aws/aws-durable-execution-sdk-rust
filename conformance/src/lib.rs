//! Shared utilities for conformance test handlers.
//!
//! This crate holds helpers consumed by individual handler binaries in the
//! `conformance/` directory tree. It is not published to crates.io.

use aws_sdk_dynamodb::types::AttributeValue;
use std::env;

/// Atomically increments and returns the attempt counter for `execution_id`
/// in the `DynamoDB` Attempts table.
///
/// Table name defaults to `"Attempts"` unless `ATTEMPTS_TABLE_NAME` env var
/// is set.
///
/// # Errors
///
/// Returns an error if the `DynamoDB` call fails.
pub async fn increment_attempt(
    execution_id: &str,
) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_dynamodb::Client::new(&config);

    let table = env::var("ATTEMPTS_TABLE_NAME").unwrap_or_else(|_| "Attempts".to_owned());

    let output = client
        .update_item()
        .table_name(table)
        .key("executionId", AttributeValue::S(execution_id.to_owned()))
        .update_expression("SET attemptCount = if_not_exists(attemptCount, :zero) + :inc")
        .expression_attribute_values(":zero", AttributeValue::N("0".to_owned()))
        .expression_attribute_values(":inc", AttributeValue::N("1".to_owned()))
        .return_values(aws_sdk_dynamodb::types::ReturnValue::UpdatedNew)
        .send()
        .await?;

    let attrs = output
        .attributes()
        .ok_or("no attributes returned from UpdateItem")?;
    let count_attr = attrs
        .get("attemptCount")
        .ok_or("attemptCount not in response")?;

    match count_attr {
        AttributeValue::N(n) => Ok(n.parse::<i32>()?),
        _ => Err("attemptCount is not a number".into()),
    }
}

/// Prints one JSON object on one stdout line: the given fields, then
/// `durableExecutionArn`.
///
/// The plugin requirements read their records from the function's log
/// group, and Lambda writes a function's stdout there.
pub fn emit_record(execution_arn: &str, fields: &[(&str, serde_json::Value)]) {
    let mut record = serde_json::Map::new();
    for (key, value) in fields {
        record.insert((*key).to_owned(), value.clone());
    }
    record.insert(
        "durableExecutionArn".to_owned(),
        serde_json::Value::String(execution_arn.to_owned()),
    );
    println!("{}", serde_json::Value::Object(record));
}

/// Formats a timestamp as ISO-8601 (`2026-10-26T11:38:42Z`), or "" when the
/// formatter rejects it.
///
/// The formatter rejects a date after 9999-12-31.
#[must_use]
pub fn iso8601(at: std::time::SystemTime) -> String {
    use aws_sdk_dynamodb::primitives::{DateTime, DateTimeFormat};
    DateTime::from(at)
        .fmt(DateTimeFormat::DateTime)
        .unwrap_or_default()
}

/// The fields of an operation info that the canonical dump writes.
pub struct DumpFields<'a> {
    /// The operation ID.
    pub id: &'a str,
    /// The operation name.
    pub name: Option<&'a str>,
    /// The type token, such as `STEP`.
    pub kind: Option<&'static str>,
    /// The raw sub-type.
    pub sub_type: Option<&'a str>,
    /// The parent operation ID.
    pub parent_id: Option<&'a str>,
    /// The status token, such as `SUCCEEDED`.
    pub status: Option<&'static str>,
    /// When the operation started.
    pub start_timestamp: Option<std::time::SystemTime>,
    /// When the operation ended.
    pub end_timestamp: Option<std::time::SystemTime>,
    /// The error message.
    pub error_message: Option<&'a str>,
    /// The attempt number.
    pub attempt: Option<u32>,
    /// Whether the hook reports a replay.
    pub is_replay: bool,
}

/// The canonical camelCase dump: `id`; then `name`, `type`, `subType`,
/// `parentId`, `status`, `startTimestamp`, `endTimestamp`, `error` and
/// `attempt`, each only when present; then `isReplay`.
#[must_use]
pub fn dump_fields(fields: &DumpFields<'_>) -> Vec<(&'static str, serde_json::Value)> {
    let text = |value: Option<&str>| value.map(|v| serde_json::Value::String(v.to_owned()));
    let time = |value: Option<std::time::SystemTime>| {
        value.map(|at| serde_json::Value::String(iso8601(at)))
    };
    let optional = [
        ("name", text(fields.name)),
        ("type", text(fields.kind)),
        ("subType", text(fields.sub_type)),
        ("parentId", text(fields.parent_id)),
        ("status", text(fields.status)),
        ("startTimestamp", time(fields.start_timestamp)),
        ("endTimestamp", time(fields.end_timestamp)),
        ("error", text(fields.error_message)),
        ("attempt", fields.attempt.map(serde_json::Value::from)),
    ];
    let mut dump = vec![("id", serde_json::Value::String(fields.id.to_owned()))];
    dump.extend(
        optional
            .into_iter()
            .filter_map(|(key, value)| value.map(|value| (key, value))),
    );
    dump.push(("isReplay", serde_json::Value::Bool(fields.is_replay)));
    dump
}

/// `dump_fields` over the accessors of an operation info.
#[must_use]
pub fn operation_dump(
    info: &aws_durable_execution_sdk::plugin::OperationInfo,
) -> Vec<(&'static str, serde_json::Value)> {
    use aws_durable_execution_sdk::{OperationKind, OperationStatus, WireError};
    dump_fields(&DumpFields {
        id: info.id(),
        name: info.name(),
        kind: info.kind().map(OperationKind::wire_str),
        sub_type: info.sub_type(),
        parent_id: info.parent_id(),
        status: info.status().map(OperationStatus::wire_str),
        start_timestamp: info.start_timestamp(),
        end_timestamp: info.end_timestamp(),
        error_message: info.error().and_then(WireError::error_message),
        attempt: info.attempt(),
        is_replay: info.is_replay(),
    })
}

#[cfg(test)]
mod tests {
    use super::{DumpFields, dump_fields, iso8601};
    use std::time::{Duration, UNIX_EPOCH};

    /// The keys of `dump`, in order.
    fn keys(dump: &[(&'static str, serde_json::Value)]) -> Vec<&'static str> {
        dump.iter().map(|(key, _)| *key).collect()
    }

    /// The value of `key` in `dump`.
    fn value<'a>(
        dump: &'a [(&'static str, serde_json::Value)],
        key: &str,
    ) -> Option<&'a serde_json::Value> {
        dump.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    #[test]
    fn iso8601_rejected_date_gives_empty_string() {
        assert_eq!(
            iso8601(UNIX_EPOCH + Duration::from_secs(400_000_000_000)),
            ""
        );
    }

    #[test]
    fn dump_fields_omits_every_absent_field() {
        let dump = dump_fields(&DumpFields {
            id: "a",
            name: None,
            kind: None,
            sub_type: None,
            parent_id: None,
            status: None,
            start_timestamp: None,
            end_timestamp: None,
            error_message: None,
            attempt: None,
            is_replay: true,
        });
        assert_eq!(keys(&dump), ["id", "isReplay"]);
        assert_eq!(
            value(&dump, "isReplay"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn dump_fields_writes_every_present_field_in_order() {
        let dump = dump_fields(&DumpFields {
            id: "a",
            name: Some("greet"),
            kind: Some("STEP"),
            sub_type: Some("Step"),
            parent_id: Some("p"),
            status: Some("SUCCEEDED"),
            start_timestamp: Some(UNIX_EPOCH + Duration::from_secs(1)),
            end_timestamp: Some(UNIX_EPOCH + Duration::from_secs(2)),
            error_message: Some("boom"),
            attempt: Some(1),
            is_replay: false,
        });
        assert_eq!(
            keys(&dump),
            [
                "id",
                "name",
                "type",
                "subType",
                "parentId",
                "status",
                "startTimestamp",
                "endTimestamp",
                "error",
                "attempt",
                "isReplay"
            ]
        );
        assert_eq!(
            value(&dump, "startTimestamp"),
            Some(&serde_json::Value::from("1970-01-01T00:00:01Z"))
        );
        assert_eq!(
            value(&dump, "endTimestamp"),
            Some(&serde_json::Value::from("1970-01-01T00:00:02Z"))
        );
        assert_eq!(value(&dump, "attempt"), Some(&serde_json::Value::from(1)));
        assert_eq!(
            value(&dump, "parentId"),
            Some(&serde_json::Value::from("p"))
        );
        assert_eq!(
            value(&dump, "error"),
            Some(&serde_json::Value::from("boom"))
        );
    }
}
