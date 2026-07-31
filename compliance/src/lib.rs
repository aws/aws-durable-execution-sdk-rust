//! Shared utilities for conformance test handlers.
//!
//! This crate holds helpers consumed by individual handler binaries in the
//! `compliance/` directory tree. It is not published to crates.io.

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
