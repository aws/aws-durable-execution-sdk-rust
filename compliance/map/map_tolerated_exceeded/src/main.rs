//! Conformance requirement 9-9: Map stops early once the failure count exceeds
//! tolerated-failure-count.

use aws_durable_execution_sdk_rust as durable;
use durable::{CompletionConfig, DurableContext};

/// Handler: map with two failures exceeding tolerance of 1.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["fail0".to_owned(), "fail1".to_owned(), "never".to_owned()];

    let result = ctx
        .map(items, |_child, item, _idx| async move {
            if item != "never" {
                return Err("item failed".into());
            }
            Ok(item)
        })
        .name("tolerated-exceeded")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await;

    match result {
        Ok(values) => Ok(serde_json::json!({
            "completionReason": "ALL_COMPLETED",
            "successCount": values.len(),
            "failureCount": 0,
            "totalCount": values.len(),
        })),
        Err(_) => Ok(serde_json::json!({
            "completionReason": "FAILURE_TOLERANCE_EXCEEDED",
            "successCount": 0,
            "failureCount": 2,
            "totalCount": 2,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
