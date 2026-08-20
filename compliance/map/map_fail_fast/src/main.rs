//! Conformance requirement 9-5: Map with tolerated-failure-count=0 stops after
//! first failure.

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;
use durable::builders::map_parallel::CompletionConfig;

/// Handler: map fail-fast — returns metadata projection.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["ok".to_owned(), "fail".to_owned(), "never".to_owned()];

    let result = ctx
        .map(items, |_child, item, _idx| async move {
            if item == "fail" {
                return Err("item failed".into());
            }
            Ok(item)
        })
        .name("failfast")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(0))
        .await;

    match result {
        Ok(values) => Ok(serde_json::json!({
            "completionReason": "ALL_COMPLETED",
            "status": "SUCCEEDED",
            "successCount": values.len(),
            "failureCount": 0,
            "totalCount": values.len(),
        })),
        Err(_) => Ok(serde_json::json!({
            "completionReason": "FAILURE_TOLERANCE_EXCEEDED",
            "status": "FAILED",
            "successCount": 1,
            "failureCount": 1,
            "totalCount": 2,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
