//! Conformance requirement 9-18: Suspension after a map that completed with a
//! failure; on replay the completed map (including the failed iteration) is
//! skipped.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;
use durable::builders::map_parallel::CompletionConfig;

/// Handler: map with one failure (tolerated), then a wait.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["ok".to_owned(), "fail".to_owned()];

    let result = ctx
        .map(items, |_child, item, _idx| async move {
            if item == "fail" {
                return Err("item failed".into());
            }
            Ok(item)
        })
        .name("fail-then-wait")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await_batch()
        .await?;

    let projection = serde_json::json!({
        "completionReason": result.reason.as_str(),
        "status": if result.has_failure() { "FAILED" } else { "SUCCEEDED" },
        "successCount": result.success_count(),
        "failureCount": result.failure_count(),
        "totalCount": result.total_count(),
    });

    ctx.wait(Duration::from_secs(1)).await?;

    Ok(projection)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
