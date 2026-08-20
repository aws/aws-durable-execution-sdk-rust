//! Conformance requirement 9-8: Map with tolerated-failure-count=1 tolerating
//! one failure, all items complete.

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;
use durable::builders::map_parallel::CompletionConfig;

/// Handler: map tolerating one failure.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["ok".to_owned(), "fail".to_owned(), "ok2".to_owned()];

    let result = ctx
        .map(items, |_child, item, _idx| async move {
            if item == "fail" {
                return Err("item failed".into());
            }
            Ok(item)
        })
        .name("tolerated")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await_batch()
        .await?;

    Ok(serde_json::json!({
        "completionReason": result.reason.as_str(),
        "status": if result.has_failure() { "FAILED" } else { "SUCCEEDED" },
        "successCount": result.success_count(),
        "failureCount": result.failure_count(),
        "totalCount": result.total_count(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
