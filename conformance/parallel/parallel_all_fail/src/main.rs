//! Conformance requirement 8-16: Parallel where all branches fail (within
//! tolerance).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: all three branches fail, tolerance is 3.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Err("fail0".into()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail1".into()) }),
        Branch::new("2", |_: DurableContext| async { Err("fail2".into()) }),
    ];

    let batch = ctx
        .parallel(branches)
        .name("all-fail")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(3))
        .await_batch()
        .await?;

    Ok(serde_json::json!({
        "completionReason": batch.reason.as_str(),
        "status": batch.status().as_str(),
        "successCount": batch.success_count(),
        "failureCount": batch.failure_count(),
        "totalCount": batch.total_count(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
