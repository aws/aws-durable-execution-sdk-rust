//! Conformance requirement 8-17: Parallel min-successful not reached (all
//! branches run).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: parallel with min-successful=3 but one branch fails.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Ok("ok0".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail".into()) }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
    ];

    let batch = ctx
        .parallel(branches)
        .name("min-not-reached")
        .max_concurrency(1)
        .completion(CompletionConfig::with_min_successful(3))
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
