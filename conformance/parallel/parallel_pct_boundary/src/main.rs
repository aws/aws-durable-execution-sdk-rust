//! Conformance requirement 8-22: Parallel tolerated-failure-percentage at the
//! boundary (exactly 25% with one failure in four branches).

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: 4 branches, one fails, boundary pct=25.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Err("fail".into()) }),
        Branch::new("1", |_: DurableContext| async { Ok("ok1".to_owned()) }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
        Branch::new("3", |_: DurableContext| async { Ok("ok3".to_owned()) }),
    ];

    let batch = ctx
        .parallel(branches)
        .name("pct-boundary")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_percentage(25))
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
