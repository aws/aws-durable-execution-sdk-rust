//! Conformance requirement 8-10: Parallel stops early once the failure count
//! exceeds the tolerated-failure-count.

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: parallel with tolerated-failure-count=1, two failures exceed it.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Err("fail0".into()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail1".into()) }),
        Branch::new("2", |_: DurableContext| async { Ok("never".to_owned()) }),
    ];

    let result = ctx
        .parallel(branches)
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
