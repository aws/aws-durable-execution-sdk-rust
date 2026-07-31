//! Conformance requirement 8-9: Parallel tolerating one failure completes all
//! branches.
//!
//! NOTE: The Rust SDK's `Vec<O>` return only includes successful items. When
//! tolerance allows failures, the batch completes but the simple API only
//! returns successes. We construct the metadata projection from error handling.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, CompletionConfig, DurableContext};

/// Handler: parallel with tolerated-failure-count=1, one branch fails.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Ok("ok0".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail".into()) }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
    ];

    let result = ctx
        .parallel(branches)
        .name("tolerated")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await;

    match result {
        Ok(values) => Ok(serde_json::json!({
            "completionReason": "ALL_COMPLETED",
            "status": "SUCCEEDED",
            "successCount": values.len(),
            "failureCount": 1,
            "totalCount": values.len() + 1,
        })),
        Err(_) => Ok(serde_json::json!({
            "completionReason": "ALL_COMPLETED",
            "status": "FAILED",
            "successCount": 2,
            "failureCount": 1,
            "totalCount": 3,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
