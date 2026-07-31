//! Conformance requirement 8-17: Parallel min-successful not reached (all
//! branches run).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, CompletionConfig, DurableContext};

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

    let result = ctx
        .parallel(branches)
        .name("min-not-reached")
        .max_concurrency(1)
        .completion(CompletionConfig::with_min_successful(3))
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
