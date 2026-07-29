//! Conformance requirement 8-22: Parallel tolerated-failure-percentage at the
//! boundary (exactly 25% with one failure in four branches).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, CompletionConfig, DurableContext};

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

    let result = ctx
        .parallel(branches)
        .name("pct-boundary")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_percentage(25))
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
            "successCount": 3,
            "failureCount": 1,
            "totalCount": 4,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
