//! Conformance requirement 8-13: Parallel stops early once the failure
//! percentage exceeds the tolerated-failure-percentage.

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: parallel with tolerated-failure-percentage=25, two of four fail.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Err("fail0".into()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail1".into()) }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
        Branch::new("3", |_: DurableContext| async { Ok("ok3".to_owned()) }),
    ];

    let result = ctx
        .parallel(branches)
        .name("tolerated-pct")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_percentage(25))
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
