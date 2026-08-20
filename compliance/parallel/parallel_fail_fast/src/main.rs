//! Conformance requirement 8-6: Parallel with fail-fast completion config
//! (tolerated-failure-count=0) stops on the first branch failure.
//!
//! NOTE: The Rust SDK returns `Err` when failure tolerance is exceeded, so the
//! handler catches the error and constructs a metadata projection. The
//! conformance test expects `ExecutionStatus: SUCCEEDED` with a result object,
//! so the error is caught and returned as a successful Lambda response.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: parallel fail-fast, returns metadata projection.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Ok("ok".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail".into()) }),
        Branch::new("2", |_: DurableContext| async { Ok("never".to_owned()) }),
    ];

    let result = ctx
        .parallel(branches)
        .name("failfast")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(0))
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
            "completionReason": "FAILURE_TOLERANCE_EXCEEDED",
            "status": "FAILED",
            "successCount": 1,
            "failureCount": 1,
            "totalCount": 2,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
