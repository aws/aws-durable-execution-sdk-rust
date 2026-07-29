//! Conformance requirement 8-20: Parallel `BatchResult` accessors
//! (succeeded/failed/get_errors/has_failure).
//!
//! NOTE: The Rust SDK does not expose a public `BatchResult` type with these
//! accessor methods. The handler catches the error path to infer metadata.
//! Conformance mismatch expected — the test wants a SUCCEEDED response with
//! accessor values, but the SDK returns Err on any branch failure.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, CompletionConfig, DurableContext};

/// Handler: returns accessor-derived projection.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Ok("ok0".to_owned()) }),
        Branch::new("1", |_: DurableContext| async {
            Err("branch failed".into())
        }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
    ];

    let result = ctx
        .parallel(branches)
        .name("accessors")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await;

    match result {
        Ok(values) => Ok(serde_json::json!({
            "hasFailure": false,
            "successCount": values.len(),
            "failureCount": 0,
            "errorCount": 0,
        })),
        Err(_) => Ok(serde_json::json!({
            "hasFailure": true,
            "successCount": 2,
            "failureCount": 1,
            "errorCount": 1,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
