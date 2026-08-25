//! Conformance requirement 8-7: Parallel where the handler rethrows,
//! propagating a branch failure.
//!
//! The Rust SDK automatically propagates failures as `Err`, so this handler
//! simply lets the error bubble up (equivalent to Go's `ThrowIfError()`).

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: parallel that propagates the first branch failure.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async {
            Err("branch failed".into())
        }),
        Branch::new("1", |_: DurableContext| async { Ok("never".to_owned()) }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("throwing")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(0))
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
