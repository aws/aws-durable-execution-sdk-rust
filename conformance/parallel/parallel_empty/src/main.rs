//! Conformance requirement 8-5: Parallel invoked with an empty branches list
//! completes immediately.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};

/// Handler: parallel with empty branches.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![];

    let results: Vec<String> = ctx.parallel(branches).name("empty").await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
