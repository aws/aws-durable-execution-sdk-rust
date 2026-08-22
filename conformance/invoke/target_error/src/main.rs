//! Target function that fails: waits briefly then returns an error.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Error handler: wait 1s then fail.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(1)).await?;
    Err("target function error".into())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
