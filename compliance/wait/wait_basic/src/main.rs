//! Conformance requirement 2-1: wait basic.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: a single wait operation with a specified duration.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(2)).await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
