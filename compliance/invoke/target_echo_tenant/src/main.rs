//! Target function with tenancy: waits briefly then echoes its input.
//! Identical to target_echo; the tenancy configuration is in the SAM template.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Echo handler with tenancy: wait 1s then return the input unchanged.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(1)).await?;
    Ok(event)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
