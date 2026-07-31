//! Companion target for the chained-invoke examples: a durable function that
//! echoes its input after a brief durable wait.
//!
//! `invoke_simple` and `comprehensive` call this function with
//! [`DurableContext::invoke`](aws_durable_execution_sdk_rust::DurableContext::invoke).
//! It is deployed in the same stack as the callers, which name it through the
//! `TARGET_FUNCTION_NAME` environment variable. A chained callee is itself an
//! ordinary durable function — nothing special marks it as a target.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

/// Echoes the input event unchanged after a one-second durable wait.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(1)).name("settle").await?;
    Ok(event)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
