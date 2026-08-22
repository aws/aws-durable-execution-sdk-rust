//! Conformance requirement 1-8: step and wait with replay.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: step followed by wait; replay skips the completed step.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: String = ctx.step(|_| async { Ok("computed".to_owned()) }).await?;
    ctx.wait(Duration::from_secs(2)).await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
