//! Conformance requirement 2-4: wait with different duration units.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: a wait using minutes, verifying conversion to seconds.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // 1 minute = 60 seconds on the wire.
    ctx.wait(Duration::from_secs(60)).await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
