//! Conformance requirement 2-5: wait with long duration (1 hour).

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: a wait with a 1-hour duration, verifying conversion to seconds.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // 1 hour = 3600 seconds on the wire.
    ctx.wait(Duration::from_secs(3600)).await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
