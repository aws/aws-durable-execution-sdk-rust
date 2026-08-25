//! Conformance requirement 2-2: wait with name.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: a wait operation with an explicit name parameter.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(2))
        .name("custom_wait_name")
        .await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
