//! Conformance requirement 5-9: replay skips succeeded invoke.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: invoke then wait — on replay the invoke returns cached result.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx.invoke::<serde_json::Value, _>(&target, event).await?;
    ctx.wait(Duration::from_secs(1)).await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
