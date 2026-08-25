//! Conformance requirement 1-5: step returning null/None.

use aws_durable_execution_sdk as durable;

/// Handler: step returning null.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: Option<()> = ctx.step(|_| async { Ok(None) }).await?;
    let _ = result;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
