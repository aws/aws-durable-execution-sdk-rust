//! Conformance requirement 5-4: invoke with null payload, target returns null.

use aws_durable_execution_sdk_rust as durable;

/// Handler: invoke with null input.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result: serde_json::Value = ctx
        .invoke::<serde_json::Value, _>(&target, serde_json::Value::Null)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
