//! Conformance requirement 5-5: invoke target fails, execution fails.

use aws_durable_execution_sdk_rust as durable;

/// Handler: invoke target that errors: propagates InvokeError.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx.invoke::<String, _>(&target, event).await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
