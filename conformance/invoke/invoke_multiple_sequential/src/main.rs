//! Conformance requirement 5-14: multiple sequential invokes.

use aws_durable_execution_sdk as durable;

/// Handler: two sequential invokes, second consumes the first's result.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target1 = std::env::var("TARGET_FUNCTION_NAME_1")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let target2 = std::env::var("TARGET_FUNCTION_NAME_2")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let first = ctx.invoke::<serde_json::Value, _>(&target1, event).await?;
    let second = ctx.invoke::<serde_json::Value, _>(&target2, first).await?;
    Ok(second)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
