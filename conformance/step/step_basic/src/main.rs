//! Conformance requirement 1-1: step basic (succeeds on first attempt).

use aws_durable_execution_sdk as durable;

/// Handler: single step that returns a greeting.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let name = event.as_str().unwrap_or("World").to_owned();
    let result = ctx
        .step(move |_| async move { Ok(format!("Hello, {name}!")) })
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
