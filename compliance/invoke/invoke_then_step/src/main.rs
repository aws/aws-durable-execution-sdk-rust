//! Conformance requirement 5-12: invoke then step.

use aws_durable_execution_sdk_rust as durable;

/// Handler: invoke target, then step processes the result.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let invoke_result = ctx.invoke::<serde_json::Value, _>(&target, event.clone()).await?;
    let step_result = ctx
        .step(move |_| async move { Ok(format!("processed: {}", invoke_result)) })
        .await?;
    Ok(step_result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
