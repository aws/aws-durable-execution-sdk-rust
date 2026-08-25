//! Conformance requirement 5-11: step then invoke.

use aws_durable_execution_sdk as durable;

/// Handler: step computes a payload, then invoke sends it to the target.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let step_result = ctx
        .step(|_| async move { Ok("step result".to_owned()) })
        .await?;
    let invoke_result = ctx.invoke::<String, _>(&target, step_result).await?;
    Ok(invoke_result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
