//! Conformance requirement 1-3: sequential steps where second depends on first.

use aws_durable_execution_sdk_rust as durable;

/// Handler: two sequential steps, second uses first's result.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let first: String = ctx.step(|_| async { Ok("first".to_owned()) }).await?;
    let second: String = ctx
        .step(move |_| async move { Ok(format!("{first}_second")) })
        .await?;
    Ok(serde_json::Value::String(second))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
