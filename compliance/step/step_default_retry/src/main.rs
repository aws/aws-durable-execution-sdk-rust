//! Conformance requirement 1-13: default retry strategy.

use aws_durable_execution_sdk_rust as durable;

/// Handler: step that fails on first two attempts and succeeds on third,
/// using the SDK's default retry (no explicit config). Tracks via DDB.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let execution_id = ctx.execution_arn().to_owned();

    let result: String = ctx
        .step(move |_sc| async move {
            let count = compliance::increment_attempt(&execution_id).await?;
            if count < 3 {
                return Err(format!("Attempt {count} failed").into());
            }
            Ok("recovered".to_owned())
        })
        .await?;

    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
