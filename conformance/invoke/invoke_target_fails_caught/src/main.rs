//! Conformance requirement 5-6: invoke target fails, caught, execution succeeds.

use aws_durable_execution_sdk as durable;

/// Handler: invoke target that errors, catch InvokeError, return fallback.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx.invoke::<String, _>(&target, event).await;
    match result {
        Ok(v) => Ok(v),
        Err(err) => {
            // Check if it's an InvokeError (target failed): catch it.
            if matches!(err.kind(), durable::OperationErrorKind::Invoke(_)) {
                Ok("fallback".to_owned())
            } else {
                Err(err.to_string().into())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
