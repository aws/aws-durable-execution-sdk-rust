//! Conformance requirement 5-10: replay re-throws failed invoke.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: invoke (fails), catch, wait, return "caught and continued".
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
            if matches!(err.kind(), durable::OperationErrorKind::Invoke(_)) {
                // Caught: continue with a wait then return.
                ctx.wait(Duration::from_secs(1)).await?;
                Ok("caught and continued".to_owned())
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
