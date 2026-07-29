//! Chained durable invocation: call another durable function and use its result.
//!
//! [`DurableContext::invoke`](aws_durable_execution_sdk_rust::DurableContext::invoke)
//! starts a child durable execution of a different function and checkpoints its
//! result. On replay the recorded result is returned without re-invoking the
//! callee, so the call happens exactly once across the whole execution.
//!
//! The target function is named through the `TARGET_FUNCTION_NAME` environment
//! variable so the same handler works against any deployed callee. The output
//! type is turbofished; the input type is inferred.

use aws_durable_execution_sdk_rust as durable;

/// Invokes the echo target with the incoming event and returns its result.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let receipt = ctx
        .invoke::<serde_json::Value, _>(&target, event)
        .name("delegate")
        .await?;
    Ok(receipt)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
