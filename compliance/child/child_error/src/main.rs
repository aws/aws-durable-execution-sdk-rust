//! Conformance requirement 3-4: child context step failure (no retry).

use aws_durable_execution_sdk_rust as durable;

/// Handler: child with a failing step (no retry).
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: String = ctx
        .run_in_child_context(|child_ctx| async move {
            let v: String = child_ctx
                .step(|_| async move { Err("Child step failed".into()) })
                .retry_strategy(Box::new(|_, _| durable::RetryDecision::Stop))
                .await?;
            Ok(v)
        })
        .name("failing-child")
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
