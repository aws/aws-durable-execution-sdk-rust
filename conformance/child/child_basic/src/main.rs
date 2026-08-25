//! Conformance requirement 3-1: child context with a single step.

use aws_durable_execution_sdk as durable;

/// Handler: child context with a single step returning the input.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let v: serde_json::Value = child_ctx
                .step(move |_| {
                    let e = event.clone();
                    async move { Ok(e) }
                })
                .await?;
            Ok(v)
        })
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
