//! Conformance requirement 3-3: child context with multiple sequential steps.

use aws_durable_execution_sdk_rust as durable;

/// Handler: child with two sequential steps.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let r1: serde_json::Value = child_ctx
                .step(move |_| {
                    let e = event.clone();
                    async move { Ok(e) }
                })
                .await?;

            let r2: serde_json::Value = child_ctx.step(move |_| async move { Ok(r1) }).await?;
            Ok(r2)
        })
        .name("multi-step")
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
