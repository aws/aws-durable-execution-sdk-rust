//! Conformance requirement 3-13: child with wait, followed by top-level step.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: child with a wait, then top-level step.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let ev = event.clone();
    let _ = ctx
        .run_in_child_context(move |child_ctx| async move {
            child_ctx.wait(Duration::from_secs(1)).await?;
            Ok(ev)
        })
        .name("wait-child")
        .await?;

    let ev2 = event.clone();
    let result: serde_json::Value = ctx.step(move |_| async move { Ok(ev2) }).await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
