//! Conformance requirement 3-18: child with step+wait, then top-level step+wait.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: child (step + wait), then top-level step + wait.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let ev = event.clone();
    let _ = ctx
        .run_in_child_context(move |child_ctx| async move {
            let ev2 = ev.clone();
            let _: serde_json::Value = child_ctx.step(move |_| async move { Ok(ev2) }).await?;

            child_ctx.wait(Duration::from_secs(2)).await?;

            Ok(ev)
        })
        .name("step-wait-child")
        .await?;

    let ev3 = event.clone();
    let result: serde_json::Value = ctx.step(move |_| async move { Ok(ev3) }).await?;

    ctx.wait(Duration::from_secs(2)).await?;

    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
