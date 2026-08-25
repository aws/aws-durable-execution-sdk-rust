//! Conformance requirement 3-10: child with step followed by wait inside.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: child containing a step then a wait.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let ev = event.clone();
    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let ev2 = ev.clone();
            let _: serde_json::Value = child_ctx.step(move |_| async move { Ok(ev2) }).await?;

            child_ctx.wait(Duration::from_secs(2)).await?;

            Ok(ev)
        })
        .name("mixed-ops")
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
