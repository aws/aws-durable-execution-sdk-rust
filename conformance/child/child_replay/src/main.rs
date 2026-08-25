//! Conformance requirement 3-9: child replay (child then wait).

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: child followed by wait; on replay child returns cached result.
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

    ctx.wait(Duration::from_secs(2)).await?;

    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
