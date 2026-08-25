//! Conformance requirement 1-9: replay skips succeeded step.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: step that logs "step executed", followed by wait. Replay returns
/// the cached value without re-executing.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: String = ctx
        .step(|_step_ctx| async {
            tracing::info!("step executed");
            Ok("cached_value".to_owned())
        })
        .await?;
    ctx.wait(Duration::from_secs(1)).await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
