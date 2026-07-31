//! Named wait: attach a stable name to a wait for observability.
//!
//! Like every operation, [`ctx.wait`] accepts an optional name via the builder
//! chain. The name surfaces in the execution history and the operation's log
//! span, so a suspended-then-resumed wait reads as `cooldown` rather than an
//! anonymous positional id. The name is metadata only and does not affect
//! operation identity.
//!
//! [`ctx.wait`]: aws_durable_execution_sdk_rust::DurableContext::wait

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

/// Waits two seconds under a human-readable name, then completes.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    tracing::info!("starting cooldown");
    ctx.wait(Duration::from_secs(2)).name("cooldown").await?;
    tracing::info!("cooldown complete");
    Ok("wait finished".to_owned())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
