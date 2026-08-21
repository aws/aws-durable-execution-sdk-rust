//! Replay-suppressed logging around a durable wait.
//!
//! When the SDK's [`ReplayFilterLayer`] is installed, user log events emitted
//! while replaying already-checkpointed work are suppressed, so a log line is
//! written once — on the invocation that first executed it — not again on every
//! resume. This example logs before a durable wait, then after it resumes: the
//! "before" line is emitted on the first invocation, and the "after" line on
//! the resumed invocation, each exactly once across the whole execution.
//!
//! [`ReplayFilterLayer`]: aws_durable_execution_sdk_rust::ReplayFilterLayer

use std::time::Duration;

use aws_durable_execution_sdk_rust::{self as durable, ReplayFilterLayer};
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Logs before and after a durable wait to show replay-suppressed emission.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    tracing::info!("before the wait");
    ctx.wait(Duration::from_secs(2)).name("pause").await?;
    tracing::info!("after the wait (resumed invocation)");
    let outcome = ctx
        .step(|_| async {
            tracing::info!("inside a post-wait step");
            Ok("done".to_owned())
        })
        .name("finish")
        .await?;
    Ok(outcome)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    // Install a JSON subscriber with the replay filter so that log events
    // inside replayed spans are suppressed.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_filter(ReplayFilterLayer::new()),
        )
        .init();

    durable::run(handler).await
}
