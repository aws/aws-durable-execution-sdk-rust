//! Guard non-durable side effects with
//! [`DurableContext::is_replaying`](aws_durable_execution_sdk_rust::DurableContext::is_replaying).
//!
//! Code between durable operations re-runs on every resume. If a side effect is
//! not itself a durable operation (a metric emit, a one-shot log), wrap it in a
//! `!ctx.is_replaying()` guard so it happens only on the invocation that is
//! making fresh progress, not on replays. This example reports the replay flag
//! before and after a durable wait forces a suspend and resume.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

/// Reports the replay flag around a durable wait.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let before = ctx.is_replaying();

    ctx.wait(Duration::from_secs(2)).name("checkpoint").await?;

    // Only perform the non-durable side effect when making fresh progress.
    let side_effect = if ctx.is_replaying() {
        "skipped (replaying prior work)"
    } else {
        "ran (fresh progress)"
    };

    Ok(format!(
        "initial_replaying={before}; side_effect_{side_effect}"
    ))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
