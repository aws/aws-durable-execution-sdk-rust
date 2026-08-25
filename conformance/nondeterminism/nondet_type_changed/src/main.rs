//! Conformance requirement 12-2: operation changes type on replay (issue #6).
//!
//! Deliberately violates the determinism contract: the first execution
//! records a step, the replay claims a wait at the same position. The SDK
//! must detect the type mismatch and fail the execution.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: first operation is a step on first execution, a wait on replay.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    if ctx.is_replaying() {
        // Replay path: a wait where a step was recorded.
        ctx.wait(Duration::from_secs(2)).name("flip").await?;
    } else {
        ctx.step(|_| async { Ok("recorded-as-step".to_owned()) })
            .name("flip")
            .await?;
    }

    // Suspend so the execution is re-invoked and replays.
    ctx.wait(Duration::from_secs(1)).await?;

    Ok(serde_json::Value::String("should-not-complete".to_owned()))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
