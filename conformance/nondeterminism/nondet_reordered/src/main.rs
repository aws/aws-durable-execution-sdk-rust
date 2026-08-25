//! Conformance requirement 12-1: reordered operations on replay (issue #6).
//!
//! Deliberately violates the determinism contract: the replay path creates
//! the two steps in the opposite order from the first execution. The SDK
//! must detect the identity mismatch (the first claim, "beta", lands on
//! the record checkpointed for "alpha") and fail the execution instead of
//! silently replaying the wrong recorded result.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: swaps step order between first execution and replay.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    if ctx.is_replaying() {
        // Replay path: intentionally reordered.
        ctx.step(|_| async { Ok("b".to_owned()) })
            .name("beta")
            .await?;
        ctx.step(|_| async { Ok("a".to_owned()) })
            .name("alpha")
            .await?;
    } else {
        ctx.step(|_| async { Ok("a".to_owned()) })
            .name("alpha")
            .await?;
        ctx.step(|_| async { Ok("b".to_owned()) })
            .name("beta")
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
