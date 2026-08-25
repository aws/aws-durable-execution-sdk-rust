//! Conformance requirement 12-3: removed operation on replay (issue #6).
//!
//! Deliberately violates the determinism contract: the replay path skips
//! the first recorded step, so its first claim ("process-data") lands on
//! the record checkpointed for "load-config". The SDK must detect the
//! identity mismatch and fail the execution instead of handing
//! "load-config"'s recorded result to "process-data".

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: drops the first step on replay.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    if !ctx.is_replaying() {
        ctx.step(|_| async { Ok("config".to_owned()) })
            .name("load-config")
            .await?;
    }
    // On replay this is the FIRST claimed operation and collides with the
    // "load-config" record.
    ctx.step(|_| async { Ok("data".to_owned()) })
        .name("process-data")
        .await?;

    // Suspend so the execution is re-invoked and replays.
    ctx.wait(Duration::from_secs(1)).await?;

    Ok(serde_json::Value::String("should-not-complete".to_owned()))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
