//! Conformance requirement 13-4: `race` task ownership (issue #7).
//!
//! Without task-ownership blessing, `race` branches fail with the
//! ownership error instead of settling normally. Both branches return the
//! same value so the winner is deterministic regardless of scheduling.

use aws_durable_execution_sdk_rust as durable;

/// Handler: races two steps that both return "done".
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let a = ctx
        .step(|_| async { Ok("done".to_owned()) })
        .name("left")
        .future();
    let b = ctx
        .step(|_| async { Ok("done".to_owned()) })
        .name("right")
        .future();

    let winner = ctx.race([a, b]).name("race").await?;
    Ok(serde_json::Value::String(winner))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
