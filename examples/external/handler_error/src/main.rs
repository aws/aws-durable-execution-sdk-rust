//! Handler-level error: returning `Err` from the handler fails the execution.
//!
//! A durable operation error can be caught and recovered inside the handler
//! (see `retry_exhaustion` and `create_callback_timeout`). An error returned
//! from the handler itself is different: it is the execution's final outcome.
//! There is no further operation to recover it, so the durable execution
//! terminates in a **failed** state carrying the returned error.
//!
//! This example runs one durable step that succeeds and is checkpointed, then
//! returns an error from the handler. On replay the step's recorded result is
//! restored, the handler runs again to the same point, and the execution ends
//! failed with the same error every time: the failure is deterministic. This
//! is the intended terminal state for this example, not a defect.

use aws_durable_execution_sdk as durable;

/// Completes a durable step, then returns an error as the execution's outcome.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // A durable step that succeeds and is checkpointed before the failure.
    ctx.step(|_| async { Ok("prepared".to_owned()) })
        .name("prepare")
        .await?;

    // Returning an error from the handler fails the whole execution. Unlike a
    // caught operation error, this is terminal.
    Err("handler failed after preparing".into())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
