//! `race`: take the first settled result, success or failure.
//!
//! [`race`] returns the first operation to settle — whether it succeeds or
//! fails. It is the durable analogue of `Promise.race`. The losing operations
//! are dropped (cancelled) once the winner settles.
//!
//! Racing is inherently nondeterministic, so the winner must be recorded:
//! `race` is a checkpointed operation that freezes which operation won. On
//! replay the recorded winner is returned without re-racing, so a resumed
//! execution can never pick a different winner than the original run. This is
//! why durable code must use `race` rather than a native `tokio::select!` over
//! durable operations, whose winner would not be recorded.
//!
//! This example races two steps and returns the winner.
//!
//! [`race`]: aws_durable_execution_sdk_rust::DurableContext::race

use aws_durable_execution_sdk_rust as durable;

/// Returns the first step to settle; the winner is checkpointed for replay.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let first = ctx
        .step(|_| async { Ok("first".to_owned()) })
        .name("first")
        .future();
    let second = ctx
        .step(|_| async { Ok("second".to_owned()) })
        .name("second")
        .future();

    let winner = ctx.race([first, second]).name("fastest").await?;
    Ok(winner)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
