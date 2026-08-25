//! `select_ok`: take the first success, ignore earlier failures.
//!
//! [`select_ok`] returns the first operation that succeeds. Operations that
//! fail before any success are ignored; if all fail, it returns an aggregate
//! error. It is the durable analogue of `Promise.any`. When the first success
//! resolves, the still-running operations are dropped (cancelled): Rust's
//! natural cancellation. The winning result is checkpointed, so replay returns
//! the same winner.
//!
//! This example races a failing operation against a succeeding one and returns
//! the successful value.
//!
//! [`select_ok`]: aws_durable_execution_sdk::DurableContext::select_ok

use aws_durable_execution_sdk as durable;

/// Returns the first successful result, skipping a failing operation.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<u32, durable::BoxError> {
    let failing = ctx
        .step(|_| async { Err::<u32, durable::BoxError>("unavailable".into()) })
        .name("primary")
        .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
        .future();
    let succeeding = ctx.step(|_| async { Ok(42u32) }).name("fallback").future();

    let winner = ctx
        .select_ok([failing, succeeding])
        .name("first-ok")
        .await?;
    Ok(winner)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
