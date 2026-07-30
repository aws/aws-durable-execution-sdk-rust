//! A spawned wait joined with a spawned step.
//!
//! `.spawn()` starts an operation eagerly on its own task. A spawned WAIT (or
//! any other parking operation) runs in its own suspension scope, so it does
//! not end the invocation the moment it parks: the invocation suspends only
//! once every spawned sibling has itself completed or parked. That is what
//! makes the "start a timer, do work alongside it" shape safe —
//!
//! ```text
//! let wait = ctx.wait(..).spawn();
//! let work = ctx.step(..).spawn();
//! let (timer, result) = tokio::join!(wait, work);
//! ```
//!
//! — the step reaches its terminal checkpoint before the invocation reports
//! PENDING on the timer, so it is not aborted mid-flight and does not
//! re-execute (duplicating its side effects) when the timer fires and the
//! execution resumes.
//!
//! On resume both operations replay from the checkpoint log, `join!` resolves,
//! and the handler returns the step's value.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

/// Joins a spawned wait with a spawned step and returns the step's value.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<i32, durable::BoxError> {
    // Both start immediately. The timer parks; the step keeps running.
    let wait = ctx.wait(Duration::from_secs(5)).name("timer").spawn();
    let work = ctx.step(|_| async { Ok(42_i32) }).name("compute").spawn();

    // The invocation suspends here (the timer has no result yet) only after
    // the step has finished. It resumes when the timer fires.
    let (timer, result) = tokio::join!(wait, work);
    timer?;
    Ok(result?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
