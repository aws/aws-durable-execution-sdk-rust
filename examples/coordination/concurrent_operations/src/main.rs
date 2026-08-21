//! Eager fan-out with `.spawn()`.
//!
//! Operation builders are lazy: a builder that is only held does no work until
//! it is awaited. `.spawn()` opts into eager execution: it starts the
//! operation immediately on its own task and returns a running
//! [`DurableFuture`](aws_durable_execution_sdk_rust::DurableFuture) you await
//! later. Several `.spawn()`ed operations therefore make progress
//! concurrently while you set up more work.
//!
//! Because each operation's identity is claimed when its builder is created,
//! before `.spawn()` starts it, identities stay deterministic no matter which
//! task runs first. `.spawn()` is the replay-safe alternative to a bare
//! `tokio::spawn` of durable work, which the SDK rejects because it would
//! escape the deterministic-identity guarantee.
//!
//! This example spawns three steps eagerly and sums their results once all have
//! completed.

use aws_durable_execution_sdk_rust as durable;

/// Spawns three steps eagerly and returns the sum of their results.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<u32, durable::BoxError> {
    // Each .spawn() starts immediately; all three run concurrently.
    let a = ctx.step(|_| async { Ok(10u32) }).name("a").spawn();
    let b = ctx.step(|_| async { Ok(20u32) }).name("b").spawn();
    let c = ctx.step(|_| async { Ok(30u32) }).name("c").spawn();

    let total = a.await? + b.await? + c.await?;
    Ok(total)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
