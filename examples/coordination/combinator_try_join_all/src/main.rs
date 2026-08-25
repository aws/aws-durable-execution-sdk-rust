//! `try_join_all`: gather all results, fail fast on the first error.
//!
//! [`try_join_all`] runs a set of durable operations concurrently and returns
//! their results as a `Vec` in input order, failing fast the moment any one
//! errors. It is the durable analogue of JavaScript's `Promise.all`, and, like
//! every combinator here, is itself a checkpointed operation, so its combined
//! result is frozen and replayed deterministically.
//!
//! Combinators take uniform [`DurableFuture`]s. Convert a builder to one with
//! `.future()`; any operation works, including waits, so a "join then wait"
//! shape needs no special support.
//!
//! This example gathers three steps into a single result vector.
//!
//! [`try_join_all`]: aws_durable_execution_sdk::DurableContext::try_join_all
//! [`DurableFuture`]: aws_durable_execution_sdk::DurableFuture

use aws_durable_execution_sdk as durable;

/// Gathers three concurrent steps, failing fast if any errors.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Vec<u32>, durable::BoxError> {
    let a = ctx.step(|_| async { Ok(1u32) }).name("a").future();
    let b = ctx.step(|_| async { Ok(2u32) }).name("b").future();
    let c = ctx.step(|_| async { Ok(3u32) }).name("c").future();

    let results = ctx.try_join_all([a, b, c]).name("gather").await?;
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
