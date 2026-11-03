//! `join_all`: collect every outcome, successes and failures alike.
//!
//! [`join_all`] runs operations concurrently and returns a `Result` per
//! operation: `Ok` with the value or `Err` with the error, never failing
//! fast. It is the durable analogue of `Promise.allSettled`, for when you
//! want every result regardless of individual failures. Errors inside the item results are preserved through
//! checkpointing, so replay reproduces the same successes and failures.
//!
//! An `Err` item is a child-context error that carries the input's recorded
//! failure in its `wire()` record, on the first run and on replay alike. So
//! match on the `wire()` fields rather than on the kind of the input that
//! failed.
//!
//! This example joins one succeeding and one failing step and summarizes both
//! outcomes, completing successfully because `join_all` does not propagate the
//! failure. Its output is `ok:1,err:operation error: child_context`.
//!
//! [`join_all`]: aws_durable_execution_sdk::DurableContext::join_all

use aws_durable_execution_sdk as durable;

/// Joins a succeeding and a failing step, summarizing every outcome.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let good = ctx.step(|_| async { Ok(1u32) }).name("good").future();
    let bad = ctx
        .step(|_| async { Err::<u32, durable::BoxError>("boom".into()) })
        .name("bad")
        .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
        .future();

    let outcomes = ctx.join_all([good, bad]).name("collect").await?;
    let summary: Vec<String> = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            Ok(value) => format!("ok:{value}"),
            Err(err) => format!("err:{err}"),
        })
        .collect();
    Ok(summary.join(","))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::init_default_subscriber()?;
    durable::run(handler).await
}
