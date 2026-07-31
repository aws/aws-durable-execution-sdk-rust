//! `join_all`: collect every outcome, successes and failures alike.
//!
//! [`join_all`] runs operations concurrently and returns a
//! [`Settled`] per operation — [`Fulfilled`](aws_durable_execution_sdk_rust::Settled::Fulfilled)
//! with the value or [`Rejected`](aws_durable_execution_sdk_rust::Settled::Rejected)
//! with the error — never failing fast. It is the durable analogue of
//! `Promise.allSettled`, for when you want every result regardless of
//! individual failures. Errors inside the settled results are preserved through
//! checkpointing, so replay reproduces the same successes and failures.
//!
//! This example joins one succeeding and one failing step and summarizes both
//! outcomes, completing successfully because `join_all` does not propagate the
//! failure.
//!
//! [`join_all`]: aws_durable_execution_sdk_rust::DurableContext::join_all
//! [`Settled`]: aws_durable_execution_sdk_rust::Settled

use aws_durable_execution_sdk_rust as durable;
use durable::Settled;

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

    let settled = ctx.join_all([good, bad]).name("collect").await?;
    let summary: Vec<String> = settled
        .into_iter()
        .map(|outcome| match outcome {
            Settled::Fulfilled(value) => format!("ok:{value}"),
            Settled::Rejected(err) => format!("err:{err}"),
            // `Settled` is non_exhaustive; treat any future variant as unknown.
            _ => "unknown".to_owned(),
        })
        .collect();
    Ok(summary.join(","))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
