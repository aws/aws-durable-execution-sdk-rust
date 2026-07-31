//! Parallel: run named branches concurrently.
//!
//! [`parallel`] takes a list of named [`Branch`]es and runs them concurrently,
//! each in its own child context. Because every operation's identity is claimed
//! when the branch is built (not when it is polled), the branches make progress
//! in any order without their operation identities colliding, and replay pairs
//! each result to the right branch deterministically. `parallel` returns the
//! branch results in input order.
//!
//! A branch body is ordinary durable code: it can run steps, waits, or nested
//! operations. This example fans out two branches — one that computes a value
//! and one that waits before returning — showing that heterogeneous work with
//! the same result type composes cleanly.
//!
//! [`parallel`]: aws_durable_execution_sdk_rust::DurableContext::parallel
//! [`Branch`]: aws_durable_execution_sdk_rust::Branch

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::Branch;

/// Fans out two branches concurrently and returns their results in order.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Vec<u32>, durable::BoxError> {
    let branches: Vec<Branch<u32>> = vec![
        Branch::new("double", |child| async move {
            let base = child.step(|_| async { Ok(21u32) }).name("compute").await?;
            Ok(base * 2)
        }),
        Branch::new("wait-then-value", |child| async move {
            child.wait(Duration::from_secs(1)).name("cooldown").await?;
            Ok(100u32)
        }),
    ];

    let results = ctx.parallel(branches).name("fan-out").await?;
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
