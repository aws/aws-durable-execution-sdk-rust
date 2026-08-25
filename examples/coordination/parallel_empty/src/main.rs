//! Parallel with no branches: the empty edge case.
//!
//! An empty branch list is legal and completes immediately with an empty result
//! vector. Handling zero elements without a special case keeps calling code
//! simple when the branch set is computed from data that may be empty.

use aws_durable_execution_sdk as durable;
use durable::Branch;

/// Runs a parallel operation with no branches and returns the result count.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<usize, durable::BoxError> {
    let branches: Vec<Branch<u32>> = Vec::new();
    let results = ctx.parallel(branches).name("empty-fan-out").await?;
    Ok(results.len())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
