//! Parallel completion policy: complete early once enough branches succeed.
//!
//! A [`CompletionConfig`] controls when a parallel (or map) batch is considered
//! done. Its `min_successful` field completes the batch as soon as that many
//! branches succeed, without waiting for the rest — useful for quorum or
//! first-N-of-M patterns. The same struct also carries failure-tolerance knobs
//! (`tolerated_failure_count`, `tolerated_failure_percentage`) for bounding how
//! many branch failures may pass before the batch aborts.
//!
//! This example runs three branches under a policy that completes once two
//! succeed, returning the results gathered by that point.
//!
//! [`CompletionConfig`]: aws_durable_execution_sdk_rust::CompletionConfig

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, CompletionConfig};

/// Runs three branches, completing as soon as two succeed.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<usize, durable::BoxError> {
    let branches: Vec<Branch<u32>> = (0..3u32)
        .map(|i| {
            Branch::new(format!("branch-{i}"), move |child| async move {
                child
                    .step(move |_| async move { Ok(i) })
                    .name("work")
                    .await
                    .map_err(Into::into)
            })
        })
        .collect();

    let results = ctx
        .parallel(branches)
        .name("quorum-fan-out")
        .completion(CompletionConfig::with_min_successful(2))
        .await?;
    Ok(results.len())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
