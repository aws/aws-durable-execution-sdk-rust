//! Map completion policy: tolerate failures across items.
//!
//! Map accepts the same [`CompletionConfig`] as parallel. Its fields let a
//! bounded number of item failures pass (`tolerated_failure_count` /
//! `tolerated_failure_percentage`) or complete the batch early once enough
//! items succeed (`min_successful`). Each item's error — including its type —
//! is preserved for inspection and survives replay, so a later invocation sees
//! the same failure it saw the first time.
//!
//! This example maps four items, one of which deliberately fails, tolerating a
//! single failure. The batch completes with the three successful results.
//!
//! [`CompletionConfig`]: aws_durable_execution_sdk_rust::CompletionConfig

use aws_durable_execution_sdk_rust as durable;
use durable::CompletionConfig;

/// Maps four items under a fault-tolerant policy (one failure allowed) and
/// returns how many succeeded.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<usize, durable::BoxError> {
    let items: Vec<u32> = (0..4).collect();

    let results = ctx
        .map(items, |child, item, _idx| async move {
            if item == 0 {
                // One item fails; the completion policy tolerates it.
                return child
                    .step(|_| async { Err::<u32, durable::BoxError>("item 0 failed".into()) })
                    .name("work")
                    .retry_strategy(Box::new(|_err, _attempt| durable::RetryDecision::Stop))
                    .await
                    .map_err(Into::into);
            }
            child
                .step(move |_| async move { Ok(item) })
                .name("work")
                .await
                .map_err(Into::into)
        })
        .name("tolerant-map")
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await?;
    Ok(results.len())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
