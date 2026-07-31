//! Map with a bounded concurrency limit.
//!
//! [`max_concurrency`] caps how many items run at once. This matters at scale:
//! mapping thousands of items without a bound would start them all
//! simultaneously and can overwhelm a downstream dependency. With a cap, at
//! most N items are in flight; as each finishes, the next starts. A suspended
//! item still holds its slot, so the effective in-flight count never exceeds
//! the cap.
//!
//! This example squares eight numbers with a concurrency cap of three and sums
//! the results, standing in for a larger high-concurrency workload.
//!
//! [`max_concurrency`]: aws_durable_execution_sdk_rust::MapBuilder::max_concurrency

use aws_durable_execution_sdk_rust as durable;

/// Maps over eight numbers with a concurrency cap of three, returning their
/// sum of squares.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<u32, durable::BoxError> {
    let items: Vec<u32> = (0..8).collect();

    let results = ctx
        .map(items, |child, item, _idx| async move {
            child
                .step(move |_| async move { Ok(item * item) })
                .name("square")
                .await
                .map_err(Into::into)
        })
        .name("bounded-map")
        .max_concurrency(3)
        .await?;
    Ok(results.into_iter().sum())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
