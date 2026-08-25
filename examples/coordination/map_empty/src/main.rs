//! Map over an empty collection: the zero-item edge case.
//!
//! Mapping an empty collection is legal and completes immediately with an empty
//! result vector. As with an empty parallel, handling zero items without a
//! special case keeps calling code simple when the input may be empty.

use aws_durable_execution_sdk as durable;

/// Maps over an empty collection and returns the result count.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<usize, durable::BoxError> {
    let items: Vec<u32> = Vec::new();

    let results = ctx
        .map(items, |child, item, _idx| async move {
            child
                .step(move |_| async move { Ok(item) })
                .name("noop")
                .await
                .map_err(Into::into)
        })
        .name("empty-map")
        .await?;
    Ok(results.len())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
