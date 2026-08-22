//! Conformance requirement 9-3: Map function receives both the item and its
//! zero-based index and uses both.

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;

/// Handler: map that adds item + index.
async fn handler(
    event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items: Vec<i64> = serde_json::from_value(event).unwrap_or_else(|_| vec![10, 20, 30]);

    let results: Vec<i64> = ctx
        .map(items, |_child, item, idx| async move {
            #[expect(clippy::cast_possible_wrap)]
            Ok(item + idx as i64)
        })
        .name("indexed")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
