//! Conformance requirement 9-2: Map invoked with the items-only form (no name
//! argument), each item returns directly.

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;

/// Handler: map without a name, items doubled.
async fn handler(
    event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items: Vec<i64> = serde_json::from_value(event).unwrap_or_else(|_| vec![1, 2]);

    let results: Vec<i64> = ctx
        .map(items, |_child, item, _idx| async move { Ok(item * 2) })
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
