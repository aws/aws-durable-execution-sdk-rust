//! Conformance requirement 9-13: Map with a custom item namer assigns
//! per-iteration names.
//!
//! Uses `.item_namer()` to assign custom display names to each iteration.

use aws_durable_execution_sdk as durable;
use durable::DurableContext;

/// Handler: map that multiplies items by 10 with custom iteration names.
async fn handler(
    event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items: Vec<i64> = serde_json::from_value(event).unwrap_or_else(|_| vec![1, 2]);
    let items_for_namer = items.clone();

    let results: Vec<i64> = ctx
        .map(items, |_child, item, _idx| async move { Ok(item * 10) })
        .name("named-items")
        .max_concurrency(1)
        .item_namer(move |idx| format!("item-{}", items_for_namer[idx]))
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
