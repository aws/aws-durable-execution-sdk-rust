//! Conformance requirement 9-11: Map with max-concurrency > 1 preserves
//! index-ordered results.

use aws_durable_execution_sdk as durable;
use durable::DurableContext;

/// Handler: map with concurrency=2, returns items in order.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["r0".to_owned(), "r1".to_owned(), "r2".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move { Ok(item) })
        .name("concurrent")
        .max_concurrency(2)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
