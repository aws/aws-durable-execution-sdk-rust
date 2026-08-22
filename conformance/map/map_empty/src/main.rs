//! Conformance requirement 9-4: Map invoked with an empty items list completes
//! immediately with no iterations.

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;

/// Handler: map with empty items.
async fn handler(
    event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items: Vec<String> = serde_json::from_value(event).unwrap_or_default();

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move { Ok(item) })
        .name("empty")
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
