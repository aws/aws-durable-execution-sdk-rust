//! Conformance requirement 9-1: Map basic: applies a function to each item,
//! each item runs a single step, all succeed.

use aws_durable_execution_sdk as durable;
use durable::DurableContext;

/// Handler: map over items with a greeting step.
async fn handler(
    event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items: Vec<String> = serde_json::from_value(event)
        .unwrap_or_else(|_| vec!["World".to_owned(), "Kiro".to_owned()]);

    let results: Vec<String> = ctx
        .map(items, |child, item, _idx| async move {
            child
                .step(move |_| async move { Ok(format!("Hello, {item}!")) })
                .await
                .map_err(Into::into)
        })
        .name("map")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
