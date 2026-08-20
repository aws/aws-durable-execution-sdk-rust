//! Conformance requirement 9-12: Map with FLAT nesting executes items in
//! virtual contexts, omitting per-iteration events.
//!
//! Uses `NestingMode::Flat` to suppress per-item context checkpoint events.

use aws_durable_execution_sdk_rust as durable;
use durable::DurableContext;
use durable::builders::map_parallel::NestingMode;

/// Handler: map flat nesting — items run steps in virtual contexts.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["fa".to_owned(), "fb".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |child, item, _idx| async move {
            child
                .step(move |_| async move { Ok(item) })
                .await
                .map_err(Into::into)
        })
        .name("flat")
        .max_concurrency(1)
        .nesting(NestingMode::Flat)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
