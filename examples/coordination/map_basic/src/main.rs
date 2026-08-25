//! Map: apply the same durable work to each item in a collection.
//!
//! [`map`] runs a closure over every item concurrently, each in its own child
//! context, and returns the results in input order. It is the data-parallel
//! counterpart to [`parallel`]'s named branches: use `map` when the work is
//! uniform across a collection.
//!
//! [`item_namer`] assigns a stable, human-readable name to each item's context
//! from its index, which makes the operation history and any per-item logs easy
//! to read. Names are derived from the index (not item content) so they stay
//! deterministic across replay.
//!
//! This example maps a short list of strings, naming each item's context.
//!
//! [`map`]: aws_durable_execution_sdk::DurableContext::map
//! [`parallel`]: aws_durable_execution_sdk::DurableContext::parallel
//! [`item_namer`]: aws_durable_execution_sdk::builders::MapBuilder::item_namer

use aws_durable_execution_sdk as durable;

/// Maps over a list of strings, naming each item's context by index.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Vec<String>, durable::BoxError> {
    let items = vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()];

    let results = ctx
        .map(items, |child, item, idx| async move {
            child
                .step(move |_| async move { Ok(format!("item-{idx}:{item}")) })
                .name("process")
                .await
                .map_err(Into::into)
        })
        .name("process-all")
        .item_namer(|idx| format!("item-{idx}"))
        .await?;
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
