//! Conformance requirement 9-6: Map where the handler rethrows, propagating an
//! item failure.
//!
//! The Rust SDK automatically propagates failures as `Err`, so this handler
//! simply lets the error bubble up (equivalent to Go's `ThrowIfError()`).

use aws_durable_execution_sdk as durable;
use durable::DurableContext;
use durable::builders::map_parallel::CompletionConfig;

/// Handler: map that propagates the first item failure.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["fail".to_owned(), "never".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move {
            if item == "fail" {
                return Err("item failed".into());
            }
            Ok(item)
        })
        .name("throwing")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(0))
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
