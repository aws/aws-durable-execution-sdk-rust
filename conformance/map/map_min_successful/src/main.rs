//! Conformance requirement 9-7: Map with a min-successful completion config
//! stops early once enough items succeed.

use aws_durable_execution_sdk as durable;
use durable::DurableContext;
use durable::builders::map_parallel::CompletionConfig;

/// Handler: map with min-successful=2 over 4 items.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec![
        "a".to_owned(),
        "b".to_owned(),
        "c".to_owned(),
        "d".to_owned(),
    ];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move { Ok(item) })
        .name("min-successful")
        .max_concurrency(1)
        .completion(CompletionConfig::with_min_successful(2))
        .await?;

    Ok(serde_json::json!({
        "completionReason": "MIN_SUCCESSFUL_REACHED",
        "successCount": results.len(),
        "totalCount": results.len(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
