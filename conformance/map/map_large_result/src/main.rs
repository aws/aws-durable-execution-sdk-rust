//! Conformance requirement 9-16: Map whose combined item results exceed the
//! large-result threshold checkpoints with replay-children.
//!
//! When the aggregate result exceeds 256KB, the SDK checkpoints the Map
//! parent with `replay_children` set. On replay, the SDK re-executes the
//! items to reconstruct the result from their terminal checkpoint records.

use aws_durable_execution_sdk as durable;
use durable::DurableContext;

/// Handler: map producing >256KB aggregate results.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items: Vec<i64> = vec![0, 1, 2, 3];

    let results: Vec<String> = ctx
        .map(items, |_child, _item, _idx| async move {
            // Each item returns ~70KB so aggregate > 256KB.
            Ok("X".repeat(70 * 1024))
        })
        .name("large")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::json!({
        "successCount": results.len(),
        "totalCount": results.len(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
