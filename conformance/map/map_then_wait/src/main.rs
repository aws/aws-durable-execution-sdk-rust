//! Conformance requirement 9-17: Suspension after a successful map; on replay
//! the completed map is skipped.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::DurableContext;

/// Handler: map followed by a wait.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["a".to_owned(), "b".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move {
            Ok(item.to_uppercase())
        })
        .name("then-wait")
        .max_concurrency(1)
        .await?;

    ctx.wait(Duration::from_secs(1)).await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
