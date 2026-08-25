//! Conformance requirement 9-15: A wait inside one map iteration suspends; on
//! replay the succeeded iteration is skipped and the suspended iteration
//! resumes.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::DurableContext;

/// Handler: map with item 0 doing a step and item 1 doing wait+step.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["r0".to_owned(), "r1".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |child, item, idx| async move {
            if idx == 0 {
                return child
                    .step(move |_| async move { Ok(item) })
                    .await
                    .map_err(Into::into);
            }
            // Item 1: wait then step.
            child.wait(Duration::from_secs(1)).await?;
            child
                .step(move |_| async move { Ok(item) })
                .await
                .map_err(Into::into)
        })
        .name("suspend")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
