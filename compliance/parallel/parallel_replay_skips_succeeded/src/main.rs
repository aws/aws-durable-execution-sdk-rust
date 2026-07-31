//! Conformance requirement 8-14: Parallel replay skips succeeded branches
//! across a wait suspension.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};

/// Handler: parallel with one branch doing a step and another doing a wait.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |child: DurableContext| async move {
            child
                .step(|_| async { Ok("b0".to_owned()) })
                .await
                .map_err(Into::into)
        }),
        Branch::new("1", |child: DurableContext| async move {
            child.wait(Duration::from_secs(2)).await?;
            Ok("b1".to_owned())
        }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("replay")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
