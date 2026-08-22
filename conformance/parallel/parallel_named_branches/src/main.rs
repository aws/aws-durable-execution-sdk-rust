//! Conformance requirement 8-3: Parallel invoked with named-branch objects.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};

/// Handler: parallel with explicitly named branches.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("first", |_: DurableContext| async { Ok("one".to_owned()) }),
        Branch::new("second", |_: DurableContext| async { Ok("two".to_owned()) }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("named")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
