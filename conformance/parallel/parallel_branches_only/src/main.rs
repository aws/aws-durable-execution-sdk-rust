//! Conformance requirement 8-2: Parallel invoked with the branches-only form
//! (no name argument).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};

/// Handler: parallel with no name, two branches returning constants.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |_: DurableContext| async { Ok("alpha".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Ok("beta".to_owned()) }),
    ];

    let results: Vec<String> = ctx.parallel(branches).max_concurrency(1).await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
