//! Conformance requirement 8-1: Parallel basic (two branches, each a single
//! step, all succeed).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};

/// Handler: two parallel branches, each returning a constant string via a step.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |child: DurableContext| async move {
            let r = child.step(|_| async { Ok("task-1".to_owned()) }).await?;
            Ok(r)
        }),
        Branch::new("1", |child: DurableContext| async move {
            let r = child.step(|_| async { Ok("task-2".to_owned()) }).await?;
            Ok(r)
        }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("parallel")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
