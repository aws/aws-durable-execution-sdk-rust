//! Conformance requirement 8-19: Parallel with invalid max-concurrency raises
//! a validation error.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};

/// Handler: parallel with max-concurrency=0 (invalid).
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |_: DurableContext| async { Ok("a".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Ok("b".to_owned()) }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("bad-concurrency")
        .max_concurrency(0)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
