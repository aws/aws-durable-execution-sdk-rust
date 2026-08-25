//! Conformance requirement 8-11: Parallel executing branches concurrently
//! returns index-ordered results.

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};

/// Handler: parallel with max-concurrency=2, three branches.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |_: DurableContext| async { Ok("r0".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Ok("r1".to_owned()) }),
        Branch::new("2", |_: DurableContext| async { Ok("r2".to_owned()) }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("concurrent")
        .max_concurrency(2)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
