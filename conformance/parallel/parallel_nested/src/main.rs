//! Conformance requirement 8-21: Nested parallel (parallel inside a parallel
//! branch).

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};

/// Handler: outer parallel with one branch that runs an inner parallel.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let outer_branches: Vec<Branch<Vec<String>>> =
        vec![Branch::new("0", |outer_ctx: DurableContext| async move {
            let inner_branches = vec![
                Branch::new("0", |inner_ctx: DurableContext| async move {
                    inner_ctx
                        .step(|_| async { Ok("i1".to_owned()) })
                        .await
                        .map_err(Into::into)
                }),
                Branch::new("1", |inner_ctx: DurableContext| async move {
                    inner_ctx
                        .step(|_| async { Ok("i2".to_owned()) })
                        .await
                        .map_err(Into::into)
                }),
            ];

            outer_ctx
                .parallel(inner_branches)
                .name("inner")
                .max_concurrency(1)
                .await
                .map_err(Into::into)
        })];

    let results: Vec<Vec<String>> = ctx
        .parallel(outer_branches)
        .name("outer")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
