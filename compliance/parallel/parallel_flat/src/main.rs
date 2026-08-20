//! Conformance requirement 8-12: Parallel with FLAT nesting executes branches
//! in virtual contexts, omitting per-branch context events.
//!
//! Uses `NestingMode::Flat` to suppress per-branch context checkpoint events.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::NestingMode;

/// Handler: parallel flat nesting — branches run steps in virtual contexts.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |child: DurableContext| async move {
            child
                .step(|_| async { Ok("fa".to_owned()) })
                .await
                .map_err(Into::into)
        }),
        Branch::new("1", |child: DurableContext| async move {
            child
                .step(|_| async { Ok("fb".to_owned()) })
                .await
                .map_err(Into::into)
        }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("flat")
        .max_concurrency(1)
        .nesting(NestingMode::Flat)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
