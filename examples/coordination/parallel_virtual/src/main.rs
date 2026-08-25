//! Parallel branches in flat (virtual) contexts.
//!
//! By default each parallel branch runs in a nested child context, giving it
//! its own operation namespace. [`NestingMode::Flat`] instead runs branches in
//! a *virtual* context: their operations share the parent's namespace rather
//! than nesting under a per-branch child. This flattens the operation history,
//! which some workloads prefer for simpler replay inspection.
//!
//! The behavior is defined by conformance requirement 8-12. This example fans
//! out three flat branches, each scaling its index.
//!
//! [`NestingMode::Flat`]: aws_durable_execution_sdk::builders::map_parallel::NestingMode::Flat

use aws_durable_execution_sdk as durable;
use durable::Branch;
use durable::builders::map_parallel::NestingMode;

/// Runs three branches in flat (virtual) contexts and returns their results.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Vec<u32>, durable::BoxError> {
    let branches: Vec<Branch<u32>> = (0..3u32)
        .map(|i| {
            Branch::new(format!("v-{i}"), move |child| async move {
                child
                    .step(move |_| async move { Ok(i * 10) })
                    .name("scale")
                    .await
                    .map_err(Into::into)
            })
        })
        .collect();

    let results = ctx
        .parallel(branches)
        .name("flat-fan-out")
        .nesting(NestingMode::Flat)
        .await?;
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
