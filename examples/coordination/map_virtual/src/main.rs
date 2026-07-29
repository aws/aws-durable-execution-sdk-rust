//! Map items in flat (virtual) contexts.
//!
//! By default each mapped item runs in its own nested child context.
//! [`NestingMode::Flat`] instead runs items in a *virtual* context, sharing the
//! parent's operation namespace rather than nesting a per-item child. This
//! flattens the operation history — the map counterpart to flat parallel
//! branches.
//!
//! The behavior is defined by conformance requirement 9-12. This example maps
//! three numbers in flat mode, incrementing each.
//!
//! [`NestingMode::Flat`]: aws_durable_execution_sdk_rust::NestingMode::Flat

use aws_durable_execution_sdk_rust as durable;
use durable::NestingMode;

/// Maps three numbers in flat (virtual) contexts and returns the results.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Vec<u32>, durable::BoxError> {
    let items: Vec<u32> = (0..3).collect();

    let results = ctx
        .map(items, |child, item, _idx| async move {
            child
                .step(move |_| async move { Ok(item + 1) })
                .name("increment")
                .await
                .map_err(Into::into)
        })
        .name("flat-map")
        .nesting(NestingMode::Flat)
        .await?;
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
