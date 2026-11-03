//! Conformance requirement 12-1: parallel with independently typed branch
//! handles.
//!
//! The handler adds three branches of different result types to a
//! `BranchSet` and runs them with `parallel_mixed`. `max_concurrency(1)`
//! runs the branches in the order they were added, which is the event order
//! 12-1 expects. Each `BranchHandle` returns its own type.

use std::collections::BTreeMap;

use aws_durable_execution_sdk as durable;

/// Handler: runs a string, an integer and a string map branch, and returns
/// their values in registration order.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let mut branches = durable::BranchSet::new();
    let string = branches.add("string", |_| async { Ok("reserved".to_owned()) });
    let integer = branches.add("integer", |_| async { Ok(200_u32) });
    let map = branches.add("map", |_| async {
        Ok(BTreeMap::from([("currency".to_owned(), "USD".to_owned())]))
    });

    ctx.parallel_mixed(branches)
        .name("typed-branches")
        .max_concurrency(1)
        .await?;

    let string: String = string.into_result()?;
    let integer: u32 = integer.into_result()?;
    let map: BTreeMap<String, String> = map.into_result()?;
    Ok(serde_json::json!([string, integer, map]))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
