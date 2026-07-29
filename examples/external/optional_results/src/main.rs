//! Operations that produce no value or an optional value.
//!
//! Rust has no `undefined`. A step that performs a side effect and returns
//! nothing uses the unit type `()`; a step that may or may not produce a value
//! returns [`Option`], which serializes to JSON `null` when `None`. Both
//! checkpoint and replay like any other result.

use aws_durable_execution_sdk_rust as durable;

/// Runs a unit-returning step and an `Option`-returning step.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // A pure side effect: nothing to return.
    ctx.step(|_| async { Ok(()) }).name("side-effect").await?;

    // An optional result: None serializes to JSON null.
    let maybe: Option<i32> = ctx
        .step(|_| async { Ok(None::<i32>) })
        .name("maybe-value")
        .await?;

    Ok(serde_json::json!({ "maybe": maybe }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
