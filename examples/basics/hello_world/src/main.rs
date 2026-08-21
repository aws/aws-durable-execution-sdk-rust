//! Hello world: the smallest durable function.
//!
//! A durable function is an ordinary Lambda handler wrapped by
//! [`durable::run`]. It becomes "durable" only when it uses operations on the
//! [`DurableContext`](aws_durable_execution_sdk_rust::DurableContext): steps,
//! waits, child contexts, and so on. This example uses none of them: it simply
//! logs a line and returns a value, which shows the minimum wiring every
//! durable function shares before any checkpointing enters the picture.
//!
//! Because there are no durable operations, the function runs to completion in
//! a single invocation and never suspends.

use aws_durable_execution_sdk_rust as durable;

/// Returns a fixed greeting. No durable operations, so no checkpoints and no
/// replay: the handler runs start to finish in one invocation.
async fn handler(
    _event: serde_json::Value,
    _ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    tracing::info!("hello world from a durable function");
    Ok("Hello, World!".to_owned())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
