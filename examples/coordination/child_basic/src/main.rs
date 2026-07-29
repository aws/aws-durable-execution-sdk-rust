//! Child contexts: an isolated, composable unit of durable work.
//!
//! [`run_in_child_context`] runs a closure in its own child
//! [`DurableContext`](aws_durable_execution_sdk_rust::DurableContext).
//! Operations inside the child are numbered in a namespace nested under the
//! parent, so a child is a self-contained unit you can build, name, and later
//! fan out concurrently without operation identities colliding. The child's
//! return value is checkpointed and replayed like any operation result.
//!
//! This example runs one child that performs two sequential steps and returns
//! a combined result. The child body returns
//! [`BoxError`](aws_durable_execution_sdk_rust::BoxError); a failing
//! inner operation converts into it with `?`.
//!
//! [`run_in_child_context`]: aws_durable_execution_sdk_rust::DurableContext::run_in_child_context

use aws_durable_execution_sdk_rust as durable;

/// Runs a single child context that fetches a name and formats a greeting.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let greeting = ctx
        .run_in_child_context(|child| async move {
            let name = child
                .step(|_| async { Ok("world".to_owned()) })
                .name("fetch-name")
                .await?;
            let greeting = child
                .step(move |_| async move { Ok(format!("hello, {name}")) })
                .name("format")
                .await?;
            Ok(greeting)
        })
        .name("greet")
        .await?;
    Ok(greeting)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
