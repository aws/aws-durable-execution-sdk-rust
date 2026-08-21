//! Basic wait: pause a durable function without holding the invocation open.
//!
//! [`ctx.wait`](aws_durable_execution_sdk_rust::DurableContext::wait) suspends
//! the execution for a
//! [`Duration`](std::time::Duration). This is not a `sleep` that blocks a
//! running Lambda: the SDK checkpoints the wait, the invocation ends, and the
//! service re-invokes the function when the duration elapses. A wait of two
//! seconds and a wait of thirty days cost the same while suspended: nothing
//! is running.
//!
//! On resume the code before the wait is replayed from checkpoints (not
//! re-executed), and execution continues after the wait.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

/// Logs a line, waits two seconds (suspending the execution), then completes.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    tracing::info!("before wait");
    ctx.wait(Duration::from_secs(2)).await?;
    tracing::info!("after wait");
    Ok("completed".to_owned())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
