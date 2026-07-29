//! Error propagation out of a child context.
//!
//! When a durable operation inside a child fails, the error propagates out of
//! the child body (via `?` converting into
//! [`BoxError`](aws_durable_execution_sdk_rust::BoxError)) and surfaces
//! at the parent as the child operation's error. The parent decides what to do
//! with it: propagate further, or — as here — catch it and continue.
//!
//! This example runs a child whose second step deliberately fails. The parent
//! catches the resulting error and returns a success value describing it, so
//! the execution completes successfully while still demonstrating that the
//! failure and its message crossed the child boundary intact.

use aws_durable_execution_sdk_rust as durable;

/// Runs a child whose step fails, then reports the caught error from the parent.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let outcome = ctx
        .run_in_child_context(|child| async move {
            child.step(|_| async { Ok(()) }).name("reserve").await?;
            // This step fails; the error propagates out of the child.
            let charged: () = child
                .step(|_| async { Err::<(), durable::BoxError>("payment declined".into()) })
                .name("charge")
                // A deliberately-failing demo step fails on the first attempt
                // instead of retrying, so the error surfaces immediately.
                .retry_strategy(Box::new(|_err, _attempt| durable::RetryDecision::Stop))
                .await?;
            Ok(charged)
        })
        .name("checkout")
        .await;

    match outcome {
        Ok(()) => Ok("checkout succeeded".to_owned()),
        Err(err) => Ok(format!("child failed as expected: {err}")),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
