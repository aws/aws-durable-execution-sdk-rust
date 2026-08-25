//! Retry a failing step until the strategy stops, then recover.
//!
//! A [`retry_strategy`](aws_durable_execution_sdk::DurableContext::step)
//! decides, per attempt, whether to retry or stop. When it returns
//! [`RetryDecision::Stop`](aws_durable_execution_sdk::RetryDecision::Stop)
//! the last error propagates as a
//! [`StepError`](aws_durable_execution_sdk::StepError). This example's step
//! always fails; the strategy allows three attempts then stops, and the handler
//! catches the exhausted error and returns a graceful summary instead of
//! failing the execution.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::{OperationErrorKind, RetryDecision};

/// Runs an always-failing step with a bounded retry policy and recovers.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let outcome: Result<i32, durable::OperationError> = ctx
        .step(
            |step_ctx| async move { Err(format!("attempt {} failed", step_ctx.attempt()).into()) },
        )
        .name("always-fails")
        .retry_strategy(|_err, attempt| {
            if attempt >= 3 {
                RetryDecision::Stop
            } else {
                RetryDecision::Retry {
                    delay: Duration::from_secs(1),
                }
            }
        })
        .await;

    match outcome {
        Ok(value) => Ok(format!("unexpected success: {value}")),
        Err(err) if matches!(err.kind(), OperationErrorKind::Step(_)) => {
            Ok(format!("retries exhausted, recovered gracefully: {err}"))
        }
        Err(err) => Err(err.to_string().into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
