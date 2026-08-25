//! Step with a retry strategy: recover from transient failures.
//!
//! A step can fail: a downstream call times out, a dependency is briefly
//! unavailable. A [`retry_strategy`] turns those transient failures into
//! automatic retries with backoff. The strategy is a function of the error and
//! the attempt number (1-based); it returns
//! [`RetryDecision::Retry`](aws_durable_execution_sdk::RetryDecision::Retry)
//! with a delay, or
//! [`RetryDecision::Stop`](aws_durable_execution_sdk::RetryDecision::Stop)
//! to give up and propagate the error.
//!
//! Between attempts the execution suspends for the delay and resumes later, so
//! a retrying step does not hold the invocation open. Each failed attempt is
//! recorded; the successful attempt's result is the one checkpointed and
//! replayed.
//!
//! This example demonstrates the pattern deterministically: the step reads the
//! 1-based attempt number from its
//! [`StepContext`](aws_durable_execution_sdk::StepContext) and fails the
//! first two attempts, so the retry path is exercised end to end and the third
//! attempt succeeds. Real code would instead fail because an actual dependency
//! call failed.
//!
//! [`retry_strategy`]: aws_durable_execution_sdk::DurableContext::step

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk::{RetryDecision, StepSemantics};

/// Runs a step that succeeds only on its third attempt, driven by a retry
/// strategy that backs off and stops after three attempts.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let result = ctx
        .step(|step_ctx| async move {
            let attempt = step_ctx.attempt();
            if attempt < 3 {
                return Err(format!("transient failure on attempt {attempt}").into());
            }
            Ok(format!("succeeded on attempt {attempt}"))
        })
        .name("flaky-call")
        .retry_strategy(|_err, attempt| {
            if attempt >= 3 {
                RetryDecision::Stop
            } else {
                // Linear backoff: 1s, 2s, ... before each retry.
                RetryDecision::Retry {
                    delay: Duration::from_secs(u64::from(attempt)),
                }
            }
        })
        .semantics(StepSemantics::AtMostOncePerRetry)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
