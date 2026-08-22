//! Conformance requirement 1-14: retry with custom config (2s initial, 3x backoff, no jitter).

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: step that fails twice then succeeds, with custom retry
/// (initial 2s, backoff 3x, no jitter). Tracks via DDB.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let execution_id = ctx.execution_arn().to_owned();

    let retry = |_err: &durable::StepError, attempt: u32| {
        const MAX_ATTEMPTS: u32 = 5;
        const INITIAL_DELAY_SECS: u64 = 2;
        const BACKOFF_RATE: u64 = 3;

        if attempt >= MAX_ATTEMPTS {
            return durable::RetryDecision::Stop;
        }
        // No jitter: deterministic delay = initial * backoff^(attempt-1).
        let delay_secs =
            INITIAL_DELAY_SECS * BACKOFF_RATE.saturating_pow(attempt.saturating_sub(1));
        durable::RetryDecision::Retry {
            delay: Duration::from_secs(delay_secs),
        }
    };

    let result: String = ctx
        .step(move |_sc| async move {
            let count = conformance::increment_attempt(&execution_id).await?;
            if count < 3 {
                return Err(format!("Attempt {count} failed").into());
            }
            Ok("finally succeeded".to_owned())
        })
        .retry_strategy(retry)
        .await?;

    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
