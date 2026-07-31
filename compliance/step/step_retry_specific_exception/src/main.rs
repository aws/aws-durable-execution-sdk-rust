//! Conformance requirement 1-15: retry specific exception.

use aws_durable_execution_sdk_rust as durable;
use std::fmt;
use std::time::Duration;

/// Custom error type that the retry strategy retries.
#[derive(Debug)]
struct TransientError {
    message: String,
}

impl fmt::Display for TransientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TransientError {}

/// Handler: step throws TransientError on first attempt; retry strategy
/// retries only TransientError. Tracks via DDB.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let execution_id = ctx.execution_arn().to_owned();

    let retry = |err: &durable::StepError, attempt: u32| {
        // Only retry if the error message indicates it's a TransientError.
        if attempt >= 3 {
            return durable::RetryDecision::Stop;
        }
        let msg = err.to_string();
        if msg.contains("TransientError") || msg.contains("Temporary failure") {
            durable::RetryDecision::Retry {
                delay: Duration::from_secs(1),
            }
        } else {
            durable::RetryDecision::Stop
        }
    };

    let result: String = ctx
        .step(move |_sc| async move {
            let count = compliance::increment_attempt(&execution_id).await?;
            if count < 2 {
                let err: durable::BoxError = Box::new(TransientError {
                    message: "Temporary failure".to_owned(),
                });
                return Err(err);
            }
            Ok("recovered from transient".to_owned())
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
