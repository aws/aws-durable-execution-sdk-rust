//! Conformance requirement 1-16: retry specific exception (non-retryable fails).

use aws_durable_execution_sdk_rust as durable;
use std::fmt;

/// Custom error type (TransientError) that the retry strategy does NOT retry.
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

/// Handler: step throws TransientError; strategy does not retry it → fails.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // Strategy only retries "ValidationError", not TransientError.
    let retry: durable::RetryStrategy = Box::new(|err, _attempt| {
        let msg = err.to_string();
        if msg.contains("ValidationError") {
            durable::RetryDecision::Retry {
                delay: std::time::Duration::from_secs(1),
            }
        } else {
            durable::RetryDecision::Stop
        }
    });

    let result: String = ctx
        .step(|_| async {
            let err: durable::BoxError = Box::new(TransientError {
                message: "transient failure".to_owned(),
            });
            Err(err)
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
