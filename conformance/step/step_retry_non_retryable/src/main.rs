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
    // Strategy only retries "ValidationError", not TransientError. The
    // wire type name travels on the TypedError wrapper reachable through
    // source().
    let retry = |err: &durable::StepError, _attempt: u32| {
        let retries = std::iter::successors(
            std::error::Error::source(err),
            |e| e.source(),
        )
        .any(|e| {
            e.downcast_ref::<durable::TypedError>()
                .is_some_and(|t| t.error_type() == "ValidationError")
        });
        if retries {
            durable::RetryDecision::Retry {
                delay: std::time::Duration::from_secs(1),
            }
        } else {
            durable::RetryDecision::Stop
        }
    };

    let result: String = ctx
        .step(|_| async {
            // TypedError records the concrete type name as the wire
            // ErrorType (a boxed error's type is otherwise erased).
            let err: durable::BoxError = Box::new(durable::TypedError::new(TransientError {
                message: "transient failure".to_owned(),
            }));
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
