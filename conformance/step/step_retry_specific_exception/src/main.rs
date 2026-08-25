//! Conformance requirement 1-15: retry specific exception.

use aws_durable_execution_sdk as durable;
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
        // Only retry if the escaping error IS a TransientError: the step
        // error's source() carries the concrete error, so the "instance
        // of" check is a downcast, not string matching.
        if attempt >= 3 {
            return durable::RetryDecision::Stop;
        }
        let is_transient = std::iter::successors(
            std::error::Error::source(err),
            |e| e.source(),
        )
        .any(|e| {
            e.downcast_ref::<TransientError>().is_some()
                || e.downcast_ref::<durable::TypedError>()
                    .is_some_and(|t| t.error_type() == "TransientError")
        });
        if is_transient {
            durable::RetryDecision::Retry {
                delay: Duration::from_secs(1),
            }
        } else {
            durable::RetryDecision::Stop
        }
    };

    let result: String = ctx
        .step(move |_sc| async move {
            let count = conformance::increment_attempt(&execution_id).await?;
            if count < 2 {
                // TypedError records the concrete type name as the wire
                // ErrorType (a boxed error's type is otherwise erased).
                let err: durable::BoxError = Box::new(durable::TypedError::new(TransientError {
                    message: "Temporary failure".to_owned(),
                }));
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
