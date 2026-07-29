//! Conformance requirement 1-12: retry exhaustion (max attempts = 4).

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: step that always fails, with 4 total attempts.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let retry: durable::RetryStrategy = Box::new(|_err, attempt| {
        if attempt >= 4 {
            durable::RetryDecision::Stop
        } else {
            durable::RetryDecision::Retry {
                delay: Duration::from_secs(1),
            }
        }
    });

    let result: String = ctx
        .step(|_| async { Err("Always fails".into()) })
        .retry_strategy(retry)
        .await?;

    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
