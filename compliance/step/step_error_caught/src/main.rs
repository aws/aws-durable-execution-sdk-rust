//! Conformance requirement 1-20: error caught and handled (try/catch).

use aws_durable_execution_sdk_rust as durable;
use durable::OperationError;

/// Handler: step fails, user code catches error, continues with fallback step.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let no_retry = |_err: &durable::StepError, _attempt: u32| durable::RetryDecision::Stop;

    let step_result: Result<String, OperationError> = ctx
        .step(|_| async { Err("Something went wrong".into()) })
        .retry_strategy(no_retry)
        .await;

    // Catch the error.
    if let Err(_e) = step_result {
        // Error caught — continue with fallback.
    }

    // Second step: fallback.
    let fallback: String = ctx
        .step(|_| async { Ok("fallback_result".to_owned()) })
        .await?;

    Ok(serde_json::Value::String(fallback))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
