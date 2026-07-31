//! Conformance requirement 1-19: step with error (fails permanently, no retries).

use aws_durable_execution_sdk_rust as durable;

/// Handler: step that always fails, with no retries.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let no_retry = |_err: &durable::StepError, _attempt: u32| durable::RetryDecision::Stop;

    let result: String = ctx
        .step(|_| async { Err("Something went wrong".into()) })
        .retry_strategy(no_retry)
        .await?;

    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
