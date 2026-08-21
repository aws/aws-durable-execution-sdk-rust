//! Conformance requirement 1-10: replay re-throws failed step.

use aws_durable_execution_sdk_rust as durable;
use durable::OperationError;
use std::time::Duration;

/// Handler: step fails permanently (no retry), caught by user code, then wait.
/// On replay, the error is re-thrown from cache without re-executing.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // No-retry strategy: stop on first attempt.
    let no_retry = |_err: &durable::StepError, _attempt: u32| durable::RetryDecision::Stop;

    let step_result: Result<String, OperationError> = ctx
        .step(|_step_ctx| async {
            tracing::info!("step executed");
            Err("Something went wrong".into())
        })
        .retry_strategy(no_retry)
        .await;

    // Catch the error and continue.
    if let Err(_e) = step_result {
        // Error caught: continue to wait.
    }

    ctx.wait(Duration::from_secs(1)).await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
