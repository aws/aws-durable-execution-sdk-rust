//! Conformance requirement 1-11: step with retry (fails then succeeds).

use aws_durable_execution_sdk as durable;
use std::time::Duration;

/// Handler: step that fails on first attempt and succeeds on second,
/// tracking attempts via DynamoDB.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let execution_id = ctx.execution_arn().to_owned();

    let retry = |_err: &durable::StepError, attempt: u32| {
        if attempt >= 3 {
            durable::RetryDecision::Stop
        } else {
            durable::RetryDecision::Retry {
                delay: Duration::from_secs(1),
            }
        }
    };

    let result: String = ctx
        .step(move |_sc| async move {
            let count = conformance::increment_attempt(&execution_id).await?;
            if count < 2 {
                return Err(format!("Attempt {count} failed").into());
            }
            Ok("Operation succeeded".to_owned())
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
