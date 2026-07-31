//! Conformance requirement 1-18: step with AtMostOncePerRetry (with retry, succeeds on second).

use aws_durable_execution_sdk_rust as durable;
use durable::StepSemantics;
use std::time::Duration;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Handler: step crashes on first attempt, succeeds on second.
/// Prints input to stdout each time. Uses DDB attempt tracking.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input = event.as_str().unwrap_or("").to_owned();
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
            let count = compliance::increment_attempt(&execution_id).await?;
            // Emit structured log with durableExecutionArn at JSON top level
            // so CloudWatch Logs Insights can filter by execution.
            let arn = &execution_id;
            tracing::info!(durableExecutionArn = %arn, "{input}");
            // Allow log flush.
            tokio::time::sleep(Duration::from_secs(1)).await;
            if count < 2 {
                std::process::exit(1);
            }
            Ok("succeeded on second attempt".to_owned())
        })
        .semantics(StepSemantics::AtMostOncePerRetry)
        .retry_strategy(retry)
        .await?;

    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .flatten_event(true)
                .with_span_list(false)
                .with_target(false)
                .with_writer(std::io::stderr),
        )
        .init();
    durable::run(handler).await
}
