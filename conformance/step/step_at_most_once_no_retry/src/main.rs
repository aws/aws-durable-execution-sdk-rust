//! Conformance requirement 1-17: step with AtMostOncePerRetry (no retry, crashes).

use aws_durable_execution_sdk as durable;
use durable::StepSemantics;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Handler: step with AtMostOncePerRetry semantics and no retry. Prints to
/// stdout then crashes (process::exit).
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input = event.as_str().unwrap_or("").to_owned();
    let arn = ctx.execution_arn().to_owned();

    let no_retry = |_err: &durable::StepError, _attempt: u32| durable::RetryDecision::Stop;

    let result: String = ctx
        .step(move |_| async move {
            // Emit structured log with durableExecutionArn at JSON top level
            // so CloudWatch Logs Insights can filter by execution.
            tracing::info!(durableExecutionArn = %arn, "{input}");
            // Allow log flush.
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            std::process::exit(1);
            #[expect(unreachable_code)]
            Ok("unreachable".to_owned())
        })
        .name("at_most_once_flaky_step")
        .semantics(StepSemantics::AtMostOncePerRetry)
        .retry_strategy(no_retry)
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
