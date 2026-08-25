//! Conformance requirement 1-7: step with context logger (optional).

use aws_durable_execution_sdk as durable;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Handler: step that logs through tracing (Rust equivalent of step context logger).
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let name = event.as_str().unwrap_or("World").to_owned();
    let arn = ctx.execution_arn().to_owned();
    let result = ctx
        .step(move |_step_ctx| async move {
            // Emit executionArn as a top-level JSON field on each log event
            // so CloudWatch Logs Insights can filter by execution.
            tracing::info!(
                executionArn = %arn,
                "Greeting step started for: {name}"
            );
            let greeting = format!("Hello, {name}!");
            tracing::info!(
                executionArn = %arn,
                "Greeting step completed with: {greeting}"
            );
            Ok(greeting)
        })
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    // Install JSON subscriber that flattens event fields into each log line.
    // Writing to stderr ensures the Lambda runtime's log pipeline indexes
    // the records in CloudWatch Logs Insights promptly (the stderr path on
    // provided.al2023 is routed through the runtime's telemetry channel).
    // The executionArn field (emitted per-event above) enables the
    // validator's filter: coalesce(durableExecutionArn, executionArn).
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
