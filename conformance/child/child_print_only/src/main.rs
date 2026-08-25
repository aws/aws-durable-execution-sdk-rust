//! Conformance requirement 3-17: child prints and returns (no durable ops).

use aws_durable_execution_sdk as durable;
use std::time::Duration;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Handler: child only prints, no durable ops inside; followed by wait.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input = event.as_str().unwrap_or("").to_owned();
    let arn = ctx.execution_arn().to_owned();

    let result = ctx
        .run_in_child_context(move |_child_ctx| {
            let val = input.clone();
            let execution_arn = arn.clone();
            async move {
                // Emit structured log with durableExecutionArn at JSON top level
                // so CloudWatch Logs Insights can filter by execution.
                tracing::info!(durableExecutionArn = %execution_arn, "{val}");
                Ok(serde_json::Value::String(val))
            }
        })
        .name("print-child")
        .await?;

    ctx.wait(Duration::from_secs(1)).await?;

    Ok(result)
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
