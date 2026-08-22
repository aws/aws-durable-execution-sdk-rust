//! Conformance requirement 3-11: child with large result payload.

use aws_durable_execution_sdk_rust as durable;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Handler: child produces a large result (>256KB), triggering replay mode.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input = event.as_str().unwrap_or("default").to_owned();

    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let input_clone = input.clone();
            let step_result: String = child_ctx.step(|_| async move { Ok(input_clone) }).await?;

            // Emit structured log with durableExecutionArn at JSON top level
            // so CloudWatch Logs Insights can filter by execution.
            let arn = child_ctx.execution_arn();
            tracing::info!(durableExecutionArn = %arn, "{step_result}");

            // Build a large result (>256KB).
            Ok(step_result.repeat(256 * 1024 / step_result.len() + 1))
        })
        .name("large-data-processor")
        .await?;

    ctx.wait(std::time::Duration::from_secs(2)).await?;

    let response = serde_json::json!({
        "success": true,
        "dataSize": result.len(),
    });
    Ok(response)
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
