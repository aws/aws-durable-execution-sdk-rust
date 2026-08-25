//! Conformance requirement 11-1: multi-page history replay (issue #5).
//!
//! Creates enough operations, each with a ~1KB payload, that the recorded
//! history spans more than one `GetDurableExecutionState` page. A wait in
//! the middle suspends the execution, so the replay after it only
//! completes correctly when the bootstrap path follows the pagination
//! marker instead of replaying against a truncated log.

use aws_durable_execution_sdk as durable;
use std::time::Duration;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Number of steps recorded before the suspension point.
const STEP_COUNT: u64 = 150;

/// Handler: 150 padded steps, a suspending wait, then a checksum result.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let arn = ctx.execution_arn().to_owned();
    let mut sum: u64 = 0;
    for i in 0..STEP_COUNT {
        // Each step returns "<index>:<~1KB padding>". The padding pushes the
        // recorded history across the service's page boundary; the index
        // prefix lets the handler verify every page replayed.
        let step_arn = arn.clone();
        let value = ctx
            .step(move |_| async move {
                // Executes exactly once per step: a truncated replay
                // re-executes steps and inflates this log count. The
                // executionArn field (top-level via the flattened JSON
                // subscriber) enables the validator's per-execution filter.
                tracing::info!(
                    executionArn = %step_arn,
                    "history-multi-page-step-executed"
                );
                Ok(format!("{i}:{}", "x".repeat(1024)))
            })
            .await?;
        let index: u64 = value
            .split(':')
            .next()
            .and_then(|prefix| prefix.parse().ok())
            .ok_or("step result lost its index prefix")?;
        sum += index;
    }

    // Suspend so the next invocation replays the full multi-page history.
    ctx.wait(Duration::from_secs(1)).await?;

    Ok(serde_json::Value::from(sum))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    // Install JSON subscriber that flattens event fields into each log line.
    // Writing to stderr ensures the Lambda runtime's log pipeline indexes
    // the records in CloudWatch Logs Insights promptly. The executionArn
    // field (emitted per-event above) enables the validator's filter:
    // coalesce(durableExecutionArn, executionArn).
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
