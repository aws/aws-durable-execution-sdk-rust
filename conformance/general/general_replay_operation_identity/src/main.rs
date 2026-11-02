//! Conformance requirement 11-1: replay rejects an operation-type mismatch.
//!
//! This handler violates the determinism contract on purpose. The first
//! invocation creates a wait named `identity-slot`. On replay it creates a
//! step with the same name in the same position. The SDK must fail the
//! execution with a non-determinism error that names the wait as expected
//! and the step as actual, and it must not run the step body.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Handler: a wait on the first invocation, a step of the same name on replay.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let arn = ctx.execution_arn().to_owned();
    if ctx.is_replaying() {
        tracing::info!(executionArn = %arn, "DETERMINISM_REPLAY_CANARY");
        let value = ctx
            .step(move |_step_ctx| async move {
                tracing::info!(executionArn = %arn, "DETERMINISM_STEP_BODY_EXECUTED");
                Ok("unexpected".to_owned())
            })
            .name("identity-slot")
            .await?;
        return Ok(serde_json::Value::String(value));
    }
    ctx.wait(Duration::from_secs(1))
        .name("identity-slot")
        .await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    // Install the JSON subscriber that flattens event fields into each log
    // line, as step_logging does. It does not filter replayed records: the
    // canary is logged during replay and the runner expects it once. The
    // runner finds each record by its executionArn field.
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
