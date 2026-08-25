//! Structured logging at every level via `tracing`.
//!
//! The SDK logs through the `tracing` facade and wraps each operation in a span
//! carrying the execution arn, operation id, attempt, and replay flag. User
//! `tracing` events inside a step inherit those fields automatically, so
//! ordinary `tracing::info!`/`warn!`/`error!` calls become correlated,
//! structured log lines. Configure the format once in `main` with
//! `lambda_runtime::tracing::init_default_subscriber()`, which honors
//! `AWS_LAMBDA_LOG_FORMAT` and `AWS_LAMBDA_LOG_LEVEL`.

use aws_durable_execution_sdk as durable;

/// Emits one event at each level from inside a step, then returns.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let outcome = ctx
        .step(|_| async {
            tracing::trace!("trace: fine-grained detail");
            tracing::debug!("debug: diagnostic detail");
            tracing::info!(stage = "processing", "info: normal progress");
            tracing::warn!("warn: something noteworthy");
            tracing::error!("error: a recoverable problem");
            Ok("logged".to_owned())
        })
        .name("emit-logs")
        .await?;
    Ok(outcome)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
