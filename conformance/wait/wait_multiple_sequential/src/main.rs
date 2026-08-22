//! Conformance requirement 2-3: multiple sequential waits.

use aws_durable_execution_sdk_rust as durable;
use serde::Serialize;
use std::time::Duration;

/// Result returned by the handler.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WaitResult {
    /// Number of completed waits.
    completed_waits: u32,
}

/// Handler: two sequential wait operations.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(2)).name("wait-1").await?;
    ctx.wait(Duration::from_secs(2)).name("wait-2").await?;
    let result = WaitResult { completed_waits: 2 };
    serde_json::to_value(result).map_err(Into::into)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
