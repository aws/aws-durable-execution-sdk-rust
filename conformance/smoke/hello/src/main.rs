//! Smoke test handler: a minimal durable function that returns immediately.
//!
//! This handler exercises the cold-start + handler-registration chain without
//! invoking any durable operations. Its purpose is to prove the deploy +
//! invoke pipeline works for Rust.

use aws_durable_execution_sdk as durable;
use serde::{Deserialize, Serialize};

/// Input event for the smoke handler.
#[derive(Debug, Deserialize)]
struct SmokeEvent {
    /// Name to greet: defaults to "world" if absent.
    #[serde(default = "default_name")]
    name: String,
}

/// Default name when the event does not include one.
fn default_name() -> String {
    "world".to_owned()
}

/// Output response from the smoke handler.
#[derive(Debug, Serialize)]
struct SmokeResponse {
    /// Greeting message.
    message: String,
    /// Whether the context reported replay mode.
    is_replaying: bool,
}

/// Handler that returns a greeting without performing any durable operations.
///
/// This validates that:
/// 1. `durable::run` correctly wires to the Lambda runtime
/// 2. Event deserialization works
/// 3. The handler can access `DurableContext` metadata
/// 4. Response serialization reaches the caller
async fn handler(
    event: SmokeEvent,
    ctx: durable::DurableContext,
) -> Result<SmokeResponse, durable::BoxError> {
    tracing::info!(
        name = %event.name,
        arn = ctx.execution_arn(),
        "smoke handler invoked"
    );

    Ok(SmokeResponse {
        message: format!("Hello, {}!", event.name),
        is_replaying: ctx.is_replaying(),
    })
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    // Initialize the Lambda-native tracing subscriber (respects
    // AWS_LAMBDA_LOG_FORMAT and AWS_LAMBDA_LOG_LEVEL env vars).
    lambda_runtime::tracing::init_default_subscriber();

    durable::run(handler).await
}
