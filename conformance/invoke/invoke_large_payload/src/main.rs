//! Conformance requirement 5-7: invoke with a large payload.

use aws_durable_execution_sdk as durable;
use serde::{Deserialize, Serialize};

/// Large payload structure.
#[derive(Serialize, Deserialize)]
struct Payload {
    data: String,
}

/// Handler: invoke with a 200KB payload.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let large = Payload {
        data: "x".repeat(200_000),
    };
    let result = ctx.invoke::<serde_json::Value, _>(&target, large).await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
