//! Conformance requirement 5-2: invoke with an explicit operation name.

use aws_durable_execution_sdk_rust as durable;
use serde::Deserialize;

/// Input with a name field and a payload.
#[derive(Deserialize)]
struct Input {
    name: String,
    payload: serde_json::Value,
}

/// Handler: invoke target with a named operation.
async fn handler(
    event: Input,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx
        .invoke::<serde_json::Value, _>(&target, event.payload)
        .name(event.name)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
