//! Conformance requirement 5-8: invoke with tenant ID.

use aws_durable_execution_sdk as durable;
use serde::Deserialize;

/// Input with tenant ID and payload.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Input {
    tenant_id: String,
    payload: serde_json::Value,
}

/// Handler: invoke target with tenant isolation.
async fn handler(
    event: Input,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx
        .invoke::<serde_json::Value, _>(&target, event.payload)
        .tenant_id(event.tenant_id)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
