//! Conformance requirement 5-15: invoke with custom payload serdes.
//!
//! The custom payload serdes uppercases string payloads before sending
//! to the target.

use aws_durable_execution_sdk_rust as durable;

/// Custom serdes that uppercases string payloads on serialization.
#[derive(Debug)]
struct UppercasePayloadSerdes;

impl durable::Serdes for UppercasePayloadSerdes {
    fn serialize_to_string(&self, json_str: &str) -> Result<String, durable::BoxError> {
        // Uppercase the JSON-serialized string value.
        Ok(json_str.to_uppercase())
    }

    fn deserialize_from_string(&self, payload: &str) -> Result<String, durable::BoxError> {
        Ok(payload.to_owned())
    }
}

/// Handler: invoke with uppercasing payload serdes.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx
        .invoke::<serde_json::Value, _>(&target, event)
        .payload_serdes(UppercasePayloadSerdes)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
