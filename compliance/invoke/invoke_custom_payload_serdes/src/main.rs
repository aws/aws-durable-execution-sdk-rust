//! Conformance requirement 5-15: invoke with custom payload serdes.
//!
//! The custom payload serdes uppercases string payloads before sending
//! to the target.

use aws_durable_execution_sdk_rust as durable;

/// Custom serdes that uppercases string payloads on serialization.
#[derive(Debug)]
struct UppercasePayloadSerdes;

impl durable::Serdes for UppercasePayloadSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &durable::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        // Uppercase the JSON rendering of the payload value.
        Ok(value.to_string().to_uppercase())
    }

    fn deserialize(
        &self,
        data: &str,
        _context: &durable::SerdesContext,
    ) -> Result<serde_json::Value, durable::BoxError> {
        Ok(serde_json::from_str(data)?)
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
