//! Conformance requirement 5-15: invoke with custom payload serdes.
//!
//! The custom payload serializer uppercases string payloads before sending
//! to the target.

use aws_durable_execution_sdk as durable;

/// Custom payload serializer that uppercases the JSON of the payload.
#[derive(Debug)]
struct UppercasePayloadSerdes;

impl durable::PayloadSerializer for UppercasePayloadSerdes {
    async fn serialize_payload<T>(
        &self,
        value: T,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError>
    where
        T: serde::Serialize + Send + 'static,
    {
        // Uppercase the JSON rendering of the payload value.
        Ok(serde_json::to_string(&value)?.to_uppercase())
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
        .invoke::<serde_json::Value>(&target, event)
        .payload_serdes(UppercasePayloadSerdes)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
