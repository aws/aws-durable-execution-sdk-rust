//! Conformance requirement 5-16: invoke with custom result serdes.
//!
//! The custom result serdes uppercases the invoke result on deserialization.

use aws_durable_execution_sdk_rust as durable;
use std::any::Any;

/// Custom serdes that uppercases the result on deserialization.
///
/// The Go SDK equivalent does `*s = strings.ToUpper(string(data))` which
/// directly assigns the uppercased raw bytes (including JSON quotes) as the
/// final string value. Our SDK applies an additional `serde_json::from_str`
/// after `deserialize_from_string`, so we must JSON-encode the uppercased
/// raw payload to preserve it through that extra parse step.
#[derive(Debug)]
struct UppercaseResultSerdes;

impl durable::Serdes for UppercaseResultSerdes {
    fn serialize(&self, _value: &dyn Any) -> Result<Vec<u8>, durable::BoxError> {
        Ok(Vec::new())
    }

    fn deserialize_bytes(
        &self,
        _bytes: &[u8],
        _type_name: &str,
    ) -> Result<Box<dyn Any + Send>, durable::BoxError> {
        Ok(Box::new(()))
    }

    fn serialize_to_string(&self, json_str: &str) -> Result<String, durable::BoxError> {
        Ok(json_str.to_owned())
    }

    fn deserialize_from_string(&self, payload: &str) -> Result<String, durable::BoxError> {
        // Uppercase the raw payload (including JSON quotes), matching the Go
        // SDK's `strings.ToUpper(string(data))` semantics. Then re-encode as
        // JSON so the SDK's subsequent `serde_json::from_str` produces the
        // uppercased value as the final typed result.
        let uppercased = payload.to_uppercase();
        Ok(serde_json::to_string(&uppercased)?)
    }
}

/// Handler: invoke with uppercasing result serdes.
async fn handler(event: String, ctx: durable::DurableContext) -> Result<String, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let result = ctx
        .invoke::<String, _>(&target, &event)
        .serdes(UppercaseResultSerdes)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
