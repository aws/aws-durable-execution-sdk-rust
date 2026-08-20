//! Conformance requirement 5-16: invoke with custom result serdes.
//!
//! The custom result serdes uppercases the invoke result on deserialization.

use aws_durable_execution_sdk_rust as durable;

/// Custom serdes that uppercases the result on deserialization.
///
/// The Go SDK equivalent does `*s = strings.ToUpper(string(data))`, which
/// assigns the uppercased raw bytes (JSON quotes included) as the final string
/// value. Returning `Value::String(data.to_uppercase())` reproduces that
/// exactly: the SDK deserializes the returned value into `String`, so no
/// re-encoding step is needed to survive an extra parse.
#[derive(Debug)]
struct UppercaseResultSerdes;

impl durable::Serdes for UppercaseResultSerdes {
    fn deserialize(
        &self,
        data: &str,
        _context: &durable::serdes::SerdesContext,
    ) -> Result<serde_json::Value, durable::BoxError> {
        Ok(serde_json::Value::String(data.to_uppercase()))
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
