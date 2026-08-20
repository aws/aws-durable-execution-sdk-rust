//! Conformance requirement 1-6: step with custom serdes (uppercase).

use aws_durable_execution_sdk_rust as durable;
use durable::Serdes;
use durable::serdes::SerdesContext;

/// Custom serdes that uppercases the serialized form.
#[derive(Debug)]
struct UppercaseSerdes;

impl Serdes for UppercaseSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(value.to_string().to_uppercase())
    }

    fn deserialize(
        &self,
        data: &str,
        _context: &SerdesContext,
    ) -> Result<serde_json::Value, durable::BoxError> {
        Ok(serde_json::from_str(data)?)
    }
}

/// Handler: step with custom serdes.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input = event.as_str().unwrap_or("").to_owned();
    let result: String = ctx
        .step(move |_| async move { Ok(input) })
        .serdes(UppercaseSerdes)
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
