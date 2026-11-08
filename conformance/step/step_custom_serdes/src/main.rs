//! Conformance requirement 1-6: step with custom serdes (uppercase).

use aws_durable_execution_sdk as durable;
use durable::Serdes;
use durable::serdes::SerdesContext;

/// Custom serdes that uppercases the serialized form.
#[derive(Debug)]
struct UppercaseSerdes;

impl Serdes<String> for UppercaseSerdes {
    fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(
            serde_json::to_string(&value)
                .map(|json| json.to_uppercase())
                .map_err(Into::into),
        )
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(serde_json::from_str(&wire).map_err(Into::into))
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
