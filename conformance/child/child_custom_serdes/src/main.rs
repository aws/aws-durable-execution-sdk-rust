//! Conformance requirement 3-14: child with custom serdes.

use aws_durable_execution_sdk as durable;

/// Uppercases the serialized result.
#[derive(Debug)]
struct UppercaseSerdes;

impl durable::Serdes<serde_json::Value> for UppercaseSerdes {
    fn serialize(
        &self,
        value: serde_json::Value,
        _context: durable::serdes::SerdesContext,
    ) -> impl Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>> + Send {
        // The value arrives typed, so the raw string needs no quote-stripping.
        std::future::ready(Ok(value.as_str().unwrap_or_default().to_uppercase()))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> impl Future<Output = Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>>> + Send
    {
        // Reverse: the raw payload is the string value itself.
        std::future::ready(Ok(serde_json::Value::String(wire)))
    }
}

/// Handler: child with custom serdes that uppercases.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let v: serde_json::Value = child_ctx
                .step(move |_| {
                    let e = event.clone();
                    async move { Ok(e) }
                })
                .await?;
            Ok(v)
        })
        .name("serdes-child")
        .serdes(UppercaseSerdes)
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
