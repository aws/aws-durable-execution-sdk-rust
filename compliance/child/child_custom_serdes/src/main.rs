//! Conformance requirement 3-14: child with custom serdes.

use aws_durable_execution_sdk_rust as durable;
use std::any::Any;

/// Uppercases the serialized result.
#[derive(Debug)]
struct UppercaseSerdes;

impl durable::Serdes for UppercaseSerdes {
    fn serialize(
        &self,
        _value: &dyn Any,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Vec::new())
    }

    fn deserialize_bytes(
        &self,
        _bytes: &[u8],
        _type_name: &str,
    ) -> Result<Box<dyn Any + Send>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(()))
    }

    fn serialize_to_string(
        &self,
        json_str: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Strip surrounding JSON quotes if present, then uppercase.
        let inner = json_str
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(json_str);
        Ok(inner.to_uppercase())
    }

    fn deserialize_from_string(
        &self,
        payload: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Reverse: wrap raw payload back in JSON string form for serde_json.
        Ok(format!("\"{}\"", payload))
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
