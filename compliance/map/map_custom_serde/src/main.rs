//! Conformance requirement 9-14: Map configured with a custom per-item
//! serializer round-trips each iteration result.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};

/// Custom serdes that wraps with "wrapped:" prefix.
///
/// Serialization prepends `"wrapped:"` to the value; deserialization strips it.
#[derive(Debug)]
struct WrapSerdes;

impl Serdes for WrapSerdes {
    fn serialize_to_string(&self, s: &str) -> Result<String, durable::BoxError> {
        Ok(format!("wrapped:{s}"))
    }

    fn deserialize_from_string(&self, s: &str) -> Result<String, durable::BoxError> {
        Ok(s.strip_prefix("wrapped:").unwrap_or(s).to_owned())
    }
}

/// Handler: map with custom serdes.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["x".to_owned(), "y".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move {
            Ok(item.to_uppercase())
        })
        .name("serdes")
        .max_concurrency(1)
        .serdes(WrapSerdes)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
