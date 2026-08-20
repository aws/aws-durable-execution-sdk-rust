//! Conformance requirement 9-14: Map configured with a custom per-item
//! serializer round-trips each iteration result.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};
use durable::serdes::SerdesContext;

/// Custom serdes that wraps with "wrapped:" prefix.
///
/// The SDK hands every serdes the value erased to `serde_json::Value`, so this
/// handler reads the string directly and prepends — the same shape as the Go
/// and Python reference handlers, with no decode step to compensate for.
#[derive(Debug)]
struct WrapSerdes;

impl Serdes for WrapSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(format!("wrapped:{}", value.as_str().unwrap_or_default()))
    }

    fn deserialize(
        &self,
        data: &str,
        _context: &SerdesContext,
    ) -> Result<serde_json::Value, durable::BoxError> {
        Ok(serde_json::Value::String(
            data.strip_prefix("wrapped:").unwrap_or(data).to_owned(),
        ))
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

#[cfg(test)]
#[allow(clippy::expect_used)] // reason: test assertions with descriptive messages
mod tests {
    use super::WrapSerdes;
    use aws_durable_execution_sdk_rust::Serdes;
    use aws_durable_execution_sdk_rust::serdes::SerdesContext;

    /// The SDK hands the serdes the item value as a `serde_json::Value`, so for
    /// item `"X"` the handler sees `Value::String("X")` and produces
    /// `wrapped:X` — the wire form requirement 9-14 asserts.
    #[test]
    fn wraps_the_item_value_and_round_trips() {
        let serdes = WrapSerdes;
        let context = SerdesContext::new("op-1", "arn:test");
        let value = serde_json::Value::String("X".to_owned());

        let payload = serdes
            .serialize(&value, &context)
            .expect("serialize must succeed");
        assert_eq!(
            payload, "wrapped:X",
            "requirement 9-14 asserts this exactly"
        );

        let back = serdes
            .deserialize(&payload, &context)
            .expect("deserialize must succeed");
        assert_eq!(back, value, "the transform must be reversible");
    }
}
