//! Conformance requirement 9-14: Map configured with a custom per-item
//! serializer round-trips each iteration result.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};
use durable::serdes::SerdesContext;

/// Custom serdes that wraps with "wrapped:" prefix.
///
/// The SDK hands the serdes the item's typed value directly, so this
/// handler prepends the prefix to the raw string — the same shape as the
/// Go and Python reference handlers, with no decode step to compensate
/// for.
#[derive(Debug)]
struct WrapSerdes;

impl Serdes<String> for WrapSerdes {
    async fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(format!("wrapped:{value}"))
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(wire.strip_prefix("wrapped:").unwrap_or(&wire).to_owned())
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

    /// The SDK hands the serdes the item value typed, so for item `"X"` the
    /// handler sees `String::from("X")` and produces `wrapped:X` — the wire
    /// form requirement 9-14 asserts.
    #[tokio::test]
    async fn wraps_the_item_value_and_round_trips() {
        let serdes = WrapSerdes;
        let context = SerdesContext::new("op-1", "arn:test");

        let payload = serdes
            .serialize("X".to_owned(), context.clone())
            .await
            .expect("serialize must succeed");
        assert_eq!(
            payload, "wrapped:X",
            "requirement 9-14 asserts this exactly"
        );

        let back: String = serdes
            .deserialize(payload, context)
            .await
            .expect("deserialize must succeed");
        assert_eq!(back, "X", "the transform must be reversible");
    }
}
