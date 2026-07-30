//! Conformance requirement 9-14: Map configured with a custom per-item
//! serializer round-trips each iteration result.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};

/// Custom serdes that wraps with "wrapped:" prefix.
///
/// A `Serdes` is a transformation *around* JSON: the SDK hands
/// `serialize_to_string` the `serde_json` encoding of the value (for a
/// `String` item that is `"X"`, quotes included) and feeds whatever
/// `deserialize_from_string` returns back to `serde_json`. The requirement
/// asserts the checkpointed iteration payload is exactly `wrapped:X`, so this
/// serdes decodes the JSON string before wrapping and re-encodes it after
/// unwrapping. Every operation path — step, child, invoke, map/parallel item
/// — hands the serdes the same shape, so this is the one rule, not a
/// map-specific accommodation.
#[derive(Debug)]
struct WrapSerdes;

impl Serdes for WrapSerdes {
    fn serialize_to_string(&self, json_str: &str) -> Result<String, durable::BoxError> {
        let raw: String = serde_json::from_str(json_str)?;
        Ok(format!("wrapped:{raw}"))
    }

    fn deserialize_from_string(&self, payload: &str) -> Result<String, durable::BoxError> {
        let raw = payload.strip_prefix("wrapped:").unwrap_or(payload);
        Ok(serde_json::to_string(raw)?)
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

    /// The payload the requirement asserts (`wrapped:X`, no JSON quotes) must
    /// come out of the serdes when it is handed what the SDK actually hands
    /// it: the `serde_json` encoding of the item result.
    #[test]
    fn wraps_the_decoded_item_and_round_trips() {
        let serdes = WrapSerdes;
        let json_str = serde_json::to_string("X").expect("a string is JSON-able");
        assert_eq!(json_str, "\"X\"", "the SDK hands over the JSON encoding");

        let payload = serdes
            .serialize_to_string(&json_str)
            .expect("serialize must succeed");
        assert_eq!(payload, "wrapped:X", "requirement 9-14 asserts this exactly");

        let back = serdes
            .deserialize_from_string(&payload)
            .expect("deserialize must succeed");
        assert_eq!(back, json_str, "the transform must be reversible");
    }
}
