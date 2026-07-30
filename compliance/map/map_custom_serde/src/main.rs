//! Conformance requirement 9-14: Map configured with a custom per-item
//! serializer round-trips each iteration result.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};

/// Custom serdes that wraps with "wrapped:" prefix.
///
/// The SDK hands map/parallel item serdes the raw item payload: for a
/// `String` value `"X"`, the serdes receives `X` (no JSON quoting). This
/// matches the JS/Python reference implementations where the handler applies
/// `wrapped:${value}` / `f"wrapped:{value}"` directly to the native value.
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

#[cfg(test)]
#[allow(clippy::expect_used)] // reason: test assertions with descriptive messages
mod tests {
    use super::WrapSerdes;
    use aws_durable_execution_sdk_rust::Serdes;

    /// The SDK hands item serdes the raw payload (no JSON quoting for
    /// strings). For `String` item `"X"`, the serdes receives `X` directly
    /// and produces `wrapped:X` — the wire form requirement 9-14 asserts.
    #[test]
    fn wraps_the_raw_item_and_round_trips() {
        let serdes = WrapSerdes;
        // The SDK extracts the raw value before passing to serdes:
        // for String "X", the serdes receives "X" (the raw content, no quotes).
        let raw = "X";

        let payload = serdes
            .serialize_to_string(raw)
            .expect("serialize must succeed");
        assert_eq!(payload, "wrapped:X", "requirement 9-14 asserts this exactly");

        let back = serdes
            .deserialize_from_string(&payload)
            .expect("deserialize must succeed");
        assert_eq!(back, raw, "the transform must be reversible");
    }
}
