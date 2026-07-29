//! Conformance requirement 9-20: Map with an operation-level serdes —
//! deserialize on replay (wait after the map).
//!
//! Uses `.result_serdes()` with the same `OpSerdes` as 9-19. The wait after
//! the map forces a replay where the SDK deserializes the checkpointed map
//! result through the custom serializer.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};
use std::any::Any;

/// Custom operation-level serializer (same as 9-19).
///
/// Serializes the entire batch result JSON into "OPSERDE:X,Y" format.
/// On deserialization, reconstructs the batch JSON from that compact form.
#[derive(Debug)]
struct OpSerdes;

impl Serdes for OpSerdes {
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
        Err("OpSerdes::deserialize_bytes not used".into())
    }

    fn serialize_to_string(
        &self,
        json_str: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Parse the batch result JSON and extract item results.
        let payload: serde_json::Value = serde_json::from_str(json_str)?;
        let results = payload
            .get("results")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        item.get("result")
                            .and_then(serde_json::Value::as_str)
                            .map(|s| {
                                // The result is a JSON-encoded string — strip outer quotes.
                                s.trim_matches('"').to_owned()
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(format!("OPSERDE:{}", results.join(",")))
    }

    fn deserialize_from_string(
        &self,
        payload: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let data = payload
            .strip_prefix("OPSERDE:")
            .ok_or("missing OPSERDE: prefix")?;
        let items: Vec<&str> = if data.is_empty() {
            Vec::new()
        } else {
            data.split(',').collect()
        };
        // Reconstruct the batch checkpoint JSON with string status/reason
        // matching the SDK's BatchCheckpointPayload format.
        let results: Vec<serde_json::Value> = items
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                serde_json::json!({
                    "index": i,
                    "status": "SUCCEEDED",
                    "result": format!("\"{val}\"")
                })
            })
            .collect();
        let payload = serde_json::json!({
            "results": results,
            "reason": "ALL_COMPLETED"
        });
        Ok(serde_json::to_string(&payload)?)
    }
}

/// Handler: map then wait (op-serde replay scenario).
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["x".to_owned(), "y".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move {
            Ok(item.to_uppercase())
        })
        .name("op-serde-replay")
        .max_concurrency(1)
        .result_serdes(OpSerdes)
        .await?;

    // Suspend after map; on replay the SDK deserializes the checkpointed result.
    ctx.wait(Duration::from_secs(1)).await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
