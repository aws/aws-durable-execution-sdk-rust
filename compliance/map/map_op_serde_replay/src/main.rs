//! Conformance requirement 9-20: Map with an operation-level serdes —
//! deserialize on replay (wait after the map).
//!
//! Uses `.result_serdes()` with the same `OpSerdes` as 9-19. The wait after
//! the map forces a replay where the SDK deserializes the checkpointed map
//! result through the custom serializer.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes, SerdesContext};

/// Custom operation-level serializer (same as 9-19).
///
/// Serializes the entire batch result into "OPSERDE:X,Y" format. On
/// deserialization, reconstructs the batch value from that compact form.
#[derive(Debug)]
struct OpSerdes;

impl Serdes for OpSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // The batch payload arrives as a structured value — no parse step.
        let results = value
            .get("results")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        item.get("result")
                            .and_then(serde_json::Value::as_str)
                            .map(|s| {
                                // Each item carries its own wire string — a
                                // JSON-encoded string here, so strip the quotes.
                                s.trim_matches('"').to_owned()
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(format!("OPSERDE:{}", results.join(",")))
    }

    fn deserialize(
        &self,
        data: &str,
        _context: &SerdesContext,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let body = data
            .strip_prefix("OPSERDE:")
            .ok_or("missing OPSERDE: prefix")?;
        let items: Vec<&str> = if body.is_empty() {
            Vec::new()
        } else {
            body.split(',').collect()
        };
        // Reconstruct the batch checkpoint value with the string status/reason
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
        Ok(serde_json::json!({
            "results": results,
            "reason": "ALL_COMPLETED"
        }))
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
