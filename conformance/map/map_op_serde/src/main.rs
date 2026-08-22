//! Conformance requirement 9-19: Map with an operation-level (whole-result)
//! serdes: serialize on a fresh operation.
//!
//! Uses `.result_serdes()` with a custom serializer that emits
//! `"OPSERDE:X,Y"` for the whole map result payload.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};
use durable::serdes::SerdesContext;

/// Custom operation-level serializer that emits `OPSERDE:<comma-joined results>`.
///
/// The whole-batch summary a `result_serdes` transforms is the SDK's
/// `BatchSummary`, an opaque type. The transform is therefore written as a
/// type-agnostic blanket `impl<T> Serdes<T>` that works through the value's
/// serde (JSON) representation: `{"results": [...], "reason": "..."}`.
#[derive(Debug)]
struct OpSerdes;

impl<T> Serdes<T> for OpSerdes
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    async fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Work through the summary's JSON representation.
        let rendered: serde_json::Value = serde_json::from_str(&serde_json::to_string(&value)?)?;
        let results = rendered
            .get("results")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        item.get("result")
                            .and_then(serde_json::Value::as_str)
                            .map(|s| {
                                // Each item carries its own wire string: a
                                // JSON-encoded string here, so strip the quotes.
                                s.trim_matches('"').to_owned()
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(format!("OPSERDE:{}", results.join(",")))
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
        let body = wire
            .strip_prefix("OPSERDE:")
            .ok_or("missing OPSERDE: prefix")?;
        let items: Vec<&str> = if body.is_empty() {
            Vec::new()
        } else {
            body.split(',').collect()
        };
        // Reconstruct the batch summary's JSON representation, then parse it
        // back into the summary type.
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
        Ok(serde_json::from_str(&serde_json::to_string(&payload)?)?)
    }
}

/// Handler: map items to uppercase with operation-level serdes.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let items = vec!["x".to_owned(), "y".to_owned()];

    let results: Vec<String> = ctx
        .map(items, |_child, item, _idx| async move {
            Ok(item.to_uppercase())
        })
        .name("op-serde")
        .max_concurrency(1)
        .result_serdes(OpSerdes)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
