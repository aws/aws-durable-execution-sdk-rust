//! Conformance requirement 9-19: Map with an operation-level (whole-result)
//! serdes — serialize on a fresh operation.
//!
//! Uses `.result_serdes()` with a custom serializer that emits
//! `"OPSERDE:X,Y"` for the whole map result payload.

use aws_durable_execution_sdk_rust as durable;
use durable::{DurableContext, Serdes};
use durable::serdes::SerdesContext;

/// Custom operation-level serializer that emits `OPSERDE:<comma-joined results>`.
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
                            .map(|s| s.trim_matches('"').to_owned())
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
        // Reverse: "OPSERDE:X,Y" → reconstruct the batch payload value.
        let body = data
            .strip_prefix("OPSERDE:")
            .ok_or("missing OPSERDE: prefix")?;
        let items: Vec<&str> = if body.is_empty() {
            Vec::new()
        } else {
            body.split(',').collect()
        };
        let results: Vec<serde_json::Value> = items
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                serde_json::json!({
                    "index": i,
                    "status": 1,
                    "result": format!("\"{val}\"")
                })
            })
            .collect();
        Ok(serde_json::json!({
            "results": results,
            "reason": 1
        }))
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
