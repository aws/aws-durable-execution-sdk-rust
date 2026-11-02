//! Conformance requirement 9-20: Map with an operation-level serdes:
//! deserialize on replay (wait after the map).
//!
//! Uses `.result_serdes()` with the same `OpSerdes` as 9-19. The wait after
//! the map forces a replay where the SDK deserializes the checkpointed map
//! result through the custom serializer.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::serdes::SerdesContext;
use durable::{DurableContext, Serdes};

/// Custom operation-level serializer that emits `OPSERDE:<comma-joined results>`.
///
/// A `result_serdes` is a `Serdes<String>` over the batch record's JSON
/// text, `{"all": [...], "completionReason": "..."}`. `serialize` reads the
/// item results from that text, and `deserialize` rebuilds the same record
/// text from them.
#[derive(Debug)]
struct OpSerdes;

/// One succeeded item of the rebuilt batch record, in the SDK's field order.
#[derive(serde::Serialize)]
struct RecordItem<'a> {
    index: usize,
    status: &'static str,
    name: String,
    result: &'a str,
}

/// The rebuilt batch record, in the SDK's field order.
#[derive(serde::Serialize)]
struct Record<'a> {
    all: Vec<RecordItem<'a>>,
    #[serde(rename = "completionReason")]
    completion_reason: &'static str,
}

impl Serdes<String> for OpSerdes {
    fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(op_serialize(&value))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(op_deserialize(&wire))
    }
}

/// Writes `OPSERDE:<comma-joined results>` from a batch record text.
fn op_serialize(value: &str) -> Result<String, durable::BoxError> {
    let record: serde_json::Value = serde_json::from_str(value)?;
    let results = record
        .get("all")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    // Each item carries its value as JSON: a string
                    // here, with the default item serdes.
                    item.get("result")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok(format!("OPSERDE:{}", results.join(",")))
}

/// Rebuilds the batch record text from `OPSERDE:<comma-joined results>`.
fn op_deserialize(wire: &str) -> Result<String, durable::BoxError> {
    let body = wire
        .strip_prefix("OPSERDE:")
        .ok_or("missing OPSERDE: prefix")?;
    let items: Vec<&str> = if body.is_empty() {
        Vec::new()
    } else {
        body.split(',').collect()
    };
    // Rebuild the batch record text.
    let record = Record {
        all: items
            .into_iter()
            .enumerate()
            .map(|(index, result)| RecordItem {
                index,
                status: "SUCCEEDED",
                name: format!("map-item-{index}"),
                result,
            })
            .collect(),
        completion_reason: "ALL_COMPLETED",
    };
    Ok(serde_json::to_string(&record)?)
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
