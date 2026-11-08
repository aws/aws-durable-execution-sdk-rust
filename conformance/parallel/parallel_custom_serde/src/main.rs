//! Conformance requirement 8-15: Parallel with a custom per-branch serde.

use aws_durable_execution_sdk as durable;
use durable::serdes::SerdesContext;
use durable::{Branch, DurableContext, Serdes};

/// Custom serdes that wraps strings as `{"wrapped": "value"}`.
#[derive(Debug)]
struct WrappedSerdes;

impl Serdes<String> for WrappedSerdes {
    fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(
            serde_json::to_string(&serde_json::json!({"wrapped": value})).map_err(Into::into),
        )
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(unwrap_value(&wire))
    }
}

/// Decodes `wire` and returns its "wrapped" field.
fn unwrap_value(wire: &str) -> Result<String, durable::BoxError> {
    let v: serde_json::Value = serde_json::from_str(wire)?;
    Ok(v.get("wrapped")
        .and_then(serde_json::Value::as_str)
        .ok_or("WrappedSerdes: missing 'wrapped' field")?
        .to_owned())
}

/// Handler: parallel with custom serdes wrapping branch results.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |_: DurableContext| async { Ok("x".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Ok("y".to_owned()) }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("serde")
        .max_concurrency(1)
        .serdes(WrappedSerdes)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
