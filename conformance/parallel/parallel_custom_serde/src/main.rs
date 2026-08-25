//! Conformance requirement 8-15: Parallel with a custom per-branch serde.

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext, Serdes};
use durable::serdes::SerdesContext;

/// Custom serdes that wraps strings as `{"wrapped": "value"}`.
#[derive(Debug)]
struct WrappedSerdes;

impl Serdes<String> for WrappedSerdes {
    async fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(serde_json::to_string(
            &serde_json::json!({"wrapped": value}),
        )?)
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        let v: serde_json::Value = serde_json::from_str(&wire)?;
        Ok(v.get("wrapped")
            .and_then(serde_json::Value::as_str)
            .ok_or("WrappedSerdes: missing 'wrapped' field")?
            .to_owned())
    }
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
