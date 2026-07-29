//! Conformance requirement 8-15: Parallel with a custom per-branch serde.

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext, Serdes};
use std::any::Any;

/// Custom serdes that wraps strings as `{"wrapped": "value"}`.
#[derive(Debug)]
struct WrappedSerdes;

impl Serdes for WrappedSerdes {
    fn serialize(&self, value: &dyn Any) -> Result<Vec<u8>, durable::BoxError> {
        let s = value
            .downcast_ref::<String>()
            .ok_or("WrappedSerdes: expected String")?;
        let wrapped = serde_json::json!({"wrapped": s});
        Ok(serde_json::to_vec(&wrapped)?)
    }

    fn deserialize_bytes(
        &self,
        bytes: &[u8],
        _type_name: &str,
    ) -> Result<Box<dyn Any + Send>, durable::BoxError> {
        let v: serde_json::Value = serde_json::from_slice(bytes)?;
        let s = v
            .get("wrapped")
            .and_then(serde_json::Value::as_str)
            .ok_or("WrappedSerdes: missing 'wrapped' field")?
            .to_owned();
        Ok(Box::new(s))
    }

    fn serialize_to_string(&self, s: &str) -> Result<String, durable::BoxError> {
        Ok(serde_json::to_string(&serde_json::json!({"wrapped": s}))?)
    }

    fn deserialize_from_string(&self, s: &str) -> Result<String, durable::BoxError> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let inner = v
            .get("wrapped")
            .and_then(serde_json::Value::as_str)
            .ok_or("WrappedSerdes: missing 'wrapped' field")?
            .to_owned();
        Ok(inner)
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
