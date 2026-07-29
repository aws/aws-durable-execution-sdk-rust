//! Configure one default [`Serdes`] for the whole execution via [`Options`].
//!
//! Instead of attaching a serdes to each operation, set it once on
//! [`Options`] and compose the handler with
//! [`wrap`](aws_durable_execution_sdk_rust::wrap). Every operation that does
//! not override its own serdes then uses this default — the construction-time
//! analogue of the per-operation `serde_basic` example.
//!
//! [`Serdes`]: aws_durable_execution_sdk_rust::Serdes
//! [`Options`]: aws_durable_execution_sdk_rust::Options

use std::any::Any;

use aws_durable_execution_sdk_rust as durable;
use durable::Serdes;

/// A serdes that uppercases the JSON wire form (illustrative).
#[derive(Debug)]
struct UppercaseSerdes;

impl Serdes for UppercaseSerdes {
    fn serialize(&self, _value: &dyn Any) -> Result<Vec<u8>, durable::BoxError> {
        Ok(Vec::new())
    }

    fn deserialize_bytes(
        &self,
        _bytes: &[u8],
        _type_name: &str,
    ) -> Result<Box<dyn Any + Send>, durable::BoxError> {
        Ok(Box::new(()))
    }

    fn serialize_to_string(&self, json_str: &str) -> Result<String, durable::BoxError> {
        Ok(json_str.to_uppercase())
    }

    fn deserialize_from_string(&self, payload: &str) -> Result<String, durable::BoxError> {
        Ok(payload.to_owned())
    }
}

/// Runs a step; its result is checkpointed through the execution-wide serdes.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let value = ctx
        .step(|_| async { Ok("hello".to_owned()) })
        .name("produce")
        .await?;
    Ok(value)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder()
        .serdes(UppercaseSerdes)
        .build()?;
    let service = durable::wrap(handler, options);
    lambda_runtime::run(lambda_runtime::service_fn(service)).await
}
