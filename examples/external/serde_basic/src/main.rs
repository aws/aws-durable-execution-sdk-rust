//! Attach a custom [`Serdes`] to a single operation.
//!
//! Every operation checkpoints its result through a serializer. The default is
//! JSON; supplying a custom [`Serdes`] lets you transform the wire form —
//! compress it, encrypt it, or (as here, illustratively) uppercase it. The SDK
//! calls the serdes at every serialization point and reverses it on replay, so
//! the transform is transparent to the rest of the handler.
//!
//! [`Serdes`]: aws_durable_execution_sdk_rust::Serdes

use std::any::Any;

use aws_durable_execution_sdk_rust as durable;
use durable::Serdes;

/// A serdes that uppercases the JSON wire form. Illustrative of the transform
/// seam; production code would compress or encrypt here.
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

/// Runs a step whose result is checkpointed through the custom serdes.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let value = ctx
        .step(|_| async { Ok("hello".to_owned()) })
        .name("produce")
        .serdes(UppercaseSerdes)
        .await?;
    Ok(value)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
