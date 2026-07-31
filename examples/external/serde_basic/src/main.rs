//! Attach a custom [`Serdes`] to a single operation.
//!
//! Every operation checkpoints its result through a serializer. The default is
//! JSON; supplying a custom [`Serdes`] lets you transform the wire form —
//! compress it, encrypt it, or (as here, illustratively) uppercase it. The SDK
//! calls the serdes at every serialization point and reverses it on replay, so
//! the transform is transparent to the rest of the handler.
//!
//! [`Serdes`]: aws_durable_execution_sdk_rust::Serdes

use aws_durable_execution_sdk_rust as durable;
use durable::{Serdes, SerdesContext};

/// A serdes that uppercases the JSON wire form. Illustrative of the transform
/// seam; production code would compress or encrypt here.
#[derive(Debug)]
struct UppercaseSerdes;

impl Serdes for UppercaseSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, durable::BoxError> {
        // The serdes receives the operation's value erased to
        // `serde_json::Value`, not pre-rendered JSON text.
        Ok(value.to_string().to_uppercase())
    }

    fn deserialize(
        &self,
        data: &str,
        _context: &SerdesContext,
    ) -> Result<serde_json::Value, durable::BoxError> {
        Ok(serde_json::from_str(data)?)
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
