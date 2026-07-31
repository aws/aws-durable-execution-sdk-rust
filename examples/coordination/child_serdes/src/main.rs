//! Custom serialization inside a child context.
//!
//! Any operation can carry a custom [`Serdes`] that controls how its result is
//! encoded for checkpointing. This is the seam for compression, encryption, or
//! an external store: the SDK calls the serdes at every serialization point and
//! reverses it on replay. A child context is a natural place to scope a serdes
//! to a subtree of work.
//!
//! This example applies an uppercasing serdes to a step inside a child. The
//! transform is illustrative — real implementations would compress or encrypt.
//! The same [`Serdes`] surface applies to large payloads without change, so a
//! serdes and a large-payload child compose directly.
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

/// Runs a step inside a child context using a custom serdes for its result.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let value = ctx
        .run_in_child_context(|child| async move {
            let value = child
                .step(|_| async { Ok("hello".to_owned()) })
                .name("produce")
                .serdes(UppercaseSerdes)
                .await?;
            Ok(value)
        })
        .name("serdes-child")
        .await?;
    Ok(value)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
