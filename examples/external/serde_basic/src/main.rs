//! Attach a custom [`Serdes`] to a single operation.
//!
//! Every operation checkpoints its result through a serializer. The default is
//! JSON; supplying a custom [`Serdes`] lets you transform the wire form:
//! compress it, encrypt it, or (as here, illustratively) reverse it. The SDK
//! calls the serdes at every serialization point and calls `deserialize` on
//! replay, so a transform whose `deserialize` exactly inverts `serialize` is
//! transparent to the rest of the handler: the step below observes `"hello"`
//! whether the value came from live execution or from a replayed checkpoint.
//!
//! [`Serdes`]: aws_durable_execution_sdk::Serdes

use aws_durable_execution_sdk as durable;
use durable::Serdes;
use durable::serdes::SerdesContext;

/// A serdes that reverses the JSON wire form: a self-inverse, lossless
/// transform. Illustrative of the transform seam; production code would
/// compress or encrypt here, pairing each `serialize` transform with its
/// exact inverse in `deserialize`.
#[derive(Debug)]
struct ReversedJsonSerdes;

impl Serdes<String> for ReversedJsonSerdes {
    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        // The serdes receives the operation's typed value directly, not
        // pre-rendered JSON text or an erased intermediate.
        Ok(serde_json::to_string(&value)?.chars().rev().collect())
    }

    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        // Undo the transform first, then parse the restored JSON: the
        // inverse of `serialize`, applied in the opposite order.
        let restored: String = wire.chars().rev().collect();
        Ok(serde_json::from_str(&restored)?)
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
        .serdes(ReversedJsonSerdes)
        .await?;
    Ok(value)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
