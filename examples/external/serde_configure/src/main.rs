//! Share one [`Serdes`] instance across operations with an `Arc`.
//!
//! Serdes are configured per operation — there is no execution-wide slot,
//! because a single erased slot cannot represent `Serdes<T>` for every
//! operation output type. To apply one configured instance across a
//! handler, wrap it in an [`Arc`](std::sync::Arc) and clone the handle into
//! each operation: `Arc<S>` forwards to `S`, so the same instance (and any
//! state or configuration it carries) serves every operation and output
//! type it supports.
//!
//! [`Serdes`]: aws_durable_execution_sdk_rust::Serdes

use std::sync::Arc;

use aws_durable_execution_sdk_rust as durable;
use durable::Serdes;
use durable::serdes::SerdesContext;

/// A serdes that uppercases the JSON wire form (illustrative). The blanket
/// implementation over every JSON-able `T` is what lets ONE instance serve
/// operations with different output types.
#[derive(Debug)]
struct UppercaseSerdes;

impl<T> Serdes<T> for UppercaseSerdes
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    async fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(serde_json::to_string(&value)?.to_uppercase())
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<T, durable::BoxError> {
        Ok(serde_json::from_str(&wire.to_lowercase())?)
    }
}

/// Runs two steps with different output types; both results are
/// checkpointed through the SAME shared serdes instance.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let shared = Arc::new(UppercaseSerdes);

    let value = ctx
        .step(|_| async { Ok("hello".to_owned()) })
        .name("produce")
        .serdes(Arc::clone(&shared))
        .await?;

    // The same instance serves a step with a different output type.
    let count: u32 = ctx
        .step(|_| async { Ok(2_u32) })
        .name("count")
        .serdes(shared)
        .await?;

    Ok(format!("{value}x{count}"))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
