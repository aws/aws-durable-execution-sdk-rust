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

use aws_durable_execution_sdk_rust as durable;
use durable::{Serdes, SerdesContext};

/// A serdes that uppercases the JSON wire form (illustrative).
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
