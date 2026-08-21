//! Named step: attach a stable name for observability.
//!
//! Every operation takes an optional [`name`] via the builder chain. The name
//! appears in the execution history and in the structured log fields the SDK
//! attaches to each operation span, so a human reading a trace sees
//! `process-data` instead of an anonymous positional id. The name is metadata
//! only: it does not affect operation identity, which is always derived from
//! call order.
//!
//! This example also shows the natural way to take typed input: declare a
//! `Deserialize` struct and let `durable::run` deserialize the event into it.
//!
//! [`name`]: aws_durable_execution_sdk_rust::DurableContext::step

use aws_durable_execution_sdk_rust as durable;
use serde::Deserialize;

/// Handler input: the payload to process.
#[derive(Debug, Deserialize)]
struct Input {
    /// Value to fold into the step's output. Defaults when absent.
    #[serde(default = "default_data")]
    data: String,
}

fn default_data() -> String {
    "default".to_owned()
}

/// Runs a named step that transforms the input.
async fn handler(event: Input, ctx: durable::DurableContext) -> Result<String, durable::BoxError> {
    let result = ctx
        .step(move |_ctx| async move { Ok(format!("processed: {}", event.data)) })
        .name("process-data")
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
