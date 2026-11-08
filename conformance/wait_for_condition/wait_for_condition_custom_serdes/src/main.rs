//! Conformance handler for requirement 6-11: custom serdes.
//! Custom serializer adds "ENC:" prefix; custom deserializer strips it.

use aws_durable_execution_sdk as durable;
use durable::builders::wait_for_condition::WaitDecision;
use durable::serdes::SerdesContext;
use durable::{BoxError, Serdes};
use std::time::Duration;

/// Custom serdes that adds/strips an "ENC:" prefix for string state.
#[derive(Debug)]
struct PrefixSerdes;

impl Serdes<String> for PrefixSerdes {
    fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        // The state arrives typed, so it needs no JSON decode first.
        std::future::ready(serde_json::to_string(&format!("ENC:{value}")).map_err(Into::into))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(strip_enc_prefix(&wire))
    }
}

/// Decodes `wire`, a JSON-quoted string like "\"ENC:hello\"", and strips
/// its "ENC:" prefix.
fn strip_enc_prefix(wire: &str) -> Result<String, BoxError> {
    let raw: String = serde_json::from_str(wire)?;
    Ok(raw.strip_prefix("ENC:").unwrap_or(&raw).to_owned())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let result = ctx
                .wait_for_condition(
                    |_ctx, state: String| async move { Ok(format!("{state}x")) },
                    String::new(),
                )
                .serdes(PrefixSerdes)
                .wait_strategy_fn(|state: String, _attempt| {
                    if state.len() >= 2 {
                        WaitDecision::complete()
                    } else {
                        WaitDecision::continue_with(Duration::from_secs(1))
                    }
                })
                .await?;
            Ok(result)
        },
    )
    .await
}
