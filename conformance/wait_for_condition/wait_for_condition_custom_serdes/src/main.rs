//! Conformance handler for requirement 6-11: custom serdes.
//! Custom serializer adds "ENC:" prefix; custom deserializer strips it.

use aws_durable_execution_sdk as durable;
use durable::{BoxError, Serdes};
use durable::builders::wait_for_condition::WaitDecision;
use durable::serdes::SerdesContext;
use std::time::Duration;

/// Custom serdes that adds/strips an "ENC:" prefix for string state.
#[derive(Debug)]
struct PrefixSerdes;

impl Serdes<String> for PrefixSerdes {
    async fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> Result<String, BoxError> {
        // The state arrives typed, so it needs no JSON decode first.
        Ok(serde_json::to_string(&format!("ENC:{value}"))?)
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<String, BoxError> {
        // `wire` is a JSON-quoted string like "\"ENC:hello\"".
        let raw: String = serde_json::from_str(&wire)?;
        Ok(raw.strip_prefix("ENC:").unwrap_or(&raw).to_owned())
    }
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
