//! Conformance handler for requirement 6-11: custom serdes.
//! Custom serializer adds "ENC:" prefix; custom deserializer strips it.

use aws_durable_execution_sdk_rust as durable;
use durable::{BoxError, Serdes};
use durable::builders::wait_for_condition::WaitDecision;
use durable::serdes::SerdesContext;
use std::time::Duration;

/// Custom serdes that adds/strips an "ENC:" prefix for string state.
#[derive(Debug)]
struct PrefixSerdes;

impl Serdes for PrefixSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, BoxError> {
        // The state arrives typed, so the raw string needs no JSON decode.
        let raw = value.as_str().unwrap_or_default();
        Ok(serde_json::to_string(&format!("ENC:{raw}"))?)
    }

    fn deserialize(
        &self,
        data: &str,
        _context: &SerdesContext,
    ) -> Result<serde_json::Value, BoxError> {
        // `data` is a JSON-quoted string like "\"ENC:hello\"".
        let raw: String = serde_json::from_str(data)?;
        let stripped = raw.strip_prefix("ENC:").unwrap_or(&raw).to_owned();
        Ok(serde_json::Value::String(stripped))
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
