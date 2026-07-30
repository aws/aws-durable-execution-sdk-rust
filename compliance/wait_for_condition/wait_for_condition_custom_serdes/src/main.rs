//! Conformance handler for requirement 6-11: custom serdes.
//! Custom serializer adds "ENC:" prefix; custom deserializer strips it.

use aws_durable_execution_sdk_rust as durable;
use durable::{BoxError, Serdes, WaitDecision};
use std::time::Duration;

/// Custom serdes that adds/strips an "ENC:" prefix for string state.
#[derive(Debug)]
struct PrefixSerdes;

impl Serdes for PrefixSerdes {
    fn serialize_to_string(&self, json_str: &str) -> Result<String, BoxError> {
        // json_str is a JSON-quoted string like "\"hello\"".
        // Parse it, add prefix, re-serialize.
        let raw: String = serde_json::from_str(json_str)?;
        let prefixed = format!("ENC:{raw}");
        Ok(serde_json::to_string(&prefixed)?)
    }

    fn deserialize_from_string(&self, payload: &str) -> Result<String, BoxError> {
        // payload is a JSON-quoted string like "\"ENC:hello\"".
        // Parse it, strip prefix, re-serialize.
        let raw: String = serde_json::from_str(payload)?;
        let stripped = raw.strip_prefix("ENC:").unwrap_or(&raw).to_owned();
        Ok(serde_json::to_string(&stripped)?)
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
                .wait_strategy_fn(Box::new(|state: String, _attempt| {
                    if state.len() >= 2 {
                        WaitDecision::complete()
                    } else {
                        WaitDecision::continue_with(Duration::from_secs(1))
                    }
                }))
                .await?;
            Ok(result)
        },
    )
    .await
}
