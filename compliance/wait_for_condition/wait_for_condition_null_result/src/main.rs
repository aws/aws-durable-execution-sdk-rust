//! Conformance handler for requirement 6-10: null result.
//! Check returns null, strategy stops immediately.

use aws_durable_execution_sdk_rust as durable;
use durable::WaitDecision;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let result: serde_json::Value = ctx
                .wait_for_condition(
                    |_ctx, _state: serde_json::Value| async move { Ok(serde_json::Value::Null) },
                    serde_json::Value::Null,
                )
                .wait_strategy_fn(Box::new(|_state: serde_json::Value, _attempt| {
                    WaitDecision::complete()
                }))
                .await?;
            Ok(result)
        },
    )
    .await
}
