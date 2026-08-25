//! Conformance handler for requirement 6-7: check function throws (uncaught).

use aws_durable_execution_sdk as durable;
use durable::builders::wait_for_condition::WaitDecision;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let result: serde_json::Value = ctx
                .wait_for_condition(
                    |_ctx, _state: serde_json::Value| async move {
                        Err("check function failed".into())
                    },
                    serde_json::Value::Null,
                )
                .wait_strategy_fn(
                    |_state: serde_json::Value, _attempt| WaitDecision::complete(),
                )
                .await?;
            Ok(result)
        },
    )
    .await
}
