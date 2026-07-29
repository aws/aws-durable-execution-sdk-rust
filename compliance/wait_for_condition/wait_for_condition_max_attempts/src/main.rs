//! Conformance handler for requirement 6-6: wait_for_condition max attempts.
//! Condition never met, strategy exhausts after 3 attempts.

use aws_durable_execution_sdk_rust as durable;
use durable::WaitDecision;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let result = ctx
                .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
                .wait_strategy_fn(Box::new(|_state: i32, attempt| {
                    if attempt >= 3 {
                        WaitDecision::exhausted("max attempts exceeded")
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
