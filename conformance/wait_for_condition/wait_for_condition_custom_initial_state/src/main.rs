//! Conformance handler for requirement 6-4: wait_for_condition custom initial state.
//! Initial state of 5, polls until threshold from input.

use aws_durable_execution_sdk_rust as durable;
use durable::builders::wait_for_condition::WaitDecision;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|threshold: i32, ctx: durable::DurableContext| async move {
        let result = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 5)
            .wait_strategy_fn(move |state: i32, _attempt| {
                if state >= threshold {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })
            .await?;
        Ok(result)
    })
    .await
}
