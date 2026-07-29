//! Conformance handler for requirement 6-12: wait_for_condition then step.
//! Poll result feeds a subsequent step that multiplies by 10.

use aws_durable_execution_sdk_rust as durable;
use durable::WaitDecision;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|threshold: i32, ctx: durable::DurableContext| async move {
        let poll_result = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
            .wait_strategy_fn(Box::new(move |state: i32, _attempt| {
                if state >= threshold {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            }))
            .await?;

        let result = ctx
            .step(move |_| async move { Ok(poll_result * 10) })
            .await?;
        Ok(result)
    })
    .await
}
