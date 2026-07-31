//! Conformance handler for requirement 6-5: wait_for_condition fixed delay.
//! Fixed 2-second delay between polls.

use aws_durable_execution_sdk_rust as durable;
use durable::WaitDecision;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|threshold: i32, ctx: durable::DurableContext| async move {
        let result = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
            .wait_strategy_fn(move |state: i32, attempt| {
                if state >= threshold {
                    WaitDecision::complete()
                } else if attempt >= 100 {
                    WaitDecision::exhausted("max attempts exceeded")
                } else {
                    WaitDecision::continue_with(Duration::from_secs(2))
                }
            })
            .await?;
        Ok(result)
    })
    .await
}
