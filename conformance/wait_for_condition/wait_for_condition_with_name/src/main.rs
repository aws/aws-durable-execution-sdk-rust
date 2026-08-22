//! Conformance handler for requirement 6-3: wait_for_condition with name.
//! Explicit operation name "poll-status".

use aws_durable_execution_sdk_rust as durable;
use durable::builders::wait_for_condition::WaitDecision;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|threshold: i32, ctx: durable::DurableContext| async move {
        let result = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
            .name("poll-status")
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
