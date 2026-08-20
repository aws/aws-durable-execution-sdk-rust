//! Conformance handler for requirement 6-2: wait_for_condition immediate.
//! Condition is already met on first check (state >= 5).

use aws_durable_execution_sdk_rust as durable;
use durable::builders::wait_for_condition::WaitDecision;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |initial_state: i32, ctx: durable::DurableContext| async move {
            let result = ctx
                .wait_for_condition(|_ctx, state: i32| async move { Ok(state) }, initial_state)
                .wait_strategy_fn(|state: i32, _attempt| {
                    if state >= 5 {
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
