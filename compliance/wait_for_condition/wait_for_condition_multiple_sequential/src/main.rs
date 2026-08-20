//! Conformance handler for requirement 6-13: multiple sequential.
//! Two sequential wait_for_condition ops, first result seeds the second.

use aws_durable_execution_sdk_rust as durable;
use durable::builders::wait_for_condition::WaitDecision;
use std::time::Duration;

fn make_strategy(threshold: i32) -> impl Fn(i32, u32) -> WaitDecision + Send + Sync {
    move |state: i32, _attempt| {
        if state >= threshold {
            WaitDecision::complete()
        } else {
            WaitDecision::continue_with(Duration::from_secs(1))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let first = ctx
                .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
                .wait_strategy_fn(make_strategy(2))
                .await?;

            let second = ctx
                .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, first)
                .wait_strategy_fn(make_strategy(4))
                .await?;

            Ok(second)
        },
    )
    .await
}
