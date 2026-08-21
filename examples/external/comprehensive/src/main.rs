//! Capstone: a single durable function exercising every core operation.
//!
//! In order it runs a [`step`], a durable [`wait`], a
//! [`run_in_child_context`], a [`map`] fan-out, a chained [`invoke`] of the
//! companion `invoke_target`, and a [`wait_for_condition`] poll, then returns
//! a summary of all of them. It is the end-to-end reference for how the pieces
//! compose in one handler. The target function is named through the
//! `TARGET_FUNCTION_NAME` environment variable.
//!
//! [`step`]: aws_durable_execution_sdk_rust::DurableContext::step
//! [`wait`]: aws_durable_execution_sdk_rust::DurableContext::wait
//! [`run_in_child_context`]: aws_durable_execution_sdk_rust::DurableContext::run_in_child_context
//! [`map`]: aws_durable_execution_sdk_rust::DurableContext::map
//! [`invoke`]: aws_durable_execution_sdk_rust::DurableContext::invoke
//! [`wait_for_condition`]: aws_durable_execution_sdk_rust::DurableContext::wait_for_condition

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::builders::wait_for_condition::WaitDecision;

/// Runs one of each core operation and returns a JSON summary.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    // 1. A durable step.
    let greeting = ctx
        .step(|_| async { Ok("hello".to_owned()) })
        .name("greet")
        .await?;

    // 2. A durable wait.
    ctx.wait(Duration::from_secs(1)).name("cooldown").await?;

    // 3. An isolated child context.
    let doubled = ctx
        .run_in_child_context(|child| async move {
            let value = child.step(|_| async { Ok(21) }).name("compute").await?;
            Ok(value * 2)
        })
        .name("branch")
        .await?;

    // 4. A bounded map fan-out.
    let scaled: Vec<i32> = ctx
        .map(vec![1, 2, 3], |_child, item: i32, _idx| async move {
            Ok(item * 10)
        })
        .name("scale")
        .await?;

    // 5. A chained invoke of the companion target.
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let echoed = ctx
        .invoke::<serde_json::Value, _>(&target, serde_json::json!({"from": "comprehensive"}))
        .name("delegate")
        .await?;

    // 6. A bounded condition poll.
    let counted = ctx
        .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
        .wait_strategy_fn(|state: i32, _attempt| {
            if state >= 2 {
                WaitDecision::complete()
            } else {
                WaitDecision::continue_with(Duration::from_secs(1))
            }
        })
        .name("gate")
        .await?;

    Ok(serde_json::json!({
        "greeting": greeting,
        "child_doubled": doubled,
        "mapped": scaled,
        "echoed": echoed,
        "condition_count": counted,
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
