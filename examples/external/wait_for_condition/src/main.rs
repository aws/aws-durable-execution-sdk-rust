//! Poll a condition until it is satisfied, with a bounded wait strategy.
//!
//! [`DurableContext::wait_for_condition`](aws_durable_execution_sdk_rust::DurableContext::wait_for_condition)
//! repeatedly runs a check, carrying state between attempts, and suspends for a
//! delay between polls so it does not hold the invocation open. The state
//! (here a counter starting at 0) is checkpointed each attempt and survives
//! across resumes. The strategy returns
//! [`WaitDecision::complete`](aws_durable_execution_sdk_rust::WaitDecision::complete)
//! once the counter reaches the requested threshold, or
//! [`WaitDecision::continue_with`](aws_durable_execution_sdk_rust::WaitDecision::continue_with)
//! to poll again after a delay.
//!
//! The threshold comes from the event and is small, so the loop is bounded to
//! a handful of polls — a real check would test an external condition (a file
//! landed, a job finished) instead of a counter.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::WaitDecision;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|threshold: i32, ctx: durable::DurableContext| async move {
        let count = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
            .wait_strategy_fn(Box::new(move |state: i32, _attempt| {
                if state >= threshold {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            }))
            .name("poll-until-ready")
            .await?;
        Ok(count)
    })
    .await
}
