//! Poll a condition until it is satisfied, with a bounded wait strategy.
//!
//! [`DurableContext::wait_for_condition`](aws_durable_execution_sdk::DurableContext::wait_for_condition)
//! repeatedly runs a check, carrying state between attempts, and waits for a
//! delay between polls. The invocation suspends during that wait only when no
//! other SDK work can make progress. This handler runs no other SDK work
//! during its waits, so its invocation suspends at each wait and the polling
//! does not hold the invocation open. The state (here a counter starting at
//! 0) is checkpointed each attempt and survives across resumes. A check
//! whose outcome is not recorded when an invocation ends runs again from
//! the initial state, so the counter then starts over. The strategy returns
//! [`WaitDecision::complete`](aws_durable_execution_sdk::builders::wait_for_condition::WaitDecision::complete)
//! once the counter reaches the requested threshold, or
//! [`WaitDecision::continue_with`](aws_durable_execution_sdk::builders::wait_for_condition::WaitDecision::continue_with)
//! to poll again after a delay.
//!
//! When no check is interrupted, the threshold from the event is small, so
//! this example finishes in a handful of polls. A real check would test an
//! external condition (a file landed, a job finished) instead of a counter.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::wait_for_condition::WaitDecision;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::init_default_subscriber()?;
    durable::run(|threshold: i32, ctx: durable::DurableContext| async move {
        let count = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
            .wait_strategy_fn(move |state: i32, _attempt| {
                if state >= threshold {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })
            .name("poll-until-ready")
            .await?;
        Ok(count)
    })
    .await
}
