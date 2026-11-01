//! Integration tests for the `LocalRunner` virtual clock and
//! `LocalRunner::checkpoint_latency`, through the crate's public API.
//!
//! The local backend keeps a virtual clock in seconds. Between invocations
//! it moves the clock to the earliest pending wait or retry end, and fires
//! only what is due then. A queued callback reply arrives before the clock
//! moves. `checkpoint_latency` delays each checkpoint response in tokio time
//! and never moves the virtual clock.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::LocalRunner;
use durable::{BoxError, DurableContext, Options, RetryDecision};

/// A runner that sends each checkpoint in its own request.
fn unbatched() -> LocalRunner {
    LocalRunner::new().options(
        Options::builder()
            .checkpoint_batching(false)
            .build()
            .expect("valid options"),
    )
}

/// The shorter of two raced waits wins, and the longer wait stays pending.
#[tokio::test]
async fn race_between_waits_returns_the_shorter_wait() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let slow = ctx
                    .run_in_child_context(|child| async move {
                        child.wait(Duration::from_secs(60)).name("minute").await?;
                        Ok("slow".to_owned())
                    })
                    .name("slow")
                    .future();
                let fast = ctx
                    .run_in_child_context(|child| async move {
                        child.wait(Duration::from_secs(1)).name("second").await?;
                        Ok("fast".to_owned())
                    })
                    .name("fast")
                    .future();
                Ok::<_, BoxError>(ctx.race([slow, fast]).await?)
            },
            (),
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output().map(String::as_str), Some("fast"));
    assert_eq!(result.invocation_count(), 2);
    let status = |name: &str| {
        result
            .operations()
            .iter()
            .find(|op| op.name() == Some(name))
            .map(|op| op.status().to_owned())
            .expect("the wait is recorded")
    };
    assert_eq!(status("second"), "Succeeded");
    assert_eq!(status("minute"), "Started");
}

/// A step retried after 300 seconds loses a race against a 1 second wait.
#[tokio::test]
async fn retry_delay_is_ordered_against_a_wait() {
    let result = unbatched()
        .run(
            |(), ctx: DurableContext| async move {
                let retried = ctx
                    .run_in_child_context(|child| async move {
                        child
                            .step(|step_ctx| async move {
                                if step_ctx.attempt() == 1 {
                                    return Err::<String, BoxError>("first attempt".into());
                                }
                                Ok("retried".to_owned())
                            })
                            .retry_strategy(|_err, _attempt| RetryDecision::Retry {
                                delay: Duration::from_secs(300),
                            })
                            .await
                            .map_err(BoxError::from)
                    })
                    .future();
                let waited = ctx
                    .run_in_child_context(|child| async move {
                        child.wait(Duration::from_secs(1)).await?;
                        Ok("waited".to_owned())
                    })
                    .future();
                Ok::<_, BoxError>(ctx.race([retried, waited]).await?)
            },
            (),
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output().map(String::as_str), Some("waited"));
    assert_eq!(result.invocation_count(), 2);
}

/// A queued callback reply wins a race against an hour-long wait, because
/// it arrives before the clock moves. So the wait never fires.
#[tokio::test]
async fn queued_callback_reply_arrives_before_the_clock_moves() {
    let result = LocalRunner::new()
        .callback_success(&"cb")
        .run(
            |(), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<String>().await?;
                let timeout = ctx
                    .run_in_child_context(|child| async move {
                        child.wait(Duration::from_secs(3600)).name("hour").await?;
                        Ok("timeout".to_owned())
                    })
                    .future();
                Ok::<_, BoxError>(ctx.race([cb.result(), timeout]).await?)
            },
            (),
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output().map(String::as_str), Some("cb"));
    let hour = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("hour"))
        .expect("the hour wait is recorded");
    assert_eq!(hour.status(), "Started");
}

/// A step that returns 42.
async fn one_step((): (), ctx: DurableContext) -> Result<u32, BoxError> {
    Ok(ctx.step(|_| async { Ok(42_u32) }).name("answer").await?)
}

/// `checkpoint_latency` delays each checkpoint response in tokio time and
/// changes nothing else. Unbatched, the step sends its START and its
/// SUCCEED in two requests, so the run takes at least twice the latency.
#[tokio::test(start_paused = true)]
async fn checkpoint_latency_delays_responses_and_changes_nothing_else() {
    let latency = Duration::from_millis(50);

    let started = tokio::time::Instant::now();
    let delayed = unbatched()
        .checkpoint_latency(latency)
        .run(one_step, ())
        .await;
    let elapsed = started.elapsed();

    let baseline = unbatched().run(one_step, ()).await;

    assert!(delayed.is_success(), "{:?}", delayed.error_message());
    assert!(
        elapsed >= latency * 2,
        "two checkpoint responses took {elapsed:?}"
    );
    assert_eq!(delayed.output(), Some(&42));
    assert_eq!(delayed.output(), baseline.output());
    assert_eq!(delayed.invocation_count(), baseline.invocation_count());
    assert_eq!(
        format!("{:?}", delayed.operations()),
        format!("{:?}", baseline.operations())
    );
}
