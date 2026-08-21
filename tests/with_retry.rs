//! Integration tests for `with_retry` (issue #13): block-level retry of a
//! multi-operation child context.
//!
//! Verifies the three semantics the operation promises:
//! - a failed attempt re-runs the WHOLE block on the next attempt, with a
//!   fresh child operation namespace (both steps re-execute, under new
//!   operation IDs);
//! - recorded results replay across invocations (the block's closure never
//!   re-runs after the operation succeeds, even when the execution suspends
//!   and re-invokes later);
//! - retry exhaustion surfaces the last attempt's error.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::test_util::LocalRunner;
use durable::{DurableContext, RetryDecision};

/// A failing first attempt re-runs BOTH steps of the block on the second
/// attempt, under a fresh operation namespace.
#[tokio::test]
#[expect(clippy::too_many_lines)] // reason: one flow (run, assert counters, assert history)
async fn failed_attempt_reruns_whole_block_with_fresh_namespace() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let step_one_runs = Arc::new(AtomicU32::new(0));
    let step_two_runs = Arc::new(AtomicU32::new(0));

    let closure_runs_h = Arc::clone(&closure_runs);
    let step_one_runs_h = Arc::clone(&step_one_runs);
    let step_two_runs_h = Arc::clone(&step_two_runs);

    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                let step_one_runs = Arc::clone(&step_one_runs_h);
                let step_two_runs = Arc::clone(&step_two_runs_h);
                async move {
                    let out = ctx
                        .with_retry(move |child| {
                            let closure_runs = Arc::clone(&closure_runs);
                            let step_one_runs = Arc::clone(&step_one_runs);
                            let step_two_runs = Arc::clone(&step_two_runs);
                            async move {
                                // 1-based count of ACTUAL closure executions.
                                // Replayed attempts do not re-run the closure,
                                // so this counts attempts, not invocations.
                                let run = closure_runs.fetch_add(1, Ordering::SeqCst) + 1;
                                let s1 = Arc::clone(&step_one_runs);
                                let a = child
                                    .step(move |_| async move {
                                        s1.fetch_add(1, Ordering::SeqCst);
                                        Ok(20_u32)
                                    })
                                    .name("s1")
                                    .await?;
                                let s2 = Arc::clone(&step_two_runs);
                                let b = child
                                    .step(move |_| async move {
                                        s2.fetch_add(1, Ordering::SeqCst);
                                        Ok(22_u32)
                                    })
                                    .name("s2")
                                    .await?;
                                if run == 1 {
                                    return Err("first attempt fails after both steps".into());
                                }
                                Ok(a + b)
                            }
                        })
                        .name("block")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_success(),
        "execution should succeed on the second attempt: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&42));

    // Both step bodies ran once per attempt: fresh namespace per attempt
    // means nothing recorded by attempt 1 replayed into attempt 2.
    assert_eq!(
        step_one_runs.load(Ordering::SeqCst),
        2,
        "step s1 must re-run on the second attempt"
    );
    assert_eq!(
        step_two_runs.load(Ordering::SeqCst),
        2,
        "step s2 must re-run on the second attempt"
    );
    assert_eq!(closure_runs.load(Ordering::SeqCst), 2);

    // The retry delay is a durable wait: the execution suspended between
    // attempts and needed more than one invocation.
    assert!(
        result.invocation_count() >= 2,
        "retry delay must suspend the execution, got {} invocation(s)",
        result.invocation_count()
    );

    // Fresh namespacing is visible in the history: each attempt recorded
    // its own s1/s2 operations under distinct operation IDs.
    let ops = result.operations();
    let s1_ids: Vec<&str> = ops
        .iter()
        .filter(|op| op.name() == Some("s1"))
        .map(durable::test_util::TestOperation::id)
        .collect();
    assert_eq!(s1_ids.len(), 2, "two s1 records, one per attempt: {ops:?}");
    assert_ne!(
        s1_ids[0], s1_ids[1],
        "attempts must not share operation IDs"
    );

    let attempt_1 = ops
        .iter()
        .find(|op| op.name() == Some("attempt-1"))
        .expect("attempt-1 context recorded");
    assert!(attempt_1.failed(), "attempt-1 must record a failure");
    let attempt_2 = ops
        .iter()
        .find(|op| op.name() == Some("attempt-2"))
        .expect("attempt-2 context recorded");
    assert!(attempt_2.succeeded(), "attempt-2 must record a success");
    let block = ops
        .iter()
        .find(|op| op.name() == Some("block"))
        .expect("with_retry operation recorded");
    assert!(block.succeeded(), "the with_retry operation must succeed");
}

/// After the block succeeds, later invocations replay the recorded result
/// without re-running the closure or its steps.
#[tokio::test]
async fn recorded_result_replays_across_invocations() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let step_runs = Arc::new(AtomicU32::new(0));

    let closure_runs_h = Arc::clone(&closure_runs);
    let step_runs_h = Arc::clone(&step_runs);

    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                let step_runs = Arc::clone(&step_runs_h);
                async move {
                    let value = ctx
                        .with_retry(move |child| {
                            let closure_runs = Arc::clone(&closure_runs);
                            let step_runs = Arc::clone(&step_runs);
                            async move {
                                closure_runs.fetch_add(1, Ordering::SeqCst);
                                let sr = Arc::clone(&step_runs);
                                let v = child
                                    .step(move |_| async move {
                                        sr.fetch_add(1, Ordering::SeqCst);
                                        Ok("recorded".to_owned())
                                    })
                                    .name("record-once")
                                    .await?;
                                Ok(v)
                            }
                        })
                        .name("block")
                        .await?;

                    // Force a suspension AFTER the block succeeds: the next
                    // invocation must replay the block from its record.
                    ctx.wait(Duration::from_secs(1)).name("suspend").await?;
                    Ok::<_, durable::BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_success(),
        "execution should succeed: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"recorded".to_owned()));
    assert!(
        result.invocation_count() >= 2,
        "the wait must force at least one re-invocation, got {}",
        result.invocation_count()
    );
    // Replay returned the frozen result: neither the closure nor the step
    // body ran a second time.
    assert_eq!(
        closure_runs.load(Ordering::SeqCst),
        1,
        "block closure must not re-run on replay"
    );
    assert_eq!(
        step_runs.load(Ordering::SeqCst),
        1,
        "step body must not re-run on replay"
    );
}

/// Exhausting the retry strategy fails the operation with the last
/// attempt's error, after running the block once per allowed attempt.
#[tokio::test]
async fn exhaustion_surfaces_last_error() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let closure_runs_h = Arc::clone(&closure_runs);

    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                async move {
                    let out: String = ctx
                        .with_retry(move |child| {
                            let closure_runs = Arc::clone(&closure_runs);
                            async move {
                                let run = closure_runs.fetch_add(1, Ordering::SeqCst) + 1;
                                child
                                    .step(move |_| async move { Ok(run) })
                                    .name("observe")
                                    .await?;
                                Err(format!("boom on attempt {run}").into())
                            }
                        })
                        .name("always-fails")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_failure(),
        "exhausted retries must fail: {result:?}"
    );
    let message = result.error_message().expect("failure carries a message");
    assert!(
        message.contains("boom on attempt 3"),
        "the LAST attempt's error must surface: {message}"
    );
    assert!(
        message.contains("retries exhausted after 3 attempts"),
        "the attempt count must surface: {message}"
    );
    assert_eq!(
        closure_runs.load(Ordering::SeqCst),
        3,
        "the block runs once per allowed attempt"
    );
}

/// A `RetryStrategyConfig` (issue #12) drives block retries the same way a
/// closure strategy does.
#[tokio::test]
async fn retry_strategy_config_shapes_block_retries() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let closure_runs_h = Arc::clone(&closure_runs);

    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                async move {
                    let out = ctx
                        .with_retry(move |child| {
                            let closure_runs = Arc::clone(&closure_runs);
                            async move {
                                let run = closure_runs.fetch_add(1, Ordering::SeqCst) + 1;
                                let v = child
                                    .step(move |_| async move { Ok(run * 10) })
                                    .name("compute")
                                    .await?;
                                if run < 2 {
                                    return Err("not yet".into());
                                }
                                Ok(v)
                            }
                        })
                        .name("configured")
                        .retry_strategy_config(
                            durable::builders::RetryStrategyConfig::builder()
                                .max_attempts(2)
                                .initial_delay(Duration::from_secs(1))
                                .build(),
                        )
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_success(),
        "config-driven retry should succeed on attempt 2: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&20));
    assert_eq!(closure_runs.load(Ordering::SeqCst), 2);
}
