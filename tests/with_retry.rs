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
//! - retry exhaustion surfaces the last attempt's error as the source of
//!   the block's child context error;
//! - the closure receives the attempt number, and `without_child_context`
//!   runs the attempts in the caller's context.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::lambda_types::EventType;
use durable::test_util::{LocalRunner, TestOperationDetails};
use durable::{
    ChildContextErrorKind, DurableContext, OperationErrorKind, OperationStatus, RetryDecision,
};

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
                        .with_retry(move |child, _attempt| {
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
                                RetryDecision::retry(Duration::from_secs(1))
                            }
                        })
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

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
                        .with_retry(move |child, _attempt| {
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
        .await
        .expect("runner");

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

/// Exhausting the retry strategy fails the operation with a child context
/// error whose source is the last attempt's error, with no extra layer,
/// after running the block once per allowed attempt.
#[tokio::test]
async fn exhaustion_returns_last_attempt_error_as_source() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let checks: Arc<Mutex<Vec<(bool, bool, bool)>>> = Arc::new(Mutex::new(Vec::new()));
    let (closure_runs_h, checks_h) = (Arc::clone(&closure_runs), Arc::clone(&checks));

    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                let checks = Arc::clone(&checks_h);
                async move {
                    let outcome: Result<String, durable::OperationError> = ctx
                        .with_retry(move |child, attempt| {
                            let closure_runs = Arc::clone(&closure_runs);
                            async move {
                                closure_runs.fetch_add(1, Ordering::SeqCst);
                                child
                                    .step(move |_| async move { Ok(attempt) })
                                    .name("observe")
                                    .await?;
                                Err(format!("boom on attempt {attempt}").into())
                            }
                        })
                        .name("always-fails")
                        .retry_strategy(retry_until(3))
                        .await;
                    let e = outcome.expect_err("exhausted retries fail");
                    let is_child = matches!(e.kind(), OperationErrorKind::ChildContext(_));
                    let last_is_source = std::error::Error::source(&e)
                        .and_then(std::error::Error::source)
                        .and_then(|s| s.downcast_ref::<durable::OperationError>())
                        .is_some_and(|last| format!("{last:#}").contains("boom on attempt 3"));
                    let no_layer = !format!("{e:#}").contains("retries exhausted");
                    checks
                        .lock()
                        .expect("lock")
                        .push((is_child, last_is_source, no_layer));
                    Err::<String, durable::BoxError>(Box::new(e))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

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
        !message.contains("retries exhausted"),
        "no exhaustion layer: {message}"
    );
    assert_eq!(*checks.lock().expect("lock"), [(true, true, true)]);
    assert_eq!(
        closure_runs.load(Ordering::SeqCst),
        3,
        "the block runs once per allowed attempt"
    );
}

/// Retries after 1 s until attempt `last`, and stops there.
fn retry_until(last: u32) -> impl Fn(&durable::StepError, u32) -> RetryDecision + Send + Sync {
    move |_err, attempt| {
        if attempt >= last {
            RetryDecision::Stop
        } else {
            RetryDecision::retry(Duration::from_secs(1))
        }
    }
}

/// The closure receives the 1-based attempt number, and a resumed attempt
/// receives the same number again.
#[tokio::test]
async fn closure_receives_attempt_numbers() {
    let seen: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_h = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let seen = Arc::clone(&seen_h);
                async move {
                    let out = ctx
                        .with_retry(move |child, attempt| {
                            let seen = Arc::clone(&seen);
                            async move {
                                seen.lock().expect("lock").push(attempt);
                                child
                                    .step(move |_| async move { Ok(attempt) })
                                    .name("work")
                                    .await?;
                                if attempt < 3 {
                                    return Err(format!("attempt {attempt} fails").into());
                                }
                                child.wait(Duration::from_secs(1)).name("inner").await?;
                                Ok(attempt)
                            }
                        })
                        .name("block")
                        .retry_strategy(retry_until(3))
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&3));
    assert_eq!(*seen.lock().expect("lock"), [1, 2, 3, 3]);
}

/// Per-attempt closure calls, strategy observations and step bodies of one
/// run of [`replay_handler_run`].
#[derive(Default)]
struct ReplayRun {
    calls: Mutex<BTreeMap<u32, u32>>,
    first_errors: Mutex<Vec<bool>>,
    a_runs: AtomicU32,
    b_runs: AtomicU32,
}

/// Runs the handler of test 3: attempt 1 runs steps `a` and `b` (or no
/// step when `steps_in_first` is false) and fails with `first`; attempt 2
/// runs step `c`, waits 1 s and returns `done`. With `unwrapped`, the
/// block runs without its own child context.
async fn replay_handler_run(
    steps_in_first: bool,
    unwrapped: bool,
) -> (Arc<ReplayRun>, durable::test_util::TestResult<String>) {
    let run = Arc::new(ReplayRun::default());
    let run_h = Arc::clone(&run);
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let run = Arc::clone(&run_h);
                async move {
                    let body_run = Arc::clone(&run);
                    let strategy_run = Arc::clone(&run);
                    let block = ctx
                        .with_retry(move |child, attempt| {
                            let run = Arc::clone(&body_run);
                            async move {
                                *run.calls.lock().expect("lock").entry(attempt).or_insert(0) += 1;
                                if attempt == 1 {
                                    if steps_in_first {
                                        let a = Arc::clone(&run);
                                        child
                                            .step(move |_| async move {
                                                a.a_runs.fetch_add(1, Ordering::SeqCst);
                                                Ok(1_u32)
                                            })
                                            .name("a")
                                            .await?;
                                        let b = Arc::clone(&run);
                                        child
                                            .step(move |_| async move {
                                                b.b_runs.fetch_add(1, Ordering::SeqCst);
                                                Ok(2_u32)
                                            })
                                            .name("b")
                                            .await?;
                                    }
                                    return Err("first".into());
                                }
                                child.step(|_| async { Ok(3_u32) }).name("c").await?;
                                child.wait(Duration::from_secs(1)).name("inner").await?;
                                Ok("done".to_owned())
                            }
                        })
                        .name("block")
                        .retry_strategy(move |err, attempt| {
                            if attempt == 1 {
                                strategy_run
                                    .first_errors
                                    .lock()
                                    .expect("lock")
                                    .push(format!("{err:#}").contains("first"));
                            }
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::retry(Duration::from_secs(1))
                            }
                        });
                    let block = if unwrapped {
                        block.without_child_context()
                    } else {
                        block
                    };
                    Ok::<_, durable::BoxError>(block.await?)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    (run, result)
}

/// A failed attempt replays from its `FAIL` record without calling the
/// closure, the strategy receives the recorded error on every invocation,
/// and attempt 2's operation IDs do not depend on attempt 1's operations.
#[tokio::test]
async fn failed_attempt_replays_from_its_record() {
    let (run, result) = replay_handler_run(true, false).await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&"done".to_owned()));
    assert!(result.invocation_count() >= 3, "{result}");
    assert_eq!(run.calls.lock().expect("lock").get(&1), Some(&1));
    assert_eq!(run.a_runs.load(Ordering::SeqCst), 1);
    assert_eq!(run.b_runs.load(Ordering::SeqCst), 1);
    let first_errors = run.first_errors.lock().expect("lock").clone();
    assert_eq!(first_errors.len(), result.invocation_count(), "{result}");
    assert!(first_errors.iter().all(|seen| *seen), "{first_errors:?}");
    let c = result.operation("c").expect("c recorded");
    let attempt_2 = result.operation("attempt-2").expect("attempt 2 recorded");
    assert_eq!(c.parent_id(), Some(attempt_2.id()));

    let (_, without_steps) = replay_handler_run(false, false).await;
    assert!(
        without_steps.is_success(),
        "{:?}",
        without_steps.error_message()
    );
    assert_eq!(
        without_steps.operation("c").expect("c recorded").id(),
        c.id(),
        "attempt 2's IDs do not depend on attempt 1's operations"
    );
}

/// A strategy delay of zero goes to the wait unchanged, and the wait
/// records the one-second minimum.
#[tokio::test]
async fn zero_delay_records_one_second_wait() {
    let runs = Arc::new(AtomicU32::new(0));
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let runs = Arc::clone(&runs);
                async move {
                    let out = ctx
                        .with_retry(move |_child, _attempt| {
                            let runs = Arc::clone(&runs);
                            async move {
                                if runs.fetch_add(1, Ordering::SeqCst) == 0 {
                                    Err::<u32, durable::BoxError>("transient".into())
                                } else {
                                    Ok(9)
                                }
                            }
                        })
                        .name("block")
                        .retry_strategy(|_err, _attempt| RetryDecision::retry(Duration::ZERO))
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    let wait = result.operation("block-backoff-1").expect("wait recorded");
    assert!(
        matches!(
            wait.details(),
            TestOperationDetails::Wait(wait)
                if wait.duration() == Some(Duration::from_secs(1))
        ),
        "{:?}",
        wait.details()
    );
}

/// Fails attempt 1 and succeeds on attempt 2, as a block named `block`,
/// unwrapped or not, inside a child `outer` between steps `before` and
/// `after`.
async fn outer_block_run(unwrapped: bool) -> durable::test_util::TestResult<u32> {
    LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| async move {
                let out = ctx
                    .run_in_child_context(move |outer| async move {
                        outer.step(|_| async { Ok(1_u32) }).name("before").await?;
                        let block = outer
                            .with_retry(|_child, attempt| async move {
                                if attempt == 1 {
                                    Err::<u32, durable::BoxError>("transient".into())
                                } else {
                                    Ok(attempt)
                                }
                            })
                            .name("block")
                            .retry_strategy(retry_until(2));
                        let value = if unwrapped {
                            block.without_child_context().await?
                        } else {
                            block.await?
                        };
                        outer.step(|_| async { Ok(3_u32) }).name("after").await?;
                        Ok(value)
                    })
                    .name("outer")
                    .await?;
                Ok::<_, durable::BoxError>(out)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

/// An unwrapped block writes no record and claims no ID: its attempts and
/// waits record the caller's parent, and `block-attempt-1` takes the ID the
/// wrapped block's own record has.
#[tokio::test]
async fn unwrapped_attempts_record_caller_parent() {
    let unwrapped = outer_block_run(true).await;
    assert!(unwrapped.is_success(), "{:?}", unwrapped.error_message());
    assert_eq!(unwrapped.output(), Some(&2));
    assert!(unwrapped.operation("block").is_none(), "{unwrapped}");
    let outer = unwrapped.operation("outer").expect("outer recorded");
    let attempt_1 = unwrapped
        .operation("block-attempt-1")
        .expect("attempt 1 recorded");
    assert_eq!(attempt_1.parent_id(), Some(outer.id()));
    let backoff = unwrapped
        .operation("block-backoff-1")
        .expect("wait recorded");
    assert_eq!(backoff.parent_id(), Some(outer.id()));
    assert_eq!(
        unwrapped
            .operation("block-attempt-2")
            .expect("attempt 2 recorded")
            .parent_id(),
        Some(outer.id())
    );

    let wrapped = outer_block_run(false).await;
    assert!(wrapped.is_success(), "{:?}", wrapped.error_message());
    assert_eq!(
        attempt_1.id(),
        wrapped.operation("block").expect("block recorded").id(),
        "attempt 1 takes the ID after `before`"
    );
}

/// What one invocation saw of an unwrapped block's exhaustion error.
#[derive(Debug, Clone)]
struct SeenError {
    child_failed: bool,
    operation_id: Option<String>,
    text: String,
}

/// An unwrapped block's exhaustion returns the last `attempt-N` error
/// unchanged, live and on replay.
#[tokio::test]
async fn unwrapped_exhaustion_returns_attempt_error() {
    let seen: Arc<Mutex<Vec<SeenError>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_h = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let seen = Arc::clone(&seen_h);
                async move {
                    ctx.run_in_child_context(move |outer| async move {
                        outer.step(|_| async { Ok(1_u32) }).name("before").await?;
                        let outcome: Result<u32, durable::OperationError> = outer
                            .with_retry(|_child, attempt| async move {
                                Err(format!("boom on attempt {attempt}").into())
                            })
                            .name("block")
                            .retry_strategy(retry_until(2))
                            .without_child_context()
                            .await;
                        let e = outcome.expect_err("exhausted retries fail");
                        seen.lock().expect("lock").push(SeenError {
                            child_failed: matches!(
                                e.kind(),
                                OperationErrorKind::ChildContext(child)
                                    if matches!(child.kind(), ChildContextErrorKind::ChildFailed { .. })
                            ),
                            operation_id: e.operation_id().map(str::to_owned),
                            text: format!("{e:#}"),
                        });
                        // A suspension after the error, so the next
                        // invocation replays the attempts' records.
                        outer.wait(Duration::from_secs(1)).name("after").await?;
                        Ok::<_, durable::BoxError>(())
                    })
                    .name("outer")
                    .await?;
                    Ok::<_, durable::BoxError>(())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    let attempt_2 = result
        .operation("block-attempt-2")
        .expect("attempt 2 recorded");
    let seen = seen.lock().expect("lock").clone();
    assert!(seen.len() >= 2, "live and replayed: {seen:?}");
    for error in &seen {
        assert!(error.child_failed, "{error:?}");
        assert_eq!(
            error.operation_id.as_deref(),
            Some(attempt_2.id()),
            "{error:?}"
        );
        assert!(error.text.contains("boom on attempt 2"), "{error:?}");
    }
}

/// Test 3's first handler, unwrapped: attempt 1's closure ran once over
/// all invocations.
#[tokio::test]
async fn unwrapped_failed_attempt_replays_from_record() {
    let (run, result) = replay_handler_run(true, true).await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert!(result.invocation_count() >= 3, "{result}");
    assert_eq!(run.calls.lock().expect("lock").get(&1), Some(&1));
    assert!(result.operation("block").is_none(), "{result}");
    assert!(result.operation("block-attempt-1").is_some(), "{result}");
}

/// An unnamed block's retry wait has no name, with or without its own
/// child context. An unwrapped block named `""` names its attempts
/// `attempt-N`, as an unnamed one does.
#[tokio::test]
async fn unnamed_waits_are_unnamed_in_both_wrap_modes() {
    for (unwrapped, name) in [(false, None), (true, None), (true, Some(""))] {
        let result = LocalRunner::new()
            .run(
                move |_: serde_json::Value, ctx: DurableContext| async move {
                    let block = ctx
                        .with_retry(|_child, attempt| async move {
                            if attempt == 1 {
                                Err::<u32, durable::BoxError>("transient".into())
                            } else {
                                Ok(attempt)
                            }
                        })
                        .retry_strategy(retry_until(2));
                    let block = match name {
                        Some(name) => block.name(name),
                        None => block,
                    };
                    let value = if unwrapped {
                        block.without_child_context().await?
                    } else {
                        block.await?
                    };
                    Ok::<_, durable::BoxError>(value)
                },
                serde_json::Value::Null,
            )
            .await
            .expect("runner");
        assert!(
            result.is_success(),
            "{unwrapped} {name:?}: {:?}",
            result.error_message()
        );
        let waits: Vec<Option<&str>> = result
            .operations()
            .iter()
            .filter(|op| op.kind() == durable::OperationKind::Wait)
            .map(|op| op.name())
            .collect();
        assert_eq!(waits, [None], "{unwrapped} {name:?}");
        assert!(
            result.operation("attempt-1").is_some(),
            "{unwrapped} {name:?}: {result}"
        );
    }
}

/// An unwrapped block whose wait between attempts fails returns the wait's
/// error unchanged. The second invocation's strategy retries where the
/// first stopped, so the wait meets the step recorded at its ID.
#[tokio::test]
async fn unwrapped_failed_wait_returns_wait_error() {
    let invocations = Arc::new(AtomicU32::new(0));
    let wait_errors: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let (invocations_h, wait_errors_h) = (Arc::clone(&invocations), Arc::clone(&wait_errors));
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let invocation = invocations_h.fetch_add(1, Ordering::SeqCst) + 1;
                let wait_errors = Arc::clone(&wait_errors_h);
                async move {
                    let outcome: Result<u32, durable::OperationError> = ctx
                        .with_retry(|_child, _attempt| async {
                            Err::<u32, durable::BoxError>("transient".into())
                        })
                        .name("block")
                        .retry_strategy(move |_err, _attempt| {
                            if invocation == 1 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::retry(Duration::from_secs(1))
                            }
                        })
                        .without_child_context()
                        .await;
                    let e = outcome.expect_err("the block fails");
                    if invocation > 1 {
                        wait_errors.lock().expect("lock").push(matches!(
                            e.kind(),
                            OperationErrorKind::NonDeterministicExecution(_)
                        ));
                        return Err::<(), durable::BoxError>(Box::new(e));
                    }
                    ctx.step(|_| async { Ok(1_u32) }).name("step").await?;
                    ctx.wait(Duration::from_secs(1)).name("suspend").await?;
                    Ok(())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_failure(), "{result}");
    assert_eq!(*wait_errors.lock().expect("lock"), [true], "{result}");
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
                        .with_retry(move |child, _attempt| {
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
        .await
        .expect("runner");

    assert!(
        result.is_success(),
        "config-driven retry should succeed on attempt 2: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&20));
    assert_eq!(closure_runs.load(Ordering::SeqCst), 2);
}

/// A custom subtype labels the block in the history: the block operation
/// and every event of it record `Checkout`, while the attempt contexts
/// keep `RunInChildContext`, the delay keeps `Wait` and the steps keep
/// `Step`, on the operations and on their events.
#[tokio::test]
async fn custom_sub_type_labels_only_the_block_in_history() {
    let work_runs = Arc::new(AtomicU32::new(0));
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let work_runs = Arc::clone(&work_runs);
                async move {
                    let out = ctx
                        .with_retry(move |child, _attempt| {
                            let work_runs = Arc::clone(&work_runs);
                            async move {
                                let v = child
                                    .step(move |_| async move {
                                        if work_runs.fetch_add(1, Ordering::SeqCst) == 0 {
                                            Err::<u32, durable::BoxError>("transient".into())
                                        } else {
                                            Ok(7)
                                        }
                                    })
                                    .name("work")
                                    .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                                    .await?;
                                Ok(v)
                            }
                        })
                        .name("block")
                        .sub_type(durable::OperationSubType::new("Checkout")?)
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::retry(Duration::from_secs(1))
                            }
                        })
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&7));

    let mut shape: Vec<(&str, Option<&str>)> = result
        .operations()
        .iter()
        .map(|op| {
            for event in op.events() {
                assert_eq!(event.sub_type(), op.sub_type(), "{event:?}");
            }
            (op.name().unwrap_or(""), op.sub_type())
        })
        .collect();
    shape.sort_unstable();
    assert_eq!(
        shape,
        [
            ("attempt-1", Some("RunInChildContext")),
            ("attempt-2", Some("RunInChildContext")),
            ("block", Some("Checkout")),
            ("block-backoff-1", Some("Wait")),
            ("work", Some("Step")),
            ("work", Some("Step")),
        ]
    );
}

/// An unnamed block, and a block named `""`, leave the wait between
/// attempts unnamed, as JS does. Each run fails attempt 1, succeeds on
/// attempt 2, and records exactly one wait, which has no name.
#[tokio::test]
async fn unnamed_block_waits_are_unnamed() {
    for name in [None, Some("")] {
        let runs = Arc::new(AtomicU32::new(0));
        let result = LocalRunner::new()
            .run(
                move |_: serde_json::Value, ctx: DurableContext| {
                    let runs = Arc::clone(&runs);
                    async move {
                        let block = ctx
                            .with_retry(move |_child, _attempt| {
                                let runs = Arc::clone(&runs);
                                async move {
                                    if runs.fetch_add(1, Ordering::SeqCst) == 0 {
                                        Err::<u32, durable::BoxError>("transient".into())
                                    } else {
                                        Ok(3)
                                    }
                                }
                            })
                            .retry_strategy(|_err, attempt| {
                                if attempt >= 2 {
                                    RetryDecision::Stop
                                } else {
                                    RetryDecision::retry(Duration::from_secs(1))
                                }
                            });
                        let block = match name {
                            Some(name) => block.name(name),
                            None => block,
                        };
                        Ok::<_, durable::BoxError>(block.await?)
                    }
                },
                serde_json::Value::Null,
            )
            .await
            .expect("runner");
        assert!(
            result.is_success(),
            "{name:?}: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&3));
        assert!(result.operation("attempt-1").expect("attempt 1").failed());
        assert!(
            result
                .operation("attempt-2")
                .expect("attempt 2")
                .succeeded()
        );

        let waits: Vec<Option<&str>> = result
            .operations()
            .iter()
            .filter(|op| op.kind() == durable::OperationKind::Wait)
            .map(|op| op.name())
            .collect();
        assert_eq!(waits, [None], "{name:?}");
        assert!(result.operation("retry-delay-1").is_none(), "{name:?}");
    }
}

/// Runs a wrapped block named `block` and a step `between`. With
/// `early_call`, the `with_retry` call comes before the step and the
/// `.await` after it; otherwise both come after the step.
async fn claim_order_run(early_call: bool) -> durable::test_util::TestResult<u32> {
    LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| async move {
                let body = |_child: DurableContext, attempt: u32| async move { Ok(attempt) };
                let early = early_call.then(|| ctx.with_retry(body).name("block"));
                ctx.step(|_| async { Ok(1_u32) }).name("between").await?;
                let block = early.unwrap_or_else(|| ctx.with_retry(body).name("block"));
                Ok::<_, durable::BoxError>(block.await?)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

/// A wrapped block claims its operation ID when the builder becomes its
/// future, not at the `with_retry` call: a step created and awaited
/// between the call and the `.await` takes the earlier ID.
#[tokio::test]
async fn wrapped_block_claims_id_when_awaited() {
    let early = claim_order_run(true).await;
    let late = claim_order_run(false).await;
    assert!(early.is_success(), "{:?}", early.error_message());
    assert!(late.is_success(), "{:?}", late.error_message());
    for name in ["between", "block"] {
        assert_eq!(
            early.operation(name).expect("recorded").id(),
            late.operation(name).expect("recorded").id(),
            "{name}"
        );
    }
}

/// With `without_child_context`, `sub_type` has no recorded effect: no
/// operation records the subtype.
#[tokio::test]
async fn unwrapped_sub_type_has_no_recorded_effect() {
    let result = LocalRunner::new()
        .run(
            |_: serde_json::Value, ctx: DurableContext| async move {
                let out = ctx
                    .with_retry(|_child, attempt| async move { Ok(attempt) })
                    .name("block")
                    .sub_type(durable::OperationSubType::new("Checkout")?)
                    .without_child_context()
                    .await?;
                Ok::<_, durable::BoxError>(out)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert!(
        result
            .operations()
            .iter()
            .all(|op| op.sub_type() != Some("Checkout")),
        "{result}"
    );
    assert_eq!(
        result
            .operation("block-attempt-1")
            .and_then(|op| op.sub_type()),
        Some("RunInChildContext")
    );
}

/// `serdes` rebuilds the builder and keeps the unwrapped mode: the block
/// still writes no record, and its attempts are named after the block in
/// the caller's context.
#[tokio::test]
async fn unwrapped_mode_survives_serdes() {
    let result = LocalRunner::new()
        .run(
            |_: serde_json::Value, ctx: DurableContext| async move {
                let out = ctx
                    .with_retry(|_child, attempt| async move {
                        if attempt == 1 {
                            Err::<u32, durable::BoxError>("transient".into())
                        } else {
                            Ok(attempt)
                        }
                    })
                    .name("block")
                    .retry_strategy(retry_until(2))
                    .without_child_context()
                    .serdes(durable::JsonSerdes)
                    .await?;
                Ok::<_, durable::BoxError>(out)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&2));
    assert!(result.operation("block").is_none(), "{result}");
    for name in ["block-attempt-1", "block-backoff-1", "block-attempt-2"] {
        let op = result.operation(name).expect("operation recorded");
        assert_eq!(op.parent_id(), None, "{name} is top-level");
    }
}

/// A replay mismatch of the wait between two attempts of a wrapped block
/// records a fatal nondeterminism error before the block's `FAIL`, so the
/// `FAIL` is refused: the block stays `STARTED`, and the handler receives
/// no block error.
///
/// The first invocation records, under the block's ID, a plain child
/// context with the block's identity: a failed `attempt-1` child, then a
/// step at the position of the first retry wait. The second invocation
/// runs the real block, which replays `attempt-1`'s failure, retries, and
/// claims the wait at the step's position, so the wait fails with a
/// nondeterminism error.
#[tokio::test]
async fn wrapped_mismatched_wait_records_no_block_failure() {
    let invocations = Arc::new(AtomicU32::new(0));
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (invocations_h, seen_h) = (Arc::clone(&invocations), Arc::clone(&seen));
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let invocation = invocations_h.fetch_add(1, Ordering::SeqCst) + 1;
                let seen = Arc::clone(&seen_h);
                async move {
                    if invocation == 1 {
                        ctx.run_in_child_context(|block| async move {
                            let _ = block
                                .run_in_child_context(|_| async {
                                    Err::<u32, durable::BoxError>("first".into())
                                })
                                .name("attempt-1")
                                .await;
                            block.step(|_| async { Ok(1_u32) }).name("intruder").await?;
                            block.wait(Duration::from_secs(1)).name("suspend").await?;
                            Ok(0_u32)
                        })
                        .name("block")
                        .await?;
                        return Ok(());
                    }
                    let outcome: Result<u32, durable::OperationError> = ctx
                        .with_retry(|_child, _attempt| async {
                            Err::<u32, durable::BoxError>("transient".into())
                        })
                        .name("block")
                        .retry_strategy(retry_until(3))
                        .await;
                    let e = outcome.expect_err("the block fails");
                    seen.lock().expect("lock").push(format!("{e:#}"));
                    Err::<(), durable::BoxError>(Box::new(e))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_failure(), "{result}");
    assert_eq!(
        result.error_type(),
        Some("NonDeterministicExecutionError"),
        "{result}"
    );
    assert_eq!(result.invocation_count(), 2);
    assert!(
        seen.lock().expect("lock").is_empty(),
        "the handler never receives the block's error"
    );
    let status = |name: &str| result.operation(name).expect("operation recorded").status();
    assert_eq!(status("block"), OperationStatus::Started);
    assert_eq!(status("attempt-1"), OperationStatus::Failed);
    let after_first: Vec<EventType> = result
        .history_events()
        .iter()
        .filter_map(|event| event.event_type().cloned())
        .skip_while(|event_type| *event_type != EventType::InvocationCompleted)
        .skip(1)
        .collect();
    assert_eq!(
        after_first,
        [
            EventType::WaitSucceeded,
            EventType::InvocationCompleted,
            EventType::ExecutionFailed,
        ]
    );
}
