//! Integration tests for non-determinism detection (issue #6).
//!
//! These tests verify that the SDK detects mismatches between the handler's
//! operation order across invocations and raises a clear error.

#![cfg(feature = "test-util")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::test_util::LocalRunner;

/// Test (a): Reordered steps between invocations raises the error.
///
/// The handler uses a shared counter to behave differently on the first vs
/// second invocation: on invocation 1 it runs step A then suspends; on
/// invocation 2 (replay) it tries step B at position 1. The SDK should
/// detect the name mismatch.
#[tokio::test]
async fn reordered_steps_detected() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        // First invocation: step "alpha" then suspend.
                        let _a = ctx
                            .step(|_| async { Ok("a".to_owned()) })
                            .name("alpha")
                            .await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok::<_, durable::BoxError>("done".to_owned())
                    } else {
                        // Second invocation (replay): step "beta" at position 1.
                        // Position 1 was checkpointed as Step named "alpha",
                        // but now we claim Step named "beta" at position 1.
                        let _b = ctx
                            .step(|_| async { Ok("b".to_owned()) })
                            .name("beta")
                            .await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok("done".to_owned())
                    }
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "expected execution to fail: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("alpha"),
        "error should mention expected name 'alpha': {msg}"
    );
    assert!(
        msg.contains("beta"),
        "error should mention claimed name 'beta': {msg}"
    );
}

/// Test (b): An operation changing type (step → wait) raises the error.
///
/// Invocation 1 runs a step at position 1. Invocation 2 runs a wait at
/// position 1. The type mismatch should be detected.
#[tokio::test]
async fn type_change_detected() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        // First invocation: a step at position 1 then suspend.
                        let _v = ctx.step(|_| async { Ok(42_i32) }).name("compute").await?;
                        ctx.wait(Duration::from_secs(1)).name("timer").await?;
                        Ok::<_, durable::BoxError>("done".to_owned())
                    } else {
                        // Second invocation: a wait at position 1 (type mismatch).
                        ctx.wait(Duration::from_secs(5)).name("compute").await?;
                        ctx.wait(Duration::from_secs(1)).name("timer").await?;
                        Ok("done".to_owned())
                    }
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "expected execution to fail: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    // The error should mention both the expected type (Step) and actual (Wait).
    assert!(
        msg.contains("Step"),
        "error should mention expected type 'Step': {msg}"
    );
    assert!(
        msg.contains("Wait"),
        "error should mention claimed type 'Wait': {msg}"
    );
}

/// Test (c): Unchanged handlers replay normally (no regression).
///
/// A deterministic handler that runs the same operations in the same order
/// should complete successfully across multiple invocations.
#[tokio::test]
async fn deterministic_handler_replays_successfully() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: durable::DurableContext| async move {
                let name = ctx
                    .step(|_| async { Ok("world".to_owned()) })
                    .name("fetch-name")
                    .await?;

                ctx.wait(Duration::from_secs(1)).name("cooldown").await?;

                let greeting = ctx
                    .step(move |_| async move { Ok(format!("hello, {name}")) })
                    .name("format")
                    .await?;

                Ok::<_, durable::BoxError>(greeting)
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "expected success but got error: {:?}",
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"hello, world".to_owned()));
    // Verify it took multiple invocations (the wait caused suspension).
    assert!(
        result.invocation_count() >= 2,
        "expected at least 2 invocations, got {}",
        result.invocation_count()
    );
}

/// Test: context-type vs step-type mismatch is detected at position 1.
#[tokio::test]
async fn context_vs_step_type_mismatch_detected() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        // First invocation: run_in_child_context (Context type) at position 1.
                        let _v = ctx
                            .run_in_child_context(|child| async move {
                                child.step(|_| async { Ok(1_i32) }).name("inner").await?;
                                Ok::<_, durable::BoxError>(1_i32)
                            })
                            .name("wrapper")
                            .await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok::<_, durable::BoxError>("done".to_owned())
                    } else {
                        // Second invocation: step (Step type) at position 1.
                        let _v = ctx.step(|_| async { Ok(1_i32) }).name("wrapper").await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok("done".to_owned())
                    }
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "expected execution to fail: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    // Context vs Step mismatch.
    assert!(
        msg.contains("Context") || msg.contains("Step"),
        "error should mention type mismatch: {msg}"
    );
}

/// Test: the `error_type` field is set to an identifiable value.
#[tokio::test]
async fn error_type_is_identifiable() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        let _a = ctx.step(|_| async { Ok(1_u32) }).name("first").await?;
                        ctx.wait(Duration::from_secs(1)).name("gap").await?;
                        Ok::<_, durable::BoxError>(1_u32)
                    } else {
                        // Different name at same position.
                        let _a = ctx.step(|_| async { Ok(1_u32) }).name("second").await?;
                        ctx.wait(Duration::from_secs(1)).name("gap").await?;
                        Ok(1_u32)
                    }
                }
            },
            (),
        )
        .await;

    assert!(result.is_failure());
    // The error type should be the concrete non-determinism error type.
    let err_type = result.error_type().unwrap_or("");
    assert_eq!(
        err_type, "NonDeterministicExecutionError",
        "error_type should be NonDeterministicExecutionError, got: '{err_type}'"
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("first") && msg.contains("second"),
        "message should name both operations: {msg}"
    );
}

/// Test: swapping a step for a `ctx.map` at the same position is detected.
///
/// Invocation 1 runs a step at position 1. Invocation 2 runs a ctx.map at
/// position 1 (type "Step" vs "Context" mismatch). The non-determinism
/// detection must fire here.
#[tokio::test]
async fn step_to_map_swap_detected() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        // First invocation: a step at position 1 then suspend.
                        let _v = ctx.step(|_| async { Ok(42_i32) }).name("work").await?;
                        ctx.wait(Duration::from_secs(1)).name("timer").await?;
                        Ok::<_, durable::BoxError>("done".to_owned())
                    } else {
                        // Second invocation: a map at position 1 (Context vs Step type mismatch).
                        let _batch = ctx
                            .map(vec![1_i32, 2, 3], |child, item, _idx| async move {
                                let v = child
                                    .step(move |_| async move { Ok(item * 2) })
                                    .name("double")
                                    .await?;
                                Ok(v)
                            })
                            .name("work")
                            .await?;
                        ctx.wait(Duration::from_secs(1)).name("timer").await?;
                        Ok("done".to_owned())
                    }
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "expected execution to fail: {:?}",
        result.error_message()
    );
    let err_type = result.error_type().unwrap_or("");
    assert_eq!(
        err_type, "NonDeterministicExecutionError",
        "expected NonDeterministicExecutionError, got: '{err_type}'"
    );
    let msg = result.error_message().unwrap_or("");
    // The error should mention both Step (expected) and Context (actual from map).
    assert!(
        msg.contains("Step") && msg.contains("Context"),
        "error should mention Step vs Context mismatch: {msg}"
    );
}

/// Test: a removed operation is detected (issue #6 "removed operation").
///
/// Invocation 1 runs an UNNAMED step, then a named step, then suspends on a
/// wait. Invocation 2 (replay) removes the unnamed step, so the named step
/// claims position 1 — whose checkpoint belongs to the unnamed step. Type
/// and sub-type are identical (both Step/Step), so only the None↔Some name
/// comparison catches it. This is the regression test for silent checkpoint
/// consumption after an operation is removed.
#[tokio::test]
async fn removed_operation_detected() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        // First invocation: unnamed step, named step, suspend.
                        let _a = ctx.step(|_| async { Ok(1_u32) }).await?;
                        let _b = ctx.step(|_| async { Ok(2_u32) }).name("second").await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok::<_, durable::BoxError>("done".to_owned())
                    } else {
                        // Replay: the unnamed step was removed. "second" now
                        // claims position 1, which stores the unnamed step.
                        let _b = ctx.step(|_| async { Ok(2_u32) }).name("second").await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok("done".to_owned())
                    }
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "expected removed operation to be detected: output={:?}, error={:?}",
        result.output(),
        result.error_message()
    );
    let err_type = result.error_type().unwrap_or("");
    assert_eq!(
        err_type, "NonDeterministicExecutionError",
        "expected NonDeterministicExecutionError, got: '{err_type}'"
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("second"),
        "error should mention the claimed name 'second': {msg}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Fatal propagation: a mismatch must fail the execution even when the
// per-operation error is swallowed by a combinator, a tolerant batch, or
// user code.
// ────────────────────────────────────────────────────────────────────────────

/// Test: a mismatch inside `join_all` fails the execution.
///
/// `join_all` stores every constituent failure as `Settled::Rejected` and
/// checkpoints the combinator as successful, so without engine-level fatal
/// tracking a replay identity mismatch inside it would produce a SUCCEEDED
/// execution. Invocation 1 parks the combinator non-terminal (a retrying
/// step holds it open); invocation 2 renames the sibling step, so its claim
/// mismatches the checkpoint.
#[tokio::test]
async fn join_all_mismatch_fails_execution() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    // Fails on attempt 1 with a retry delay so the combinator
                    // suspends non-terminal on the first invocation.
                    let flaky = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            Ok(2_u32)
                        })
                        .name("flaky")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                durable::RetryDecision::Stop
                            } else {
                                durable::RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .future();
                    // Renamed on the second invocation: replay identity
                    // mismatch at this position.
                    let name = if call == 0 { "stable" } else { "renamed" };
                    let stable = ctx.step(|_| async { Ok(1_u32) }).name(name).future();

                    let settled = ctx.join_all([flaky, stable]).name("gather").await?;
                    let _ = settled; // outcomes inspected nowhere — join_all never fails fast
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a mismatch inside join_all must fail the execution: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("stable") && msg.contains("renamed"),
        "error should name both identities: {msg}"
    );
}

/// Test: a mismatch in a `select_ok` loser fails the execution even when a
/// sibling branch succeeds.
///
/// `select_ok` returns the first success and drops losers, so a mismatching
/// branch would otherwise be silently out-raced. Both branches park on
/// invocation 1; on invocation 2 the renamed branch rejects with the
/// mismatch while the other succeeds.
#[tokio::test]
async fn select_ok_mismatch_fails_execution_despite_sibling_success() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    // Both steps fail attempt 1 with a retry delay so
                    // select_ok parks non-terminal on invocation 1. The
                    // winner sleeps briefly on attempt 2 so the mismatching
                    // loser is polled to its rejection first.
                    let winner = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            tokio::time::sleep(Duration::from_millis(20)).await;
                            Ok(1_u32)
                        })
                        .name("winner")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                durable::RetryDecision::Stop
                            } else {
                                durable::RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .future();
                    let name = if call == 0 { "loser" } else { "loser-renamed" };
                    let loser = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            Ok(2_u32)
                        })
                        .name(name)
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                durable::RetryDecision::Stop
                            } else {
                                durable::RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .future();

                    let first = ctx.select_ok([winner, loser]).name("race-ok").await?;
                    let _ = first;
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a mismatch in a select_ok loser must fail the execution: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
}

/// Test: a mismatch in a tolerated map branch fails the execution.
///
/// A `CompletionConfig` tolerating every failure would otherwise let the
/// batch — and therefore the execution — succeed while a branch failed for
/// a replay identity mismatch (stringified through the `ChildFnError`
/// boundary on the way).
#[tokio::test]
async fn tolerated_map_branch_mismatch_fails_execution() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    let batch = ctx
                        .map(vec![0_u32, 1], move |child, item, idx| async move {
                            // Renamed on the second invocation → mismatch.
                            let name = if call == 0 {
                                format!("s{idx}")
                            } else {
                                format!("changed{idx}")
                            };
                            let v = child
                                .step(move |_| async move { Ok(item + 1) })
                                .name(name)
                                .await?;
                            child.wait(Duration::from_secs(1)).name("hold").await?;
                            Ok(v)
                        })
                        .name("batch")
                        .completion(durable::CompletionConfig::with_tolerated_failure_count(2))
                        .await_batch()
                        .await?;
                    let _ = batch; // tolerated failures — batch reports success
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a tolerated map-branch mismatch must fail the execution: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
}

/// Test: a mismatch swallowed by user code still fails the execution.
///
/// The handler ignores the child-context error entirely and returns `Ok` —
/// the execution must still fail with the dedicated error, because replay
/// integrity is broken however the per-operation error was handled.
#[tokio::test]
async fn handler_swallowed_mismatch_still_fails_execution() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    let inner_name = if call == 0 { "inner" } else { "inner-renamed" };
                    let swallowed = ctx
                        .run_in_child_context(move |child| async move {
                            let v = child.step(|_| async { Ok(1_u32) }).name(inner_name).await?;
                            child.wait(Duration::from_secs(1)).name("hold").await?;
                            Ok(v)
                        })
                        .name("wrapper")
                        .await;
                    let _ = swallowed; // deliberately ignored
                    Ok::<_, durable::BoxError>("swallowed".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a swallowed mismatch must still fail the execution: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Scheduler independence: the mismatch must be detected even when a
// short-circuiting combinator settles a sibling FIRST and aborts the
// mismatching loser before it is ever polled. Identity validation runs
// eagerly when each constituent future is finalized (`.future()`), so
// none of these tests slow the winner down artificially.
// ────────────────────────────────────────────────────────────────────────────

/// Builds the retry strategy shared by the scheduler-independence tests:
/// one 1-second retry so the combinator parks non-terminal on invocation 1,
/// then stop.
fn park_once_retry(_err: &durable::StepError, attempt: u32) -> durable::RetryDecision {
    if attempt >= 2 {
        durable::RetryDecision::Stop
    } else {
        durable::RetryDecision::Retry {
            delay: Duration::from_secs(1),
        }
    }
}

/// Test: `select_ok` with an IMMEDIATELY ready winner still detects a
/// mismatching loser.
///
/// On invocation 2 the winner succeeds on its first poll with no delay, so
/// `select_ok` aborts the loser as fast as the scheduler allows — under
/// lazy (poll-time) validation the loser's mismatch could go unobserved.
/// Eager validation at `.future()` creation records the fatal before the
/// combinator even starts racing.
#[tokio::test]
async fn select_ok_immediate_winner_still_detects_loser_mismatch() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    // Fails attempt 1 (parks the combinator), succeeds
                    // attempt 2 IMMEDIATELY — no sleep.
                    let winner = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            Ok(1_u32)
                        })
                        .name("winner")
                        .retry_strategy(park_once_retry)
                        .future();
                    // Renamed on invocation 2 → replay identity mismatch.
                    // Fails attempt 1 like the winner so the combinator
                    // parks non-terminal on invocation 1.
                    let name = if call == 0 { "loser" } else { "loser-renamed" };
                    let loser = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            Ok(2_u32)
                        })
                        .name(name)
                        .retry_strategy(park_once_retry)
                        .future();

                    let first = ctx.select_ok([winner, loser]).name("first-ok").await?;
                    let _ = first;
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a loser mismatch must fail the execution even with an immediate winner: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("loser") && msg.contains("loser-renamed"),
        "error should name both identities: {msg}"
    );
}

/// Test: `race` with an IMMEDIATELY settling winner still detects a
/// mismatching loser.
///
/// `race` aborts every loser on the FIRST settled outcome, so its abort
/// window is the widest of the combinators. The winner settles on its first
/// poll of invocation 2 with no delay.
#[tokio::test]
async fn race_immediate_winner_still_detects_loser_mismatch() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    let winner = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            Ok(1_u32)
                        })
                        .name("winner")
                        .retry_strategy(park_once_retry)
                        .future();
                    // Renamed on invocation 2 → replay identity mismatch.
                    // Fails attempt 1 like the winner so the combinator
                    // parks non-terminal on invocation 1.
                    let name = if call == 0 { "loser" } else { "loser-renamed" };
                    let loser = ctx
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() < 2 {
                                return Err::<u32, durable::BoxError>("transient".into());
                            }
                            Ok(2_u32)
                        })
                        .name(name)
                        .retry_strategy(park_once_retry)
                        .future();

                    let first = ctx.race([winner, loser]).name("fastest").await?;
                    let _ = first;
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a loser mismatch must fail the execution even when race settles instantly: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
}

/// Test: `try_join_all` reports the DEDICATED error when a sibling fails
/// first and the mismatching branch is aborted before being polled.
///
/// On invocation 2 the "failing" branch fails terminally on its first poll,
/// triggering `try_join_all`'s fail-fast `abort_all()`. The renamed sibling
/// may never be polled, so under lazy validation the execution would fail
/// with a generic `CombinatorError` instead of the dedicated one.
#[tokio::test]
async fn try_join_all_sibling_failure_still_detects_mismatch() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    // Fails attempt 1 with a retry (parks the combinator on
                    // invocation 1), then fails attempt 2 terminally and
                    // IMMEDIATELY — triggering fail-fast loser abort.
                    let failing = ctx
                        .step(|_| async { Err::<u32, durable::BoxError>("permanent".into()) })
                        .name("failing")
                        .retry_strategy(park_once_retry)
                        .future();
                    let name = if call == 0 { "stable" } else { "renamed" };
                    let mismatching = ctx
                        .step(|_| async { Ok(2_u32) })
                        .name(name)
                        .retry_strategy(park_once_retry)
                        .future();

                    let all = ctx
                        .try_join_all([failing, mismatching])
                        .name("gather")
                        .await?;
                    let _ = all;
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "the execution must fail: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "the DEDICATED error must win over the sibling's combinator failure: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("stable") && msg.contains("renamed"),
        "error should name both identities: {msg}"
    );
}

/// Test: a mismatching future that is dropped WITHOUT ever being polled is
/// still detected.
///
/// This deterministically pins the eager-validation property the three
/// combinator tests above rely on: identity is validated when the
/// `DurableFuture` is finalized (`.future()`), not when it is first polled.
/// A combinator's `abort_all()` cancelling an unpolled loser is exactly the
/// "never polled" case — under lazy (poll-time) validation this handler
/// would complete successfully and the mismatch would vanish.
#[tokio::test]
async fn unpolled_dropped_future_mismatch_still_detected() {
    let invocation = Arc::new(AtomicU32::new(0));
    let inv_clone = Arc::clone(&invocation);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let inv = Arc::clone(&inv_clone);
                async move {
                    let call = inv.fetch_add(1, Ordering::SeqCst);
                    if call == 0 {
                        // First invocation: step "original" runs and
                        // checkpoints, then the wait suspends.
                        let _a = ctx.step(|_| async { Ok(1_u32) }).name("original").await?;
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok::<_, durable::BoxError>("done".to_owned())
                    } else {
                        // Replay: claim position 1 with a DIFFERENT name,
                        // finalize the future, and drop it unpolled.
                        let renamed = ctx.step(|_| async { Ok(1_u32) }).name("renamed").future();
                        drop(renamed);
                        // The wait at position 2 replays its recorded
                        // completion, so the handler resolves successfully —
                        // only the recorded fatal can fail the execution.
                        ctx.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok("done".to_owned())
                    }
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_failure(),
        "a mismatch in an unpolled dropped future must fail the execution: output={:?}",
        result.output()
    );
    assert_eq!(
        result.error_type().unwrap_or(""),
        "NonDeterministicExecutionError",
        "error: {:?}",
        result.error_message()
    );
    let msg = result.error_message().unwrap_or("");
    assert!(
        msg.contains("original") && msg.contains("renamed"),
        "error should name both identities: {msg}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Empty names: an unchanged handler using empty item/branch names must
// replay deterministically (regression for the Some("")↔None false positive).
// ────────────────────────────────────────────────────────────────────────────

/// Test: a map whose `item_namer` returns empty strings suspends and
/// resumes without a false non-determinism failure.
///
/// The claim computes `Some("")` from the namer, but `build_child_update`
/// omits `Name` for empty strings so the checkpoint stores `None`; the
/// comparison must treat the two as the same identity.
#[tokio::test]
async fn empty_map_item_name_replays_successfully() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: durable::DurableContext| async move {
                let out = ctx
                    .map(vec![1_u32, 2], |child, item, _idx| async move {
                        let v = child
                            .step(move |_| async move { Ok(item * 10) })
                            .name("work")
                            .await?;
                        child.wait(Duration::from_secs(1)).name("hold").await?;
                        Ok(v)
                    })
                    .name("batch")
                    .item_namer(|_i| String::new())
                    .await?;
                Ok::<_, durable::BoxError>(out.iter().sum::<u32>())
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "empty item names must replay deterministically: {:?}",
        result.error_message()
    );
    assert_eq!(result.output(), Some(&30_u32));
    assert!(
        result.invocation_count() >= 2,
        "the wait must have suspended at least once, got {}",
        result.invocation_count()
    );
}

/// Test: a parallel branch named `""` suspends and resumes without a false
/// non-determinism failure. Parallel always claims branch names through its
/// namer, so an empty `Branch` name hits the same `Some("")`↔`None` path.
#[tokio::test]
async fn empty_parallel_branch_name_replays_successfully() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: durable::DurableContext| async move {
                let branches = vec![
                    durable::Branch::new("", |child: durable::DurableContext| async move {
                        let v = child.step(|_| async { Ok(1_u32) }).name("a").await?;
                        child.wait(Duration::from_secs(1)).name("hold").await?;
                        Ok(v)
                    }),
                    durable::Branch::new("named", |child: durable::DurableContext| async move {
                        let v = child.step(|_| async { Ok(2_u32) }).name("b").await?;
                        child.wait(Duration::from_secs(1)).name("hold").await?;
                        Ok(v)
                    }),
                ];
                let out = ctx.parallel(branches).name("fan-out").await?;
                Ok::<_, durable::BoxError>(out.iter().sum::<u32>())
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "an empty parallel branch name must replay deterministically: {:?}",
        result.error_message()
    );
    assert_eq!(result.output(), Some(&3_u32));
    assert!(
        result.invocation_count() >= 2,
        "the waits must have suspended at least once, got {}",
        result.invocation_count()
    );
}
