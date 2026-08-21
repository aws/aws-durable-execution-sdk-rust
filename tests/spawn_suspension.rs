//! Regression tests for suspension-scope accounting around `.spawn()` and
//! the awaited combinators (issues #48 and #49).
//!
//! Issue #48: dropping a `.spawn()` handle before its task's first poll used
//! to abort the task before its settling guard existed, leaving a phantom
//! outstanding spawn that parked the owner scope forever — the next
//! `ctx.wait` could never suspend and the invocation deadlocked.
//!
//! Issue #49: inputs of a directly awaited combinator used to park the
//! root suspension scope, so a losing input's wait suspended the
//! invocation even after another input settled. Each input now runs in
//! its own suspension scope, so a losing park stays isolated.

#![cfg(feature = "test-util")]
#![allow(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::test_util::LocalRunner;

/// Reproducer from issue #48. On a current-thread runtime the spawned task
/// cannot be polled until the handler yields, so dropping the handle aborts
/// the task before its first poll. The settling guard must still fire and
/// balance the spawn count, or the following `ctx.wait` can never suspend
/// the invocation.
#[tokio::test(flavor = "current_thread")]
async fn dropping_unpolled_spawn_does_not_deadlock_next_wait() {
    let runner = LocalRunner::new();
    let execution = runner.run(
        |(), ctx: durable::DurableContext| async move {
            // On a current-thread runtime, this spawned task cannot be polled
            // until the current handler yields.
            let handle = ctx.step(|_| async { Ok(()) }).spawn();

            // Dropping the handle aborts the task before its first poll.
            drop(handle);

            // This must still be able to suspend the invocation.
            ctx.wait(Duration::from_secs(1)).await?;

            Ok::<_, durable::BoxError>(())
        },
        (),
    );

    let result = tokio::time::timeout(Duration::from_secs(1), execution)
        .await
        .expect(
            "execution deadlocked because the dropped spawn remained \
             registered as outstanding",
        );

    assert!(
        result.is_success(),
        "execution should complete after the wait: {:?}",
        result.error_message(),
    );
}

/// Reproducer from issue #49. A directly awaited `race` must not let its
/// losing input park the root suspension scope: the fast step settles and
/// wins, so the losing 60 second wait — isolated in its own constituent
/// scope — must not suspend the invocation, and the execution completes in
/// a single invocation.
#[tokio::test(flavor = "current_thread")]
async fn race_does_not_propagate_loser_suspension_to_root() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: durable::DurableContext| async move {
                let waiting = ctx.wait(Duration::from_mins(1)).future();

                let fast = ctx
                    .step(|_| async {
                        // Give the wait input time to park before this resolves.
                        tokio::task::yield_now().await;
                        Ok(())
                    })
                    .future();

                ctx.race([waiting, fast]).await?;

                Ok::<_, durable::BoxError>(())
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "the immediately successful input should win: {:?}",
        result.error_message(),
    );
    assert_eq!(
        result.invocation_count(),
        1,
        "a losing wait must not suspend the root invocation",
    );
}

/// `select_ok` counterpart of the issue #49 reproducer: the immediately
/// successful input wins, and the losing wait — parked in its own
/// constituent scope — must not suspend the root invocation.
#[tokio::test(flavor = "current_thread")]
async fn select_ok_does_not_propagate_loser_suspension_to_root() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: durable::DurableContext| async move {
                let waiting = ctx.wait(Duration::from_mins(1)).future();

                let fast = ctx
                    .step(|_| async {
                        // Give the wait input time to park before this resolves.
                        tokio::task::yield_now().await;
                        Ok(())
                    })
                    .future();

                ctx.select_ok([waiting, fast]).await?;

                Ok::<_, durable::BoxError>(())
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "the immediately successful input should win: {:?}",
        result.error_message(),
    );
    assert_eq!(
        result.invocation_count(),
        1,
        "a losing wait must not suspend the root invocation",
    );
}

/// `try_join_all` counterpart: a settling error fails the join fast, and
/// the parked wait sibling must not suspend the root invocation — the
/// combined failure is recorded in the same (single) invocation.
#[tokio::test(flavor = "current_thread")]
async fn try_join_all_fail_fast_ignores_parked_sibling() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: durable::DurableContext| async move {
                let waiting = ctx.wait(Duration::from_mins(1)).future();

                let failing = ctx
                    .step(|_| async {
                        tokio::task::yield_now().await;
                        Err::<(), durable::BoxError>("boom".into())
                    })
                    .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
                    .future();

                ctx.try_join_all([waiting, failing]).await?;

                Ok::<_, durable::BoxError>(())
            },
            (),
        )
        .await;

    assert!(
        !result.is_success(),
        "the failing input must fail the join fast",
    );
    assert_eq!(
        result.invocation_count(),
        1,
        "a parked sibling must not suspend the root invocation when the \
         join outcome is already decided",
    );
}

/// When no input can make progress, the combinator itself must suspend —
/// and resume once the backend advances. A `race` over two waits parks
/// both inputs, suspends exactly once, and the shorter wait wins on the
/// resumed invocation.
#[tokio::test(flavor = "current_thread")]
async fn race_with_all_inputs_parked_suspends_and_resumes() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: durable::DurableContext| async move {
                let short = ctx.wait(Duration::from_secs(1)).future();
                let long = ctx.wait(Duration::from_mins(1)).future();

                ctx.race([short, long]).await?;

                Ok::<_, durable::BoxError>(())
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "the shorter wait should win after resumption: {:?}",
        result.error_message(),
    );
    assert!(
        result.invocation_count() >= 2,
        "a race whose inputs all parked must suspend the invocation and \
         resume later, not spin or complete early",
    );
}

/// `join_all` must keep waiting for a parked input: the settled sibling's
/// progress is preserved across the suspension, and the collection
/// completes on a later invocation with both outcomes.
#[tokio::test(flavor = "current_thread")]
async fn join_all_suspends_until_parked_input_resolves() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: durable::DurableContext| async move {
                let waiting = ctx.wait(Duration::from_secs(1)).future();

                let fast = ctx
                    .step(|_| async {
                        tokio::task::yield_now().await;
                        Ok(())
                    })
                    .future();

                let settled = ctx.join_all([waiting, fast]).await?;
                let fulfilled = settled
                    .iter()
                    .filter(|s| matches!(s, durable::Settled::Fulfilled(())))
                    .count();

                Ok::<_, durable::BoxError>(fulfilled)
            },
            (),
        )
        .await;

    assert_eq!(
        result.output(),
        Some(&2),
        "both inputs must settle as fulfilled: {:?}",
        result.error_message(),
    );
    assert!(
        result.invocation_count() >= 2,
        "join_all must suspend while an input is parked and complete on a \
         later invocation",
    );
}
