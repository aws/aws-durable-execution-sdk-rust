//! Regression tests for suspension-scope accounting around `.spawn()` and
//! the awaited combinators (issues #48 and #49).
//!
//! Issue #48: dropping a `.spawn()` handle before its task's first poll used
//! to abort the task before its settling guard existed, leaving a phantom
//! outstanding spawn that parked the owner scope forever — the next
//! `ctx.wait` could never suspend and the invocation deadlocked.
//!
//! Issue #49 (known failing, guarded here as `#[ignore]`): inputs of a
//! directly awaited combinator park the root suspension scope, so a losing
//! input's wait suspends the invocation even after another input settled.

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

/// Reproducer from issue #49. A directly awaited `race` lets its losing
/// input park the root suspension scope: the 60 second wait suspends the
/// invocation even though the fast step already settled, so the execution
/// takes a second invocation where one suffices.
#[tokio::test(flavor = "current_thread")]
#[ignore = "known failing - tracked by #49"]
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
