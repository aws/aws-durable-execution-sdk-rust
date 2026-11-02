//! Regression tests for suspension-scope accounting around `.spawn()` and
//! the awaited combinators (issues #48 and #49).
//!
//! Issue #48: dropping a `.spawn()` handle before its task's first poll used
//! to abort the task before its settling guard existed, leaving a phantom
//! outstanding spawn that parked the owner scope forever: the next
//! `ctx.wait` could never suspend and the invocation deadlocked.
//!
//! Issue #49: inputs of a directly awaited combinator used to park the
//! root suspension scope, so a losing input's wait suspended the
//! invocation even after another input settled. The invocation now
//! suspends only when no SDK work can make progress, so a losing wait
//! never suspends it once another input settled.
//!
//! `tokio::join!` over a lazy operation that waits and a running step lets
//! the step finish before the invocation suspends.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk as durable;
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
        )
        .expect("runner");

    assert!(
        result.is_success(),
        "execution should complete after the wait: {:?}",
        result.error_message(),
    );
}

/// Reproducer from issue #49. A directly awaited `race` must not let its
/// losing input park the root suspension scope: the fast step settles and
/// wins, so the losing 60 second wait, isolated in its own constituent
/// scope, must not suspend the invocation, and the execution completes in
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
        .await
        .expect("runner");

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
/// successful input wins, and the losing wait, parked in its own
/// constituent scope, must not suspend the root invocation.
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
        .await
        .expect("runner");

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
/// the parked wait sibling must not suspend the root invocation: the
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
        .await
        .expect("runner");

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

/// When no input can make progress, the combinator itself must suspend,
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
        .await
        .expect("runner");

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

                let outcomes = ctx.join_all([waiting, fast]).await?;
                let fulfilled = outcomes.iter().filter(|s| matches!(s, Ok(()))).count();

                Ok::<_, durable::BoxError>(fulfilled)
            },
            (),
        )
        .await
        .expect("runner");

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

/// Counts the starts and ends of a step body.
#[derive(Default)]
struct BodyRuns {
    started: std::sync::atomic::AtomicUsize,
    finished: std::sync::atomic::AtomicUsize,
}

impl BodyRuns {
    fn started(&self) -> usize {
        self.started.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn finished(&self) -> usize {
        self.finished.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// A step body that sleeps 100 ms and counts its starts and ends.
async fn slow_submit(runs: std::sync::Arc<BodyRuns>) -> Result<u32, durable::BoxError> {
    runs.started
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(100)).await;
    runs.finished
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    Ok(1)
}

/// A lazy callback result joined with a running step: the step finishes
/// before the invocation suspends on the callback.
#[tokio::test(flavor = "current_thread")]
async fn join_lazy_callback_result_with_running_step() {
    let runs = std::sync::Arc::new(BodyRuns::default());
    let handler_runs = std::sync::Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |(), ctx: durable::DurableContext| {
                let runs = std::sync::Arc::clone(&handler_runs);
                async move {
                    let cb = ctx.create_callback::<String>().name("approval").await?;
                    let (approval, submitted) = tokio::join!(
                        cb.result(),
                        ctx.step(move |_| slow_submit(runs)).name("submit").future()
                    );
                    submitted?;
                    Ok::<_, durable::BoxError>(approval?)
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_suspended(), "no reply was queued");
    assert_eq!(result.invocation_count(), 1);
    assert_eq!(
        result
            .operation("submit")
            .map(durable::test_util::TestOperation::status),
        Some(durable::OperationStatus::Succeeded)
    );
    assert_eq!(runs.started(), 1, "the body started once");
    assert_eq!(runs.finished(), 1, "the body finished once");
}

/// A lazy wait joined with a running step: the step finishes before the
/// invocation suspends on the wait, so it runs only once.
#[tokio::test(flavor = "current_thread")]
async fn join_lazy_wait_with_running_step() {
    let runs = std::sync::Arc::new(BodyRuns::default());
    let handler_runs = std::sync::Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |(), ctx: durable::DurableContext| {
                let runs = std::sync::Arc::clone(&handler_runs);
                async move {
                    let (timer, submitted) = tokio::join!(
                        ctx.wait(Duration::from_secs(1)).name("timer").future(),
                        ctx.step(move |_| slow_submit(runs)).name("submit").future()
                    );
                    timer?;
                    submitted?;
                    Ok::<_, durable::BoxError>(())
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2);
    assert_eq!(runs.started(), 1, "the body started once");
}

/// The winner of the issue #49 race is checkpointed, and a replay returns
/// the same winner without running the winning body again.
#[tokio::test(flavor = "current_thread")]
async fn race_winner_is_recorded_and_replayed() {
    let winners = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let fast_runs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let handler_winners = std::sync::Arc::clone(&winners);
    let handler_runs = std::sync::Arc::clone(&fast_runs);
    let result = LocalRunner::new()
        .run(
            move |(), ctx: durable::DurableContext| {
                let winners = std::sync::Arc::clone(&handler_winners);
                let runs = std::sync::Arc::clone(&handler_runs);
                async move {
                    // The race needs one output type: the losing wait runs
                    // in a child context that returns a string.
                    let waiting = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(Duration::from_mins(1)).await?;
                            Ok("slow".to_owned())
                        })
                        .future();
                    let fast = ctx
                        .step(move |_| async move {
                            runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            tokio::task::yield_now().await;
                            Ok("fast".to_owned())
                        })
                        .name("fast")
                        .future();
                    let winner = ctx.race([waiting, fast]).name("race").await?;
                    winners.lock().expect("lock").push(winner);
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>(())
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2);
    assert_eq!(
        *winners.lock().expect("lock"),
        vec!["fast".to_owned(), "fast".to_owned()],
        "the replay returned the same winner"
    );
    assert_eq!(
        fast_runs.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the winning body ran once"
    );
    assert_eq!(
        result
            .operation("race")
            .map(durable::test_util::TestOperation::status),
        Some(durable::OperationStatus::Succeeded)
    );
}
