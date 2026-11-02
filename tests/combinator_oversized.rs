//! A combinator whose combined result is larger than the 256 KiB payload
//! limit records `ReplayChildren` instead of the result, and replay rebuilds
//! the result from the inputs (issue #112).
//!
//! Each handler runs one combinator, records what it saw, then waits 1 s.
//! So the combinator runs live in the first invocation and replays from its
//! record in the second.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, TestOperationDetails, TestResult};
use durable::{OperationErrorKind, RetryDecision};

/// What each invocation saw, one entry per invocation.
type Seen<T> = Arc<Mutex<Vec<T>>>;

/// Asserts that the operation named `name` is a context whose record
/// carries `ReplayChildren` and the payload `payload`.
fn assert_replay_children_record<O>(result: &TestResult<O>, name: &str, payload: &str) {
    let op = result.operation(name).expect("the combinator operation");
    assert!(
        matches!(op.details(), TestOperationDetails::Context(c) if c.replay_children()),
        "{name} records ReplayChildren: {:?}",
        op.details()
    );
    assert_eq!(op.result(), Some(payload));
}

/// A step body that counts its runs in `runs` and returns `fill` repeated
/// 200 KiB times.
fn big_step(
    ctx: &durable::DurableContext,
    name: &str,
    fill: &'static str,
    runs: &'static AtomicUsize,
) -> durable::DurableFuture<String> {
    ctx.step(move |_| async move {
        runs.fetch_add(1, Ordering::SeqCst);
        Ok(fill.repeat(200 * 1024))
    })
    .name(name)
    .future()
}

#[tokio::test]
async fn try_join_all_oversized_result_replays() {
    static A_RUNS: AtomicUsize = AtomicUsize::new(0);
    static B_RUNS: AtomicUsize = AtomicUsize::new(0);

    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: durable::DurableContext| async move {
                let a = big_step(&ctx, "a", "a", &A_RUNS);
                let b = big_step(&ctx, "b", "b", &B_RUNS);
                let values = ctx.try_join_all([a, b]).name("gather").await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, durable::BoxError>(values.iter().map(String::len).sum::<usize>())
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.output(), Some(&409_600));
    assert_eq!(result.invocation_count(), 2);
    assert_replay_children_record(&result, "gather", "");
    assert_eq!(A_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(B_RUNS.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn join_all_oversized_result_replays() {
    static A_RUNS: AtomicUsize = AtomicUsize::new(0);
    static B_RUNS: AtomicUsize = AtomicUsize::new(0);

    // Whether the `Err` item was a `ChildContext` error, per invocation.
    let seen: Seen<bool> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let seen = Arc::clone(&handler_seen);
                async move {
                    let a = big_step(&ctx, "a", "a", &A_RUNS);
                    let b = big_step(&ctx, "b", "b", &B_RUNS);
                    let c = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom".into()) })
                        .name("c")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();
                    let outcomes = ctx.join_all([a, b, c]).name("gather").await?;
                    for outcome in &outcomes {
                        if let Err(err) = outcome {
                            seen.lock()
                                .expect("lock")
                                .push(matches!(err.kind(), OperationErrorKind::ChildContext(_)));
                        }
                    }
                    ctx.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok::<_, durable::BoxError>(outcomes.iter().filter(|o| o.is_ok()).count())
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.output(), Some(&2));
    assert_eq!(result.invocation_count(), 2);
    assert_replay_children_record(&result, "gather", "");
    assert_eq!(*seen.lock().expect("lock"), vec![true, true]);
}

/// A child context whose body counts its runs in `runs` and returns 300 KiB.
/// The child's own record therefore carries `ReplayChildren` too.
fn big_child(
    ctx: &durable::DurableContext,
    runs: &'static AtomicUsize,
) -> durable::DurableFuture<String> {
    ctx.run_in_child_context(move |_child| async move {
        runs.fetch_add(1, Ordering::SeqCst);
        Ok("w".repeat(300 * 1024))
    })
    .name("winner")
    .future()
}

/// A step whose body counts its runs in `runs` and then never finishes.
fn pending_step(
    ctx: &durable::DurableContext,
    runs: &'static AtomicUsize,
) -> durable::DurableFuture<String> {
    ctx.step(move |_| async move {
        runs.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        Ok(String::new())
    })
    .name("loser")
    .future()
}

/// Asserts the `(loser runs, winner runs)` entries of a race-like handler:
/// the winner body ran once live and once more to rebuild its value, and
/// the loser body did not run on replay.
fn assert_counters(seen: &Seen<(usize, usize)>) {
    let seen = seen.lock().expect("lock");
    assert_eq!(seen.len(), 2, "one entry per invocation: {seen:?}");
    assert_eq!(seen[0].1, 1, "the winner body ran once live");
    assert_eq!(
        seen[1].1, 2,
        "the winner body ran again to rebuild its value"
    );
    assert_eq!(seen[0].0, seen[1].0, "the loser body did not run on replay");
}

#[tokio::test]
async fn race_oversized_winner_replays_only_the_winner() {
    static WINNER_RUNS: AtomicUsize = AtomicUsize::new(0);
    static LOSER_RUNS: AtomicUsize = AtomicUsize::new(0);

    let seen: Seen<(usize, usize)> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let seen = Arc::clone(&handler_seen);
                async move {
                    let winner = big_child(&ctx, &WINNER_RUNS);
                    let loser = pending_step(&ctx, &LOSER_RUNS);
                    let value = ctx.race([winner, loser]).name("first").await?;
                    seen.lock().expect("lock").push((
                        LOSER_RUNS.load(Ordering::SeqCst),
                        WINNER_RUNS.load(Ordering::SeqCst),
                    ));
                    ctx.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok::<_, durable::BoxError>(value.len())
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.output(), Some(&307_200));
    assert_replay_children_record(&result, "first", "{\"winner\":0}");
    assert_counters(&seen);
}

#[tokio::test]
async fn select_ok_oversized_winner_replays_only_the_winner() {
    static WINNER_RUNS: AtomicUsize = AtomicUsize::new(0);
    static LOSER_RUNS: AtomicUsize = AtomicUsize::new(0);

    let seen: Seen<(usize, usize)> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let seen = Arc::clone(&handler_seen);
                async move {
                    let failing = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom".into()) })
                        .name("failing")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();
                    let winner = big_child(&ctx, &WINNER_RUNS);
                    let loser = pending_step(&ctx, &LOSER_RUNS);
                    let value = ctx
                        .select_ok([failing, winner, loser])
                        .name("first-ok")
                        .await?;
                    seen.lock().expect("lock").push((
                        LOSER_RUNS.load(Ordering::SeqCst),
                        WINNER_RUNS.load(Ordering::SeqCst),
                    ));
                    ctx.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok::<_, durable::BoxError>(value.len())
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.output(), Some(&307_200));
    assert_replay_children_record(&result, "first-ok", "{\"winner\":1}");
    assert_counters(&seen);
}
