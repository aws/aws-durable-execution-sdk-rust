//! Early completion of a batch: an item that is still running or parked
//! when the completion rule ends the batch is reported as `Started`, in the
//! live invocation and on replay, and a custom completion reason survives
//! replay.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{CompletionConfig, CompletionDecision, CompletionOutcome};
use durable::test_util::LocalRunner;

/// The item statuses each invocation saw, formatted with `{:?}`.
type Seen = Arc<Mutex<Vec<Vec<String>>>>;

/// Runs `map([0, 1, 2])` named `m` with `min_successful(2)`. Every item
/// runs one step that returns `"x".repeat(size)`; item 2 first waits 5 s.
/// After the batch, the handler records the item statuses, then waits 1 s,
/// so the batch is replayed in a second invocation. Returns what each
/// invocation recorded.
async fn run_parked_item_batch(size: usize) -> Vec<Vec<String>> {
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let seen = Arc::clone(&handler_seen);
                async move {
                    let batch = ctx
                        .map(vec![0_u32, 1, 2], move |child, item, _idx| async move {
                            if item == 2 {
                                child.wait(Duration::from_secs(5)).name("park").await?;
                            }
                            let value = child
                                .step(move |_| async move { Ok("x".repeat(size)) })
                                .name("value")
                                .await?;
                            Ok(value)
                        })
                        .name("m")
                        .completion(CompletionConfig::with_min_successful(2))
                        .await_batch()
                        .await?;
                    let statuses = batch
                        .items
                        .iter()
                        .map(|item| format!("{:?}", item.status))
                        .collect();
                    seen.lock().expect("lock").push(statuses);
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>(())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    seen.lock().expect("lock").clone()
}

const PARKED_STARTED: [&str; 3] = ["Succeeded", "Succeeded", "Started"];

#[tokio::test]
async fn parked_item_is_started_live_and_on_replay_small() {
    assert_eq!(
        run_parked_item_batch(10).await,
        [PARKED_STARTED, PARKED_STARTED],
        "the live invocation and the replay report the parked item as Started"
    );
}

#[tokio::test]
async fn parked_item_is_started_live_and_on_replay_oversized() {
    // Two 150 KiB results exceed the batch record limit, so the batch
    // records its decision record and the replay reads the started set
    // from it.
    assert_eq!(
        run_parked_item_batch(150 * 1024).await,
        [PARKED_STARTED, PARKED_STARTED],
        "the live invocation and the replay report the parked item as Started"
    );
}

/// The reason and status each invocation saw.
type Outcomes = Arc<Mutex<Vec<(String, String)>>>;

/// Runs `map([0, 1, 2])` with `max_concurrency(1)` and a predicate that
/// completes with `outcome` once two items have settled. Every item runs one
/// step that returns `"x".repeat(size)`. After the batch, the handler
/// records the reason and the status, then waits 1 s, so the batch is
/// replayed in a second invocation. Returns what each invocation recorded.
async fn run_custom_completion(size: usize, outcome: CompletionOutcome) -> Vec<(String, String)> {
    let seen: Outcomes = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let seen = Arc::clone(&handler_seen);
                async move {
                    let batch = ctx
                        .map(vec![0_u32, 1, 2], move |child, _item, _idx| async move {
                            let value = child
                                .step(move |_| async move { Ok("x".repeat(size)) })
                                .name("value")
                                .await?;
                            Ok(value)
                        })
                        .name("m")
                        .max_concurrency(1)
                        .completion(CompletionConfig::with_completion_predicate(move |stats| {
                            if stats.settled() >= 2 {
                                CompletionDecision::Complete(outcome)
                            } else {
                                CompletionDecision::Continue
                            }
                        }))
                        .await_batch()
                        .await?;
                    seen.lock().expect("lock").push((
                        batch.reason.as_str().to_owned(),
                        format!("{:?}", batch.status()),
                    ));
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>(())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    seen.lock().expect("lock").clone()
}

fn twice(reason: &str, status: &str) -> [(String, String); 2] {
    let entry = (reason.to_owned(), status.to_owned());
    [entry.clone(), entry]
}

#[tokio::test]
async fn custom_succeeded_survives_small_replay() {
    assert_eq!(
        run_custom_completion(10, CompletionOutcome::Succeeded).await,
        twice("CUSTOM_COMPLETION_SUCCEEDED", "Succeeded"),
        "the live invocation and the replay record the same reason"
    );
}

#[tokio::test]
async fn custom_failed_survives_small_replay() {
    assert_eq!(
        run_custom_completion(10, CompletionOutcome::Failed).await,
        twice("CUSTOM_COMPLETION_FAILED", "Failed"),
        "the live invocation and the replay record the same reason"
    );
}

#[tokio::test]
async fn custom_failed_survives_oversized_replay() {
    // Two 150 KiB results exceed the batch record limit, so the batch
    // records its decision record and the replay reads the reason from it.
    assert_eq!(
        run_custom_completion(150 * 1024, CompletionOutcome::Failed).await,
        twice("CUSTOM_COMPLETION_FAILED", "Failed"),
        "the live invocation and the replay record the same reason"
    );
}

/// What each invocation saw: the reason, and how many times the predicate
/// had been called in the whole run when the batch returned.
type Calls = Arc<Mutex<Vec<(String, usize)>>>;

/// Runs `map([0, 1, 2])` named `m` with `max_concurrency(1)`, `nesting`,
/// and a predicate that completes as failed once two items have settled.
/// Every item runs two steps that return `size` bytes each, and returns
/// both joined. After the
/// batch, the handler records the reason and the predicate call count,
/// then waits 1 s, so the batch is replayed in a second invocation.
async fn run_counted_predicate(
    nesting: durable::builders::map_parallel::NestingMode,
    size: usize,
) -> Vec<(String, usize)> {
    let seen: Calls = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let handler_seen = Arc::clone(&seen);
    let handler_calls = Arc::clone(&calls);
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let seen = Arc::clone(&handler_seen);
                let calls = Arc::clone(&handler_calls);
                let predicate_calls = Arc::clone(&calls);
                async move {
                    let batch = ctx
                        .map(vec![0_u32, 1, 2], move |child, _item, _idx| async move {
                            let first = child
                                .step(move |_| async move { Ok("x".repeat(size)) })
                                .name("first")
                                .await?;
                            let second = child
                                .step(move |_| async move { Ok("y".repeat(size)) })
                                .name("second")
                                .await?;
                            Ok(first + &second)
                        })
                        .name("m")
                        .max_concurrency(1)
                        .nesting(nesting)
                        .completion(CompletionConfig::with_completion_predicate(move |stats| {
                            predicate_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            if stats.settled() >= 2 {
                                CompletionDecision::Complete(CompletionOutcome::Failed)
                            } else {
                                CompletionDecision::Continue
                            }
                        }))
                        .await_batch()
                        .await?;
                    seen.lock().expect("lock").push((
                        batch.reason.as_str().to_owned(),
                        calls.load(std::sync::atomic::Ordering::SeqCst),
                    ));
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>(())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    seen.lock().expect("lock").clone()
}

/// Asserts that the replay recorded the live reason, and that the
/// predicate was not called again while the batch was rebuilt.
fn assert_rebuild_skips_predicate(seen: &[(String, usize)]) {
    assert_eq!(seen.len(), 2, "two invocations: {seen:?}");
    let (live_reason, live_calls) = seen.first().expect("live invocation");
    let (replay_reason, replay_calls) = seen.get(1).expect("replay invocation");
    assert_eq!(live_reason, "CUSTOM_COMPLETION_FAILED");
    assert_eq!(
        replay_reason, live_reason,
        "the replay reads the recorded reason"
    );
    assert!(*live_calls > 0, "the live run asks the predicate");
    assert_eq!(
        replay_calls, live_calls,
        "rebuilding the completed batch calls the predicate no more"
    );
}

#[tokio::test]
async fn flat_oversized_rebuild_does_not_call_predicate() {
    // Two 160 KiB item results exceed the batch record limit, so the batch
    // records its decision record. Flat items have no child records, so
    // the replay rebuilds the batch by re-running the recorded items.
    let seen = run_counted_predicate(
        durable::builders::map_parallel::NestingMode::Flat,
        80 * 1024,
    )
    .await;
    assert_rebuild_skips_predicate(&seen);
}

#[tokio::test]
async fn oversized_child_rebuild_does_not_call_predicate() {
    // A 300 KiB item result exceeds the child record limit, so each child
    // records ReplayChildren and the replay rebuilds the batch by
    // re-running the recorded items under the decision record.
    let seen = run_counted_predicate(
        durable::builders::map_parallel::NestingMode::Normal,
        150 * 1024,
    )
    .await;
    assert_rebuild_skips_predicate(&seen);
}
