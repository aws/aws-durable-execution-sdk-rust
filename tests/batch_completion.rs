//! Early completion of a batch: an item that is still running or parked
//! when the completion rule ends the batch is reported as `Started`, in the
//! live invocation and on replay.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::CompletionConfig;
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
