//! A `map` whose result is over 256 KiB records `ReplayChildren`, so a
//! later invocation rebuilds the result from the item records. Each test
//! runs a map, then waits 1 s, so invocation 1 runs the map live and
//! invocation 2 replays it. One replay decodes each item whose record holds
//! its value once, and runs only the bodies of the items whose own record
//! carries `ReplayChildren`.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{CompletionConfig, CompletionDecision, CompletionOutcome};
use durable::serdes::SerdesContext;
use durable::test_util::{LocalRunner, TestOperationDetails, TestResult};
use durable::{BoxError, DurableContext, JsonSerdes, Serdes};

/// `JsonSerdes` that logs `(invocation, operation ID)` for every decode.
#[derive(Clone)]
struct DecodeLog {
    invocation: Arc<AtomicUsize>,
    decodes: Arc<Mutex<Vec<(usize, String)>>>,
}

impl DecodeLog {
    fn new() -> Self {
        Self {
            invocation: Arc::new(AtomicUsize::new(0)),
            decodes: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The decodes of operation `id` in invocation `invocation`.
    fn count(&self, invocation: usize, id: &str) -> usize {
        let decodes = self.decodes.lock().expect("lock");
        decodes
            .iter()
            .filter(|(inv, op)| *inv == invocation && op == id)
            .count()
    }
}

impl Serdes<String> for DecodeLog {
    async fn serialize(&self, value: String, ctx: SerdesContext) -> Result<String, BoxError> {
        JsonSerdes.serialize(value, ctx).await
    }

    async fn deserialize(&self, wire: String, ctx: SerdesContext) -> Result<String, BoxError> {
        let invocation = self.invocation.load(Ordering::SeqCst);
        let id = ctx.operation_id().to_owned();
        self.decodes.lock().expect("lock").push((invocation, id));
        JsonSerdes.deserialize(wire, ctx).await
    }
}

/// Runs a map named `m` over `lens`, whose item `i` returns `lens[i]`
/// bytes, with `max_concurrency`, then waits 1 s. Returns the result, the
/// serdes and the item bodies that ran as `(invocation, index)`, sorted,
/// because concurrent item bodies run in any order.
async fn map_of_lengths_then_wait(
    lens: Vec<usize>,
    max_concurrency: usize,
) -> (TestResult<usize>, DecodeLog, Vec<(usize, usize)>) {
    let serdes = DecodeLog::new();
    let bodies: Arc<Mutex<Vec<(usize, usize)>>> = Arc::default();
    let handler_serdes = serdes.clone();
    let handler_bodies = Arc::clone(&bodies);
    let result = LocalRunner::new()
        .run(
            move |(), ctx: DurableContext| {
                let serdes = handler_serdes.clone();
                let bodies = Arc::clone(&handler_bodies);
                let lens = lens.clone();
                async move {
                    let invocation = serdes.invocation.fetch_add(1, Ordering::SeqCst) + 1;
                    let out = ctx
                        .map(lens, move |_child, len, idx| {
                            let bodies = Arc::clone(&bodies);
                            async move {
                                bodies.lock().expect("lock").push((invocation, idx));
                                Ok("b".repeat(len))
                            }
                        })
                        .name("m")
                        .max_concurrency(max_concurrency)
                        .serdes(serdes)
                        .await?;
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(out.len())
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    let mut bodies = bodies.lock().expect("lock").clone();
    bodies.sort_unstable();
    (result, serdes, bodies)
}

/// The IDs of the children of `m` whose record holds the result `payload`.
fn items_with_result(result: &TestResult<usize>, payload: &str) -> Vec<String> {
    let map = result.operation("m").expect("the map record");
    result
        .children(map)
        .filter(|c| c.result() == Some(payload))
        .map(|c| c.id().to_owned())
        .collect()
}

/// Item 0 is stored inline, and item 1, 300 KiB, records `ReplayChildren`,
/// so the batch does too. One replay decodes item 0 once and runs item 1's
/// body again to rebuild its value.
async fn replay_decodes_inline_item_once(max_concurrency: usize) {
    let (result, serdes, bodies) =
        map_of_lengths_then_wait(vec![0, 300 * 1024], max_concurrency).await;
    assert_eq!(result.output(), Some(&2), "{result}");
    let inline = items_with_result(&result, "\"\"");
    assert_eq!(inline.len(), 1, "item 0 is stored inline: {result}");
    let item0 = inline.first().expect("item 0");
    assert_eq!(serdes.count(2, item0), 1, "one replay decodes item 0 once");
    assert_eq!(
        bodies,
        vec![(1, 0), (1, 1), (2, 1)],
        "only item 1 is rebuilt"
    );
}

#[tokio::test]
async fn oversized_batch_replay_decodes_each_inline_item_once() {
    replay_decodes_inline_item_once(2).await;
}

#[tokio::test]
async fn oversized_sequential_batch_replay_decodes_each_inline_item_once() {
    replay_decodes_inline_item_once(1).await;
}

/// Three items of 100 KiB each fit their own record while the batch result
/// does not, so the batch records `ReplayChildren`. Its replay decodes each
/// item once and runs no item body.
#[tokio::test]
async fn oversized_batch_of_inline_items_replays_without_running_bodies() {
    for max_concurrency in [1, 3] {
        let (result, serdes, bodies) =
            map_of_lengths_then_wait(vec![100 * 1024; 3], max_concurrency).await;
        assert_eq!(result.output(), Some(&3), "{result}");
        let items = items_with_result(&result, &format!("\"{}\"", "b".repeat(100 * 1024)));
        assert_eq!(items.len(), 3, "every item is stored inline: {result}");
        for id in &items {
            assert_eq!(serdes.count(2, id), 1, "one replay decodes each item once");
        }
        assert!(
            bodies.iter().all(|(invocation, _)| *invocation == 1),
            "no item body runs on replay: {bodies:?}"
        );
    }
}

/// A map over an empty input within `plugin_child_operations_depth`
/// records its result with `ReplayChildren`, so the next invocation
/// rebuilds it from a replay plan with no item. The rebuild returns the
/// recorded reason and does not ask the completion predicate again.
#[tokio::test]
async fn zero_item_plan_replay_keeps_recorded_reason_without_reinvoking_predicate() {
    let asks = Arc::new(AtomicUsize::new(0));
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let (handler_asks, handler_seen) = (Arc::clone(&asks), Arc::clone(&seen));
    let options = durable::Options::builder()
        .plugin_child_operations_depth(1)
        .build()
        .expect("valid options");
    let result = LocalRunner::new()
        .options(options)
        .run(
            move |(), ctx: DurableContext| {
                let asks = Arc::clone(&handler_asks);
                let seen = Arc::clone(&handler_seen);
                async move {
                    let batch = ctx
                        .map(
                            Vec::<u32>::new(),
                            |_child, item, _idx| async move { Ok(item) },
                        )
                        .name("m")
                        .completion(CompletionConfig::with_completion_predicate(move |_| {
                            asks.fetch_add(1, Ordering::SeqCst);
                            CompletionDecision::complete(CompletionOutcome::Succeeded)
                        }))
                        .await_batch()
                        .await?;
                    let reason = batch.reason.as_str().to_owned();
                    seen.lock().expect("lock").push(reason);
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(())
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert!(result.is_success(), "{result} {:?}", result.error_message());
    let map = result.operation("m").expect("the map record");
    assert!(
        matches!(map.details(), TestOperationDetails::Context(c) if c.replay_children()),
        "the map records ReplayChildren: {result}"
    );
    assert_eq!(
        *seen.lock().expect("lock"),
        ["CUSTOM_COMPLETION_SUCCEEDED"; 2]
    );
    assert_eq!(asks.load(Ordering::SeqCst), 1, "only the live map asks");
}
