//! The operation IDs of `map`, `parallel` and `parallel_mixed` items.
//!
//! A batch takes exactly one ID from its caller, its own. Item `i` of the
//! batch has the positional ID `<batch>-<i+1>`, in the batch's own
//! namespace, in both nesting modes. So an operation the caller starts
//! beside a batch never shifts the items' IDs, and replay finds every item
//! and every later operation at the ID it was recorded at.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{CompletionConfig, NestingMode};
use durable::serdes::SerdesContext;
use durable::test_util::{LocalRunner, TestOperation, TestOperationDetails, TestResult};
use durable::{Branch, BranchSet, OperationKind, Serdes};
use sha2::{Digest, Sha256};

/// The wire ID of a positional ID: its SHA-256 digest in lowercase hex.
fn wire(positional: &str) -> String {
    use std::fmt::Write as _;
    Sha256::digest(positional.as_bytes())
        .iter()
        .fold(String::new(), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

/// The operation recorded at `positional`, if any.
fn at<'a, O>(result: &'a TestResult<O>, positional: &str) -> Option<&'a TestOperation> {
    let id = wire(positional);
    result.operations().iter().find(|op| op.id() == id)
}

/// The operation recorded at `positional`; panics when there is none.
fn expect_at<'a, O>(result: &'a TestResult<O>, positional: &str) -> &'a TestOperation {
    let op = at(result, positional);
    assert!(op.is_some(), "no operation at {positional}: {result}");
    op.expect("checked above")
}

/// Asserts that `positional` holds an operation of `kind` named `name`
/// whose parent is `parent` (a positional ID, or `None` for the root).
fn assert_op<O>(
    result: &TestResult<O>,
    positional: &str,
    kind: OperationKind,
    name: &str,
    parent: Option<&str>,
) {
    let op = expect_at(result, positional);
    assert_eq!(op.kind(), kind, "kind at {positional}: {result}");
    assert_eq!(op.name(), Some(name), "name at {positional}: {result}");
    assert_eq!(
        op.parent_id().map(str::to_owned),
        parent.map(wire),
        "parent at {positional}: {result}"
    );
}

/// How many times each step body ran, by label.
type Runs = Arc<Mutex<BTreeMap<String, usize>>>;

fn bump(runs: &Runs, label: impl Into<String>) {
    *runs.lock().expect("lock").entry(label.into()).or_default() += 1;
}

fn runs_of(runs: &Runs) -> BTreeMap<String, usize> {
    runs.lock().expect("lock").clone()
}

fn once(labels: &[&str]) -> BTreeMap<String, usize> {
    labels.iter().map(|l| ((*l).to_owned(), 1)).collect()
}

/// Test 1: a map started with `.spawn()`, then two steps, then the map is
/// awaited, then a one-second wait. Replay must find `s2` at `3`, not at
/// the ID of the map's first item.
#[tokio::test]
async fn spawned_map_beside_steps_replays() {
    let runs: Runs = Arc::new(Mutex::new(BTreeMap::new()));
    let handler_runs = Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let a = ctx
                        .map(
                            vec!["a0".to_owned(), "a1".to_owned()],
                            |_c, s, _| async move { Ok(s) },
                        )
                        .name("a")
                        .spawn();
                    let r1 = Arc::clone(&runs);
                    let s1 = ctx
                        .step(move |_| {
                            bump(&r1, "s1");
                            async { Ok("s1-value".to_owned()) }
                        })
                        .name("s1")
                        .await?;
                    let r2 = Arc::clone(&runs);
                    let s2 = ctx
                        .step(move |_| {
                            bump(&r2, "s2");
                            async { Ok("s2-value".to_owned()) }
                        })
                        .name("s2")
                        .await?;
                    let va = a.await?;
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>((va, s1, s2))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(
        runs_of(&runs),
        once(&["s1", "s2"]),
        "each step body runs once"
    );
    assert_eq!(
        result.output(),
        Some(&(
            vec!["a0".to_owned(), "a1".to_owned()],
            "s1-value".to_owned(),
            "s2-value".to_owned()
        ))
    );
    assert_op(&result, "1", OperationKind::Context, "a", None);
    assert_op(
        &result,
        "1-1",
        OperationKind::Context,
        "map-item-0",
        Some("1"),
    );
    assert_op(
        &result,
        "1-2",
        OperationKind::Context,
        "map-item-1",
        Some("1"),
    );
    assert_op(&result, "2", OperationKind::Step, "s1", None);
    assert_op(&result, "3", OperationKind::Step, "s2", None);
    assert_op(&result, "4", OperationKind::Wait, "after", None);
}

/// Test 2: the same with a `parallel` of two branches.
#[tokio::test]
async fn spawned_parallel_beside_steps_replays() {
    let runs: Runs = Arc::new(Mutex::new(BTreeMap::new()));
    let handler_runs = Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let p = ctx
                        .parallel(vec![
                            Branch::new("b0", |_c| async move { Ok(10_u32) }),
                            Branch::new("b1", |_c| async move { Ok(20_u32) }),
                        ])
                        .name("p")
                        .spawn();
                    let r1 = Arc::clone(&runs);
                    let s1 = ctx
                        .step(move |_| {
                            bump(&r1, "s1");
                            async { Ok(1_u32) }
                        })
                        .name("s1")
                        .await?;
                    let r2 = Arc::clone(&runs);
                    let s2 = ctx
                        .step(move |_| {
                            bump(&r2, "s2");
                            async { Ok(2_u32) }
                        })
                        .name("s2")
                        .await?;
                    let vp = p.await?;
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>((vp, s1, s2))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(
        runs_of(&runs),
        once(&["s1", "s2"]),
        "each step body runs once"
    );
    assert_eq!(result.output(), Some(&(vec![10, 20], 1, 2)));
    assert_op(&result, "1", OperationKind::Context, "p", None);
    assert_op(&result, "1-1", OperationKind::Context, "b0", Some("1"));
    assert_op(&result, "1-2", OperationKind::Context, "b1", Some("1"));
    assert_op(&result, "2", OperationKind::Step, "s1", None);
    assert_op(&result, "3", OperationKind::Step, "s2", None);
    assert_op(&result, "4", OperationKind::Wait, "after", None);
}

/// One serdes call: `"write"` or `"read"`, the value, the invocation it
/// ran in and the positional ID of the operation it saw.
type SerdesLog = Arc<Mutex<Vec<(&'static str, String, usize, String)>>>;

/// A string serdes that records each call with the operation it saw.
struct RecordingSerdes {
    log: SerdesLog,
    invocation: Arc<AtomicUsize>,
}

impl RecordingSerdes {
    fn new(log: &SerdesLog, invocation: &Arc<AtomicUsize>) -> Self {
        Self {
            log: Arc::clone(log),
            invocation: Arc::clone(invocation),
        }
    }

    fn record(&self, dir: &'static str, value: &str, context: &SerdesContext) {
        // Recover the positional ID from the wire ID: every ID these tests
        // record is at most two levels deep with counters below 10.
        let id = context.operation_id();
        let positional = (1..10)
            .flat_map(|a| {
                std::iter::once(a.to_string()).chain((1..10).map(move |b| format!("{a}-{b}")))
            })
            .find(|p| wire(p) == id)
            .unwrap_or_else(|| id.to_owned());
        self.log.lock().expect("lock").push((
            dir,
            value.to_owned(),
            self.invocation.load(Ordering::SeqCst),
            positional,
        ));
    }
}

impl Serdes<String> for RecordingSerdes {
    // reason: the serdes records synchronously; the async-fn form is what user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn serialize(
        &self,
        value: String,
        context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        self.record("write", &value, &context);
        Ok(serde_json::to_string(&value)?)
    }

    // reason: the serdes records synchronously; the async-fn form is what user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn deserialize(
        &self,
        wire: String,
        context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        let value: String = serde_json::from_str(&wire)?;
        self.record("read", &value, &context);
        Ok(value)
    }
}

/// Test 3: two maps started with `.spawn()` and joined, then a wait. Every
/// value is read on replay at the ID it was written at.
#[tokio::test]
async fn overlapping_maps_read_their_own_ids() {
    let log: SerdesLog = Arc::new(Mutex::new(Vec::new()));
    let invocation = Arc::new(AtomicUsize::new(0));
    let (handler_log, handler_invocation) = (Arc::clone(&log), Arc::clone(&invocation));
    let result = LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let log = Arc::clone(&handler_log);
                let invocation = Arc::clone(&handler_invocation);
                async move {
                    invocation.fetch_add(1, Ordering::SeqCst);
                    let a = ctx
                        .map(
                            vec!["a0".to_owned(), "a1".to_owned()],
                            |_c, s, _| async move { Ok(s) },
                        )
                        .serdes(RecordingSerdes::new(&log, &invocation))
                        .name("a")
                        .spawn();
                    let b = ctx
                        .map(
                            vec!["b0".to_owned(), "b1".to_owned()],
                            |_c, s, _| async move { Ok(s) },
                        )
                        .serdes(RecordingSerdes::new(&log, &invocation))
                        .name("b")
                        .spawn();
                    let (ra, rb) = tokio::join!(a, b);
                    let out = (ra?, rb?);
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(
        result.output(),
        Some(&(
            vec!["a0".to_owned(), "a1".to_owned()],
            vec!["b0".to_owned(), "b1".to_owned()]
        ))
    );
    let expected_id = |value: &str| match value {
        "a0" => Some("1-1"),
        "a1" => Some("1-2"),
        "b0" => Some("2-1"),
        "b1" => Some("2-2"),
        _ => None,
    };
    let log = log.lock().expect("lock").clone();
    for (dir, value, _, id) in &log {
        assert_eq!(
            Some(id.as_str()),
            expected_id(value),
            "{dir} of {value}: {log:?}"
        );
    }
    let replay_reads = log
        .iter()
        .filter(|(dir, _, invocation, _)| *dir == "read" && *invocation == 2)
        .count();
    assert_eq!(replay_reads, 4, "the replay reads every value: {log:?}");
}

/// The batch a [`batch_item_layout`] case runs.
#[derive(Clone, Copy, Debug)]
enum Batch {
    Map,
    Parallel,
    ParallelMixed,
}

/// Runs `batch` as operation `1` with two items or branches that each run
/// one step `inner`, under `nesting`, then a step `after`.
async fn run_layout_case(batch: Batch, nesting: NestingMode) -> TestResult<()> {
    LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| async move {
                match batch {
                    Batch::Map => {
                        ctx.map(vec![1_u32, 2], |c, i, _| async move {
                            Ok(c.step(move |_| async move { Ok(i) }).name("inner").await?)
                        })
                        .name("batch")
                        .nesting(nesting)
                        .await?;
                    }
                    Batch::Parallel => {
                        ctx.parallel(vec![
                            Branch::new("b0", |c| async move {
                                Ok(c.step(|_| async { Ok(1_u32) }).name("inner").await?)
                            }),
                            Branch::new("b1", |c| async move {
                                Ok(c.step(|_| async { Ok(2_u32) }).name("inner").await?)
                            }),
                        ])
                        .name("batch")
                        .nesting(nesting)
                        .await?;
                    }
                    Batch::ParallelMixed => {
                        let mut branches = BranchSet::new();
                        let _b0 = branches.add("b0", |c| async move {
                            Ok(c.step(|_| async { Ok(1_u32) }).name("inner").await?)
                        });
                        let _b1 = branches.add("b1", |c| async move {
                            Ok(c.step(|_| async { Ok("two".to_owned()) })
                                .name("inner")
                                .await?)
                        });
                        ctx.parallel_mixed(branches)
                            .name("batch")
                            .nesting(nesting)
                            .await?;
                    }
                }
                ctx.step(|_| async { Ok(0_u32) }).name("after").await?;
                Ok::<(), durable::BoxError>(())
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

/// Test 4: the item layout of `map`, `parallel` and `parallel_mixed`, in
/// both nesting modes.
#[tokio::test]
async fn batch_item_layout() {
    for batch in [Batch::Map, Batch::Parallel, Batch::ParallelMixed] {
        let item_names = match batch {
            Batch::Map => ["map-item-0", "map-item-1"],
            Batch::Parallel | Batch::ParallelMixed => ["b0", "b1"],
        };
        for nesting in [NestingMode::Normal, NestingMode::Flat] {
            let case = format!("{batch:?} {nesting:?}");
            let result = run_layout_case(batch, nesting).await;
            assert!(
                result.is_success(),
                "{case}: the execution succeeds: {result}"
            );
            assert_op(&result, "1", OperationKind::Context, "batch", None);
            for (item, name) in ["1-1", "1-2"].into_iter().zip(item_names) {
                let step = format!("{item}-1");
                if nesting == NestingMode::Flat {
                    assert!(
                        at(&result, item).is_none(),
                        "{case}: no record at {item}: {result}"
                    );
                    assert_op(&result, &step, OperationKind::Step, "inner", Some("1"));
                } else {
                    assert_op(&result, item, OperationKind::Context, name, Some("1"));
                    assert_op(&result, &step, OperationKind::Step, "inner", Some(item));
                }
            }
            assert_op(&result, "2", OperationKind::Step, "after", None);
            let expected = if nesting == NestingMode::Flat { 4 } else { 6 };
            assert_eq!(result.operations().len(), expected, "{case}: {result}");
        }
    }
}

/// A map of three items that each run one step, then a step `after`.
async fn run_sequential_map(completion: Option<CompletionConfig>) -> TestResult<()> {
    LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let completion = completion.clone();
                async move {
                    let mut map = ctx
                        .map(vec![1_u32, 2, 3], |c, i, _| async move {
                            Ok(c.step(move |_| async move { Ok(i) }).name("inner").await?)
                        })
                        .name("m")
                        .max_concurrency(1);
                    if let Some(completion) = completion {
                        map = map.completion(completion);
                    }
                    map.await?;
                    ctx.step(|_| async { Ok(0_u32) }).name("after").await?;
                    Ok::<(), durable::BoxError>(())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

/// Test 5: items admitted one at a time take the same IDs.
#[tokio::test]
async fn sequential_items_take_the_same_ids() {
    let result = run_sequential_map(None).await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    for (i, item) in ["1-1", "1-2", "1-3"].into_iter().enumerate() {
        assert_op(
            &result,
            item,
            OperationKind::Context,
            &format!("map-item-{i}"),
            Some("1"),
        );
        assert_op(
            &result,
            &format!("{item}-1"),
            OperationKind::Step,
            "inner",
            Some(item),
        );
    }
    assert_op(&result, "2", OperationKind::Step, "after", None);
}

/// Test 6: an early completion records nothing for the items it never
/// started, and the caller's next operation is the batch's ID plus one.
#[tokio::test]
async fn early_completion_leaves_later_items_unrecorded() {
    let result = run_sequential_map(Some(CompletionConfig::with_min_successful(1))).await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_op(
        &result,
        "1-1",
        OperationKind::Context,
        "map-item-0",
        Some("1"),
    );
    assert!(at(&result, "1-2").is_none(), "no record at 1-2: {result}");
    assert!(at(&result, "1-3").is_none(), "no record at 1-3: {result}");
    assert_op(&result, "2", OperationKind::Step, "after", None);
    assert_eq!(result.operations().len(), 4, "{result}");
}

/// Test 7: an awaited empty batch, and a batch that is created and
/// dropped without being polled, each take only their own ID.
#[tokio::test]
async fn empty_and_dropped_batches_take_only_their_own_id() {
    let empty = LocalRunner::new()
        .run(
            |_e: serde_json::Value, ctx: durable::DurableContext| async move {
                ctx.map(Vec::<u32>::new(), |_c, i, _| async move { Ok(i) })
                    .name("empty")
                    .await?;
                ctx.step(|_| async { Ok(0_u32) }).name("after").await?;
                Ok::<(), durable::BoxError>(())
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(empty.is_success(), "the execution succeeds: {empty}");
    assert_op(&empty, "1", OperationKind::Context, "empty", None);
    assert_op(&empty, "2", OperationKind::Step, "after", None);

    let dropped = LocalRunner::new()
        .run(
            |_e: serde_json::Value, ctx: durable::DurableContext| async move {
                let unpolled = ctx
                    .map(vec![1_u32, 2, 3], |_c, i, _| async move { Ok(i) })
                    .name("dropped")
                    .future();
                drop(unpolled);
                ctx.step(|_| async { Ok(0_u32) }).name("after").await?;
                Ok::<(), durable::BoxError>(())
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(dropped.is_success(), "the execution succeeds: {dropped}");
    assert!(
        at(&dropped, "1").is_none(),
        "the dropped batch records nothing: {dropped}"
    );
    assert_op(&dropped, "2", OperationKind::Step, "after", None);
    assert_eq!(dropped.operations().len(), 1, "{dropped}");
}

/// Test 8: two overlapping maps whose items each fit the checkpoint limit
/// but whose batch records exceed it, so replay reads each batch from its
/// item records, each return their own values.
#[tokio::test]
async fn oversized_batch_replays_beside_another_batch() {
    const SIZE: usize = 150 * 1024;
    let result = LocalRunner::new()
        .run(
            |_e: serde_json::Value, ctx: durable::DurableContext| async move {
                let a = ctx
                    .map(vec!['a', 'b'], |_c, ch, _| async move {
                        Ok(ch.to_string().repeat(SIZE))
                    })
                    .name("a")
                    .spawn();
                let b = ctx
                    .map(vec!['c', 'd'], |_c, ch, _| async move {
                        Ok(ch.to_string().repeat(SIZE))
                    })
                    .name("b")
                    .spawn();
                let (ra, rb) = tokio::join!(a, b);
                let out: Vec<Vec<char>> = [ra?, rb?]
                    .iter()
                    .map(|values| {
                        values
                            .iter()
                            .map(|v| {
                                assert_eq!(v.len(), SIZE);
                                v.chars().next().expect("non-empty")
                            })
                            .collect()
                    })
                    .collect();
                ctx.wait(Duration::from_secs(1)).name("after").await?;
                Ok::<_, durable::BoxError>(out)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(result.output(), Some(&vec![vec!['a', 'b'], vec!['c', 'd']]));
    for batch in ["1", "2"] {
        assert!(
            matches!(
                expect_at(&result, batch).details(),
                TestOperationDetails::Context(c) if c.replay_children()
            ),
            "batch {batch} records ReplayChildren: {result}"
        );
    }
    assert_op(
        &result,
        "1-1",
        OperationKind::Context,
        "map-item-0",
        Some("1"),
    );
    assert_op(
        &result,
        "2-2",
        OperationKind::Context,
        "map-item-1",
        Some("2"),
    );
}

/// Test 9: each branch of a `parallel_mixed` decodes with its own serdes at
/// `<batch>-<i+1>`, live and on replay.
#[tokio::test]
async fn parallel_mixed_branch_serdes_by_index() {
    let logs: [SerdesLog; 2] = [
        Arc::new(Mutex::new(Vec::new())),
        Arc::new(Mutex::new(Vec::new())),
    ];
    let invocation = Arc::new(AtomicUsize::new(0));
    let (handler_logs, handler_invocation) = (logs.clone(), Arc::clone(&invocation));
    let result = LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let logs = handler_logs.clone();
                let invocation = Arc::clone(&handler_invocation);
                async move {
                    invocation.fetch_add(1, Ordering::SeqCst);
                    let mut branches = BranchSet::new();
                    let first = branches.add_with_serdes(
                        "first",
                        RecordingSerdes::new(&logs[0], &invocation),
                        |_c| async { Ok("one".to_owned()) },
                    );
                    let second = branches.add_with_serdes(
                        "second",
                        RecordingSerdes::new(&logs[1], &invocation),
                        |_c| async { Ok("two".to_owned()) },
                    );
                    ctx.parallel_mixed(branches).name("mixed").await?;
                    let out = (first.into_result()?, second.into_result()?);
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(result.output(), Some(&("one".to_owned(), "two".to_owned())));
    for (branch, (log, (id, value))) in logs
        .iter()
        .zip([("1-1", "one"), ("1-2", "two")])
        .enumerate()
    {
        let log = log.lock().expect("lock").clone();
        assert!(!log.is_empty(), "branch {branch} serdes ran");
        for (dir, seen, _, seen_id) in &log {
            assert_eq!(
                (seen.as_str(), seen_id.as_str()),
                (value, id),
                "branch {branch} {dir}: {log:?}"
            );
        }
        assert!(
            log.iter()
                .any(|(dir, _, inv, _)| *dir == "read" && *inv == 2),
            "branch {branch} decodes on replay: {log:?}"
        );
    }
}

/// Test 12: the handler of test 1, where each item runs a step and then a
/// one-second wait, so the map is still running when the first invocation
/// ends.
#[tokio::test]
async fn spawned_unfinished_map_beside_steps_replays() {
    let runs: Runs = Arc::new(Mutex::new(BTreeMap::new()));
    let handler_runs = Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let item_runs = Arc::clone(&runs);
                    let a = ctx
                        .map(vec![0_u32, 1], move |c, i, _| {
                            let runs = Arc::clone(&item_runs);
                            async move {
                                let v = c
                                    .step(move |_| {
                                        bump(&runs, format!("item{i}.in"));
                                        async move { Ok(i * 10) }
                                    })
                                    .name("in")
                                    .await?;
                                c.wait(Duration::from_secs(1)).name("item-wait").await?;
                                Ok(v)
                            }
                        })
                        .name("a")
                        .spawn();
                    let r1 = Arc::clone(&runs);
                    let s1 = ctx
                        .step(move |_| {
                            bump(&r1, "s1");
                            async { Ok(1_u32) }
                        })
                        .name("s1")
                        .await?;
                    let r2 = Arc::clone(&runs);
                    let s2 = ctx
                        .step(move |_| {
                            bump(&r2, "s2");
                            async { Ok(2_u32) }
                        })
                        .name("s2")
                        .await?;
                    let va = a.await?;
                    Ok::<_, durable::BoxError>((va, s1, s2))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(
        runs_of(&runs),
        once(&["item0.in", "item1.in", "s1", "s2"]),
        "each step body runs once"
    );
    assert_eq!(result.output(), Some(&(vec![0, 10], 1, 2)));
    assert_op(
        &result,
        "1-1",
        OperationKind::Context,
        "map-item-0",
        Some("1"),
    );
    assert_op(
        &result,
        "1-2",
        OperationKind::Context,
        "map-item-1",
        Some("1"),
    );
    assert_op(&result, "3", OperationKind::Step, "s2", None);
}

/// Test 13: a map whose items each start an inner map with `.spawn()`, run
/// two steps, await the inner map and wait one second.
#[tokio::test]
async fn nested_spawned_map_beside_steps_replays() {
    let runs: Runs = Arc::new(Mutex::new(BTreeMap::new()));
    let handler_runs = Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |_e: serde_json::Value, ctx: durable::DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let out = ctx
                        .map(vec![0_u32, 1], move |c, i, _| {
                            let runs = Arc::clone(&runs);
                            async move {
                                let inner = c
                                    .map(
                                        vec![1_u32, 2],
                                        move |_c, v, _| async move { Ok(v + 10 * i) },
                                    )
                                    .name("inner")
                                    .spawn();
                                let r1 = Arc::clone(&runs);
                                let s1 = c
                                    .step(move |_| {
                                        bump(&r1, format!("o{i}.s1"));
                                        async move { Ok(i * 100 + 1) }
                                    })
                                    .name("s1")
                                    .await?;
                                let r2 = Arc::clone(&runs);
                                let s2 = c
                                    .step(move |_| {
                                        bump(&r2, format!("o{i}.s2"));
                                        async move { Ok(i * 100 + 2) }
                                    })
                                    .name("s2")
                                    .await?;
                                let iv = inner.await?;
                                c.wait(Duration::from_secs(1)).name("item-wait").await?;
                                Ok((s1, s2, iv))
                            }
                        })
                        .name("outer")
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(
        runs_of(&runs),
        once(&["o0.s1", "o0.s2", "o1.s1", "o1.s2"]),
        "each step body runs once"
    );
    assert_eq!(
        result.output(),
        Some(&vec![(1, 2, vec![1, 2]), (101, 102, vec![11, 12])])
    );
    assert_op(
        &result,
        "1-1-1",
        OperationKind::Context,
        "inner",
        Some("1-1"),
    );
    assert_op(
        &result,
        "1-1-1-1",
        OperationKind::Context,
        "map-item-0",
        Some("1-1-1"),
    );
    assert_op(
        &result,
        "1-1-1-2",
        OperationKind::Context,
        "map-item-1",
        Some("1-1-1"),
    );
    assert_op(&result, "1-1-2", OperationKind::Step, "s1", Some("1-1"));
    assert_op(&result, "1-1-3", OperationKind::Step, "s2", Some("1-1"));
    assert_op(&result, "1-2-3", OperationKind::Step, "s2", Some("1-2"));
}
