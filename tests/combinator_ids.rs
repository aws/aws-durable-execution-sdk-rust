//! The operation ID of a combinator.
//!
//! A combinator (`try_join_all`, `join_all`, `select_ok`, `race`) claims
//! its operation ID when it is first awaited or polled, or when it is
//! spawned, as a JS combinator claims its ID when it is first awaited. The
//! context method only collects the inputs. So:
//!
//! 1. A combinator held while the handler runs another operation takes the
//!    ID after that operation.
//! 2. A combinator that is dropped unawaited claims no ID and records
//!    nothing.
//! 3. Inputs that an iterator creates take their IDs before the combinator.
//! 4. A still-unclaimed combinator passed to another combinator, and a
//!    combinator first started inside another context's body, fail the
//!    execution with a context usage error.
//!
//! Except for the failure cases, each handler ends with a one-second wait
//! named `tail` and a step named `after`. The wait suspends the first
//! invocation, so the next invocation replays the combinator.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::panic)] // reason: test assertions

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, TestOperationDetails, TestResult};
use durable::{Branch, DurableContext, DurableFuture, OperationStatus, RetryDecision};
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

/// Asserts that each named operation is recorded at its positional ID.
fn assert_ids<O>(result: &TestResult<O>, expected: &[(&str, &str)]) {
    for (name, positional) in expected {
        let op = result.operation(name);
        assert!(op.is_some(), "no operation named {name}: {result}");
        assert_eq!(
            op.expect("checked above").id(),
            wire(positional),
            "{name} is at {positional}: {result}"
        );
    }
}

/// Asserts that no operation has the name `name`.
fn assert_unrecorded<O>(result: &TestResult<O>, name: &str) {
    assert!(
        result.operation(name).is_none(),
        "{name} records nothing: {result}"
    );
}

/// How many times each step body ran, by label.
type Runs = Arc<Mutex<BTreeMap<String, usize>>>;

fn new_runs() -> Runs {
    Arc::new(Mutex::new(BTreeMap::new()))
}

fn bump(runs: &Runs, label: &str) {
    *runs
        .lock()
        .expect("lock")
        .entry(label.to_owned())
        .or_default() += 1;
}

fn runs_of(runs: &Runs) -> BTreeMap<String, usize> {
    runs.lock().expect("lock").clone()
}

fn count(runs: &Runs, label: &str) -> usize {
    runs_of(runs).get(label).copied().unwrap_or(0)
}

fn once(labels: &[&str]) -> BTreeMap<String, usize> {
    labels.iter().map(|l| ((*l).to_owned(), 1)).collect()
}

/// A step named `label` that counts its body runs and returns `label`.
fn step(ctx: &DurableContext, runs: &Runs, label: &'static str) -> DurableFuture<String> {
    let runs = Arc::clone(runs);
    ctx.step(move |_| {
        bump(&runs, label);
        async move { Ok(label.to_owned()) }
    })
    .name(label)
    .future()
}

/// The `tail` wait and the `after` step that end every succeeding handler.
async fn tail_and_after(ctx: &DurableContext, runs: &Runs) -> Result<(), durable::BoxError> {
    ctx.wait(Duration::from_secs(1)).name("tail").await?;
    step(ctx, runs, "after").await?;
    Ok(())
}

/// Runs `handler` with `LocalRunner` and an empty event.
async fn run<O, F, Fut>(handler: F) -> TestResult<O>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, durable::BoxError>> + Send + 'static,
{
    LocalRunner::new().run(handler, ()).await.expect("runner")
}

/// The combinator a deferred-claim test uses.
#[derive(Clone, Copy)]
enum Method {
    TryJoinAll,
    JoinAll,
    SelectOk,
    Race,
}

/// Steps `a` and `b`, a combinator `all` over them created before step
/// `s1` and awaited after it, then `tail` and `after`.
async fn deferred(method: Method) -> (TestResult<()>, Runs) {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let a = step(&ctx, &runs, "a");
            let b = step(&ctx, &runs, "b");
            match method {
                Method::TryJoinAll => {
                    let all = ctx.try_join_all([a, b]).name("all");
                    step(&ctx, &runs, "s1").await?;
                    assert_eq!(all.await?, ["a", "b"]);
                }
                Method::JoinAll => {
                    let all = ctx.join_all([a, b]).name("all");
                    step(&ctx, &runs, "s1").await?;
                    let outcomes = all.await?;
                    assert!(outcomes.iter().all(Result::is_ok));
                }
                Method::SelectOk => {
                    let all = ctx.select_ok([a, b]).name("all");
                    step(&ctx, &runs, "s1").await?;
                    all.await?;
                }
                Method::Race => {
                    let all = ctx.race([a, b]).name("all");
                    step(&ctx, &runs, "s1").await?;
                    all.await?;
                }
            }
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("a", "1"),
            ("b", "2"),
            ("s1", "3"),
            ("all", "4"),
            ("tail", "5"),
            ("after", "6"),
        ],
    );
    assert_eq!(count(&runs, "s1"), 1);
    assert_eq!(count(&runs, "after"), 1);
    (result, runs)
}

/// Test 1.
#[tokio::test]
async fn deferred_try_join_all_claims_id_at_first_poll() {
    let (_, runs) = deferred(Method::TryJoinAll).await;
    assert_eq!(runs_of(&runs), once(&["a", "b", "s1", "after"]));
}

/// Test 2.
#[tokio::test]
async fn deferred_join_all_claims_id_at_first_poll() {
    let (_, runs) = deferred(Method::JoinAll).await;
    assert_eq!(runs_of(&runs), once(&["a", "b", "s1", "after"]));
}

/// Test 3. A losing input may not run, so only `s1` and `after` have fixed
/// body counts.
#[tokio::test]
async fn deferred_select_ok_claims_id_at_first_poll() {
    deferred(Method::SelectOk).await;
}

/// Test 4.
#[tokio::test]
async fn deferred_race_claims_id_at_first_poll() {
    deferred(Method::Race).await;
}

/// Test 5.
#[tokio::test]
async fn never_awaited_combinator_claims_no_id() {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let a = step(&ctx, &runs, "a");
            let b = step(&ctx, &runs, "b");
            let all = ctx.try_join_all([a, b]).name("all");
            step(&ctx, &runs, "s1").await?;
            drop(all);
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(&result, &[("s1", "3"), ("tail", "4"), ("after", "5")]);
    assert_unrecorded(&result, "all");
    assert_eq!(runs_of(&runs), once(&["s1", "after"]), "a and b never run");
}

/// Test 6.
#[tokio::test]
async fn iterator_built_inputs_claim_before_combinator() {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let values = ctx
                .try_join_all(
                    ["i0", "i1", "i2"]
                        .iter()
                        .copied()
                        .map(|n| step(&ctx, &runs, n)),
                )
                .name("all")
                .await?;
            assert_eq!(values, ["i0", "i1", "i2"]);
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("i0", "1"),
            ("i1", "2"),
            ("i2", "3"),
            ("all", "4"),
            ("tail", "5"),
            ("after", "6"),
        ],
    );
    assert_eq!(runs_of(&runs), once(&["i0", "i1", "i2", "after"]));
}

/// Test 7.
#[tokio::test]
async fn spawn_after_deferral_claims_id_at_spawn() {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let a = step(&ctx, &runs, "a");
            let b = step(&ctx, &runs, "b");
            let all = ctx.try_join_all([a, b]).name("all");
            step(&ctx, &runs, "s1").await?;
            let handle = all.spawn();
            assert_eq!(handle.await?, ["a", "b"]);
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("a", "1"),
            ("b", "2"),
            ("s1", "3"),
            ("all", "4"),
            ("tail", "5"),
            ("after", "6"),
        ],
    );
    assert_eq!(runs_of(&runs), once(&["a", "b", "s1", "after"]));
}

/// Test 8. The first invocation suspends inside `all` on `w`. The second
/// resumes `all` at 3 and suspends on `tail`. The third runs `after`.
#[tokio::test]
async fn suspended_deferred_combinator_resumes_at_same_id() {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").future();
            let all = ctx.try_join_all([w]).name("all");
            step(&ctx, &runs, "s1").await?;
            all.await?;
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 3, "{result}");
    assert_ids(
        &result,
        &[
            ("w", "1"),
            ("s1", "2"),
            ("all", "3"),
            ("tail", "4"),
            ("after", "5"),
        ],
    );
    assert_eq!(runs_of(&runs), once(&["s1", "after"]));
}

/// The rule-6 message for an unclaimed inner combinator.
fn unclaimed_input_message(method: &str, context: &str, outer: &str) -> String {
    format!(
        "context usage error in \"{method}\": the combinator was created on context \
         \"{context}\", but combinator \"{outer}\" started it as an input. Start the inner \
         combinator with .spawn() and pass the returned handle to the outer combinator."
    )
}

/// How the nested-combinator tests treat `inner` and `outer`.
#[derive(Clone, Copy)]
enum Nested {
    /// `inner` from `.future()`, `outer` awaited.
    Unclaimed,
    /// `inner` from `.future()`, `outer` dropped unawaited.
    DroppedOuter,
    /// `inner` from `.spawn()`, `outer` awaited.
    SpawnedInner,
}

/// Steps `a`, `b`, `c`, `inner = race([a, b])`, `outer = try_join_all([inner,
/// c])`, step `s1`, then `outer` per `nested`, then `tail` and `after`.
async fn nested(nested: Nested) -> (TestResult<()>, Runs) {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let a = step(&ctx, &runs, "a");
            let b = step(&ctx, &runs, "b");
            let c = step(&ctx, &runs, "c");
            let inner = ctx.race([a, b]).name("inner");
            let inner = match nested {
                Nested::SpawnedInner => inner.spawn(),
                Nested::Unclaimed | Nested::DroppedOuter => inner.future(),
            };
            let outer = ctx.try_join_all([inner, c]).name("outer");
            step(&ctx, &runs, "s1").await?;
            match nested {
                Nested::DroppedOuter => drop(outer),
                Nested::Unclaimed | Nested::SpawnedInner => {
                    outer.await?;
                }
            }
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    (result, runs)
}

/// Test 9.
#[tokio::test]
async fn unclaimed_combinator_input_is_rejected() {
    let (result, runs) = nested(Nested::Unclaimed).await;
    assert!(result.is_failure(), "{result}");
    assert_eq!(result.error_type(), Some("Error"), "{result}");
    assert_eq!(
        result.error_message(),
        Some(unclaimed_input_message("race", "root", "5").as_str()),
        "{result}"
    );
    assert_eq!(result.invocation_count(), 1, "{result}");
    assert_ids(&result, &[("s1", "4"), ("outer", "5")]);
    let outer = result.operation("outer").expect("checked above");
    assert_eq!(outer.status(), OperationStatus::Started, "{result}");
    assert_unrecorded(&result, "inner");
    assert_unrecorded(&result, "after");
    assert_eq!(runs_of(&runs), once(&["s1"]), "no input body runs");
}

/// Test 10.
#[tokio::test]
async fn dropped_outer_with_unclaimed_input_claims_nothing() {
    let (result, runs) = nested(Nested::DroppedOuter).await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(&result, &[("s1", "4"), ("tail", "5"), ("after", "6")]);
    assert_unrecorded(&result, "inner");
    assert_unrecorded(&result, "outer");
    assert_eq!(runs_of(&runs), once(&["s1", "after"]), "no input body runs");
}

/// Test 11.
#[tokio::test]
async fn spawned_combinator_input_is_accepted() {
    let (result, runs) = nested(Nested::SpawnedInner).await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("inner", "4"),
            ("s1", "5"),
            ("outer", "6"),
            ("tail", "7"),
            ("after", "8"),
        ],
    );
    assert_eq!(count(&runs, "c"), 1);
    assert_eq!(count(&runs, "s1"), 1);
    assert_eq!(count(&runs, "after"), 1);
}

/// Where test 12 first starts a combinator created on the root context.
#[derive(Clone, Copy, Debug)]
enum OtherBody {
    ChildAwait,
    ChildSpawn,
    WithRetry,
    MapItem,
    ParallelBranch,
}

/// The existing context usage message for a root combinator started inside
/// the body of context `active`.
fn other_body_message(active: &str) -> String {
    format!(
        "context usage error in \"try_join_all\": the operation was called on context \
         \"root\", but the SDK is running the body of context \"{active}\". Inside \
         run_in_child_context, with_retry, map and parallel, call operations on the context \
         passed to the closure."
    )
}

/// Takes the combinator out of its shared slot and awaits it.
async fn await_slot(slot: Slot) -> Result<Vec<String>, durable::BoxError> {
    let all = slot.lock().expect("lock").take().expect("started once");
    Ok(all.await?)
}

type Slot = Arc<Mutex<Option<durable::builders::TryJoinAllBuilder<String>>>>;

/// Test 12.
#[tokio::test]
async fn combinator_started_in_other_body_is_rejected() {
    let cases = [
        (OtherBody::ChildAwait, "2"),
        (OtherBody::ChildSpawn, "2"),
        (OtherBody::WithRetry, "2-1"),
        (OtherBody::MapItem, "2-1"),
        (OtherBody::ParallelBranch, "2-1"),
    ];
    for (case, active) in cases {
        let result = run(move |(), ctx: DurableContext| async move {
            let runs = new_runs();
            let a = step(&ctx, &runs, "a");
            let slot: Slot = Arc::new(Mutex::new(Some(ctx.try_join_all([a]).name("all"))));
            match case {
                OtherBody::ChildAwait => {
                    ctx.run_in_child_context(move |_child| await_slot(slot))
                        .name("child")
                        .await?;
                }
                OtherBody::ChildSpawn => {
                    ctx.run_in_child_context(move |_child| async move {
                        let all = slot.lock().expect("lock").take().expect("started once");
                        Ok(all.spawn().await?)
                    })
                    .name("child")
                    .await?;
                }
                OtherBody::WithRetry => {
                    ctx.with_retry(move |_child| await_slot(Arc::clone(&slot)))
                        .name("retry")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .await?;
                }
                OtherBody::MapItem => {
                    ctx.map(vec![0_u32], move |_child, _item, _idx| {
                        await_slot(Arc::clone(&slot))
                    })
                    .name("map")
                    .await?;
                }
                OtherBody::ParallelBranch => {
                    ctx.parallel(vec![Branch::new("branch", move |_child| await_slot(slot))])
                        .name("parallel")
                        .await?;
                }
            }
            Ok::<_, durable::BoxError>(())
        })
        .await;
        assert!(result.is_failure(), "{case:?}: {result}");
        assert_eq!(result.error_type(), Some("Error"), "{case:?}: {result}");
        assert_eq!(
            result.error_message(),
            Some(other_body_message(active).as_str()),
            "{case:?}: {result}"
        );
        assert_eq!(result.invocation_count(), 1, "{case:?}: {result}");
    }
}

/// Test 13. `tokio::join!` polls its futures in argument order on its first
/// poll, so `x` claims before `y`.
#[tokio::test]
async fn joined_deferred_combinators_claim_in_join_order() {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let a = step(&ctx, &runs, "a");
            let b = step(&ctx, &runs, "b");
            let x = ctx.try_join_all([a]).name("x");
            let y = ctx.try_join_all([b]).name("y");
            step(&ctx, &runs, "s1").await?;
            let (x, y) = tokio::join!(x, y);
            assert_eq!(x?, ["a"]);
            assert_eq!(y?, ["b"]);
            tail_and_after(&ctx, &runs).await
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("a", "1"),
            ("b", "2"),
            ("s1", "3"),
            ("x", "4"),
            ("y", "5"),
            ("tail", "6"),
            ("after", "7"),
        ],
    );
    assert_eq!(runs_of(&runs), once(&["a", "b", "s1", "after"]));
}

/// Test 14.
#[tokio::test]
async fn deferred_failed_combinator_uses_first_poll_id() {
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let a_runs = Arc::clone(&runs);
            let a = ctx
                .step(move |_| {
                    bump(&a_runs, "a");
                    async { Err::<String, durable::BoxError>("boom".into()) }
                })
                .name("a")
                .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                .future();
            let b = step(&ctx, &runs, "b");
            let all = ctx.try_join_all([a, b]).name("all");
            step(&ctx, &runs, "s1").await?;
            let failed = all.await.is_err();
            tail_and_after(&ctx, &runs).await?;
            Ok(failed)
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.output(), Some(&true), "all fails on both runs");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("a", "1"),
            ("b", "2"),
            ("s1", "3"),
            ("all", "4"),
            ("tail", "5"),
            ("after", "6"),
        ],
    );
    let all = result.operation("all").expect("checked above");
    assert_eq!(all.status(), OperationStatus::Failed, "{result}");
    assert_eq!(count(&runs, "a"), 1, "a replays from its record");
    assert_eq!(count(&runs, "s1"), 1);
    assert_eq!(count(&runs, "after"), 1);
}

/// Test 15. Two 200 KiB results exceed the 256 KiB record limit, so `all`
/// records `ReplayChildren`, and the replay rebuilds the result from the
/// inputs' records.
#[tokio::test]
async fn deferred_oversized_combinator_uses_first_poll_id() {
    const LEN: usize = 200 * 1024;
    let runs = new_runs();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let big = |label: &'static str| {
                let runs = Arc::clone(&runs);
                ctx.step(move |_| {
                    bump(&runs, label);
                    async { Ok("x".repeat(LEN)) }
                })
                .name(label)
                .future()
            };
            let all = ctx.try_join_all([big("a"), big("b")]).name("all");
            step(&ctx, &runs, "s1").await?;
            let lens: Vec<usize> = all.await?.iter().map(String::len).collect();
            tail_and_after(&ctx, &runs).await?;
            Ok(lens)
        }
    })
    .await;
    assert!(result.is_success(), "{result}");
    assert_eq!(result.output(), Some(&vec![LEN, LEN]), "replay rebuilds it");
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_ids(
        &result,
        &[
            ("a", "1"),
            ("b", "2"),
            ("s1", "3"),
            ("all", "4"),
            ("tail", "5"),
            ("after", "6"),
        ],
    );
    let all = result.operation("all").expect("checked above");
    let TestOperationDetails::Context(details) = all.details() else {
        panic!("all is a context: {result}");
    };
    assert!(details.replay_children(), "{result}");
    assert_eq!(runs_of(&runs), once(&["a", "b", "s1", "after"]));
}
