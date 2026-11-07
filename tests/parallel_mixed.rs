//! `parallel_mixed` runs branches whose results have different types, and
//! each `BranchHandle` returns its branch's own type.
//!
//! A handler that checks replay reads its handles, records what they
//! returned, and then waits 1 s. So the parallel runs live in the first
//! invocation and replays from its record in the second. A branch body
//! increments a counter, so a counter of 1 means replay did not run the
//! branch again.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::panic)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{
    CompletionConfig, CompletionDecision, CompletionOutcome, CompletionReason, NestingMode,
};
use durable::serdes::{FileSystemSerdes, SerdesContext};
use durable::test_util::{
    ExecutionStatus, LocalRunner, TestOperation, TestOperationDetails, TestResult,
};
use durable::{
    BoxError, Branch, BranchSet, ChildContextErrorKind, DurableContext, OperationError,
    OperationErrorKind, RetryDecision, Serdes, StepErrorKind, TypedError, WireError,
};
use serde::{Deserialize, Serialize};

/// A shared log that handlers append to in every invocation.
type Log<T> = Arc<Mutex<Vec<T>>>;

fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}

fn entries<T: Clone>(log: &Log<T>) -> Vec<T> {
    log.lock().expect("lock").clone()
}

fn push<T>(log: &Log<T>, value: T) {
    log.lock().expect("lock").push(value);
}

/// Runs `handler` with the unit event and returns the result.
async fn run<F, Fut>(handler: F) -> TestResult<()>
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    LocalRunner::new()
        .run(handler, ())
        .await
        .expect("the run finishes")
}

/// Waits 1 s, so the handler runs a second time and replays.
async fn wait_then_replay(ctx: &DurableContext) -> Result<(), BoxError> {
    ctx.wait(Duration::from_secs(1)).name("after").await?;
    Ok(())
}

/// The recorded operation named `name`.
fn op<'a>(result: &'a TestResult<()>, name: &str) -> &'a TestOperation {
    result
        .operation(name)
        .unwrap_or_else(|| panic!("operation {name} is recorded: {result}"))
}

/// The batch record of the parallel named `name`, parsed as JSON.
fn record(result: &TestResult<()>, name: &str) -> serde_json::Value {
    let payload = op(result, name)
        .result()
        .expect("the parallel has a payload");
    serde_json::from_str(payload).expect("the batch record is JSON")
}

/// The `result` field of item `index` in a batch record.
fn item_result(record: &serde_json::Value, index: usize) -> &serde_json::Value {
    record
        .pointer(&format!("/all/{index}/result"))
        .unwrap_or_else(|| panic!("item {index} has a result: {record}"))
}

/// The recorded failure an operation error carries.
fn wire_of(err: &OperationError) -> WireError {
    err.wire().expect("the error carries a record").clone()
}

/// Whether `err` is a `ChildContext` error of kind `NotCompleted`.
fn is_not_completed(err: &OperationError) -> bool {
    matches!(
        err.kind(),
        OperationErrorKind::ChildContext(child)
            if matches!(child.kind(), ChildContextErrorKind::NotCompleted { .. })
    )
}

/// Whether `err` is a `ChildContext` error of kind `ChildFailed`.
fn is_child_failed(err: &OperationError) -> bool {
    matches!(
        err.kind(),
        OperationErrorKind::ChildContext(child)
            if matches!(child.kind(), ChildContextErrorKind::ChildFailed { .. })
    )
}

/// A counter shared by a branch body across invocations.
fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn count(counter: &AtomicUsize) -> usize {
    counter.load(Ordering::SeqCst)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Reservation {
    sku: String,
    quantity: u32,
}

#[tokio::test]
async fn typed_handles_return_struct_integer_and_string() {
    let seen: Log<(Reservation, u64, String)> = log();
    let counters = [counter(), counter(), counter()];
    let (seen_h, counters_h) = (Arc::clone(&seen), counters.clone());
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let [c0, c1, c2] = counters_h.clone();
        async move {
            let mut set = BranchSet::new();
            let reservation = set.add("reserve", move |_ctx| async move {
                c0.fetch_add(1, Ordering::SeqCst);
                Ok(Reservation {
                    sku: "sku-1".to_owned(),
                    quantity: 2,
                })
            });
            let number = set.add("count", move |_ctx| async move {
                c1.fetch_add(1, Ordering::SeqCst);
                Ok(7_u64)
            });
            let text = set.add("label", move |_ctx| async move {
                c2.fetch_add(1, Ordering::SeqCst);
                Ok("ready".to_owned())
            });
            ctx.parallel_mixed(set).name("pm").await?;
            push(
                &seen,
                (
                    reservation.into_result()?,
                    number.into_result()?,
                    text.into_result()?,
                ),
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let expected = (
        Reservation {
            sku: "sku-1".to_owned(),
            quantity: 2,
        },
        7_u64,
        "ready".to_owned(),
    );
    assert_eq!(entries(&seen), vec![expected.clone(), expected]);
    for c in &counters {
        assert_eq!(count(c), 1, "replay does not run a completed branch");
    }
}

#[tokio::test]
async fn oversized_branch_replays_like_parallel() {
    let seen: Log<(String, u32)> = log();
    let steps = [counter(), counter()];
    let (seen_h, steps_h) = (Arc::clone(&seen), steps.clone());
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let [s0, s1] = steps_h.clone();
        async move {
            let mut set = BranchSet::new();
            let big = set.add("big", move |child: DurableContext| async move {
                let a = child
                    .step(move |_| async move {
                        s0.fetch_add(1, Ordering::SeqCst);
                        Ok("a".repeat(150_000))
                    })
                    .name("first")
                    .await?;
                let b = child
                    .step(move |_| async move {
                        s1.fetch_add(1, Ordering::SeqCst);
                        Ok("b".repeat(150_000))
                    })
                    .name("second")
                    .await?;
                Ok(format!("{a}{b}"))
            });
            let small = set.add("small", |_ctx| async { Ok(1_u32) });
            ctx.parallel_mixed(set).name("pm").await?;
            push(&seen, (big.into_result()?, small.into_result()?));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let expected = (format!("{}{}", "a".repeat(150_000), "b".repeat(150_000)), 1);
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations read the handles");
    for entry in &seen {
        assert_eq!(entry, &expected);
    }
    for s in &steps {
        assert_eq!(count(s), 1, "replay does not run a completed step");
    }
}

/// A struct whose fields serialize in an order other than sorted order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Unsorted {
    zeta: u32,
    alpha: String,
}

/// The fields of an operation that the history comparison covers.
#[derive(Debug, PartialEq)]
struct Shape {
    id: String,
    name: Option<String>,
    kind: String,
    sub_type: Option<String>,
    status: String,
    parent_id: Option<String>,
    payload: Option<serde_json::Value>,
}

fn shapes(result: &TestResult<()>) -> Vec<Shape> {
    result
        .operations()
        .iter()
        .map(|op| Shape {
            id: op.id().to_owned(),
            name: op.name().map(str::to_owned),
            kind: format!("{:?}", op.kind()),
            sub_type: op.sub_type().map(str::to_owned),
            status: format!("{:?}", op.status()),
            parent_id: op.parent_id().map(str::to_owned),
            payload: op
                .result()
                .map(|p| serde_json::from_str(p).expect("every payload is JSON")),
        })
        .collect()
}

#[tokio::test]
async fn history_equals_parallel_over_json_values() {
    let mixed = run(|(), ctx| async move {
        let mut set = BranchSet::new();
        let _object = set.add("object", |_ctx| async {
            Ok(Unsorted {
                zeta: 1,
                alpha: "a".to_owned(),
            })
        });
        let _number = set.add("number", |_ctx| async { Ok(5_u32) });
        let _text = set.add("text", |_ctx| async { Ok("hi".to_owned()) });
        ctx.parallel_mixed(set).name("pm").await?;
        wait_then_replay(&ctx).await
    })
    .await;
    let homogeneous = run(|(), ctx| async move {
        let branches: Vec<Branch<serde_json::Value>> = vec![
            Branch::new("object", |_ctx| async {
                Ok(serde_json::json!({"zeta": 1, "alpha": "a"}))
            }),
            Branch::new("number", |_ctx| async { Ok(serde_json::json!(5)) }),
            Branch::new("text", |_ctx| async { Ok(serde_json::json!("hi")) }),
        ];
        let _values = ctx.parallel(branches).name("pm").await?;
        wait_then_replay(&ctx).await
    })
    .await;
    assert!(mixed.is_success(), "parallel_mixed succeeds: {mixed}");
    assert!(homogeneous.is_success(), "parallel succeeds: {homogeneous}");
    assert_eq!(shapes(&mixed), shapes(&homogeneous));
    let (mixed_record, homogeneous_record) = (record(&mixed, "pm"), record(&homogeneous, "pm"));
    for index in 0..3 {
        // A struct and a `serde_json::Value` can write object keys in
        // different orders, so each result is compared as parsed JSON.
        let parse = |value: &serde_json::Value| -> serde_json::Value {
            serde_json::from_str(&value.to_string()).expect("a result is JSON")
        };
        assert_eq!(
            parse(item_result(&mixed_record, index)),
            parse(item_result(&homogeneous_record, index)),
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Order {
    id: u64,
}

/// A serdes for `Order` only: JSON of the ID, prefixed with `order:`.
struct OrderSerdes;

impl Serdes<Order> for OrderSerdes {
    fn serialize(
        &self,
        value: Order,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(
            serde_json::to_string(&value.id)
                .map(|id| format!("order:{id}"))
                .map_err(BoxError::from),
        )
    }

    fn deserialize(
        &self,
        wire: String,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<Order, BoxError>> + Send {
        std::future::ready(
            wire.strip_prefix("order:")
                .ok_or_else(|| BoxError::from("missing order: prefix"))
                .and_then(|id| serde_json::from_str(id).map_err(BoxError::from))
                .map(|id| Order { id }),
        )
    }
}

/// One serdes call: the serdes label, the method and the operation ID.
type Call = (&'static str, &'static str, String);

/// Forwards to `inner` and logs each call.
struct Recording<S> {
    label: &'static str,
    inner: S,
    calls: Log<Call>,
}

impl<T, S> Serdes<T> for Recording<S>
where
    T: Send + 'static,
    S: Serdes<T>,
{
    async fn serialize(&self, value: T, ctx: SerdesContext) -> Result<String, BoxError> {
        push(
            &self.calls,
            (self.label, "serialize", ctx.operation_id().to_owned()),
        );
        self.inner.serialize(value, ctx).await
    }

    async fn deserialize(&self, wire: String, ctx: SerdesContext) -> Result<T, BoxError> {
        push(
            &self.calls,
            (self.label, "deserialize", ctx.operation_id().to_owned()),
        );
        self.inner.deserialize(wire, ctx).await
    }
}

/// The value of the `add` branch of `each_branch_uses_its_own_serdes`.
/// `JsonSerdes` serializes a value by calling its `Serialize` once, so
/// this `Serialize` logs the call of the branch's `JsonSerdes`. A value
/// that `Deserialize` produced logs nothing.
#[derive(Debug, Clone, Deserialize)]
#[serde(from = "u32")]
struct JsonLogged {
    value: u32,
    calls: Option<Log<Call>>,
}

impl From<u32> for JsonLogged {
    fn from(value: u32) -> Self {
        Self { value, calls: None }
    }
}

impl Serialize for JsonLogged {
    fn serialize<Ser: serde::Serializer>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error> {
        if let Some(calls) = &self.calls {
            push(calls, ("json", "serialize", String::new()));
        }
        self.value.serialize(serializer)
    }
}

#[tokio::test]
async fn each_branch_uses_its_own_serdes() {
    let dir = std::env::temp_dir().join(format!(
        "each_branch_uses_its_own_serdes-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create the serdes directory");
    let base = dir.to_str().expect("the directory is UTF-8").to_owned();
    let seen: Log<(u32, Order, String)> = log();
    let calls: Log<Call> = log();
    let (seen_h, calls_h) = (Arc::clone(&seen), Arc::clone(&calls));
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let calls = Arc::clone(&calls_h);
        let base = base.clone();
        async move {
            let mut set = BranchSet::new();
            let json_calls = Arc::clone(&calls);
            let json = set.add("json", move |_ctx| async move {
                Ok(JsonLogged {
                    value: 3,
                    calls: Some(json_calls),
                })
            });
            let order = set.add_with_serdes(
                "order",
                Recording {
                    label: "order",
                    inner: OrderSerdes,
                    calls: Arc::clone(&calls),
                },
                |_ctx| async { Ok(Order { id: 42 }) },
            );
            let file = set.add_with_serdes(
                "file",
                Recording {
                    label: "file",
                    inner: FileSystemSerdes::new(base),
                    calls,
                },
                |_ctx| async { Ok("stored in a file".to_owned()) },
            );
            ctx.parallel_mixed(set).name("pm").await?;
            push(
                &seen,
                (
                    json.into_result()?.value,
                    order.into_result()?,
                    file.into_result()?,
                ),
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let expected = (3_u32, Order { id: 42 }, "stored in a file".to_owned());
    assert_eq!(entries(&seen), vec![expected.clone(), expected]);

    let record = record(&result, "pm");
    // The `add` branch: its JSON value, as the child payload holds it.
    let json_payload = op(&result, "json").result().expect("json payload");
    assert_eq!(json_payload, "3");
    assert_eq!(item_result(&record, 0), &serde_json::json!(3));
    // Each `add_with_serdes` branch: its child payload, in a JSON string.
    let order_payload = op(&result, "order").result().expect("order payload");
    assert_eq!(order_payload, "order:42");
    assert_eq!(item_result(&record, 1), &serde_json::json!(order_payload));
    let file_payload = op(&result, "file").result().expect("file payload");
    assert_eq!(item_result(&record, 2), &serde_json::json!(file_payload));

    // Each custom serdes serialized its branch once, with the branch's ID.
    // The `JsonSerdes` of the `add` branch serialized its value once.
    let calls = entries(&calls);
    let json_serializes = calls
        .iter()
        .filter(|(label, method, _)| *label == "json" && *method == "serialize")
        .count();
    assert_eq!(json_serializes, 1, "json serializes once: {calls:?}");
    for (label, id) in [
        ("order", op(&result, "order").id()),
        ("file", op(&result, "file").id()),
    ] {
        let serializes: Vec<&Call> = calls
            .iter()
            .filter(|(l, method, _)| *l == label && *method == "serialize")
            .collect();
        assert_eq!(serializes.len(), 1, "{label} serializes once: {calls:?}");
        assert_eq!(serializes.first().map(|c| c.2.as_str()), Some(id));
    }
    std::fs::remove_dir_all(&dir).expect("remove the serdes directory");
}

#[tokio::test]
async fn serdes_calls_match_parallel() {
    let parallel_calls: Log<Call> = log();
    let calls_h = Arc::clone(&parallel_calls);
    let parallel = run(move |(), ctx| {
        let calls = Arc::clone(&calls_h);
        async move {
            push(&calls, ("invocation", "", String::new()));
            let serdes = Recording {
                label: "item",
                inner: durable::JsonSerdes,
                calls,
            };
            let branches = vec![
                Branch::new("a", |_ctx| async { Ok(1_u32) }),
                Branch::new("b", |_ctx| async { Ok(2_u32) }),
            ];
            let _values = ctx
                .parallel(branches)
                .name("pm")
                .max_concurrency(1)
                .serdes(serdes)
                .await?;
            wait_then_replay(&ctx).await
        }
    })
    .await;
    let mixed_calls: Log<Call> = log();
    let calls_h = Arc::clone(&mixed_calls);
    let mixed = run(move |(), ctx| {
        let calls = Arc::clone(&calls_h);
        async move {
            push(&calls, ("invocation", "", String::new()));
            let serdes = Arc::new(Recording {
                label: "item",
                inner: durable::JsonSerdes,
                calls,
            });
            let mut set = BranchSet::new();
            let _a = set.add_with_serdes("a", Arc::clone(&serdes), |_ctx| async { Ok(1_u32) });
            let _b = set.add_with_serdes("b", serdes, |_ctx| async { Ok(2_u32) });
            ctx.parallel_mixed(set)
                .name("pm")
                .max_concurrency(1)
                .await?;
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(parallel.is_success(), "parallel succeeds: {parallel}");
    assert!(mixed.is_success(), "parallel_mixed succeeds: {mixed}");
    let parallel_calls = entries(&parallel_calls);
    assert_eq!(
        parallel_calls
            .iter()
            .filter(|(label, _, _)| *label == "invocation")
            .count(),
        2,
        "two invocations ran: {parallel_calls:?}"
    );
    assert_eq!(entries(&mixed_calls), parallel_calls);
}

#[tokio::test]
async fn max_concurrency_runs_branches_in_order() {
    let events: Log<String> = log();
    let seen: Log<Vec<u32>> = log();
    let (events_h, seen_h) = (Arc::clone(&events), Arc::clone(&seen));
    let result = run(move |(), ctx| {
        let events = Arc::clone(&events_h);
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let handles: Vec<_> = (0..3_u32)
                .map(|i| {
                    let events = Arc::clone(&events);
                    set.add(
                        format!("branch-{i}"),
                        move |child: DurableContext| async move {
                            push(&events, format!("start-{i}"));
                            let value = child
                                .step(move |_| async move { Ok(i * 10) })
                                .name("work")
                                .await?;
                            push(&events, format!("end-{i}"));
                            Ok(value)
                        },
                    )
                })
                .collect();
            ctx.parallel_mixed(set)
                .name("pm")
                .max_concurrency(1)
                .await?;
            let values = handles
                .into_iter()
                .map(durable::BranchHandle::into_result)
                .collect::<Result<Vec<u32>, _>>()?;
            push(&seen, values);
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(
        entries(&events),
        ["start-0", "end-0", "start-1", "end-1", "start-2", "end-2"]
    );
    assert_eq!(entries(&seen), vec![vec![0, 10, 20], vec![0, 10, 20]]);
}

#[tokio::test]
async fn max_concurrency_zero_fails_the_execution() {
    let after = counter();
    let after_h = Arc::clone(&after);
    let result = run(move |(), ctx| {
        let after = Arc::clone(&after_h);
        async move {
            let mut set = BranchSet::new();
            let _a = set.add("a", |_ctx| async { Ok(1_u32) });
            let _b = set.add("b", |_ctx| async { Ok("b".to_owned()) });
            ctx.parallel_mixed(set)
                .name("pm")
                .max_concurrency(0)
                .await?;
            after.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    })
    .await;
    assert_eq!(result.status(), ExecutionStatus::Failed, "{result}");
    let error = result.error().expect("the execution records an error");
    assert_eq!(error.error_type(), Some("Error"));
    assert_eq!(
        error.error_message(),
        Some(
            "Invalid max_concurrency: 0. Must be a positive number or unset for unlimited concurrency."
        )
    );
    assert!(
        result.operation("pm").is_none(),
        "the parallel records nothing"
    );
    assert_eq!(count(&after), 0, "the handler never resumes");
}

/// What a handler observed in one invocation of a failing batch: the
/// `.await` error's record, and what each handle returned.
#[derive(Debug, Clone, PartialEq)]
struct FailedRun {
    await_wire: WireError,
    await_child_failed: bool,
    handles: Vec<Result<String, (bool, Option<WireError>)>>,
}

/// Summarizes a handle's result for comparison across invocations: a
/// value, or whether the error is `NotCompleted` and its record.
fn summarize<T: ToString>(
    result: Result<T, OperationError>,
) -> Result<String, (bool, Option<WireError>)> {
    result
        .map(|value| value.to_string())
        .map_err(|err| (is_not_completed(&err), err.wire().cloned()))
}

#[tokio::test]
async fn fail_fast_returns_the_branch_error() {
    let seen: Log<FailedRun> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let failing = set.add("failing", |_ctx| async {
                Err::<u32, BoxError>("boom".into())
            });
            let waiting = set.add("waiting", |child: DurableContext| async move {
                child
                    .step(|_| async {
                        std::future::pending::<()>().await;
                        Ok("never".to_owned())
                    })
                    .name("forever")
                    .await
                    .map_err(BoxError::from)
            });
            let outcome = ctx.parallel_mixed(set).name("pm").await;
            let err = outcome.expect_err("the failed branch fails the batch");
            push(
                &seen,
                FailedRun {
                    await_wire: wire_of(&err),
                    await_child_failed: is_child_failed(&err),
                    handles: vec![
                        summarize(failing.into_result()),
                        summarize(waiting.into_result()),
                    ],
                },
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations ran: {seen:?}");
    for run in &seen {
        assert!(run.await_child_failed, "the error is ChildFailed: {run:?}");
        assert_eq!(run.await_wire.error_message(), Some("boom"));
        let Some(Err((false, Some(failing)))) = run.handles.first() else {
            panic!("handle 0 holds the branch error: {run:?}");
        };
        assert_eq!(failing.error_type(), run.await_wire.error_type());
        assert_eq!(failing.error_message(), run.await_wire.error_message());
        assert!(
            matches!(run.handles.get(1), Some(Err((true, _)))),
            "handle 1 has no result: {run:?}"
        );
    }
    assert_eq!(seen.first(), seen.get(1), "replay returns the same");
}

#[tokio::test]
async fn step_failure_in_a_branch_records_the_step_type() {
    let seen: Log<WireError> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let pay = set.add("pay", |child: DurableContext| async move {
                child
                    .step(|_| async {
                        Err::<u32, BoxError>(
                            TypedError::with_type("PaymentError", "declined").into(),
                        )
                    })
                    .name("charge")
                    .retry_strategy(|_, _| RetryDecision::Stop)
                    .await?;
                Ok(1_u32)
            });
            let _outcome = ctx.parallel_mixed(set).name("pm").await;
            let err = pay.into_result().expect_err("the branch failed");
            push(&seen, wire_of(&err));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let fail = op(&result, "pay").error().expect("the branch records FAIL");
    assert_eq!(fail.error_type(), Some("PaymentError"));
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations ran");
    for wire in &seen {
        assert_eq!(wire.error_type(), Some("PaymentError"));
        assert_eq!(wire.error_message(), fail.error_message());
    }
}

/// A serdes for `()` that writes the empty string.
struct EmptyUnitSerdes;

impl Serdes<()> for EmptyUnitSerdes {
    fn serialize(
        &self,
        (): (),
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(Ok(String::new()))
    }

    fn deserialize(
        &self,
        wire: String,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<(), BoxError>> + Send {
        std::future::ready(if wire.is_empty() {
            Ok(())
        } else {
            Err(format!("expected an empty string, got {wire:?}").into())
        })
    }
}

#[tokio::test]
async fn empty_encoded_branch_handle_replays_its_value() {
    let seen: Log<((), u32)> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let unit = set.add_with_serdes("unit", EmptyUnitSerdes, |_ctx| async { Ok(()) });
            let number = set.add("number", |_ctx| async { Ok(1_u32) });
            ctx.parallel_mixed(set).name("pm").await?;
            push(&seen, (unit.into_result()?, number.into_result()?));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(entries(&seen), vec![((), 1), ((), 1)]);
}

#[tokio::test]
async fn failed_handle_carries_recorded_failure() {
    let seen: Log<WireError> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let pay = set.add("pay", |_ctx| async {
                Err::<u32, BoxError>(
                    TypedError::with_type("PaymentError", "declined")
                        .with_error_data(r#"{"code":42}"#)
                        .into(),
                )
            });
            let _outcome = ctx.parallel_mixed(set).name("pm").await;
            let err = pay.into_result().expect_err("the branch failed");
            push(&seen, wire_of(&err));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let fail = op(&result, "pay").error().expect("the branch records FAIL");
    assert_eq!(fail.error_type(), Some("PaymentError"));
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations ran");
    for wire in &seen {
        assert_eq!(wire.error_type(), Some("PaymentError"));
        assert_eq!(wire.error_message(), fail.error_message());
        assert_eq!(wire.error_data(), Some(r#"{"code":42}"#));
        assert_eq!(wire.error_data(), fail.error_data());
        assert_eq!(wire.stack_trace(), fail.stack_trace());
        assert!(
            wire.stack_trace().is_empty(),
            "the SDK records no stack trace"
        );
    }
}

/// A type with neither `Serialize` nor `DeserializeOwned`.
#[derive(Debug, Clone, PartialEq)]
struct Opaque(u32);

/// A serdes for `Opaque`: its number as decimal text.
struct OpaqueSerdes;

impl Serdes<Opaque> for OpaqueSerdes {
    fn serialize(
        &self,
        value: Opaque,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(Ok(value.0.to_string()))
    }

    fn deserialize(
        &self,
        wire: String,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<Opaque, BoxError>> + Send {
        std::future::ready(wire.parse().map(Opaque).map_err(BoxError::from))
    }
}

#[tokio::test]
async fn non_serde_type_through_custom_serdes() {
    let seen: Log<(Opaque, Vec<Opaque>)> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let mixed = set.add_with_serdes("mixed", OpaqueSerdes, |_ctx| async { Ok(Opaque(5)) });
            ctx.parallel_mixed(set).name("pm").await?;
            let values = ctx
                .parallel(vec![Branch::new("plain", |_ctx| async { Ok(Opaque(6)) })])
                .name("p")
                .serdes(OpaqueSerdes)
                .await?;
            push(&seen, (mixed.into_result()?, values));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let expected = (Opaque(5), vec![Opaque(6)]);
    assert_eq!(entries(&seen), vec![expected.clone(), expected]);
}

#[tokio::test]
async fn parallel_mixed_on_the_wrong_context_names_the_method() {
    let result = run(|(), ctx| async move {
        let parent = ctx.clone();
        ctx.run_in_child_context(move |_child| async move {
            parent.parallel_mixed(BranchSet::new()).name("pm").await?;
            Ok(())
        })
        .name("child")
        .await?;
        Ok(())
    })
    .await;
    assert!(result.is_failure(), "the execution fails: {result}");
    let message = result.error_message().unwrap_or_default();
    assert!(
        message.starts_with("context usage error in \"parallel_mixed\""),
        "the error names the method: {message}"
    );
}

#[tokio::test]
async fn renamed_parallel_mixed_fails_replay() {
    let invocation = counter();
    let invocation_h = Arc::clone(&invocation);
    let result = run(move |(), ctx| {
        let name = if invocation_h.fetch_add(1, Ordering::SeqCst) == 0 {
            "alpha"
        } else {
            "beta"
        };
        async move {
            let mut set = BranchSet::new();
            let _value = set.add("only", |_ctx| async { Ok(1_u32) });
            ctx.parallel_mixed(set).name(name).await?;
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_failure(), "the replay fails: {result}");
    let message = result.error_message().unwrap_or_default();
    assert!(
        message.contains("alpha") && message.contains("beta"),
        "the error names both names: {message}"
    );
}

/// A `parallel` with `max_concurrency(1)` runs beside `parallel_mixed` on
/// the same context, so the two batches take child IDs from the context in
/// turn. Each branch of `parallel_mixed` decodes with its own serdes, on
/// the first run and on replay.
#[tokio::test]
async fn parallel_mixed_beside_another_parallel_decodes_each_branch() {
    let seen: Log<(u32, Order, Vec<u32>)> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let other = ctx
                .parallel(vec![
                    Branch::new("other-0", |_ctx| async { Ok(10_u32) }),
                    Branch::new("other-1", |_ctx| async { Ok(11_u32) }),
                ])
                .name("other");
            let mut set = BranchSet::new();
            let number = set.add("number", |_ctx| async { Ok(1_u32) });
            let order =
                set.add_with_serdes("order", OrderSerdes, |_ctx| async { Ok(Order { id: 42 }) });
            let mixed = ctx.parallel_mixed(set).name("pm").max_concurrency(1);
            let (other, mixed) = tokio::join!(other.into_future(), mixed.into_future());
            let other = other?;
            mixed?;
            push(&seen, (number.into_result()?, order.into_result()?, other));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(
        result.is_success(),
        "the execution succeeds: {:?} {result}",
        result.error_message()
    );
    let expected = (1_u32, Order { id: 42 }, vec![10_u32, 11]);
    assert_eq!(entries(&seen), vec![expected.clone(), expected]);
}

/// Branches of 100 KB and 200 KB: each child record holds its value, but
/// the batch record, which holds both, is too large to store. So replay
/// decodes each branch from its child record, with its own branch's serdes.
#[tokio::test]
async fn oversized_batch_record_replays_each_branch_from_its_child() {
    let seen: Log<(String, Order, String)> = log();
    let counters = [counter(), counter(), counter()];
    let (seen_h, counters_h) = (Arc::clone(&seen), counters.clone());
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let [c0, c1, c2] = counters_h.clone();
        async move {
            let mut set = BranchSet::new();
            let first = set.add("first", move |_ctx| async move {
                c0.fetch_add(1, Ordering::SeqCst);
                Ok("a".repeat(100_000))
            });
            let order = set.add_with_serdes("order", OrderSerdes, move |_ctx| async move {
                c1.fetch_add(1, Ordering::SeqCst);
                Ok(Order { id: 42 })
            });
            let third = set.add("third", move |_ctx| async move {
                c2.fetch_add(1, Ordering::SeqCst);
                Ok("c".repeat(200_000))
            });
            ctx.parallel_mixed(set).name("pm").await?;
            push(
                &seen,
                (
                    first.into_result()?,
                    order.into_result()?,
                    third.into_result()?,
                ),
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(
        result.is_success(),
        "the execution succeeds: {:?}",
        result.error_message()
    );
    let expected = ("a".repeat(100_000), Order { id: 42 }, "c".repeat(200_000));
    assert_eq!(entries(&seen), vec![expected.clone(), expected]);
    for c in &counters {
        assert_eq!(count(c), 1, "replay does not run a completed branch");
    }
}

/// A serdes that logs `(method, operation ID, value)` for each call. Before
/// it decodes, it yields once, so a batch that is replaying lets another
/// batch run before it finishes.
struct IdLog {
    calls: Log<(&'static str, String, String)>,
}

impl Serdes<String> for IdLog {
    fn serialize(
        &self,
        value: String,
        ctx: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        push(
            &self.calls,
            ("serialize", ctx.operation_id().to_owned(), value.clone()),
        );
        std::future::ready(Ok(value))
    }

    async fn deserialize(&self, wire: String, ctx: SerdesContext) -> Result<String, BoxError> {
        tokio::task::yield_now().await;
        push(
            &self.calls,
            ("deserialize", ctx.operation_id().to_owned(), wire.clone()),
        );
        Ok(wire)
    }
}

/// Runs two batches side by side on one context, `a0, a1` and `b0, b1`,
/// each branch through `serdes`. With `mixed`, both are `parallel_mixed`
/// batches; otherwise both are `parallel` batches with `serdes` as the
/// item serdes. Returns the result, the branch values each invocation
/// read, and the serdes calls sorted.
async fn overlapping_batches(
    mixed: bool,
) -> (
    TestResult<()>,
    Vec<Vec<String>>,
    Vec<(&'static str, String, String)>,
) {
    let calls: Log<(&'static str, String, String)> = log();
    let seen: Log<Vec<String>> = log();
    let serdes = Arc::new(IdLog {
        calls: Arc::clone(&calls),
    });
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let serdes = Arc::clone(&serdes);
        async move {
            let values = if mixed {
                let mut first = BranchSet::new();
                let mut second = BranchSet::new();
                let mut handles = Vec::new();
                for (set, names) in [(&mut first, ["a0", "a1"]), (&mut second, ["b0", "b1"])] {
                    for name in names {
                        handles.push(set.add_with_serdes(
                            name,
                            Arc::clone(&serdes),
                            move |_ctx| async move { Ok(name.to_owned()) },
                        ));
                    }
                }
                let first = ctx.parallel_mixed(first).name("first");
                let second = ctx.parallel_mixed(second).name("second");
                let (first, second) = tokio::join!(first.into_future(), second.into_future());
                first?;
                second?;
                let mut values = Vec::new();
                for handle in handles {
                    values.push(handle.into_result()?);
                }
                values
            } else {
                let branches = |names: [&'static str; 2]| -> Vec<Branch<String>> {
                    names
                        .into_iter()
                        .map(|name| {
                            Branch::new(name, move |_ctx| async move { Ok(name.to_owned()) })
                        })
                        .collect()
                };
                let first = ctx
                    .parallel(branches(["a0", "a1"]))
                    .name("first")
                    .serdes(Arc::clone(&serdes));
                let second = ctx
                    .parallel(branches(["b0", "b1"]))
                    .name("second")
                    .serdes(Arc::clone(&serdes));
                let (first, second) = tokio::join!(first.into_future(), second.into_future());
                let mut values = first?;
                values.extend(second?);
                values
            };
            push(&seen, values);
            wait_then_replay(&ctx).await
        }
    })
    .await;
    let mut calls = entries(&calls);
    calls.sort();
    (result, entries(&seen), calls)
}

/// Two batches run side by side on one context. `parallel_mixed` calls
/// each branch's serdes with the same operation IDs as `parallel` calls
/// its item serdes, on the first run and on replay. Each handle returns
/// its own branch's value in both invocations, so the item serdes found
/// the right branch for each call.
#[tokio::test]
async fn overlapping_batches_call_serdes_as_parallel_does() {
    let (mixed, mixed_seen, mixed_calls) = overlapping_batches(true).await;
    let (plain, plain_seen, plain_calls) = overlapping_batches(false).await;
    for result in [&mixed, &plain] {
        assert!(
            result.is_success(),
            "the execution succeeds: {:?} {result}",
            result.error_message()
        );
    }
    let values: Vec<String> = ["a0", "a1", "b0", "b1"].map(str::to_owned).to_vec();
    assert_eq!(mixed_seen, vec![values.clone(), values.clone()]);
    assert_eq!(plain_seen, vec![values.clone(), values]);
    assert!(
        mixed_calls
            .iter()
            .any(|(method, _, _)| *method == "deserialize"),
        "replay decodes the branches: {mixed_calls:?}"
    );
    assert_eq!(mixed_calls, plain_calls);
}

/// With `max_concurrency(1)`, a failed first branch ends the batch before
/// the second branch starts. The operation after the batch gets the same
/// ID on the first run and on replay, so replay does not run it again.
#[tokio::test]
async fn operation_after_a_sequential_early_stop_replays() {
    let next_runs = counter();
    let unstarted: Log<bool> = log();
    let (next_h, unstarted_h) = (Arc::clone(&next_runs), Arc::clone(&unstarted));
    let result = run(move |(), ctx| {
        let next = Arc::clone(&next_h);
        let unstarted = Arc::clone(&unstarted_h);
        async move {
            let mut set = BranchSet::new();
            let _failing = set.add("failing", |_ctx| async {
                Err::<u32, BoxError>("boom".into())
            });
            let later = set.add("later", |_ctx| async { Ok(2_u32) });
            let outcome = ctx.parallel_mixed(set).name("pm").max_concurrency(1).await;
            assert!(outcome.is_err(), "the failed branch fails the batch");
            push(
                &unstarted,
                later.into_result().is_err_and(|err| is_not_completed(&err)),
            );
            ctx.step(move |_| async move {
                next.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .name("next")
            .await?;
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(
        result.is_success(),
        "the execution succeeds: {:?} {result}",
        result.error_message()
    );
    assert_eq!(entries(&unstarted), vec![true, true]);
    assert_eq!(count(&next_runs), 1, "replay does not run the step again");
}

/// The history shapes of `result`, ordered by operation ID, so operations
/// that run side by side compare in a fixed order.
fn sorted_shapes(result: &TestResult<()>) -> Vec<Shape> {
    let mut shapes = shapes(result);
    shapes.sort_by(|a, b| a.id.cmp(&b.id));
    shapes
}

/// A step that runs beside the batch is created after the batch is
/// called and before it starts. `parallel_mixed` and `parallel` record the
/// same history, including every operation ID.
#[tokio::test]
async fn history_equals_parallel_with_an_interleaved_operation() {
    let mixed = run(|(), ctx| async move {
        let mut set = BranchSet::new();
        let _number = set.add("number", |_ctx| async { Ok(5_u32) });
        let _text = set.add("text", |_ctx| async { Ok("hi".to_owned()) });
        let batch = ctx.parallel_mixed(set).name("pm");
        let side = ctx.step(|_| async { Ok(1_u32) }).name("side");
        let (batch, side) = tokio::join!(batch.into_future(), side.into_future());
        batch?;
        side?;
        ctx.step(|_| async { Ok(2_u32) }).name("next").await?;
        wait_then_replay(&ctx).await
    })
    .await;
    let homogeneous = run(|(), ctx| async move {
        let branches: Vec<Branch<serde_json::Value>> = vec![
            Branch::new("number", |_ctx| async { Ok(serde_json::json!(5)) }),
            Branch::new("text", |_ctx| async { Ok(serde_json::json!("hi")) }),
        ];
        let batch = ctx.parallel(branches).name("pm");
        let side = ctx.step(|_| async { Ok(1_u32) }).name("side");
        let (batch, side) = tokio::join!(batch.into_future(), side.into_future());
        batch?;
        side?;
        ctx.step(|_| async { Ok(2_u32) }).name("next").await?;
        wait_then_replay(&ctx).await
    })
    .await;
    assert!(mixed.is_success(), "parallel_mixed succeeds: {mixed}");
    assert!(homogeneous.is_success(), "parallel succeeds: {homogeneous}");
    assert_eq!(sorted_shapes(&mixed), sorted_shapes(&homogeneous));
}

/// Two batches run side by side, each awaited under `tokio::join!`.
/// `parallel_mixed` and `parallel` record the same history, including the
/// child IDs of both batches.
#[tokio::test]
async fn history_equals_parallel_beside_another_batch() {
    let mixed = run(|(), ctx| async move {
        let mut first = BranchSet::new();
        let _a0 = first.add("a0", |_ctx| async { Ok(0_u32) });
        let _a1 = first.add("a1", |_ctx| async { Ok("a1".to_owned()) });
        let mut second = BranchSet::new();
        let _b0 = second.add("b0", |_ctx| async { Ok(true) });
        let _b1 = second.add("b1", |_ctx| async { Ok(1_u32) });
        let first = ctx.parallel_mixed(first).name("first");
        let second = ctx.parallel_mixed(second).name("second");
        let (first, second) = tokio::join!(first.into_future(), second.into_future());
        first?;
        second?;
        wait_then_replay(&ctx).await
    })
    .await;
    let homogeneous = run(|(), ctx| async move {
        let first: Vec<Branch<serde_json::Value>> = vec![
            Branch::new("a0", |_ctx| async { Ok(serde_json::json!(0)) }),
            Branch::new("a1", |_ctx| async { Ok(serde_json::json!("a1")) }),
        ];
        let second: Vec<Branch<serde_json::Value>> = vec![
            Branch::new("b0", |_ctx| async { Ok(serde_json::json!(true)) }),
            Branch::new("b1", |_ctx| async { Ok(serde_json::json!(1)) }),
        ];
        let first = ctx.parallel(first).name("first");
        let second = ctx.parallel(second).name("second");
        let (first, second) = tokio::join!(first.into_future(), second.into_future());
        first?;
        second?;
        wait_then_replay(&ctx).await
    })
    .await;
    assert!(mixed.is_success(), "parallel_mixed succeeds: {mixed}");
    assert!(homogeneous.is_success(), "parallel succeeds: {homogeneous}");
    assert_eq!(sorted_shapes(&mixed), sorted_shapes(&homogeneous));
}

/// With `max_concurrency(1)`, a failed first branch ends the batch before
/// the second branch starts. `parallel_mixed` and `parallel` record the
/// same history, including the ID of the step after the batch.
#[tokio::test]
async fn history_equals_parallel_after_a_sequential_early_stop() {
    let mixed = run(|(), ctx| async move {
        let mut set = BranchSet::new();
        let _failing = set.add("failing", |_ctx| async {
            Err::<u32, BoxError>("boom".into())
        });
        let _later = set.add("later", |_ctx| async { Ok("later".to_owned()) });
        let outcome = ctx.parallel_mixed(set).name("pm").max_concurrency(1).await;
        assert!(outcome.is_err(), "the failed branch fails the batch");
        ctx.step(|_| async { Ok(2_u32) }).name("next").await?;
        wait_then_replay(&ctx).await
    })
    .await;
    let homogeneous = run(|(), ctx| async move {
        let branches: Vec<Branch<serde_json::Value>> = vec![
            Branch::new("failing", |_ctx| async {
                Err::<serde_json::Value, BoxError>("boom".into())
            }),
            Branch::new("later", |_ctx| async { Ok(serde_json::json!("later")) }),
        ];
        let outcome = ctx.parallel(branches).name("pm").max_concurrency(1).await;
        assert!(outcome.is_err(), "the failed branch fails the batch");
        ctx.step(|_| async { Ok(2_u32) }).name("next").await?;
        wait_then_replay(&ctx).await
    })
    .await;
    assert!(mixed.is_success(), "parallel_mixed succeeds: {mixed}");
    assert!(homogeneous.is_success(), "parallel succeeds: {homogeneous}");
    assert_eq!(sorted_shapes(&mixed), sorted_shapes(&homogeneous));
}

/// What one invocation saw: the `.await` outcome of the parallel and a
/// summary of each handle.
#[derive(Debug, Clone, PartialEq)]
struct CompletionRun {
    reason: Option<CompletionReason>,
    await_error: Option<String>,
    handles: Vec<Result<String, (bool, Option<WireError>)>>,
}

/// The kind name of a `ChildContext` error, or `None` for any other kind.
fn child_kind(err: &OperationError) -> Option<String> {
    match err.kind() {
        OperationErrorKind::ChildContext(child) => Some(match child.kind() {
            ChildContextErrorKind::ChildFailed { .. } => "ChildFailed".to_owned(),
            ChildContextErrorKind::BatchCompletionFailed { .. } => {
                "BatchCompletionFailed".to_owned()
            }
            other => format!("{other:?}"),
        }),
        _ => None,
    }
}

/// The message of the record a handle's error carries.
fn handle_message(handle: Option<&Result<String, (bool, Option<WireError>)>>) -> Option<String> {
    match handle {
        Some(Err((false, Some(wire)))) => wire.error_message().map(str::to_owned),
        _ => None,
    }
}

fn handle_not_completed(handle: Option<&Result<String, (bool, Option<WireError>)>>) -> bool {
    matches!(handle, Some(Err((true, _))))
}

#[tokio::test]
async fn tolerated_failure_fills_every_handle() {
    let seen: Log<CompletionRun> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let first = set.add("first", |_ctx| async { Ok(10_u32) });
            let failing = set.add("failing", |_ctx| async {
                Err::<u32, BoxError>("boom".into())
            });
            let last = set.add("last", |_ctx| async { Ok("two".to_owned()) });
            let outcome = ctx
                .parallel_mixed(set)
                .name("pm")
                .completion(CompletionConfig::with_tolerated_failure_count(1))
                .await;
            let failed = failing.into_result();
            assert!(
                failed.as_ref().is_err_and(is_child_failed),
                "handle 1 holds a ChildFailed error: {failed:?}"
            );
            push(
                &seen,
                CompletionRun {
                    reason: outcome.as_ref().ok().copied(),
                    await_error: outcome.err().map(|err| err.to_string()),
                    handles: vec![
                        summarize(first.into_result()),
                        summarize(failed),
                        summarize(last.into_result()),
                    ],
                },
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations ran: {seen:?}");
    for run in &seen {
        assert_eq!(run.reason, Some(CompletionReason::AllCompleted), "{run:?}");
        assert_eq!(run.handles.first(), Some(&Ok("10".to_owned())), "{run:?}");
        assert_eq!(handle_message(run.handles.get(1)).as_deref(), Some("boom"));
        assert_eq!(run.handles.get(2), Some(&Ok("two".to_owned())), "{run:?}");
    }
    assert_eq!(seen.first(), seen.get(1), "replay returns the same");
}

/// Runs a `min_successful(1)` batch whose branch 0 returns `value`,
/// branch 1 runs a step that never ends, and branch 2 waits 5 s. Returns
/// what each invocation saw and the run's result.
async fn min_successful_run<T>(value: T) -> (Vec<CompletionRun>, TestResult<()>)
where
    T: Clone + ToString + Serialize + serde::de::DeserializeOwned + Send + Sync + 'static,
{
    let seen: Log<CompletionRun> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let value = value.clone();
        async move {
            let mut set = BranchSet::new();
            let first = set.add("first", move |_ctx| async move { Ok(value) });
            let running = set.add("running", |child: DurableContext| async move {
                child
                    .step(|_| async {
                        std::future::pending::<()>().await;
                        Ok(1_u32)
                    })
                    .name("forever")
                    .await
                    .map_err(BoxError::from)
            });
            let waiting = set.add("waiting", |child: DurableContext| async move {
                child.wait(Duration::from_secs(5)).name("long").await?;
                Ok(2_u32)
            });
            let outcome = ctx
                .parallel_mixed(set)
                .name("pm")
                .completion(CompletionConfig::with_min_successful(1))
                .await;
            push(
                &seen,
                CompletionRun {
                    reason: outcome.as_ref().ok().copied(),
                    await_error: outcome.err().map(|err| err.to_string()),
                    handles: vec![
                        summarize(first.into_result()),
                        summarize(running.into_result()),
                        summarize(waiting.into_result()),
                    ],
                },
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    (entries(&seen), result)
}

/// Whether the context record of `op` carries `ReplayChildren`.
fn replays_children(op: &TestOperation) -> bool {
    matches!(op.details(), TestOperationDetails::Context(c) if c.replay_children())
}

fn assert_min_successful_runs(seen: &[CompletionRun], expected: &str) {
    assert_eq!(seen.len(), 2, "two invocations ran: {seen:?}");
    for run in seen {
        assert_eq!(
            run.reason,
            Some(CompletionReason::MinSuccessfulReached),
            "{run:?}"
        );
        assert_eq!(run.handles.first(), Some(&Ok(expected.to_owned())));
        assert!(handle_not_completed(run.handles.get(1)), "{run:?}");
        assert!(handle_not_completed(run.handles.get(2)), "{run:?}");
    }
}

#[tokio::test]
async fn min_successful_leaves_running_and_waiting_branches_not_completed() {
    let (seen, result) = min_successful_run(1_u32).await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert!(!replays_children(op(&result, "pm")), "the record is inline");
    assert_min_successful_runs(&seen, "1");

    let big = "x".repeat(300_000);
    let (seen, result) = min_successful_run(big.clone()).await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert!(
        replays_children(op(&result, "pm")),
        "the oversized record replays its children"
    );
    assert_min_successful_runs(&seen, &big);
}

#[tokio::test]
async fn exceeded_tolerance_returns_the_first_branch_error() {
    let seen: Log<(Option<String>, Option<String>, CompletionRun)> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let a = set.add("a", |_ctx| async { Err::<u32, BoxError>("a".into()) });
            let b = set.add("b", |_ctx| async { Err::<String, BoxError>("b".into()) });
            let c = set.add("c", |_ctx| async { Ok(3_u32) });
            let outcome = ctx
                .parallel_mixed(set)
                .name("pm")
                .max_concurrency(1)
                .completion(CompletionConfig::with_tolerated_failure_count(1))
                .await;
            let err = outcome.as_ref().err();
            push(
                &seen,
                (
                    err.and_then(child_kind),
                    err.and_then(|err| err.wire()?.error_message().map(str::to_owned)),
                    CompletionRun {
                        reason: outcome.as_ref().ok().copied(),
                        await_error: None,
                        handles: vec![
                            summarize(a.into_result()),
                            summarize(b.into_result()),
                            summarize(c.into_result()),
                        ],
                    },
                ),
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations ran: {seen:?}");
    for (kind, message, run) in &seen {
        assert_eq!(kind.as_deref(), Some("ChildFailed"), "{run:?}");
        assert_eq!(message.as_deref(), Some("a"), "{run:?}");
        assert_eq!(handle_message(run.handles.first()).as_deref(), Some("a"));
        assert_eq!(handle_message(run.handles.get(1)).as_deref(), Some("b"));
        assert!(handle_not_completed(run.handles.get(2)), "{run:?}");
    }
    assert_eq!(seen.first(), seen.get(1), "replay returns the same");
}

#[tokio::test]
async fn custom_completion_failure_publishes_settled_handles() {
    let seen: Log<(Option<String>, CompletionRun)> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let h0 = set.add("zero", |_ctx| async { Ok(0_u32) });
            let h1 = set.add("one", |_ctx| async { Ok(1_u32) });
            let h2 = set.add("two", |_ctx| async { Ok(2_u32) });
            let outcome = ctx
                .parallel_mixed(set)
                .name("pm")
                .max_concurrency(1)
                .completion(CompletionConfig::with_completion_predicate(|stats| {
                    if stats.settled() >= 2 {
                        CompletionDecision::Complete(CompletionOutcome::Failed)
                    } else {
                        CompletionDecision::Continue
                    }
                }))
                .await;
            push(
                &seen,
                (
                    outcome.as_ref().err().and_then(child_kind),
                    CompletionRun {
                        reason: outcome.as_ref().ok().copied(),
                        await_error: None,
                        handles: vec![
                            summarize(h0.into_result()),
                            summarize(h1.into_result()),
                            summarize(h2.into_result()),
                        ],
                    },
                ),
            );
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let seen = entries(&seen);
    assert_eq!(seen.len(), 2, "two invocations ran: {seen:?}");
    for (kind, run) in &seen {
        assert_eq!(kind.as_deref(), Some("BatchCompletionFailed"), "{run:?}");
        assert_eq!(run.handles.first(), Some(&Ok("0".to_owned())), "{run:?}");
        assert_eq!(run.handles.get(1), Some(&Ok("1".to_owned())), "{run:?}");
        assert!(handle_not_completed(run.handles.get(2)), "{run:?}");
    }
    assert_eq!(seen.first(), seen.get(1), "replay returns the same");
}

#[tokio::test]
async fn spawn_outside_runtime_returns_error() {
    let seen: Log<Option<String>> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let mut set = BranchSet::new();
            let _handle = set.add("only", |_ctx| async { Ok(1_u32) });
            let builder = ctx.parallel_mixed(set).name("pm");
            let polled = std::thread::spawn(move || {
                assert!(
                    tokio::runtime::Handle::try_current().is_err(),
                    "the thread has no Tokio runtime"
                );
                let future = std::pin::pin!(builder.spawn());
                match future.poll(&mut std::task::Context::from_waker(std::task::Waker::noop())) {
                    std::task::Poll::Ready(Err(err)) => Some(err),
                    std::task::Poll::Ready(Ok(_)) | std::task::Poll::Pending => None,
                }
            })
            .join()
            .expect("spawn does not panic");
            let err = polled.expect("the first poll returns the error");
            assert!(
                matches!(
                    err.kind(),
                    OperationErrorKind::Step(step)
                        if matches!(step.kind(), StepErrorKind::ExecutionFailed { .. })
                ),
                "the error is the ownership error: {err:?}"
            );
            let mut chain = vec![err.to_string()];
            let mut source = std::error::Error::source(&err);
            while let Some(cause) = source {
                chain.push(cause.to_string());
                source = cause.source();
            }
            let chain = chain.join(": ");
            push(&seen, Some(chain));
            Ok(())
        }
    })
    .await;
    // The thread does not own the context, so the call fails the execution
    // before the runtime check.
    assert!(result.is_failure(), "the execution fails: {result}");
    assert!(
        result
            .error_message()
            .is_some_and(|m| m.starts_with("context usage error in \"spawn\"")),
        "the execution fails with a context usage error: {:?}",
        result.error_message()
    );
    let seen = entries(&seen);
    assert_eq!(seen.len(), 1, "one invocation ran: {seen:?}");
    let message = seen.first().cloned().flatten().expect("an error message");
    assert!(
        message.contains("does not own the context"),
        "the message names the cause: {message}"
    );
    assert!(
        result.operation("pm").is_none(),
        "the parallel records nothing"
    );
}

#[tokio::test]
async fn spawned_handles_fill_on_await() {
    let seen: Log<(bool, Option<CompletionReason>, Option<String>)> = log();
    let seen_h = Arc::clone(&seen);
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        async move {
            let started = Arc::new(AtomicUsize::new(0));
            let notify = Arc::new(tokio::sync::Notify::new());
            let mut set = BranchSet::new();
            let (s0, n0) = (Arc::clone(&started), Arc::clone(&notify));
            let h0 = set.add("zero", move |_ctx| async move {
                if s0.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                    n0.notify_one();
                }
                Ok(0_u32)
            });
            let (s1, n1) = (Arc::clone(&started), Arc::clone(&notify));
            let h1 = set.add("one", move |_ctx| async move {
                if s1.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                    n1.notify_one();
                }
                Ok("one".to_owned())
            });
            let batch = ctx.parallel_mixed(set).name("pm").spawn();
            notify.notified().await;
            let before = h0.into_result().is_err_and(|err| is_not_completed(&err));
            let reason = batch.await.ok();
            push(&seen, (before, reason, h1.into_result().ok()));
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(
        entries(&seen),
        vec![(
            true,
            Some(CompletionReason::AllCompleted),
            Some("one".to_owned())
        )]
    );
}

/// A `Serdes<String>` for the batch record: adds `summary:` before the
/// record's JSON text, and removes it on replay.
struct PrefixSerdes;

impl Serdes<String> for PrefixSerdes {
    fn serialize(
        &self,
        value: String,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(Ok(format!("summary:{value}")))
    }

    fn deserialize(
        &self,
        wire: String,
        _ctx: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(
            wire.strip_prefix("summary:")
                .map(str::to_owned)
                .ok_or_else(|| BoxError::from("missing summary: prefix")),
        )
    }
}

#[tokio::test]
async fn result_serdes_encodes_the_record() {
    let seen: Log<(u32, String)> = log();
    let counters = [counter(), counter()];
    let (seen_h, counters_h) = (Arc::clone(&seen), counters.clone());
    let result = run(move |(), ctx| {
        let seen = Arc::clone(&seen_h);
        let [c0, c1] = counters_h.clone();
        async move {
            let mut set = BranchSet::new();
            let number = set.add("number", move |_ctx| async move {
                c0.fetch_add(1, Ordering::SeqCst);
                Ok(5_u32)
            });
            let text = set.add("text", move |_ctx| async move {
                c1.fetch_add(1, Ordering::SeqCst);
                Ok("hi".to_owned())
            });
            ctx.parallel_mixed(set)
                .name("pm")
                .result_serdes(PrefixSerdes)
                .await?;
            push(&seen, (number.into_result()?, text.into_result()?));
            wait_then_replay(&ctx).await
        }
    })
    .await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    let payload = op(&result, "pm")
        .result()
        .expect("the parallel has a payload");
    assert!(payload.starts_with("summary:"), "the payload: {payload}");
    assert_eq!(
        entries(&seen),
        vec![(5, "hi".to_owned()), (5, "hi".to_owned())],
        "every handle returns its value in both invocations"
    );
    for c in &counters {
        assert_eq!(count(c), 1, "replay reads the record");
    }
}

#[tokio::test]
async fn flat_nesting_history_equals_parallel() {
    let mixed = run(|(), ctx| async move {
        let mut set = BranchSet::new();
        let _object = set.add("object", |_ctx| async {
            Ok(Unsorted {
                zeta: 1,
                alpha: "a".to_owned(),
            })
        });
        let _number = set.add("number", |_ctx| async { Ok(5_u32) });
        let _text = set.add("text", |_ctx| async { Ok("hi".to_owned()) });
        ctx.parallel_mixed(set)
            .name("pm")
            .nesting(NestingMode::Flat)
            .await?;
        wait_then_replay(&ctx).await
    })
    .await;
    let homogeneous = run(|(), ctx| async move {
        let branches: Vec<Branch<serde_json::Value>> = vec![
            Branch::new("object", |_ctx| async {
                Ok(serde_json::json!({"zeta": 1, "alpha": "a"}))
            }),
            Branch::new("number", |_ctx| async { Ok(serde_json::json!(5)) }),
            Branch::new("text", |_ctx| async { Ok(serde_json::json!("hi")) }),
        ];
        let _values = ctx
            .parallel(branches)
            .name("pm")
            .nesting(NestingMode::Flat)
            .await?;
        wait_then_replay(&ctx).await
    })
    .await;
    assert!(mixed.is_success(), "parallel_mixed succeeds: {mixed}");
    assert!(homogeneous.is_success(), "parallel succeeds: {homogeneous}");
    assert_eq!(shapes(&mixed), shapes(&homogeneous));
    assert!(
        mixed
            .operations()
            .iter()
            .all(|op| op.sub_type() != Some("ParallelBranch")),
        "Flat nesting records no branch context: {mixed}"
    );
    let (mixed_record, homogeneous_record) = (record(&mixed, "pm"), record(&homogeneous, "pm"));
    for index in 0..3 {
        let parse = |value: &serde_json::Value| -> serde_json::Value {
            serde_json::from_str(&value.to_string()).expect("a result is JSON")
        };
        assert_eq!(
            parse(item_result(&mixed_record, index)),
            parse(item_result(&homogeneous_record, index)),
        );
    }
}

/// What the summary of [`parallel_mixed_summary_run`] saw in one call.
#[derive(Debug, Clone, PartialEq)]
struct SummaryCall {
    text_len: Option<usize>,
    numbers_len: Option<usize>,
    other_set: Option<String>,
    failed: Vec<(usize, String, Option<String>)>,
    reason: CompletionReason,
}

/// What the handles of [`parallel_mixed_summary_run`] returned in one
/// invocation: the text's length and the number count.
type HandleValues = (usize, usize);

/// Runs a `parallel_mixed` named `pm` with a summary. Branch `text`
/// returns `text_len` bytes of text at once, branch `numbers` waits 1 s
/// when `wait` is set and returns `numbers_len` numbers, and branch
/// `fails` fails within the tolerance. When `wait` is set, the parent waits
/// after the parallel. Returns each summary call, the handles' values of
/// each invocation that read them, and the run.
async fn parallel_mixed_summary_run(
    text_len: usize,
    numbers_len: usize,
    wait: bool,
) -> (Vec<SummaryCall>, Vec<HandleValues>, TestResult<()>) {
    let calls: Log<SummaryCall> = log();
    let handles: Log<HandleValues> = log();
    let (calls_h, handles_h) = (Arc::clone(&calls), Arc::clone(&handles));
    let result = run(move |(), ctx| {
        let calls = Arc::clone(&calls_h);
        let handles = Arc::clone(&handles_h);
        async move {
            let mut set = BranchSet::new();
            let text = set.add("text", move |_ctx| async move { Ok("t".repeat(text_len)) });
            let numbers = set.add("numbers", move |child: DurableContext| async move {
                if wait {
                    child.wait(Duration::from_secs(1)).name("slow").await?;
                }
                Ok(vec![1_234_567_u32; numbers_len])
            });
            let _fails = set.add("fails", |_ctx| async {
                Err::<u32, BoxError>("boom".into())
            });
            let mut other = BranchSet::new();
            let other_text = other.add("text", |_ctx| async { Ok("other".to_owned()) });
            let (text_key, numbers_key, other_key) = (text.key(), numbers.key(), other_text.key());
            ctx.parallel_mixed(set)
                .name("pm")
                .completion(CompletionConfig::with_tolerated_failure_count(1))
                .summary(move |batch| {
                    let call = SummaryCall {
                        text_len: batch.value(&text_key).map(String::len),
                        numbers_len: batch.value(&numbers_key).map(Vec::len),
                        other_set: batch.value(&other_key).cloned(),
                        failed: batch
                            .failed()
                            .map(|item| (item.index, item.name.clone(), item.error_message.clone()))
                            .collect(),
                        reason: batch.reason,
                    };
                    let output = format!("{call:?}");
                    push(&calls, call);
                    output
                })
                .await?;
            push(
                &handles,
                (text.into_result()?.len(), numbers.into_result()?.len()),
            );
            if wait {
                wait_then_replay(&ctx).await?;
            }
            Ok(())
        }
    })
    .await;
    (entries(&calls), entries(&handles), result)
}

/// Two branches of about 150 KiB each make the batch record too large to
/// store. The summary runs once, in the invocation that completes the
/// batch, and reads both values through their keys, although `text`
/// completed in the invocation before. The rebuild in the third invocation
/// does not call it, and the handles return their values both times.
#[tokio::test]
async fn oversized_parallel_mixed_records_the_summary() {
    let (calls, handles, result) = parallel_mixed_summary_run(150_000, 19_000, true).await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 3, "{result}");
    let expected = SummaryCall {
        text_len: Some(150_000),
        numbers_len: Some(19_000),
        other_set: None,
        failed: vec![(2, "fails".to_owned(), Some("boom".to_owned()))],
        reason: CompletionReason::AllCompleted,
    };
    assert_eq!(calls, vec![expected.clone()]);
    assert_eq!(handles, vec![(150_000, 19_000), (150_000, 19_000)]);

    let payload = op(&result, "pm")
        .result()
        .expect("the parallel has a payload");
    let record: serde_json::Value = serde_json::from_str(payload).expect("the record is JSON");
    assert_eq!(
        record.get("type"),
        Some(&serde_json::json!("ParallelResult"))
    );
    assert_eq!(record.get("totalCount"), Some(&serde_json::json!(3)));
    assert_eq!(record.get("successCount"), Some(&serde_json::json!(2)));
    assert_eq!(record.get("failureCount"), Some(&serde_json::json!(1)));
    assert_eq!(record.get("itemStatuses"), Some(&serde_json::json!("SSF")));
    let output = format!("{expected:?}");
    assert_eq!(record.get("summary"), Some(&serde_json::json!(output)));
    let last = format!(",\"summary\":{}}}", serde_json::Value::from(output));
    assert!(
        payload.ends_with(&last),
        "summary is the last key: {payload}"
    );
}

/// A batch record under the limit is stored as usual: the summary is not
/// called, and the record has no `summary` key.
#[tokio::test]
async fn parallel_mixed_under_the_limit_skips_the_summary() {
    let (calls, handles, result) = parallel_mixed_summary_run(10, 3, false).await;
    assert!(result.is_success(), "the execution succeeds: {result}");
    assert_eq!(result.invocation_count(), 1, "{result}");
    assert_eq!(calls, Vec::new());
    assert_eq!(handles, vec![(10, 3)]);
    let record = record(&result, "pm");
    assert!(record.get("summary").is_none(), "{record}");
    assert!(
        record.get("all").is_some(),
        "the record is inline: {record}"
    );
}
