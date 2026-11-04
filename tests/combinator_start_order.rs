//! The order in which a combinator begins its `.future()` inputs and its
//! own start step.
//!
//! JS starts an operation at its call, and a combinator is a lazy promise
//! that queues its own `START` when it is first awaited. So the inputs'
//! first checkpoints come before the combinator's, and an input's
//! "queue, then hook" start hook fires before the combinator's hook. A
//! combinator begins each `.future()` input's start phase, in input order,
//! before its own start step, as these tests check for `try_join_all`,
//! `join_all`, `select_ok` and `race`.
//!
//! Each test runs every case and asserts only after the last one, so one
//! run reports every combinator that fails.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::unwrap_used)] // reason: test assertions

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::lambda_types::{
    ExecutionDetails, Operation, OperationAction, OperationStatus, OperationType, OperationUpdate,
};
use durable::observability::{TARGET, event_names, field_names};
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::serdes::SerdesContext;
use durable::test_util::{LocalRunner, TestResult};
use durable::{
    BoxError, CheckpointOutput, DurableContext, DurableFuture, ExecutionClient,
    ExecutionClientError, ExecutionClientErrorScope, GetStateOutput, InvocationPayload, Options,
    RetryDecision, Serdes, lambda_runtime,
};
use tracing_subscriber::layer::SubscriberExt;

/// Serializes the capture of test 3 against the other tests of this
/// binary. The other tests run without a subscriber on their own threads,
/// and while they run, the thread-scoped subscriber of test 3 can miss
/// events (`tracing` caches callsite interest across threads). Test 3 takes
/// the lock exclusively, every other test shares it.
static CAPTURE: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

/// The four combinators, each named `all` in the handlers below.
#[derive(Debug, Clone, Copy)]
enum Combinator {
    TryJoinAll,
    JoinAll,
    SelectOk,
    Race,
}

const COMBINATORS: [Combinator; 4] = [
    Combinator::TryJoinAll,
    Combinator::JoinAll,
    Combinator::SelectOk,
    Combinator::Race,
];

impl Combinator {
    /// Awaits the combinator `all` over `inputs`, and drops its result.
    async fn run<O>(self, ctx: &DurableContext, inputs: Vec<DurableFuture<O>>)
    where
        O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    {
        match self {
            Self::TryJoinAll => drop(ctx.try_join_all(inputs).name("all").await),
            Self::JoinAll => drop(ctx.join_all(inputs).name("all").await),
            Self::SelectOk => drop(ctx.select_ok(inputs).name("all").await),
            Self::Race => drop(ctx.race(inputs).name("all").await),
        }
    }
}

/// How many times each step body ran, by name.
type Runs = Arc<Mutex<BTreeMap<String, usize>>>;

fn runs_of(runs: &Runs, name: &str) -> usize {
    runs.lock().unwrap().get(name).copied().unwrap_or(0)
}

/// A step named `name` from `.future()` that counts its body runs. With
/// `fail`, the body fails and the step does not retry.
fn step(
    ctx: &DurableContext,
    runs: &Runs,
    name: &'static str,
    fail: bool,
) -> DurableFuture<String> {
    let runs = Arc::clone(runs);
    ctx.step(move |_| {
        *runs.lock().unwrap().entry(name.to_owned()).or_default() += 1;
        async move {
            if fail {
                Err::<String, BoxError>(format!("{name} failed").into())
            } else {
                Ok(name.to_owned())
            }
        }
    })
    .name(name)
    .retry_strategy(|_err, _attempt| RetryDecision::Stop)
    .future()
}

/// Records every `on_operation_start` call as (invocation number, name,
/// `is_replay`), counting invocations from 1, and the request ID of each
/// invocation.
#[derive(Clone, Default)]
struct StartHooks {
    count: Arc<AtomicUsize>,
    hooks: Arc<Mutex<Vec<(usize, String, bool)>>>,
    request_ids: Arc<Mutex<Vec<String>>>,
}

struct StartHookPlugin {
    invocation: usize,
    hooks: Arc<Mutex<Vec<(usize, String, bool)>>>,
}

impl Plugin for StartHookPlugin {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.hooks.lock().unwrap().push((
            self.invocation,
            info.name().unwrap_or("-").to_owned(),
            info.is_replay(),
        ));
        Ok(None)
    }
}

impl PluginFactory for StartHooks {
    fn create_plugin(&self, info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let invocation = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        self.request_ids
            .lock()
            .unwrap()
            .push(info.request_id().to_owned());
        Ok(Some(Box::new(StartHookPlugin {
            invocation,
            hooks: Arc::clone(&self.hooks),
        })))
    }
}

impl StartHooks {
    /// The start hooks of invocation `invocation`, in call order.
    fn of(&self, invocation: usize) -> Vec<(String, bool)> {
        self.hooks
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _, _)| *n == invocation)
            .map(|(_, name, replay)| (name.clone(), *replay))
            .collect()
    }

    /// The request ID of invocation `invocation`.
    fn request_id(&self, invocation: usize) -> Option<String> {
        self.request_ids
            .lock()
            .unwrap()
            .get(invocation - 1)
            .cloned()
    }
}

/// Runs `handler` on `LocalRunner` with the start-hook recorder.
async fn run_local<F, Fut>(hooks: &StartHooks, handler: F) -> TestResult<()>
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    LocalRunner::new()
        .options(
            Options::builder()
                .plugin(hooks.clone())
                .build()
                .expect("valid options"),
        )
        .run(handler, ())
        .await
        .expect("the runner completes")
}

/// The names of `result`'s operations, in first-checkpoint order.
fn names<O>(result: &TestResult<O>) -> Vec<String> {
    result
        .operations()
        .iter()
        .map(|op| op.name().unwrap_or("-").to_owned())
        .collect()
}

/// The handler of tests 1 to 3: steps `a` and `b` from `.future()`, the
/// combinator `all` over them, a one-second wait `tail` and a step `after`.
/// With `fail_b`, step `b` fails.
fn steps_handler(
    combinator: Combinator,
    runs: &Runs,
    fail_b: bool,
) -> impl Fn((), DurableContext) -> Pin<Box<dyn Future<Output = Result<(), BoxError>> + Send>>
+ Send
+ Sync
+ 'static {
    let runs = Arc::clone(runs);
    move |(), ctx| {
        let runs = Arc::clone(&runs);
        Box::pin(async move {
            let a = step(&ctx, &runs, "a", false);
            let b = step(&ctx, &runs, "b", fail_b);
            combinator.run(&ctx, vec![a, b]).await;
            ctx.wait(Duration::from_secs(1)).name("tail").await?;
            step(&ctx, &runs, "after", false).await?;
            Ok(())
        })
    }
}

/// The handler of tests 4 and 5: a one-second wait `a` and a two-second
/// wait `b` from `.future()`, the combinator `all` over them, and a step
/// `after`.
async fn waits_handler(combinator: Combinator, ctx: DurableContext) -> Result<(), BoxError> {
    let a = ctx.wait(Duration::from_secs(1)).name("a").future();
    let b = ctx.wait(Duration::from_secs(2)).name("b").future();
    combinator.run(&ctx, vec![a, b]).await;
    ctx.step(|_| async { Ok(()) }).name("after").await?;
    Ok(())
}

fn owned(entries: &[(&str, bool)]) -> Vec<(String, bool)> {
    entries
        .iter()
        .map(|(name, replay)| ((*name).to_owned(), *replay))
        .collect()
}

#[tokio::test]
async fn inputs_checkpoint_before_combinator_start() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let runs = Runs::default();
        let result = run_local(
            &StartHooks::default(),
            steps_handler(combinator, &runs, false),
        )
        .await;
        let names = names(&result);
        if names != ["a", "b", "all", "tail", "after"] {
            failures.push(format!("{combinator:?}: {names:?}"));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn input_queue_time_start_hooks_before_combinator_start_hook() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let runs = Runs::default();
        let hooks = StartHooks::default();
        run_local(&hooks, steps_handler(combinator, &runs, false)).await;
        let first = hooks.of(1);
        let expected = owned(&[("a", false), ("b", false), ("all", false), ("tail", false)]);
        if first != expected {
            failures.push(format!("{combinator:?}: {first:?}"));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

/// A `MakeWriter` that captures subscriber output in a shared buffer.
#[derive(Clone)]
struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// The `operation_replayed` lifecycle events in `output`, as
/// (`requestId`, `operationName`).
fn replayed_events(output: &str) -> Vec<(String, String)> {
    output
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| {
            v.get("message").and_then(serde_json::Value::as_str)
                == Some(event_names::OPERATION_REPLAYED)
                && v.get("target").and_then(serde_json::Value::as_str) == Some(TARGET)
        })
        .map(|v| {
            let field = |name: &str| {
                v.get(name)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            (
                field(field_names::REQUEST_ID),
                field(field_names::OPERATION_NAME),
            )
        })
        .collect()
}

#[tokio::test]
async fn final_record_replay_starts_no_input() {
    let _capture = CAPTURE.write().await;
    let cases: Vec<(Combinator, bool)> = COMBINATORS
        .into_iter()
        .map(|c| (c, false))
        .chain(std::iter::once((Combinator::TryJoinAll, true)))
        .collect();
    let mut failures = Vec::new();
    for (combinator, fail_b) in cases {
        let case = format!("{combinator:?} fail_b={fail_b}");
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .json()
                .flatten_event(true)
                .with_writer(CaptureWriter(Arc::clone(&buffer))),
        );
        let guard = tracing::subscriber::set_default(subscriber);
        let runs = Runs::default();
        let hooks = StartHooks::default();
        let result = run_local(&hooks, steps_handler(combinator, &runs, fail_b)).await;
        drop(guard);
        if result.invocation_count() != 2 {
            failures.push(format!("{case}: {} invocations", result.invocation_count()));
            continue;
        }
        let second = hooks.of(2);
        if second.iter().any(|(name, _)| name == "a" || name == "b") {
            failures.push(format!("{case}: invocation 2 start hooks {second:?}"));
        }
        let request_id = hooks.request_id(2).expect("invocation 2 has a request ID");
        let output = String::from_utf8_lossy(&buffer.lock().unwrap()).to_string();
        let replayed: Vec<String> = replayed_events(&output)
            .into_iter()
            .filter(|(rid, _)| *rid == request_id)
            .map(|(_, name)| name)
            .collect();
        if replayed.iter().any(|name| name == "a" || name == "b") {
            failures.push(format!("{case}: invocation 2 replayed {replayed:?}"));
        }
        if !replayed.iter().any(|name| name == "all") {
            failures.push(format!("{case}: no replay event for all: {replayed:?}"));
        }
        for name in ["a", "b"] {
            if runs_of(&runs, name) > 1 {
                failures.push(format!("{case}: {name} ran {} times", runs_of(&runs, name)));
            }
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn started_combinator_input_replay_hook_before_combinator_replay_hook() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let hooks = StartHooks::default();
        run_local(&hooks, move |(), ctx| waits_handler(combinator, ctx)).await;
        let second = hooks.of(2);
        let position = |entry: (&str, bool)| {
            second
                .iter()
                .position(|(name, replay)| name == entry.0 && *replay == entry.1)
        };
        match (position(("b", true)), position(("all", true))) {
            (Some(b), Some(all)) if b < all => {}
            _ => failures.push(format!("{combinator:?}: {second:?}")),
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn waiting_input_starts_precede_combinator_start_before_suspension() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let result = run_local(&StartHooks::default(), move |(), ctx| {
            waits_handler(combinator, ctx)
        })
        .await;
        let names = names(&result);
        if names != ["a", "b", "all", "after"] {
            failures.push(format!("{combinator:?}: {names:?}"));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

/// A shared, ordered event log.
#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    fn push(&self, entry: String) {
        self.0.lock().unwrap().push(entry);
    }

    fn entries(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }

    fn position(&self, entry: &str) -> Option<usize> {
        self.entries().iter().position(|e| e == entry)
    }

    fn contains(&self, entry: &str) -> bool {
        self.position(entry).is_some()
    }

    /// Whether `first` and `second` are both logged, `first` before.
    fn before(&self, first: &str, second: &str) -> bool {
        matches!(
            (self.position(first), self.position(second)),
            (Some(f), Some(s)) if f < s
        )
    }
}

/// A gate that holds back what waits on it until it opens.
#[derive(Clone)]
struct Gate(Arc<tokio::sync::Semaphore>);

impl Default for Gate {
    fn default() -> Self {
        Self(Arc::new(tokio::sync::Semaphore::new(0)))
    }
}

impl Gate {
    async fn pass(&self) {
        let _permit = self.0.acquire().await.expect("the gate is open");
    }

    fn open(&self) {
        self.0.add_permits(tokio::sync::Semaphore::MAX_PERMITS / 2);
    }
}

/// A transport that logs `send <ACTION> <name>` for each update it
/// receives, and `response <name>` for each `START` when that request's
/// future completes, after one yield. A request that carries the `START`
/// of a held name waits for the gate first. With `fail_first`, the first
/// call fails with an execution-scoped error.
struct Transport {
    log: Log,
    held: Vec<&'static str>,
    gate: Gate,
    fail_first: bool,
    calls: AtomicUsize,
}

impl Transport {
    fn new(log: &Log) -> Self {
        Self {
            log: log.clone(),
            held: Vec::new(),
            gate: Gate::default(),
            fail_first: false,
            calls: AtomicUsize::new(0),
        }
    }
}

impl ExecutionClient for Transport {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
        for update in &updates {
            self.log.push(format!(
                "send {} {}",
                update.action().as_str(),
                update.name().unwrap_or("-")
            ));
        }
        let starts: Vec<String> = updates
            .iter()
            .filter(|u| u.action() == &OperationAction::Start)
            .map(|u| u.name().unwrap_or("-").to_owned())
            .collect();
        let held = starts.iter().any(|name| self.held.contains(&name.as_str()));
        let fail = self.fail_first && self.calls.fetch_add(1, Ordering::SeqCst) == 0;
        let log = self.log.clone();
        let gate = self.gate.clone();
        Box::pin(async move {
            if fail {
                return Err(ExecutionClientError::new(
                    ExecutionClientErrorScope::Execution,
                    "rejected",
                ));
            }
            tokio::task::yield_now().await;
            if held {
                gate.pass().await;
            }
            for name in starts {
                log.push(format!("response {name}"));
            }
            Ok(CheckpointOutput::new(
                Some("t".to_owned()),
                Vec::new(),
                None,
            ))
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        _marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
        Box::pin(async {
            Ok(GetStateOutput::new(
                vec![
                    Operation::builder()
                        .id("root")
                        .r#type(OperationType::Execution)
                        .status(OperationStatus::Started)
                        .start_timestamp(durable::lambda_types::DateTime::from_secs(0))
                        .execution_details(ExecutionDetails::builder().input_payload("{}").build())
                        .build()
                        .expect("every required Operation field is set"),
                ],
                None,
            ))
        })
    }
}

fn payload() -> InvocationPayload {
    serde_json::from_value(serde_json::json!({
        "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:combinator-start",
        "CheckpointToken": "t-0",
        "InitialExecutionState": {
            "Operations": [{
                "Id": "root",
                "Type": "Execution",
                "Status": "STARTED",
                "ExecutionDetails": { "InputPayload": "{}" }
            }]
        },
    }))
    .expect("a valid payload")
}

/// A plugin that logs `hook <name>` for each `on_operation_start`.
#[derive(Clone)]
struct HookLog(Log);

impl Plugin for HookLog {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.0.push(format!("hook {}", info.name().unwrap_or("-")));
        Ok(None)
    }
}

impl PluginFactory for HookLog {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Runs `handler` once through `wrap` over `transport`, with checkpoint
/// batching off and the hook log, and returns the response as JSON.
async fn invoke<F, Fut>(transport: Transport, handler: F) -> serde_json::Value
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, BoxError>> + Send + 'static,
{
    let log = transport.log.clone();
    let service = durable::wrap(
        handler,
        Options::builder()
            .execution_client(Arc::new(transport))
            .checkpoint_batching(false)
            .plugin(HookLog(log))
            .build()
            .expect("valid options"),
    );
    let event = lambda_runtime::LambdaEvent::new(payload(), lambda_runtime::Context::default());
    let response = service(event).await.expect("the invocation returns");
    serde_json::to_value(&response).expect("the response serializes")
}

#[tokio::test]
async fn combinator_start_hook_fires_before_start_response() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let log = Log::default();
        invoke(Transport::new(&log), move |_, ctx| async move {
            let runs = Runs::default();
            let a = step(&ctx, &runs, "a", false);
            let b = step(&ctx, &runs, "b", false);
            combinator.run(&ctx, vec![a, b]).await;
            Ok(String::new())
        })
        .await;
        if !log.before("hook all", "send START all") || !log.before("hook all", "response all") {
            failures.push(format!("{combinator:?}: {:?}", log.entries()));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn unclaimed_input_start_failure_is_unrecoverable() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let log = Log::default();
        let runs = Runs::default();
        let mut transport = Transport::new(&log);
        transport.fail_first = true;
        let handler_runs = Arc::clone(&runs);
        let response = invoke(transport, move |_, ctx| {
            let runs = Arc::clone(&handler_runs);
            async move {
                let c = step(&ctx, &runs, "c", false);
                let d = step(&ctx, &runs, "d", false);
                let inner = ctx.race([c, d]).name("inner").future();
                let s = step(&ctx, &runs, "s", false);
                combinator.run(&ctx, vec![inner, s]).await;
                Ok(String::new())
            }
        })
        .await;
        let error = |field: &str| {
            response
                .get("Error")
                .and_then(|e| e.get(field))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let sends: Vec<String> = log
            .entries()
            .into_iter()
            .filter(|e| e.starts_with("send "))
            .collect();
        let ran: usize = ["c", "d", "s"].iter().map(|n| runs_of(&runs, n)).sum();
        if error("ErrorType") != "CheckpointUnrecoverableExecutionError"
            || !error("ErrorMessage").starts_with("[Unrecoverable Execution] Checkpoint failed:")
            || sends != ["send START all"]
            || ran != 0
        {
            failures.push(format!(
                "{combinator:?}: response={response} sends={sends:?} runs={ran}"
            ));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

/// A payload serdes that logs `serialize <name>` when it is entered, then
/// waits for its gate before it serializes.
struct GatedSerdes {
    name: &'static str,
    log: Log,
    gate: Gate,
}

impl Serdes<String> for GatedSerdes {
    async fn serialize(&self, value: String, _context: SerdesContext) -> Result<String, BoxError> {
        self.log.push(format!("serialize {}", self.name));
        self.gate.pass().await;
        Ok(serde_json::to_string(&value)?)
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(serde_json::from_str(&wire).map_err(BoxError::from))
    }
}

/// Waits up to two seconds for `ready`. Returns whether it held.
async fn eventually(mut ready: impl FnMut() -> bool) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while !ready() {
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    true
}

#[tokio::test]
async fn invoke_inputs_first_poll_before_combinator_start_hook() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let log = Log::default();
        let gate_a = Gate::default();
        let gate_b = Gate::default();
        let (handler_log, handler_a, handler_b) = (log.clone(), gate_a.clone(), gate_b.clone());
        let invocation = invoke(Transport::new(&log), move |_, ctx| {
            let (log, gate_a, gate_b) = (handler_log.clone(), handler_a.clone(), handler_b.clone());
            async move {
                let a = ctx
                    .invoke::<String, String>("target-a", "in".to_owned())
                    .name("a")
                    .payload_serdes(GatedSerdes {
                        name: "a",
                        log: log.clone(),
                        gate: gate_a,
                    })
                    .future();
                let b = ctx
                    .invoke::<String, String>("target-b", "in".to_owned())
                    .name("b")
                    .payload_serdes(GatedSerdes {
                        name: "b",
                        log,
                        gate: gate_b,
                    })
                    .future();
                combinator.run(&ctx, vec![a, b]).await;
                Ok(String::new())
            }
        });
        let observe = async {
            let reached = eventually(|| log.contains("hook all")).await;
            let held_ok = reached
                && log.before("serialize a", "serialize b")
                && log.before("serialize b", "hook all")
                && !log.contains("send START a")
                && !log.contains("send START b");
            let entries = log.entries();
            gate_a.open();
            gate_b.open();
            (held_ok, entries)
        };
        let (_, (held_ok, entries)) = tokio::join!(invocation, observe);
        if !held_ok {
            failures.push(format!("{combinator:?}: {entries:?}"));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn condition_inputs_first_poll_before_combinator_start_hook() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let log = Log::default();
        let mut transport = Transport::new(&log);
        transport.held = vec!["a", "b"];
        let gate = transport.gate.clone();
        let invocation = invoke(transport, move |_, ctx| async move {
            let a = ctx
                .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0)
                .name("a")
                .future();
            let b = ctx
                .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0)
                .name("b")
                .future();
            combinator.run(&ctx, vec![a, b]).await;
            Ok(String::new())
        });
        let observe = async {
            let reached = eventually(|| log.contains("hook all")).await;
            let held_ok = reached
                && log.before("hook a", "hook b")
                && log.before("hook b", "hook all")
                && !log.contains("response a")
                && !log.contains("response b");
            let entries = log.entries();
            gate.open();
            (held_ok, entries)
        };
        let (response, (held_ok, entries)) = tokio::join!(invocation, observe);
        let status = response.get("Status").and_then(serde_json::Value::as_str);
        if !held_ok || status != Some("SUCCEEDED") {
            failures.push(format!("{combinator:?}: status={status:?} {entries:?}"));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn spawned_input_starts_once() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let runs = Runs::default();
        let hooks = StartHooks::default();
        let handler_runs = Arc::clone(&runs);
        let result = run_local(&hooks, move |(), ctx| {
            let runs = Arc::clone(&handler_runs);
            async move {
                let a = {
                    let runs = Arc::clone(&runs);
                    ctx.step(move |_| {
                        *runs.lock().unwrap().entry("a".to_owned()).or_default() += 1;
                        async { Ok("a".to_owned()) }
                    })
                    .name("a")
                    .spawn()
                };
                let b = step(&ctx, &runs, "b", false);
                combinator.run(&ctx, vec![a, b]).await;
                ctx.wait(Duration::from_secs(1)).name("tail").await?;
                Ok(())
            }
        })
        .await;
        let records = names(&result).iter().filter(|n| *n == "a").count();
        let first_hooks = hooks.of(1).iter().filter(|(n, _)| n == "a").count();
        let all_hooks = hooks
            .hooks
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, n, _)| n == "a")
            .count();
        if records != 1 || first_hooks != 1 || all_hooks != 1 || runs_of(&runs, "a") != 1 {
            failures.push(format!(
                "{combinator:?}: records={records} hooks={first_hooks}/{all_hooks} runs={}",
                runs_of(&runs, "a")
            ));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

#[tokio::test]
async fn refused_input_never_resolves() {
    let _capture = CAPTURE.read().await;
    let mut failures = Vec::new();
    for combinator in COMBINATORS {
        let runs = Runs::default();
        let hooks = StartHooks::default();
        let outcome: Arc<Mutex<Option<String>>> = Arc::default();
        let (handler_runs, handler_outcome) = (Arc::clone(&runs), Arc::clone(&outcome));
        let result = run_local(&hooks, move |(), ctx| {
            let (runs, outcome) = (Arc::clone(&handler_runs), Arc::clone(&handler_outcome));
            async move {
                let slot: Arc<Mutex<Option<DurableFuture<String>>>> = Arc::default();
                {
                    let (slot, runs) = (Arc::clone(&slot), Arc::clone(&runs));
                    ctx.run_in_child_context(move |child| async move {
                        *slot.lock().unwrap() = Some(step(&child, &runs, "x", false));
                        Ok(())
                    })
                    .name("scope")
                    .await?;
                }
                let x = slot.lock().unwrap().take().expect("the child stored x");
                let b = step(&ctx, &runs, "b", false);
                let inputs = vec![x, b];
                let got = match combinator {
                    Combinator::SelectOk => Some(format!(
                        "{:?}",
                        ctx.select_ok(inputs)
                            .name("all")
                            .await
                            .map_err(|e| e.to_string())
                    )),
                    Combinator::Race => Some(format!(
                        "{:?}",
                        ctx.race(inputs)
                            .name("all")
                            .await
                            .map_err(|e| e.to_string())
                    )),
                    Combinator::TryJoinAll => tokio::time::timeout(
                        Duration::from_millis(300),
                        ctx.try_join_all(inputs).name("all").into_future(),
                    )
                    .await
                    .ok()
                    .map(|r| format!("{:?}", r.map_err(|e| e.to_string()))),
                    Combinator::JoinAll => tokio::time::timeout(
                        Duration::from_millis(300),
                        ctx.join_all(inputs).name("all").into_future(),
                    )
                    .await
                    .ok()
                    .map(|r| format!("{:?}", r.map(|v| v.len()).map_err(|e| e.to_string()))),
                };
                *outcome.lock().unwrap() = got;
                Ok(())
            }
        })
        .await;
        let got = outcome.lock().unwrap().clone();
        let expected = match combinator {
            Combinator::SelectOk | Combinator::Race => Some(r#"Ok("b")"#.to_owned()),
            Combinator::TryJoinAll | Combinator::JoinAll => None,
        };
        let b_succeeded = result
            .operation("b")
            .is_some_and(|op| op.status() == durable::OperationStatus::Succeeded);
        let x_hooks = hooks
            .hooks
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, n, _)| n == "x")
            .count();
        if got != expected
            || !b_succeeded
            || result.operation("x").is_some()
            || x_hooks != 0
            || runs_of(&runs, "x") != 0
        {
            failures.push(format!(
                "{combinator:?}: outcome={got:?} b_succeeded={b_succeeded} x_record={} \
                 x_hooks={x_hooks} x_runs={}",
                result.operation("x").is_some(),
                runs_of(&runs, "x")
            ));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}
