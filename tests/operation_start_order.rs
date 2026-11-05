//! The order of an operation's `START` write and its `on_operation_start`
//! hook, through `wrap` and a caller-supplied transport.
//!
//! JS fires `onOperationStart` after the `START` response for a step under
//! `AtMostOncePerRetry`, a wait and a callback, and right after it queues
//! `START` for a default step and `waitForCondition`. A queued JS `START`
//! is sent only after the hook returns. These tests pin that order on the
//! current-thread and the multi-thread runtime.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::unwrap_used)] // reason: test assertions

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::lambda_types::{
    CallbackDetails, DateTime, ExecutionDetails, Operation, OperationAction, OperationStatus,
    OperationType, OperationUpdate,
};
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::{
    BoxError, CheckpointOutput, DurableContext, ExecutionClient, ExecutionClientError,
    GetStateOutput, InvocationPayload, Options, StepSemantics, lambda_runtime,
};

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

    fn contains(&self, entry: &str) -> bool {
        self.entries().iter().any(|e| e == entry)
    }

    /// The index of `entry`, failing the test when it is missing.
    fn index(&self, entry: &str) -> usize {
        let entries = self.entries();
        entries
            .iter()
            .position(|e| e == entry)
            .ok_or_else(|| format!("{entry:?} is missing from {entries:?}"))
            .unwrap()
    }
}

/// A transport that logs `send <ACTION> <name>` for each update it
/// receives, and `response <name>` for each `START` when that request's
/// future completes, after one yield. A callback `START` is answered with
/// the callback's record. With a hold, the response of a request that
/// carries the `START` of the held name waits for a semaphore permit.
struct Transport {
    log: Log,
    hold: Option<(String, Arc<tokio::sync::Semaphore>)>,
}

impl Transport {
    fn new(log: Log) -> Self {
        Self { log, hold: None }
    }

    /// The transport of [`Self::new`], holding the response of the
    /// `START` of `name` until `semaphore` grants a permit.
    fn holding(log: Log, name: &str, semaphore: Arc<tokio::sync::Semaphore>) -> Self {
        Self {
            log,
            hold: Some((name.to_owned(), semaphore)),
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
        let starts: Vec<&OperationUpdate> = updates
            .iter()
            .filter(|u| u.action() == &OperationAction::Start)
            .collect();
        let names: Vec<String> = starts
            .iter()
            .map(|u| u.name().unwrap_or("-").to_owned())
            .collect();
        let records: Vec<Operation> = starts
            .iter()
            .filter(|u| u.r#type() == &OperationType::Callback)
            .map(|u| callback_record(u.id()))
            .collect();
        let hold = self
            .hold
            .as_ref()
            .filter(|(held, _)| names.contains(held))
            .map(|(_, semaphore)| Arc::clone(semaphore));
        let log = self.log.clone();
        Box::pin(async move {
            tokio::task::yield_now().await;
            if let Some(semaphore) = hold {
                let _permit = semaphore.acquire().await.expect("the semaphore is open");
            }
            for name in names {
                log.push(format!("response {name}"));
            }
            Ok(CheckpointOutput::new(Some("t".to_owned()), records, None))
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        _marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
        Box::pin(async { Ok(GetStateOutput::new(vec![execution_operation()], None)) })
    }
}

/// The `STARTED` record of the callback `id`, with a callback ID.
fn callback_record(id: &str) -> Operation {
    Operation::builder()
        .id(id)
        .r#type(OperationType::Callback)
        .status(OperationStatus::Started)
        .start_timestamp(DateTime::from_secs(0))
        .callback_details(CallbackDetails::builder().callback_id("cb-id").build())
        .build()
        .expect("every required Operation field is set")
}

fn execution_operation() -> Operation {
    Operation::builder()
        .id("root")
        .r#type(OperationType::Execution)
        .status(OperationStatus::Started)
        .start_timestamp(DateTime::from_secs(0))
        .execution_details(ExecutionDetails::builder().input_payload("{}").build())
        .build()
        .expect("every required Operation field is set")
}

fn payload() -> InvocationPayload {
    serde_json::from_value(serde_json::json!({
        "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:start-order",
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

/// A plugin that logs `hook <name>` when `on_operation_start` begins, and
/// `hook-return <name>` as the hook's last statement.
#[derive(Clone)]
struct HookLog(Log);

impl Plugin for HookLog {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        let name = info.name().unwrap_or("-");
        self.0.push(format!("hook {name}"));
        self.0.push(format!("hook-return {name}"));
        Ok(None)
    }
}

impl PluginFactory for HookLog {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Runs `handler` once through `wrap` over the logging transport, with the
/// plugin `factory`, and returns the invocation's status.
async fn invoke<F, Fut, P>(log: &Log, batching: bool, factory: P, handler: F) -> String
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, BoxError>> + Send + 'static,
    P: PluginFactory,
{
    invoke_with(
        Transport::new(log.clone()),
        payload(),
        batching,
        factory,
        handler,
    )
    .await
}

/// Runs `handler` once through `wrap` over `transport`, with the initial
/// state of `payload` and the plugin `factory`, and returns the
/// invocation's status.
async fn invoke_with<F, Fut, P>(
    transport: Transport,
    payload: InvocationPayload,
    batching: bool,
    factory: P,
    handler: F,
) -> String
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, BoxError>> + Send + 'static,
    P: PluginFactory,
{
    let service = durable::wrap(
        handler,
        Options::builder()
            .execution_client(Arc::new(transport))
            .checkpoint_batching(batching)
            .plugin(factory)
            .build()
            .expect("valid options"),
    );
    let event = lambda_runtime::LambdaEvent::new(payload, lambda_runtime::Context::default());
    let response = service(event).await.expect("the invocation returns");
    let response = serde_json::to_value(&response).expect("the response serializes");
    response
        .get("Status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Runs `handler` with checkpoint batching disabled and the hook log.
async fn invoke_unbatched<F, Fut>(log: &Log, handler: F) -> String
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, BoxError>> + Send + 'static,
{
    invoke(log, false, HookLog(log.clone()), handler).await
}

#[tokio::test]
async fn default_step_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        Ok(ctx.step(|_| async { Ok("v".to_owned()) }).name("a").await?)
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook a") < log.index("response a"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn at_most_once_step_start_hook_fires_after_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        Ok(ctx
            .step(|_| async { Ok("v".to_owned()) })
            .name("a")
            .semantics(StepSemantics::AtMostOncePerRetry)
            .await?)
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("response a") < log.index("hook a"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn wait_start_hook_fires_after_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.wait(Duration::from_secs(1)).name("w").await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "PENDING");
    assert!(
        log.index("response w") < log.index("hook w"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn create_callback_start_hook_fires_after_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        let cb = ctx.create_callback::<String>().name("cb").await?;
        Ok(cb.id().to_owned())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("response cb") < log.index("hook cb"),
        "{:?}",
        log.entries()
    );
}

/// The "queue, then hook" operations of test 7, each named `a`.
#[derive(Debug, Clone, Copy)]
enum QueueThenHook {
    Step,
    WaitForCondition,
}

impl QueueThenHook {
    async fn run(self, ctx: DurableContext) -> Result<String, BoxError> {
        match self {
            Self::Step => Ok(ctx.step(|_| async { Ok("v".to_owned()) }).name("a").await?),
            Self::WaitForCondition => {
                let state = ctx
                    .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0)
                    .name("a")
                    .await?;
                Ok(state.to_string())
            }
        }
    }
}

/// Every case of test 7: each "queue, then hook" operation, with
/// checkpoint batching disabled and enabled.
fn cases() -> Vec<(QueueThenHook, bool)> {
    [QueueThenHook::Step, QueueThenHook::WaitForCondition]
        .into_iter()
        .flat_map(|op| [(op, false), (op, true)])
        .collect()
}

#[tokio::test]
async fn queue_then_hook_returns_before_start_send_current_thread() {
    for (op, batching) in cases() {
        let log = Log::default();
        let status = invoke(&log, batching, HookLog(log.clone()), move |_, ctx| {
            op.run(ctx)
        })
        .await;
        assert_eq!(status, "SUCCEEDED", "{op:?} batching={batching}");
        assert!(
            log.index("hook-return a") < log.index("send START a"),
            "{op:?} batching={batching}: {:?}",
            log.entries()
        );
    }
}

/// The state of [`BlockingHook`]: whether its hook was entered, and
/// whether the test released it.
#[derive(Default)]
struct Gate {
    entered: bool,
    released: bool,
}

/// A plugin whose `on_operation_start` of `a` blocks, in
/// `block_in_place`, until the test releases it or five seconds pass.
#[derive(Clone, Default)]
struct BlockingHook(Arc<(Mutex<Gate>, Condvar)>);

impl BlockingHook {
    /// Waits, without blocking the runtime, until the hook is entered.
    async fn entered(&self) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while !self.0.0.lock().unwrap().entered {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the hook was not entered"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn release(&self) {
        self.0.0.lock().unwrap().released = true;
        self.0.1.notify_all();
    }
}

impl Plugin for BlockingHook {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        if info.name() == Some("a") {
            tokio::task::block_in_place(|| {
                let (gate, condvar) = &*self.0;
                let mut state = gate.lock().unwrap();
                state.entered = true;
                let _unused = condvar
                    .wait_timeout_while(state, Duration::from_secs(5), |s| !s.released)
                    .unwrap();
            });
        }
        Ok(None)
    }
}

impl PluginFactory for BlockingHook {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(clippy::disallowed_methods)] // reason: the invocation runs beside the test's observer
async fn queue_then_hook_blocks_start_send_until_hook_dispatch() {
    let mut failures = Vec::new();
    for (op, batching) in cases() {
        let log = Log::default();
        let hook = BlockingHook::default();
        let invocation = tokio::spawn({
            let log = log.clone();
            let hook = hook.clone();
            async move { invoke(&log, batching, hook, move |_, ctx| op.run(ctx)).await }
        });
        hook.entered().await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let sent_while_blocked = log.contains("send START a");
        hook.release();
        let status = invocation.await.expect("the invocation task completes");
        let sent_after = log.contains("send START a");
        if sent_while_blocked || !sent_after || status != "SUCCEEDED" {
            failures.push(format!(
                "{op:?} batching={batching}: sent while blocked={sent_while_blocked}, \
                 sent after={sent_after}, status={status}, log={:?}",
                log.entries()
            ));
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}

/// The names of the operations `result` recorded, in first-checkpoint
/// order.
fn names<O>(result: &durable::test_util::TestResult<O>) -> Vec<String> {
    result
        .operations()
        .iter()
        .map(|op| op.name().unwrap_or("-").to_owned())
        .collect()
}

#[tokio::test]
async fn spawned_step_start_precedes_next_operation() {
    let result = durable::test_util::LocalRunner::new()
        .run(
            |_: serde_json::Value, ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
                ctx.step(|_| async { Ok(1_u32) }).name("s1").await?;
                a.await?;
                Ok::<_, BoxError>(())
            },
            serde_json::Value::Null,
        )
        .await
        .expect("the runner completes");
    assert_eq!(names(&result), ["a", "s1"]);
}

/// Runs `handler` on `LocalRunner` with `options` and returns its result.
async fn run_local<F, Fut>(options: Options, handler: F) -> durable::test_util::TestResult<()>
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    run_runner(
        durable::test_util::LocalRunner::new().options(options),
        handler,
    )
    .await
}

/// Runs `handler` on `runner` and returns its result.
async fn run_runner<F, Fut>(
    runner: durable::test_util::LocalRunner,
    handler: F,
) -> durable::test_util::TestResult<()>
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    let result = runner
        .run(handler, serde_json::Value::Null)
        .await
        .expect("the runner completes");
    assert!(
        result.output().is_some(),
        "the execution succeeds: {:?}",
        result.error()
    );
    result
}

/// The names of `result`'s operations that are `first` or `s1`, in
/// first-checkpoint order.
fn order(result: &durable::test_util::TestResult<()>, first: &str) -> Vec<String> {
    names(result)
        .into_iter()
        .filter(|name| name == first || name == "s1")
        .collect()
}

/// Awaits a step `s1` on `ctx`.
async fn s1(ctx: &DurableContext) -> Result<(), BoxError> {
    ctx.step(|_| async { Ok(1_u32) }).name("s1").await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawned_step_start_precedes_next_operation_multi_thread() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
        s1(&ctx).await?;
        a.await?;
        Ok(())
    })
    .await;
    assert_eq!(names(&result), ["a", "s1"]);
}

#[tokio::test]
async fn spawned_wait_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
        s1(&ctx).await?;
        w.await?;
        Ok(())
    })
    .await;
    assert_eq!(
        names(&result).get(..2),
        Some(&["w".to_owned(), "s1".to_owned()][..])
    );
}

#[tokio::test]
async fn spawned_child_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let c = ctx
            .run_in_child_context(|child| async move {
                Ok(child.step(|_| async { Ok(1_u32) }).name("s").await?)
            })
            .name("c")
            .spawn();
        s1(&ctx).await?;
        c.await?;
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "c"), ["c", "s1"]);
}

/// The boxed future of one item of [`map_m`].
type ItemFuture = Pin<Box<dyn Future<Output = Result<u32, BoxError>> + Send>>;

/// The item body type of [`map_m`].
type ItemBody = fn(DurableContext, u32, usize) -> ItemFuture;

/// The builder [`map_m`] returns.
type MapM = durable::builders::MapBuilder<u32, u32, ItemBody, ItemFuture>;

/// The body of each item of [`map_m`]: a step that returns the item.
fn map_item(child: DurableContext, item: u32, _index: usize) -> ItemFuture {
    Box::pin(async move {
        Ok(child
            .step(move |_| async move { Ok(item) })
            .name("item")
            .await?)
    })
}

/// A map `m` of two items, each running a step.
fn map_m(ctx: &DurableContext) -> MapM {
    ctx.map(vec![1_u32, 2], map_item as ItemBody).name("m")
}

#[tokio::test]
async fn spawned_map_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let m = map_m(&ctx).spawn();
        s1(&ctx).await?;
        m.await?;
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "m"), ["m", "s1"]);
}

#[tokio::test]
async fn spawn_batch_map_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let m = map_m(&ctx).spawn_batch();
        s1(&ctx).await?;
        m.await?;
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "m"), ["m", "s1"]);
}

/// Two unnamed branches that each return a value.
fn two_branches() -> Vec<durable::Branch<u32>> {
    vec![
        durable::Branch::unnamed(|_child| async { Ok(1_u32) }),
        durable::Branch::unnamed(|_child| async { Ok(2_u32) }),
    ]
}

#[tokio::test]
async fn spawned_parallel_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let p = ctx.parallel(two_branches()).name("p").spawn();
        s1(&ctx).await?;
        p.await?;
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "p"), ["p", "s1"]);
}

#[tokio::test]
async fn spawn_batch_parallel_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let p = ctx.parallel(two_branches()).name("p").spawn_batch();
        s1(&ctx).await?;
        p.await?;
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "p"), ["p", "s1"]);
}

#[tokio::test]
async fn spawned_parallel_mixed_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let mut branches = durable::BranchSet::new();
        let handle = branches.add("pb", |_child| async { Ok(1_u32) });
        let pm = ctx.parallel_mixed(branches).name("pm").spawn();
        s1(&ctx).await?;
        pm.await?;
        assert_eq!(handle.into_result()?, 1);
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "pm"), ["pm", "s1"]);
}

#[tokio::test]
async fn spawned_create_callback_start_precedes_next_operation() {
    let runner = durable::test_util::LocalRunner::new().callback_for(
        "cb",
        durable::test_util::CallbackReply::success(&"ok").expect("the reply encodes"),
    );
    let result = run_runner(runner, |_, ctx| async move {
        let cb = ctx.create_callback::<String>().name("cb").spawn();
        s1(&ctx).await?;
        let cb = cb.await?;
        assert_eq!(cb.result().await?, "ok");
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "cb"), ["cb", "s1"]);
}

#[tokio::test]
async fn spawned_wait_for_callback_start_precedes_next_operation() {
    let runner = durable::test_util::LocalRunner::new().callback_for(
        "wcb",
        durable::test_util::CallbackReply::success(&"ok").expect("the reply encodes"),
    );
    let result = run_runner(runner, |_, ctx| async move {
        let wcb = ctx
            .wait_for_callback::<String, _, _>(|_, _id| async { Ok(()) })
            .name("wcb")
            .spawn();
        s1(&ctx).await?;
        assert_eq!(wcb.await?, "ok");
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "wcb"), ["wcb", "s1"]);
}

#[tokio::test]
async fn spawned_with_retry_start_precedes_next_operation() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let wr = ctx
            .with_retry(
                |child| async move { Ok(child.step(|_| async { Ok(1_u32) }).name("in").await?) },
            )
            .name("wr")
            .spawn();
        s1(&ctx).await?;
        wr.await?;
        Ok(())
    })
    .await;
    assert_eq!(order(&result, "wr"), ["wr", "s1"]);
}

#[tokio::test]
async fn spawned_step_start_hook_precedes_next_operation_hook() {
    let log = Log::default();
    let options = Options::builder()
        .plugin(HookLog(log.clone()))
        .build()
        .expect("valid options");
    run_local(options, |_, ctx| async move {
        let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
        s1(&ctx).await?;
        a.await?;
        Ok(())
    })
    .await;
    let hooks: Vec<String> = log
        .entries()
        .into_iter()
        .filter(|e| e == "hook a" || e == "hook s1")
        .collect();
    assert_eq!(hooks, ["hook a", "hook s1"]);
}

/// The wire ID of a positional ID: its SHA-256 digest in lowercase hex.
fn wire(positional: &str) -> String {
    use sha2::Digest as _;
    use std::fmt::Write as _;
    sha2::Sha256::digest(positional.as_bytes())
        .iter()
        .fold(String::new(), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

#[tokio::test]
async fn child_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.run_in_child_context(|child| async move {
            Ok(child.step(|_| async { Ok(1_u32) }).name("c.s").await?)
        })
        .name("c")
        .await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook c") < log.index("response c"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn map_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.map(vec![1_u32], |_child, item, _| async move { Ok(item) })
            .name("m")
            .await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook m") < log.index("response m"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn spawned_child_replays_without_second_start() {
    let result = run_local(Options::default(), |_, ctx| async move {
        let c = ctx
            .run_in_child_context(|child| async move {
                child.wait(Duration::from_secs(1)).name("cw").await?;
                Ok(1_u32)
            })
            .name("c")
            .spawn();
        s1(&ctx).await?;
        c.await?;
        Ok(())
    })
    .await;
    assert_eq!(result.invocation_count(), 2);
    let c: Vec<_> = result
        .operations()
        .iter()
        .filter(|op| op.name() == Some("c"))
        .collect();
    assert_eq!(c.len(), 1, "one record for c");
    let c_id = c.first().expect("c is recorded").id().to_owned();
    let starts = result
        .history_events()
        .iter()
        .filter(|event| {
            event.id() == Some(c_id.as_str())
                && event.event_type() == Some(&durable::lambda_types::EventType::ContextStarted)
        })
        .count();
    assert_eq!(starts, 1, "the second invocation queues no START for c");
}

#[tokio::test]
async fn map_item_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.map(vec![1_u32], |_child, item, _| async move { Ok(item) })
            .name("m")
            .await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook map-item-0") < log.index("response map-item-0"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn parallel_branch_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.parallel(vec![durable::Branch::new("b0", |_child| async {
            Ok(1_u32)
        })])
        .name("p")
        .await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook b0") < log.index("response b0"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn parallel_mixed_branch_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        let mut branches = durable::BranchSet::new();
        let _handle = branches.add("pb0", |_child| async { Ok(1_u32) });
        ctx.parallel_mixed(branches).name("pm").await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook pb0") < log.index("response pb0"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn wait_for_callback_start_hook_fires_before_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        let value = ctx
            .wait_for_callback::<String, _, _>(|_, _id| async { Ok(()) })
            .name("wcb")
            .await?;
        Ok(value)
    })
    .await;
    assert_eq!(status, "PENDING");
    assert!(
        log.index("hook wcb") < log.index("response wcb"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn child_outcome_waits_for_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.run_in_child_context(|child| async move {
            Ok(child.step(|_| async { Ok(1_u32) }).name("c.s").await?)
        })
        .name("c")
        .await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("response c") < log.index("send SUCCEED c"),
        "{:?}",
        log.entries()
    );
}

#[tokio::test]
async fn map_item_outcome_waits_for_start_response() {
    let log = Log::default();
    let status = invoke_unbatched(&log, |_, ctx| async move {
        ctx.map(vec![1_u32], |_child, item, _| async move { Ok(item) })
            .name("m")
            .await?;
        Ok(String::new())
    })
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("response map-item-0") < log.index("send SUCCEED map-item-0"),
        "{:?}",
        log.entries()
    );
}

/// The initial state of test 23: the execution and a `STARTED` step `a`,
/// the first operation of the root.
fn started_step_payload() -> InvocationPayload {
    serde_json::from_value(serde_json::json!({
        "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:start-order",
        "CheckpointToken": "t-0",
        "InitialExecutionState": {
            "Operations": [
                {
                    "Id": "root",
                    "Type": "Execution",
                    "Status": "STARTED",
                    "ExecutionDetails": { "InputPayload": "{}" }
                },
                {
                    "Id": wire("1"),
                    "Type": "Step",
                    "SubType": "Step",
                    "Name": "a",
                    "Status": "STARTED",
                    "StepDetails": {}
                }
            ]
        },
    }))
    .expect("a valid payload")
}

#[tokio::test(flavor = "current_thread")]
async fn spawned_step_replay_hook_precedes_post_spawn_marker() {
    let log = Log::default();
    let handler_log = log.clone();
    let status = invoke_with(
        Transport::new(log.clone()),
        started_step_payload(),
        false,
        HookLog(log.clone()),
        move |_, ctx| {
            let log = handler_log.clone();
            async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
                log.push("spawned".to_owned());
                a.await?;
                Ok(String::new())
            }
        },
    )
    .await;
    assert_eq!(status, "SUCCEEDED");
    assert!(
        log.index("hook a") < log.index("spawned"),
        "{:?}",
        log.entries()
    );
    assert!(!log.contains("send START a"), "{:?}", log.entries());
}

#[tokio::test]
#[expect(clippy::disallowed_methods)] // reason: the test calls `.spawn()` from a task the SDK did not start
async fn spawn_from_unblessed_task_is_rejected_before_start() {
    let log = Log::default();
    let handler_log = log.clone();
    let status = invoke_unbatched(&log, move |_, ctx| {
        let log = handler_log.clone();
        async move {
            let body_log = log.clone();
            let outcome = tokio::spawn(async move {
                ctx.step(move |_| {
                    let body_log = body_log.clone();
                    async move {
                        body_log.push("body x".to_owned());
                        Ok(1_u32)
                    }
                })
                .name("x")
                .spawn()
                .await
            })
            .await
            .expect("the bare task completes");
            let err = outcome.expect_err("an unblessed caller is refused");
            let mut chain = err.to_string();
            let mut source = std::error::Error::source(&err);
            while let Some(cause) = source {
                chain = format!("{chain}: {cause}");
                source = cause.source();
            }
            log.push(format!("error {chain}"));
            Ok(String::new())
        }
    })
    .await;
    // The refused `.spawn()` fails the execution with a context usage error.
    assert_eq!(status, "FAILED");
    let entries = log.entries();
    assert!(
        entries
            .iter()
            .any(|e| e.starts_with("error ") && e.contains("tokio::spawn")),
        "{entries:?}"
    );
    for absent in ["hook x", "send START x", "body x"] {
        assert!(!log.contains(absent), "{absent}: {entries:?}");
    }
}

/// The plugin of test 28: logs `hook <name>` in `on_operation_start`.
/// The test's body logs `body_started` and notifies `started`.
async fn body_before_response<F, Fut>(target: &str, handler: F)
where
    F: Fn(Log, Arc<tokio::sync::Notify>, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, BoxError>> + Send + 'static,
{
    let log = Log::default();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(0));
    let started = Arc::new(tokio::sync::Notify::new());
    let handler = Arc::new(handler);
    let invocation = {
        let log = log.clone();
        let started = Arc::clone(&started);
        let transport = Transport::holding(log.clone(), target, Arc::clone(&semaphore));
        let factory = HookLog(log.clone());
        async move {
            invoke_with(transport, payload(), false, factory, move |_, ctx| {
                handler(log.clone(), Arc::clone(&started), ctx)
            })
            .await
        }
    };
    let observe = async {
        let notified = tokio::time::timeout(Duration::from_secs(2), started.notified()).await;
        let responded = log.contains(&format!("response {target}"));
        semaphore.add_permits(1);
        (notified.is_ok(), responded)
    };
    let (status, (notified, responded)) = tokio::join!(invocation, observe);
    assert!(notified, "{target}: the body started: {:?}", log.entries());
    assert!(
        !responded,
        "{target}: the body started before the START response: {:?}",
        log.entries()
    );
    assert_eq!(status, "SUCCEEDED", "{target}: {:?}", log.entries());
    let hook = log.index(&format!("hook {target}"));
    let body = log.index("body_started");
    let response = log.index(&format!("response {target}"));
    assert!(
        hook < body && body < response,
        "{target}: {:?}",
        log.entries()
    );
}

/// The first statement of a body in test 28.
fn body_started(log: &Log, started: &tokio::sync::Notify) {
    log.push("body_started".to_owned());
    started.notify_one();
}

#[tokio::test]
async fn child_body_starts_before_start_response() {
    body_before_response("c", |log, started, ctx| async move {
        ctx.run_in_child_context(move |_child| async move {
            body_started(&log, &started);
            Ok(1_u32)
        })
        .name("c")
        .await?;
        Ok(String::new())
    })
    .await;
}

#[tokio::test]
async fn map_body_starts_before_start_response() {
    body_before_response("m", |log, started, ctx| async move {
        ctx.map(vec![1_u32], move |_child, item, _| {
            let log = log.clone();
            let started = Arc::clone(&started);
            async move {
                body_started(&log, &started);
                Ok(item)
            }
        })
        .name("m")
        .await?;
        Ok(String::new())
    })
    .await;
}

#[tokio::test]
async fn map_item_body_starts_before_start_response() {
    body_before_response("map-item-0", |log, started, ctx| async move {
        ctx.map(vec![1_u32], move |_child, item, _| {
            let log = log.clone();
            let started = Arc::clone(&started);
            async move {
                body_started(&log, &started);
                Ok(item)
            }
        })
        .name("m")
        .await?;
        Ok(String::new())
    })
    .await;
}

#[tokio::test]
async fn parallel_branch_body_starts_before_start_response() {
    body_before_response("b0", |log, started, ctx| async move {
        ctx.parallel(vec![durable::Branch::new("b0", move |_child| async move {
            body_started(&log, &started);
            Ok(1_u32)
        })])
        .name("p")
        .await?;
        Ok(String::new())
    })
    .await;
}

#[tokio::test]
async fn parallel_mixed_branch_body_starts_before_start_response() {
    body_before_response("pb0", |log, started, ctx| async move {
        let mut branches = durable::BranchSet::new();
        let _handle = branches.add("pb0", move |_child| async move {
            body_started(&log, &started);
            Ok(1_u32)
        });
        ctx.parallel_mixed(branches).name("pm").await?;
        Ok(String::new())
    })
    .await;
}

/// A plugin that logs `operation <name>` in `on_operation_start` and
/// `attempt <name>` in `on_attempt_start`.
#[derive(Clone)]
struct AttemptLog(Log);

impl Plugin for AttemptLog {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.0
            .push(format!("operation {}", info.name().unwrap_or("-")));
        Ok(None)
    }

    fn on_attempt_start(&self, info: &durable::plugin::AttemptInfo) -> HookWork {
        self.0.push(format!(
            "attempt {}",
            info.operation().name().unwrap_or("-")
        ));
        Ok(None)
    }
}

impl PluginFactory for AttemptLog {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn spawned_step_attempt_hook_stays_on_operation_task() {
    let log = Log::default();
    let handler_log = log.clone();
    let options = Options::builder()
        .plugin(AttemptLog(log.clone()))
        .build()
        .expect("valid options");
    run_local(options, move |_, ctx| {
        let log = handler_log.clone();
        async move {
            let body_log = log.clone();
            let a = ctx
                .step(move |_| {
                    let body_log = body_log.clone();
                    async move {
                        body_log.push("body a".to_owned());
                        Ok(1_u32)
                    }
                })
                .name("a")
                .spawn();
            log.push("spawn returned".to_owned());
            a.await?;
            Ok(())
        }
    })
    .await;
    let entries = log.entries();
    let at = |entry: &str| log.index(entry);
    assert!(
        at("operation a") < at("spawn returned")
            && at("spawn returned") < at("attempt a")
            && at("attempt a") < at("body a"),
        "{entries:?}"
    );
}

/// The child-backed "queue, then hook" operations of test 31, and the
/// name of the target operation whose hook blocks.
#[derive(Debug, Clone, Copy)]
enum ChildBacked {
    Child,
    MapParent,
    MapItem,
    WaitForCallback,
}

impl ChildBacked {
    fn target(self) -> &'static str {
        match self {
            Self::Child => "c",
            Self::MapParent => "m",
            Self::MapItem => "map-item-0",
            Self::WaitForCallback => "wcb",
        }
    }

    fn expected_status(self) -> &'static str {
        match self {
            Self::WaitForCallback => "PENDING",
            _ => "SUCCEEDED",
        }
    }

    async fn run(self, ctx: DurableContext) -> Result<String, BoxError> {
        match self {
            Self::Child => {
                ctx.run_in_child_context(|child| async move {
                    Ok(child.step(|_| async { Ok(1_u32) }).name("s").await?)
                })
                .name("c")
                .await?;
            }
            Self::MapParent | Self::MapItem => {
                ctx.map(vec![1_u32], |_child, item, _| async move { Ok(item) })
                    .name("m")
                    .await?;
            }
            Self::WaitForCallback => {
                return Ok(ctx
                    .wait_for_callback::<String, _, _>(|_, _id| async { Ok(()) })
                    .name("wcb")
                    .await?);
            }
        }
        Ok(String::new())
    }
}

/// A plugin whose `on_operation_start` of `target` blocks, in
/// `block_in_place`, until the test releases it or five seconds pass.
#[derive(Clone)]
struct BlockingTarget {
    target: &'static str,
    hook: BlockingHook,
}

impl Plugin for BlockingTarget {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        if info.name() == Some(self.target) {
            tokio::task::block_in_place(|| {
                let (gate, condvar) = &*self.hook.0;
                let mut state = gate.lock().unwrap();
                state.entered = true;
                let _unused = condvar
                    .wait_timeout_while(state, Duration::from_secs(5), |s| !s.released)
                    .unwrap();
            });
        }
        Ok(None)
    }
}

impl PluginFactory for BlockingTarget {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(clippy::disallowed_methods)] // reason: the invocation runs beside the test's observer
async fn child_backed_queue_then_hook_blocks_start_send_until_hook_dispatch() {
    let mut failures = Vec::new();
    for op in [
        ChildBacked::Child,
        ChildBacked::MapParent,
        ChildBacked::MapItem,
        ChildBacked::WaitForCallback,
    ] {
        for batching in [false, true] {
            let log = Log::default();
            let hook = BlockingHook::default();
            let target = op.target();
            let invocation = tokio::spawn({
                let log = log.clone();
                let factory = BlockingTarget {
                    target,
                    hook: hook.clone(),
                };
                async move { invoke(&log, batching, factory, move |_, ctx| op.run(ctx)).await }
            });
            hook.entered().await;
            tokio::time::sleep(Duration::from_millis(200)).await;
            let start = format!("send START {target}");
            let sent_while_blocked = log.contains(&start);
            hook.release();
            let status = invocation.await.expect("the invocation task completes");
            let sent_after = log.contains(&start);
            if sent_while_blocked || !sent_after || status != op.expected_status() {
                failures.push(format!(
                    "{op:?} batching={batching}: sent while blocked={sent_while_blocked}, \
                     sent after={sent_after}, status={status}, log={:?}",
                    log.entries()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "failed cases: {failures:#?}");
}
