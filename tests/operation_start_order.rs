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

/// A transport that logs `send START <name>` when it receives a `START`,
/// and `response <name>` when that request's future completes, after one
/// yield. A callback `START` is answered with the callback's record.
struct Transport(Log);

impl ExecutionClient for Transport {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
        let starts: Vec<&OperationUpdate> = updates
            .iter()
            .filter(|u| u.action() == &OperationAction::Start)
            .collect();
        let names: Vec<String> = starts
            .iter()
            .map(|u| u.name().unwrap_or("-").to_owned())
            .collect();
        for name in &names {
            self.0.push(format!("send START {name}"));
        }
        let records: Vec<Operation> = starts
            .iter()
            .filter(|u| u.r#type() == &OperationType::Callback)
            .map(|u| callback_record(u.id()))
            .collect();
        let log = self.0.clone();
        Box::pin(async move {
            tokio::task::yield_now().await;
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
    let service = durable::wrap(
        handler,
        Options::builder()
            .execution_client(Arc::new(Transport(log.clone())))
            .checkpoint_batching(batching)
            .plugin(factory)
            .build()
            .expect("valid options"),
    );
    let event = lambda_runtime::LambdaEvent::new(payload(), lambda_runtime::Context::default());
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
