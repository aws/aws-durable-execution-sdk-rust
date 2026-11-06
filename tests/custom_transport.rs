//! A caller-supplied checkpoint transport drives `wrap` end to end.
//!
//! The transport here uses only items of `aws_durable_execution_sdk`: the
//! public `ExecutionClient` trait, its error and output types, and the
//! service model types of `lambda_types`. It implements no `Debug`.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::error::Error as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;

use aws_durable_execution_sdk::lambda_types::{
    CallbackDetails, DateTime, ExecutionDetails, Operation, OperationAction, OperationStatus,
    OperationType, OperationUpdate,
};
use aws_durable_execution_sdk::plugin::{
    HookFuture, InvocationEndInfo, InvocationInfo, Plugin, PluginFactory,
};
use aws_durable_execution_sdk::{
    BoxError, CheckpointOutput, DurableContext, ExecutionClient, ExecutionClientError,
    ExecutionClientErrorScope, GetStateOutput, InvocationPayload, InvocationStatus, Options,
    lambda_runtime,
};

/// A transport that records every update and answers from a script.
///
/// `checkpoint` returns tokens `t-1`, `t-2` and so on. When
/// `checkpoint_error` is set, it returns that error for the first request
/// that carries an update with the action `fail_on` (any request when
/// `fail_on` is `None`), and succeeds for every other request. `get_state`
/// returns one page with the execution operation and no marker, or
/// `state_error` when it is set.
struct ScriptedTransport {
    updates: Mutex<Vec<OperationUpdate>>,
    tokens: Mutex<u32>,
    checkpoint_error: Option<ExecutionClientError>,
    fail_on: Option<OperationAction>,
    failed_once: Mutex<bool>,
    state_error: Option<ExecutionClientError>,
}

impl ScriptedTransport {
    fn new() -> Self {
        Self {
            updates: Mutex::new(Vec::new()),
            tokens: Mutex::new(0),
            checkpoint_error: None,
            fail_on: None,
            failed_once: Mutex::new(false),
            state_error: None,
        }
    }

    fn failing_checkpoint(error: ExecutionClientError, fail_on: Option<OperationAction>) -> Self {
        Self {
            checkpoint_error: Some(error),
            fail_on,
            ..Self::new()
        }
    }

    fn failing_state(error: ExecutionClientError) -> Self {
        Self {
            state_error: Some(error),
            ..Self::new()
        }
    }

    fn recorded(&self) -> Vec<OperationUpdate> {
        self.updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl ExecutionClient for ScriptedTransport {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
        let matches = self
            .fail_on
            .as_ref()
            .is_none_or(|action| updates.iter().any(|u| u.action() == action));
        self.updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(updates);
        let mut failed_once = self
            .failed_once
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = match &self.checkpoint_error {
            Some(error) if matches && !*failed_once => {
                *failed_once = true;
                Err(error.clone())
            }
            _ => {
                let mut tokens = self
                    .tokens
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *tokens += 1;
                Ok(CheckpointOutput::new(
                    Some(format!("t-{tokens}")),
                    Vec::new(),
                    None,
                ))
            }
        };
        Box::pin(async move { result })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        _marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
        let result = match &self.state_error {
            Some(error) => Err(error.clone()),
            None => Ok(GetStateOutput::new(vec![execution_operation()], None)),
        };
        Box::pin(async move { result })
    }
}

/// The execution operation, built from `lambda_types` alone.
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

/// A durable invocation envelope, with a `NextMarker` when `paginated`.
fn payload(paginated: bool) -> InvocationPayload {
    let mut state = serde_json::json!({
        "Operations": [{
            "Id": "root",
            "Type": "Execution",
            "Status": "STARTED",
            "ExecutionDetails": { "InputPayload": "{}" }
        }]
    });
    if paginated {
        state["NextMarker"] = serde_json::json!("page-2");
    }
    serde_json::from_value::<InvocationPayload>(serde_json::json!({
        "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:custom",
        "CheckpointToken": "t-0",
        "InitialExecutionState": state,
    }))
    .expect("a valid payload")
}

/// Runs a handler with one step returning `7` over `transport`, and
/// returns the response as JSON. Records whether the handler ran.
async fn invoke(
    transport: Arc<ScriptedTransport>,
    paginated: bool,
) -> (Result<serde_json::Value, lambda_runtime::Error>, bool) {
    use std::sync::atomic::{AtomicBool, Ordering};

    let ran = Arc::new(AtomicBool::new(false));
    let handler_ran = Arc::clone(&ran);
    let service = aws_durable_execution_sdk::wrap(
        move |_event: serde_json::Value, ctx: DurableContext| {
            handler_ran.store(true, Ordering::SeqCst);
            async move {
                let value: u32 = ctx.step(|_| async { Ok(7_u32) }).name("seven").await?;
                Ok::<u32, BoxError>(value)
            }
        },
        Options::builder()
            .execution_client(transport)
            .build()
            .expect("execution_client alone is valid"),
    );
    let event =
        lambda_runtime::LambdaEvent::new(payload(paginated), lambda_runtime::Context::default());
    let result = service(event)
        .await
        .map(|response| serde_json::to_value(&response).expect("the response serializes"));
    (result, ran.load(Ordering::SeqCst))
}

fn status(response: &serde_json::Value) -> Option<&str> {
    response.get("Status").and_then(serde_json::Value::as_str)
}

fn error_field<'a>(response: &'a serde_json::Value, field: &str) -> Option<&'a str> {
    response
        .get("Error")
        .and_then(|e| e.get(field))
        .and_then(serde_json::Value::as_str)
}

#[tokio::test]
async fn custom_transport_records_a_step() {
    let transport = Arc::new(ScriptedTransport::new());
    let (result, _) = invoke(Arc::clone(&transport), false).await;

    let response = result.expect("the invocation succeeds");
    assert_eq!(status(&response), Some("SUCCEEDED"), "{response}");
    assert_eq!(
        response.get("Result").and_then(serde_json::Value::as_str),
        Some("7")
    );
    assert!(
        transport.recorded().iter().any(|u| {
            u.r#type() == &OperationType::Step && u.action() == &OperationAction::Succeed
        }),
        "the transport recorded the step's SUCCEED: {:?}",
        transport.recorded()
    );
}

#[tokio::test]
async fn execution_scoped_checkpoint_error_fails_the_execution() {
    let transport = Arc::new(ScriptedTransport::failing_checkpoint(
        ExecutionClientError::new(ExecutionClientErrorScope::Execution, "rejected"),
        Some(OperationAction::Succeed),
    ));
    let (result, _) = invoke(Arc::clone(&transport), false).await;

    let response = result.expect("an execution-scoped error returns an envelope");
    assert_eq!(status(&response), Some("FAILED"), "{response}");
    assert_eq!(
        error_field(&response, "ErrorType"),
        Some("CheckpointUnrecoverableExecutionError")
    );
    assert_eq!(
        error_field(&response, "ErrorMessage"),
        Some("[Unrecoverable Execution] Checkpoint failed: rejected")
    );
    let recorded = transport.recorded();
    assert!(
        recorded
            .iter()
            .all(|u| u.action() != &OperationAction::Fail),
        "no FAIL update is sent: {recorded:?}"
    );
    let last = recorded.last().expect("the transport received updates");
    assert_eq!(last.r#type(), &OperationType::Step);
    assert_eq!(last.action(), &OperationAction::Succeed, "{recorded:?}");
}

#[tokio::test]
async fn invocation_scoped_checkpoint_error_fails_the_invocation() {
    let transport = Arc::new(ScriptedTransport::failing_checkpoint(
        ExecutionClientError::new(ExecutionClientErrorScope::Invocation, "throttled"),
        None,
    ));
    let (result, _) = invoke(transport, false).await;

    assert!(
        result.is_err(),
        "an invocation-scoped error fails the invocation: {result:?}"
    );
}

#[tokio::test]
async fn execution_scoped_state_read_error_returns_failed() {
    let transport = Arc::new(ScriptedTransport::failing_state(ExecutionClientError::new(
        ExecutionClientErrorScope::Execution,
        "denied",
    )));
    let (result, ran) = invoke(transport, true).await;

    let response = result.expect("an execution-scoped state read error returns an envelope");
    assert_eq!(status(&response), Some("FAILED"), "{response}");
    assert_eq!(
        error_field(&response, "ErrorType"),
        Some("DurableExecutionClientError")
    );
    assert_eq!(error_field(&response, "ErrorMessage"), Some("denied"));
    assert!(!ran, "the handler must not run");
}

#[tokio::test]
async fn invocation_scoped_state_read_error_fails_the_invocation() {
    let transport = Arc::new(ScriptedTransport::failing_state(ExecutionClientError::new(
        ExecutionClientErrorScope::Invocation,
        "unavailable",
    )));
    let (result, ran) = invoke(transport, true).await;

    assert!(
        result.is_err(),
        "an invocation-scoped error fails the invocation: {result:?}"
    );
    assert!(!ran, "the handler must not run");
}

/// A test error type whose identity survives the error's `Arc` source.
#[derive(Debug)]
struct Probe;

impl std::fmt::Display for Probe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("probe")
    }
}

impl std::error::Error for Probe {}

#[tokio::test]
async fn custom_transport_source_survives_clone() {
    let error =
        ExecutionClientError::with_source(ExecutionClientErrorScope::Invocation, "m", Probe);
    let transport = Arc::new(ScriptedTransport::failing_checkpoint(error, None));
    let (result, _) = invoke(Arc::clone(&transport), false).await;

    assert!(
        result.is_err(),
        "an invocation-scoped error fails the invocation: {result:?}"
    );
    let clone = transport
        .checkpoint_error
        .clone()
        .expect("the transport holds its error");
    assert!(
        clone
            .source()
            .and_then(|s| s.downcast_ref::<Probe>())
            .is_some(),
        "the clone's source downcasts to the original error type"
    );
}

// A failed checkpoint write is recorded once, and the SDK sends nothing
// after it.

/// Decides whether a request fails: the error, or `None` to accept it.
type Reject = dyn Fn(&[OperationUpdate]) -> Option<ExecutionClientError> + Send + Sync;

/// A transport that records every request and answers each one after a
/// 50 ms `tokio::time::sleep`, with the error `reject` returns for it, or
/// with a token. A `START` of a callback is answered with the callback's
/// operation and its ID. With `state_error`, an accepted request that
/// carries a `SUCCEED` gets a pagination marker, and every `get_state`
/// fails with `state_error`. Each `get_state` is recorded with its marker
/// in `state_reads`.
struct Answering {
    requests: Mutex<Vec<Vec<OperationUpdate>>>,
    state_reads: Mutex<Vec<Option<String>>>,
    tokens: Mutex<u32>,
    reject: Box<Reject>,
    state_error: Option<ExecutionClientError>,
}

impl Answering {
    fn new(
        reject: impl Fn(&[OperationUpdate]) -> Option<ExecutionClientError> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::new(Vec::new()),
            state_reads: Mutex::new(Vec::new()),
            tokens: Mutex::new(0),
            reject: Box::new(reject),
            state_error: None,
        })
    }

    /// Fails the request that carries `name`'s update with `action` with
    /// an error of `scope`, and accepts every other one.
    fn failing(
        name: &'static str,
        action: OperationAction,
        scope: ExecutionClientErrorScope,
    ) -> Arc<Self> {
        Self::new(move |updates| {
            has(updates, name, &action).then(|| ExecutionClientError::new(scope, "rejected"))
        })
    }

    fn accepting() -> Arc<Self> {
        Self::new(|_| None)
    }

    fn requests(&self) -> Vec<Vec<OperationUpdate>> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn updates(&self) -> Vec<OperationUpdate> {
        self.requests().into_iter().flatten().collect()
    }

    /// The updates of the requests the transport accepted.
    fn accepted(&self) -> Vec<OperationUpdate> {
        self.requests()
            .into_iter()
            .filter(|request| (self.reject)(request).is_none())
            .flatten()
            .collect()
    }

    /// Waits until the transport received a request that `matches`.
    async fn until_received(&self, matches: impl Fn(&[OperationUpdate]) -> bool) {
        while !self.requests().iter().any(|request| matches(request)) {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    }

    /// Asserts that the last request carries `name`'s update with
    /// `action`: no request follows it.
    fn assert_last_request_has(&self, name: &str, action: &OperationAction) {
        let requests = self.requests();
        let last = requests.last().expect("a request was sent");
        assert!(has(last, name, action), "{requests:?}");
    }
}

impl ExecutionClient for Answering {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(updates.clone());
        let result = if let Some(error) = (self.reject)(&updates) {
            Err(error)
        } else {
            let mut tokens = self
                .tokens
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *tokens += 1;
            let operations = updates
                .iter()
                .filter(|u| {
                    u.r#type() == &OperationType::Callback && u.action() == &OperationAction::Start
                })
                .map(started_callback)
                .collect();
            let marker = (self.state_error.is_some()
                && updates
                    .iter()
                    .any(|u| u.action() == &OperationAction::Succeed))
            .then(|| "page-2".to_owned());
            Ok(CheckpointOutput::new(
                Some(format!("t-{tokens}")),
                operations,
                marker,
            ))
        };
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            result
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
        self.state_reads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(marker.map(ToOwned::to_owned));
        let result = match &self.state_error {
            Some(error) => Err(error.clone()),
            None => Ok(GetStateOutput::new(Vec::new(), None)),
        };
        Box::pin(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            result
        })
    }
}

/// The `STARTED` operation of a callback `START`, with an ID.
fn started_callback(update: &OperationUpdate) -> Operation {
    let mut builder = Operation::builder()
        .id(update.id())
        .r#type(OperationType::Callback)
        .status(OperationStatus::Started)
        .start_timestamp(DateTime::from_secs(0))
        .callback_details(CallbackDetails::builder().callback_id("cb-1").build());
    if let Some(name) = update.name() {
        builder = builder.name(name);
    }
    builder
        .build()
        .expect("every required Operation field is set")
}

/// Whether `updates` hold `name`'s update with `action`.
fn has(updates: &[OperationUpdate], name: &str, action: &OperationAction) -> bool {
    updates
        .iter()
        .any(|u| u.name() == Some(name) && u.action() == action)
}

/// The status and the error type of one `on_invocation_end`.
type End = (InvocationStatus, Option<String>);

/// Records every `on_invocation_end`.
#[derive(Clone, Default)]
struct EndRecorder(Arc<Mutex<Vec<End>>>);

impl EndRecorder {
    fn ends(&self) -> Vec<End> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl PluginFactory for EndRecorder {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl Plugin for EndRecorder {
    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((
                info.status(),
                info.execution_error()
                    .and_then(aws_durable_execution_sdk::WireError::error_type)
                    .map(str::to_owned),
            ));
        Box::pin(std::future::ready(Ok(())))
    }
}

/// A durable invocation envelope whose state holds the execution and
/// `operations`.
fn payload_with(operations: Vec<serde_json::Value>) -> InvocationPayload {
    let mut all = vec![serde_json::json!({
        "Id": "root",
        "Type": "Execution",
        "Status": "STARTED",
        "ExecutionDetails": { "InputPayload": "{}" }
    })];
    all.extend(operations);
    serde_json::from_value::<InvocationPayload>(serde_json::json!({
        "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:custom",
        "CheckpointToken": "t-0",
        "InitialExecutionState": { "Operations": all },
    }))
    .expect("a valid payload")
}

/// The step records that `accepted`, the updates the service accepted,
/// create, as the service would hand them to the next invocation.
fn operations_from(accepted: &[OperationUpdate]) -> Vec<serde_json::Value> {
    let mut operations: Vec<serde_json::Value> = Vec::new();
    for update in accepted {
        let status = match update.action() {
            OperationAction::Start => "STARTED",
            OperationAction::Succeed => "SUCCEEDED",
            OperationAction::Fail => "FAILED",
            _ => continue,
        };
        let mut details = serde_json::json!({ "Attempt": 1 });
        if let Some(payload) = update.payload() {
            details["Result"] = serde_json::json!(payload);
        }
        if let Some(error) = update.error() {
            details["Error"] = serde_json::json!({
                "ErrorType": error.error_type(),
                "ErrorMessage": error.error_message(),
            });
        }
        let record = serde_json::json!({
            "Id": update.id(),
            "Type": update.r#type().as_str(),
            "SubType": update.sub_type(),
            "Name": update.name(),
            "Status": status,
            "StepDetails": details,
        });
        operations.retain(|op| op["Id"] != update.id());
        operations.push(record);
    }
    operations
}

/// Runs `handler` once over `transport`, with the recording plugin, and
/// with `checkpoint_batching(false)` when `unbatched`.
async fn invoke_handler<F, Fut>(
    transport: &Arc<Answering>,
    unbatched: bool,
    operations: Vec<serde_json::Value>,
    handler: F,
) -> (
    Result<serde_json::Value, lambda_runtime::Error>,
    EndRecorder,
)
where
    F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, BoxError>> + Send,
{
    let recorder = EndRecorder::default();
    let mut options = Options::builder()
        .execution_client(Arc::clone(transport) as Arc<dyn ExecutionClient>)
        .plugin(recorder.clone());
    if unbatched {
        options = options.checkpoint_batching(false);
    }
    let service = aws_durable_execution_sdk::wrap(handler, options.build().expect("valid options"));
    let event = lambda_runtime::LambdaEvent::new(
        payload_with(operations),
        lambda_runtime::Context::default(),
    );
    let result = service(event)
        .await
        .map(|response| serde_json::to_value(&response).expect("the response serializes"));
    (result, recorder)
}

const EXECUTION_FAILED: &str = "CheckpointUnrecoverableExecutionError";
const INVOCATION_FAILED: &str = "CheckpointUnrecoverableInvocationError";

/// Asserts a `FAILED` answer with `CheckpointUnrecoverableExecutionError`,
/// and a `Failed` end with it.
fn assert_execution_failed(
    result: &Result<serde_json::Value, lambda_runtime::Error>,
    recorder: &EndRecorder,
) {
    let response = result
        .as_ref()
        .expect("an execution-scoped failure returns an envelope");
    assert_eq!(status(response), Some("FAILED"), "{response}");
    assert_eq!(error_field(response, "ErrorType"), Some(EXECUTION_FAILED));
    assert_eq!(
        recorder.ends(),
        [(InvocationStatus::Failed, Some(EXECUTION_FAILED.to_owned()))]
    );
}

/// Asserts a failed invocation with `CheckpointUnrecoverableInvocationError`,
/// and a `Retrying` end with it.
fn assert_invocation_failed(
    result: &Result<serde_json::Value, lambda_runtime::Error>,
    recorder: &EndRecorder,
) {
    assert!(result.is_err(), "the invocation fails: {result:?}");
    assert_eq!(
        recorder.ends(),
        [(
            InvocationStatus::Retrying,
            Some(INVOCATION_FAILED.to_owned())
        )]
    );
}

/// Runs a handler whose child body spawns step `orphan`, waits until the
/// transport received `orphan`'s `SUCCEED`, and returns, and that then
/// runs step `next`.
async fn orphan_then_next(
    transport: &Arc<Answering>,
) -> (
    Result<serde_json::Value, lambda_runtime::Error>,
    EndRecorder,
) {
    let seen = Arc::clone(transport);
    invoke_handler(transport, true, Vec::new(), move |_event, ctx| {
        let seen = Arc::clone(&seen);
        async move {
            ctx.run_in_child_context(move |child| async move {
                drop(child.step(|_| async { Ok(1_u32) }).name("orphan").spawn());
                seen.until_received(|r| has(r, "orphan", &OperationAction::Succeed))
                    .await;
                Ok(0_u32)
            })
            .name("c")
            .await?;
            ctx.step(|_| async { Ok(2_u32) }).name("next").await?;
            Ok("done".to_owned())
        }
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn stopped_child_step_write_failure_fails_execution_unbatched() {
    let transport = Answering::failing(
        "orphan",
        OperationAction::Succeed,
        ExecutionClientErrorScope::Execution,
    );
    let (result, recorder) = orphan_then_next(&transport).await;
    assert_execution_failed(&result, &recorder);
    transport.assert_last_request_has("orphan", &OperationAction::Succeed);
}

/// Runs `combine` over steps `a` and `b`, whose bodies return in that
/// order: `b`'s body waits for `a`'s, so `b`'s `SUCCEED` is queued
/// behind `a`'s.
async fn ordered_pair<C, CFut>(
    transport: &Arc<Answering>,
    combine: C,
) -> (
    Result<serde_json::Value, lambda_runtime::Error>,
    EndRecorder,
)
where
    C: Fn(
            DurableContext,
            aws_durable_execution_sdk::DurableFuture<u32>,
            aws_durable_execution_sdk::DurableFuture<u32>,
        ) -> CFut
        + Send
        + Sync
        + 'static,
    CFut: Future<Output = Result<u32, aws_durable_execution_sdk::OperationError>> + Send,
{
    let combine = Arc::new(combine);
    invoke_handler(transport, true, Vec::new(), move |_event, ctx| {
        let combine = Arc::clone(&combine);
        async move {
            let order = Arc::new(tokio::sync::Notify::new());
            let a_done = Arc::clone(&order);
            let a = ctx
                .step(move |_| {
                    let a_done = Arc::clone(&a_done);
                    async move {
                        a_done.notify_one();
                        Ok(1_u32)
                    }
                })
                .name("a")
                .future();
            let b = ctx
                .step(move |_| {
                    let after_a = Arc::clone(&order);
                    async move {
                        after_a.notified().await;
                        Ok(2_u32)
                    }
                })
                .name("b")
                .future();
            let winner = combine(ctx.clone(), a, b).await?;
            Ok(winner.to_string())
        }
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn race_loser_write_failure_fails_execution_unbatched() {
    let transport = Answering::failing(
        "b",
        OperationAction::Succeed,
        ExecutionClientErrorScope::Execution,
    );
    let (result, recorder) = ordered_pair(&transport, |ctx, a, b| async move {
        ctx.race([a, b]).name("r").await
    })
    .await;
    assert_execution_failed(&result, &recorder);
    transport.assert_last_request_has("b", &OperationAction::Succeed);
}

#[tokio::test(start_paused = true)]
async fn select_ok_loser_write_failure_fails_execution_unbatched() {
    let transport = Answering::failing(
        "b",
        OperationAction::Succeed,
        ExecutionClientErrorScope::Execution,
    );
    let (result, recorder) = ordered_pair(&transport, |ctx, a, b| async move {
        ctx.select_ok([a, b]).name("r").await
    })
    .await;
    assert_execution_failed(&result, &recorder);
    transport.assert_last_request_has("b", &OperationAction::Succeed);
}

#[tokio::test(start_paused = true)]
async fn queued_write_behind_failed_write_is_not_sent_unbatched() {
    let transport = Answering::failing(
        "a",
        OperationAction::Succeed,
        ExecutionClientErrorScope::Execution,
    );
    let (result, recorder) = ordered_pair(&transport, |_ctx, a, b| async move {
        let (a, b) = tokio::join!(a, b);
        Ok(a? + b?)
    })
    .await;
    assert_execution_failed(&result, &recorder);
    assert!(
        !has(&transport.updates(), "b", &OperationAction::Succeed),
        "b's SUCCEED is never sent: {:?}",
        transport.requests()
    );
    assert!(
        transport
            .updates()
            .iter()
            .all(|u| u.action() != &OperationAction::Fail)
    );
}

#[tokio::test(start_paused = true)]
async fn first_failure_decides_unbatched() {
    let transport = Answering::new(|updates| {
        if has(updates, "orphan", &OperationAction::Succeed) {
            Some(ExecutionClientError::new(
                ExecutionClientErrorScope::Invocation,
                "throttled",
            ))
        } else if has(updates, "next", &OperationAction::Succeed) {
            Some(ExecutionClientError::new(
                ExecutionClientErrorScope::Execution,
                "rejected",
            ))
        } else {
            None
        }
    });
    let (result, recorder) = orphan_then_next(&transport).await;
    assert_invocation_failed(&result, &recorder);
    assert!(
        !has(&transport.updates(), "next", &OperationAction::Start),
        "next's START is never sent: {:?}",
        transport.requests()
    );
}

/// Runs a handler with one step `s` returning `"value"`, that turns any
/// step error into `Ok("caught")`, and counts the body runs in `runs`.
async fn caught_step(
    transport: &Arc<Answering>,
    unbatched: bool,
    operations: Vec<serde_json::Value>,
    runs: &Arc<std::sync::atomic::AtomicU32>,
) -> (
    Result<serde_json::Value, lambda_runtime::Error>,
    EndRecorder,
) {
    let runs = Arc::clone(runs);
    invoke_handler(transport, unbatched, operations, move |_event, ctx| {
        let runs = Arc::clone(&runs);
        async move {
            let value = ctx
                .step(move |_| {
                    let runs = Arc::clone(&runs);
                    async move {
                        runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Ok("value".to_owned())
                    }
                })
                .name("s")
                .await;
            Ok(value.unwrap_or_else(|_| "caught".to_owned()))
        }
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn failed_write_sends_nothing_more_unbatched() {
    let transport = Answering::failing(
        "s",
        OperationAction::Succeed,
        ExecutionClientErrorScope::Execution,
    );
    let runs = Arc::default();
    let (result, recorder) = caught_step(&transport, true, Vec::new(), &runs).await;
    assert_execution_failed(&result, &recorder);
    let requests = transport.requests();
    assert_eq!(requests.len(), 2, "{requests:?}");
    assert!(has(&requests[0], "s", &OperationAction::Start));
    assert!(has(&requests[1], "s", &OperationAction::Succeed));
}

#[tokio::test(start_paused = true)]
async fn failed_state_read_after_accepted_write_fails_execution() {
    let transport = Arc::new(Answering {
        requests: Mutex::new(Vec::new()),
        state_reads: Mutex::new(Vec::new()),
        tokens: Mutex::new(0),
        reject: Box::new(|_| None),
        state_error: Some(ExecutionClientError::new(
            ExecutionClientErrorScope::Execution,
            "denied",
        )),
    });
    let runs = Arc::default();
    let (result, recorder) = caught_step(&transport, false, Vec::new(), &runs).await;
    assert_execution_failed(&result, &recorder);
    transport.assert_last_request_has("s", &OperationAction::Succeed);
    // The SDK reads the whole state from its first page.
    assert_eq!(
        *transport
            .state_reads
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec![None],
        "exactly one state read follows the accepted SUCCEED"
    );
    assert!(
        transport
            .updates()
            .iter()
            .all(|u| u.action() != &OperationAction::Fail),
        "{:?}",
        transport.requests()
    );
}

#[tokio::test(start_paused = true)]
async fn lost_failed_response_replays_last_accepted_record() {
    use std::sync::atomic::Ordering;

    let first = Answering::failing(
        "s",
        OperationAction::Succeed,
        ExecutionClientErrorScope::Execution,
    );
    let runs = Arc::default();
    let (result, recorder) = caught_step(&first, true, Vec::new(), &runs).await;
    assert_execution_failed(&result, &recorder);
    assert_eq!(runs.load(Ordering::SeqCst), 1);

    // The FAILED response is lost, and the service invokes the function
    // again from the records it accepted.
    let second = Answering::accepting();
    let (result, _recorder) =
        caught_step(&second, true, operations_from(&first.accepted()), &runs).await;
    let response = result.expect("the second invocation answers");
    assert_eq!(status(&response), Some("SUCCEEDED"), "{response}");
    assert_eq!(
        response.get("Result").and_then(serde_json::Value::as_str),
        Some("\"value\""),
        "the second invocation never reaches the catch: {response}"
    );
    assert_eq!(runs.load(Ordering::SeqCst), 2, "the body runs again");
}

#[tokio::test(start_paused = true)]
async fn refresh_failure_stops_later_writes_unbatched() {
    let transport = Answering::new(|updates| {
        updates
            .is_empty()
            .then(|| ExecutionClientError::new(ExecutionClientErrorScope::Invocation, "throttled"))
    });
    let seen = Arc::clone(&transport);
    let (result, recorder) = invoke_handler(&transport, true, Vec::new(), move |_event, ctx| {
        let seen = Arc::clone(&seen);
        async move {
            let cb = ctx.create_callback::<String>().name("cb").await?;
            let t = ctx
                .step(move |_| {
                    let seen = Arc::clone(&seen);
                    async move {
                        // Return only once the refresh is in flight, so
                        // the SUCCEED is queued behind it.
                        seen.until_received(<[OperationUpdate]>::is_empty).await;
                        Ok(1_u32)
                    }
                })
                .name("t")
                .spawn();
            let (callback, t) = tokio::join!(cb.result(), t);
            Ok(format!("{callback:?} {t:?}"))
        }
    })
    .await;
    assert_invocation_failed(&result, &recorder);
    assert!(
        !has(&transport.updates(), "t", &OperationAction::Succeed),
        "t's SUCCEED is never sent: {:?}",
        transport.requests()
    );
    let requests = transport.requests();
    assert!(
        requests.last().is_some_and(Vec::is_empty),
        "no request follows the refresh: {requests:?}"
    );
}
