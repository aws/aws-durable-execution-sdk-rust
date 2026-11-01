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
    DateTime, ExecutionDetails, Operation, OperationAction, OperationStatus, OperationType,
    OperationUpdate,
};
use aws_durable_execution_sdk::{
    BoxError, CheckpointOutput, DurableContext, ExecutionClient, ExecutionClientError,
    ExecutionClientErrorScope, GetStateOutput, InvocationPayload, Options, lambda_runtime,
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
    let fail = recorded
        .iter()
        .find(|u| u.r#type() == &OperationType::Step && u.action() == &OperationAction::Fail);
    assert!(
        fail.is_some(),
        "the step's terminal FAIL was recorded: {recorded:?}"
    );
    let fail = fail.expect("checked above");
    assert_eq!(
        fail.error().and_then(|e| e.error_type()),
        Some("CheckpointFailedError")
    );
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
