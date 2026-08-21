//! Execution client abstraction for checkpoint operations.
//!
//! This module provides the internal trait that the engine calls to
//! checkpoint operation results and retrieve execution state. The
//! production implementation is backed by `aws-sdk-lambda`.

use std::pin::Pin;
#[cfg(test)]
use std::sync::Mutex;

use aws_sdk_lambda::types::{Operation, OperationType, OperationUpdate};

use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};

// ────────────────────────────────────────────────────────────────────────────
// `ExecutionClient` Trait
// ────────────────────────────────────────────────────────────────────────────

/// Checkpoint client error type wrapping the underlying cause.
///
/// `Clone` because a coalesced checkpoint batch publishes one result to
/// every operation that contributed updates to it.
#[derive(Debug, Clone)]
pub(crate) struct ClientError {
    message: String,
    /// Whether the underlying failure was classified retryable.
    ///
    /// The production client relies on the aws-sdk's standard retry, so by
    /// the time an error carries this flag the SDK has already exhausted
    /// its transport-level attempts. The flag decides the *recovery scope*
    /// (see [`classify_checkpoint_error`]): a retryable failure fails the
    /// invocation — the durable service re-invokes, which is the same
    /// recovery as an interruption — while a non-retryable failure is a
    /// permanent rejection that persists a terminal `FAIL` for the
    /// operation and then fails the execution
    /// ([`DurableContext::checkpoint_failure_unrecoverable`](crate::context::DurableContext::checkpoint_failure_unrecoverable)).
    retryable: bool,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "durable: client: {}", self.message)
    }
}

impl std::error::Error for ClientError {}

impl ClientError {
    /// Whether this error was classified retryable — the invocation fails
    /// and the durable service re-invokes. `false` means a permanent
    /// rejection: the terminal-`FAIL`-then-fail-the-execution path.
    pub(crate) fn is_retryable(&self) -> bool {
        self.retryable
    }

    pub(crate) fn non_retryable(message: String) -> Self {
        Self {
            message,
            retryable: false,
        }
    }

    /// Creates a non-retryable error from a string slice (convenience).
    pub(crate) fn new_non_retryable(message: &str) -> Self {
        Self {
            message: message.to_owned(),
            retryable: false,
        }
    }

    pub(crate) fn from_retryable(message: String) -> Self {
        Self {
            message,
            retryable: true,
        }
    }
}

/// The output from a successful checkpoint call.
#[derive(Debug, Clone)]
pub(crate) struct CheckpointOutput {
    /// The new checkpoint token for subsequent calls.
    pub(crate) checkpoint_token: String,
    /// Updated operations from the backend (may be empty).
    pub(crate) updated_operations: Vec<Operation>,
    /// Pagination marker — if present, more operations are available via
    /// `get_state` and the caller must paginate.
    pub(crate) next_marker: Option<String>,
}

/// The output from loading execution state.
#[derive(Debug)]
pub(crate) struct GetStateOutput {
    /// All operations returned from the backend (paginated internally).
    pub(crate) operations: Vec<Operation>,
}

/// The internal abstraction over checkpoint API operations.
///
/// The engine calls this trait for persisting operation updates and
/// retrieving execution state. The production implementation uses
/// `aws-sdk-lambda`; the test double uses an in-memory store.
///
/// Object-safe (no generics, no `Self: Sized` constraints) so it can be
/// stored as `Box<dyn ExecutionClient>`.
pub(crate) trait ExecutionClient: Send + Sync + std::fmt::Debug {
    /// Checkpoints operation updates, rotating the token.
    ///
    /// Transient-failure retry is the transport's responsibility: the
    /// production implementation relies on the aws-sdk's standard retry,
    /// and an error returned here is final for this invocation.
    fn checkpoint(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>>;

    /// Retrieves the full execution state (all operations), following
    /// pagination internally.
    fn get_state(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>;
}

// ────────────────────────────────────────────────────────────────────────────
// Production Implementation (`aws-sdk-lambda`)
// ────────────────────────────────────────────────────────────────────────────

/// Production `ExecutionClient` backed by `aws_sdk_lambda::Client`.
///
/// Retry is delegated entirely to the wrapped SDK client's own retry
/// strategy (standard retry by default: jittered exponential backoff, 3
/// attempts). This matches the sibling JS and Python SDKs, which rely on
/// their AWS SDKs' built-in retry. A call that exhausts the SDK's retries
/// fails the invocation; the durable execution service re-invokes the
/// handler, which is the recovery path — longer in-process retry would
/// only delay it. Callers who need a different retry policy supply a
/// preconfigured client via [`Options`](crate::Options)'s `lambda_client`.
#[derive(Debug, Clone)]
pub(crate) struct LambdaExecutionClient {
    client: aws_sdk_lambda::Client,
}

impl LambdaExecutionClient {
    /// Creates a new client wrapping the provided Lambda SDK client.
    pub(crate) fn new(client: aws_sdk_lambda::Client) -> Self {
        Self { client }
    }
}

impl ExecutionClient for LambdaExecutionClient {
    fn checkpoint(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>> {
        let arn = execution_arn.to_owned();
        let token = checkpoint_token.to_owned();
        Box::pin(async move {
            let result = self
                .client
                .checkpoint_durable_execution()
                .durable_execution_arn(&arn)
                .checkpoint_token(&token)
                .set_updates(Some(updates))
                .send()
                .await;

            match result {
                Ok(output) => {
                    let new_token = output.checkpoint_token.unwrap_or_default();
                    if new_token.is_empty() {
                        return Err(ClientError::non_retryable(
                            "backend returned no checkpoint token".to_owned(),
                        ));
                    }
                    let (updated_ops, next_marker) = match output.new_execution_state {
                        Some(state) => (
                            state.operations.unwrap_or_default(),
                            state.next_marker.filter(|m| !m.is_empty()),
                        ),
                        None => (Vec::new(), None),
                    };
                    Ok(CheckpointOutput {
                        checkpoint_token: new_token,
                        updated_operations: updated_ops,
                        next_marker,
                    })
                }
                // The SDK's standard retry has already retried everything
                // transient; the final error is classified into a recovery
                // scope (invocation vs execution) — see
                // `classify_checkpoint_error`.
                Err(err) => Err(classify_checkpoint_error(&err)),
            }
        })
    }

    fn get_state(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>> {
        let arn = execution_arn.to_owned();
        let token = checkpoint_token.to_owned();
        Box::pin(async move {
            let mut all_operations = Vec::new();
            let mut marker: Option<String> = None;

            loop {
                let mut builder = self
                    .client
                    .get_durable_execution_state()
                    .durable_execution_arn(&arn)
                    .checkpoint_token(&token);

                if let Some(ref m) = marker {
                    builder = builder.marker(m.as_str());
                }

                // As with `checkpoint`, the SDK's standard retry owns the
                // transient-failure retry; the final error maps into a
                // retryable `ClientError`.
                let output = builder
                    .send()
                    .await
                    .map_err(|err| ClientError::from_retryable(format!("{err}")))?;

                all_operations.extend(output.operations);

                match output.next_marker {
                    Some(ref m) if !m.is_empty() => {
                        marker = Some(m.clone());
                    }
                    _ => break,
                }
            }

            Ok(GetStateOutput {
                operations: all_operations,
            })
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Checkpoint error classification
// ────────────────────────────────────────────────────────────────────────────

/// The message prefix the service puts on an `InvalidParameterValueException`
/// that rejects a stale checkpoint token.
///
/// A stale token resolves on re-invocation (the service issues a fresh one),
/// so this specific rejection is classified retryable — the carve-out the
/// Java SDK's `DurableApiErrorClassifier` applies.
const STALE_TOKEN_MESSAGE_PREFIX: &str = "Invalid Checkpoint Token";

/// Classifies a final `CheckpointDurableExecution` error into a recovery
/// scope, carried as [`ClientError::is_retryable`].
///
/// By the time this runs, the aws-sdk's standard retry has exhausted its
/// transport-level attempts, so the question is no longer "try again now"
/// but *which recovery converges*:
///
/// - **Retryable** — fail the invocation; the durable service re-invokes
///   and replay resumes from the recorded state. This is the same recovery
///   as an interruption. Applied to transport-level failures (dispatch,
///   timeout, malformed response), throttling, 5xx, and the stale-token
///   carve-out ([`STALE_TOKEN_MESSAGE_PREFIX`]).
/// - **Non-retryable** — the service permanently rejected this write (for
///   example an oversized payload). Re-invoking re-runs the body and hits
///   the same rejection, so the caller persists a small terminal `FAIL`
///   for the operation and fails the execution instead.
///
/// **Unknown errors default to retryable.** The original classification
/// (pre-#43) defaulted unknown to non-retryable, with the rationale — from
/// the r3726032098 review thread — that the surfaced error only failed the
/// invocation and the service re-invoked anyway, so the default was
/// harmless. That rationale belongs to the old model: under this model a
/// non-retryable classification fails the *execution*, so a novel
/// transient error variant defaulting to non-retryable would kill
/// executions. Java and Python default unknown to retryable for the same
/// reason. The `checkpoint_error_variant_canary` test pins the variant set
/// this function classifies explicitly, so a new upstream variant fails CI
/// for reclassification instead of falling silently into the default.
///
/// Generic over the response type `R` so tests can classify a
/// `SdkError::service_error(err, ())` without constructing an HTTP
/// response.
fn classify_checkpoint_error<R: std::fmt::Debug>(
    err: &aws_sdk_lambda::error::SdkError<
        aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError,
        R,
    >,
) -> ClientError {
    use aws_sdk_lambda::error::SdkError;
    use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError as E;

    let message = format!("{}", aws_sdk_lambda::error::DisplayErrorContext(err));

    let SdkError::ServiceError(service_err) = err else {
        // Dispatch failures, timeouts, and malformed responses are
        // channel-level: nothing was permanently rejected.
        return ClientError::from_retryable(message);
    };

    match service_err.err() {
        // Throttling and service-side 5xx: transient by definition.
        E::TooManyRequestsException(_) | E::ServiceException(_) => {
            ClientError::from_retryable(message)
        }
        // A parameter rejection is deterministic — the same write fails the
        // same way on every lap — EXCEPT the stale-token rejection, which a
        // re-invocation resolves with a fresh token.
        E::InvalidParameterValueException(inner) => {
            if inner
                .message()
                .is_some_and(|m| m.starts_with(STALE_TOKEN_MESSAGE_PREFIX))
            {
                ClientError::from_retryable(message)
            } else {
                ClientError::non_retryable(message)
            }
        }
        // KMS misconfiguration on the function: deterministic until an
        // operator fixes the key, so re-invoking loops without progress.
        E::KmsAccessDeniedException(_)
        | E::KmsDisabledException(_)
        | E::KmsInvalidStateException(_)
        | E::KmsNotFoundException(_) => ClientError::non_retryable(message),
        // Unknown errors are RETRYABLE — see the function docs for why the
        // pre-#43 non-retryable default (r3726032098) no longer applies.
        // The variant canary test keeps this arm from silently absorbing
        // new modeled variants.
        _ => ClientError::from_retryable(message),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// In-Memory Test Double
// ────────────────────────────────────────────────────────────────────────────

/// Injection point for controlling test double behavior per call.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) enum TestResponse {
    /// Return success with the given operations.
    Success(Vec<Operation>),
    /// Return success with given operations and a pagination marker,
    /// indicating more operations are available via `get_state`.
    SuccessPaginated(Vec<Operation>, String),
    /// Return a retryable failure.
    RetryableError(String),
    /// Return a non-retryable failure.
    NonRetryableError(String),
}

/// In-memory test double for `ExecutionClient`.
///
/// Supports injecting failures and recording calls for assertions.
/// Sufficient for unit tests — not the full `test-util` `LocalRunner`.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct InMemoryExecutionClient {
    /// Pre-loaded state returned by `get_state`.
    state_operations: Mutex<Vec<Operation>>,
    /// Queue of responses for checkpoint calls; if empty, returns success.
    checkpoint_responses: Mutex<Vec<TestResponse>>,
    /// Counter for checkpoint calls made (including retries).
    pub(crate) checkpoint_call_count: Mutex<u32>,
    /// Counter for `get_state` calls made.
    pub(crate) get_state_call_count: Mutex<u32>,
    /// When set, `get_state` fails with a non-retryable error carrying this
    /// message (used to test that a persisted checkpoint's lifecycle events
    /// survive a failed pagination fetch).
    get_state_failure: Mutex<Option<String>>,
    /// Token counter for generating unique tokens.
    token_counter: Mutex<u32>,
    /// All operation updates received across checkpoint calls.
    recorded_updates: Mutex<Vec<OperationUpdate>>,
}

#[cfg(test)]
impl InMemoryExecutionClient {
    /// Creates a new test double with the given pre-loaded state.
    pub(crate) fn new(state_operations: Vec<Operation>) -> Self {
        Self {
            state_operations: Mutex::new(state_operations),
            checkpoint_responses: Mutex::new(Vec::new()),
            checkpoint_call_count: Mutex::new(0),
            get_state_call_count: Mutex::new(0),
            get_state_failure: Mutex::new(None),
            token_counter: Mutex::new(0),
            recorded_updates: Mutex::new(Vec::new()),
        }
    }

    /// Enqueues a response for the next checkpoint call.
    pub(crate) fn enqueue_checkpoint_response(&self, response: TestResponse) {
        let mut responses = self
            .checkpoint_responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        responses.push(response);
    }

    /// Makes every subsequent `get_state` call fail non-retryably with the
    /// given message.
    pub(crate) fn fail_get_state(&self, message: &str) {
        let mut failure = self
            .get_state_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *failure = Some(message.to_owned());
    }

    /// Returns all operation updates recorded across checkpoint calls.
    pub(crate) fn recorded_updates(&self) -> Vec<OperationUpdate> {
        self.recorded_updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[cfg(test)]
impl ExecutionClient for InMemoryExecutionClient {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>> {
        {
            let mut count = self
                .checkpoint_call_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *count += 1;
        }

        // Record the updates for test assertions.
        {
            let mut recorded = self
                .recorded_updates
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            recorded.extend(updates);
        }

        let response = {
            let mut responses = self
                .checkpoint_responses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if responses.is_empty() {
                None
            } else {
                Some(responses.remove(0))
            }
        };

        Box::pin(async move {
            match response {
                Some(TestResponse::RetryableError(msg)) => Err(ClientError::from_retryable(msg)),
                Some(TestResponse::NonRetryableError(msg)) => Err(ClientError::non_retryable(msg)),
                Some(TestResponse::Success(ops)) => {
                    let mut counter = self
                        .token_counter
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *counter += 1;
                    Ok(CheckpointOutput {
                        checkpoint_token: format!("token-{counter}"),
                        updated_operations: ops,
                        next_marker: None,
                    })
                }
                Some(TestResponse::SuccessPaginated(ops, marker)) => {
                    let mut counter = self
                        .token_counter
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *counter += 1;
                    Ok(CheckpointOutput {
                        checkpoint_token: format!("token-{counter}"),
                        updated_operations: ops,
                        next_marker: Some(marker),
                    })
                }
                None => {
                    let mut counter = self
                        .token_counter
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *counter += 1;
                    Ok(CheckpointOutput {
                        checkpoint_token: format!("token-{counter}"),
                        updated_operations: Vec::new(),
                        next_marker: None,
                    })
                }
            }
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>> {
        {
            let mut count = self
                .get_state_call_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *count += 1;
        }

        let failure = self
            .get_state_failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(message) = failure {
            return Box::pin(async move { Err(ClientError::non_retryable(message)) });
        }

        let ops = self
            .state_operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();

        Box::pin(async move { Ok(GetStateOutput { operations: ops }) })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Checkpoint Log Population Helper
// ────────────────────────────────────────────────────────────────────────────

/// Converts a single SDK `Operation` into a `(wire_id, CheckpointRecord)` pair.
fn operation_to_record(op: &Operation) -> (String, CheckpointRecord) {
    let attempt = op.step_details.as_ref().map_or(0, |s| {
        #[allow(clippy::cast_sign_loss)] // reason: attempt is non-negative from backend
        {
            s.attempt.max(0) as u32
        }
    });
    let record = CheckpointRecord {
        id: op.id().to_owned(),
        status: sdk_status_to_checkpoint_status(op.status()),
        result: extract_result(op),
        error_type: extract_error_type(op),
        error_message: extract_error_message(op),
        error_data: extract_wire_error_field(op, |e| e.error_data.clone()),
        stack_trace: extract_wire_error_field(op, |e| {
            (!e.stack_trace().is_empty()).then(|| e.stack_trace().to_vec())
        }),
        attempt,
        // ChainedInvoke payloads live in their own record fields — mirroring
        // parse_single_operation in lib.rs, which is the JSON-path reference
        // for this mapping. invoke.rs replays exclusively from these.
        invoke_result: op
            .chained_invoke_details
            .as_ref()
            .and_then(|i| i.result.clone()),
        invoke_error_type: op
            .chained_invoke_details
            .as_ref()
            .and_then(|i| i.error.as_ref())
            .and_then(|e| e.error_type.clone()),
        invoke_error_message: op
            .chained_invoke_details
            .as_ref()
            .and_then(|i| i.error.as_ref())
            .and_then(|e| e.error_message.clone()),
        invoke_error_data: op
            .chained_invoke_details
            .as_ref()
            .and_then(|i| i.error.as_ref())
            .and_then(|e| e.error_data.clone()),
        invoke_stack_trace: op
            .chained_invoke_details
            .as_ref()
            .and_then(|i| i.error.as_ref())
            .and_then(|e| (!e.stack_trace().is_empty()).then(|| e.stack_trace().to_vec())),
        // Large child contexts require ContextDetails.ReplayChildren to
        // signal re-execution of children; mirror the JSON path.
        replay_children: op
            .context_details
            .as_ref()
            .and_then(|c| c.replay_children)
            .unwrap_or(false),
        callback_id: op
            .callback_details
            .as_ref()
            .and_then(|cb| cb.callback_id.clone()),
        op_type: Some(operation_type_to_string(&op.r#type)),
        sub_type: op.sub_type.clone(),
        op_name: op.name.clone(),
    };
    (op.id().to_owned(), record)
}

/// Converts an `OperationType` enum to a string for identity comparison.
fn operation_type_to_string(op_type: &OperationType) -> String {
    match op_type {
        OperationType::Callback => "Callback".to_owned(),
        OperationType::ChainedInvoke => "ChainedInvoke".to_owned(),
        OperationType::Context => "Context".to_owned(),
        OperationType::Execution => "Execution".to_owned(),
        OperationType::Step => "Step".to_owned(),
        OperationType::Wait => "Wait".to_owned(),
        // Future-proof: unknown variants use a placeholder.
        _ => "Unknown".to_owned(),
    }
}

/// Converts SDK `Operation` records into a `CheckpointLog`.
///
/// Keyed by wire ID (the operation's `id` field from the service).
pub(crate) fn operations_to_checkpoint_log(operations: &[Operation]) -> CheckpointLog {
    let records: Vec<(String, CheckpointRecord)> = operations
        .iter()
        // Skip Execution-type operations for consistency with
        // parse_inline_operations, which filters them in the JSON path.
        .filter(|op| op.r#type != OperationType::Execution)
        .map(operation_to_record)
        .collect();
    CheckpointLog::from_records(records)
}

/// Merges updated operations from a checkpoint response into an existing log.
///
/// After a checkpoint call, the backend may return operations with
/// backend-assigned fields (e.g. `callback_id`). These must be visible to
/// subsequent reads in the same invocation.
pub(crate) fn merge_operations_into_log(log: &CheckpointLog, operations: &[Operation]) {
    for op in operations {
        let (wire_id, record) = operation_to_record(op);
        log.insert(wire_id, record);
    }
}

/// Resolves the bootstrap checkpoint log, following the pagination marker.
///
/// This is the single production decision point for bootstrap pagination:
/// when the initial execution state carried a `NextMarker`, the inline
/// first page is incomplete, so the complete state is fetched through
/// [`ExecutionClient::get_state`] (which follows markers until exhausted).
/// Without a marker the inline page already holds the full history and is
/// used as-is. Both `run`/`wrap` and the `test-util` `LocalRunner` call
/// this helper, so tests exercise exactly the code production runs.
pub(crate) async fn resolve_bootstrap_log(
    client: &dyn ExecutionClient,
    execution_arn: &str,
    checkpoint_token: &str,
    first_page: CheckpointLog,
    next_marker: Option<&str>,
) -> Result<CheckpointLog, ClientError> {
    if next_marker.is_none() {
        return Ok(first_page);
    }
    let full_state = client.get_state(execution_arn, checkpoint_token).await?;
    Ok(operations_to_checkpoint_log(&full_state.operations))
}

/// Maps SDK `OperationStatus` to internal `CheckpointStatus`.
fn sdk_status_to_checkpoint_status(
    status: &aws_sdk_lambda::types::OperationStatus,
) -> CheckpointStatus {
    match status {
        aws_sdk_lambda::types::OperationStatus::Pending => CheckpointStatus::Pending,
        aws_sdk_lambda::types::OperationStatus::Ready => CheckpointStatus::Ready,
        aws_sdk_lambda::types::OperationStatus::Succeeded => CheckpointStatus::Succeeded,
        aws_sdk_lambda::types::OperationStatus::Failed => CheckpointStatus::Failed,
        aws_sdk_lambda::types::OperationStatus::Cancelled => CheckpointStatus::Cancelled,
        aws_sdk_lambda::types::OperationStatus::TimedOut => CheckpointStatus::TimedOut,
        aws_sdk_lambda::types::OperationStatus::Stopped => CheckpointStatus::Stopped,
        // Started and future variants: non-terminal safe default.
        _ => CheckpointStatus::Started,
    }
}

/// Extracts the result payload from an `Operation`.
///
/// `ChainedInvoke` results are deliberately excluded: they map to
/// `invoke_result` (see `operation_to_record`), matching the JSON path in
/// `parse_single_operation`.
fn extract_result(op: &Operation) -> Option<String> {
    op.step_details
        .as_ref()
        .and_then(|s| s.result.clone())
        .or_else(|| op.context_details.as_ref().and_then(|c| c.result.clone()))
        .or_else(|| {
            op.callback_details
                .as_ref()
                .and_then(|cb| cb.result.clone())
        })
}

/// Extracts the error type from an `Operation`.
///
/// `ChainedInvoke` errors are deliberately excluded: they map to
/// `invoke_error_type` (see `operation_to_record`), matching the JSON path
/// in `parse_single_operation`.
fn extract_error_type(op: &Operation) -> Option<String> {
    op.step_details
        .as_ref()
        .and_then(|s| s.error.as_ref())
        .and_then(|e| e.error_type.clone())
        .or_else(|| {
            op.context_details
                .as_ref()
                .and_then(|c| c.error.as_ref())
                .and_then(|e| e.error_type.clone())
        })
        .or_else(|| {
            op.callback_details
                .as_ref()
                .and_then(|cb| cb.error.as_ref())
                .and_then(|e| e.error_type.clone())
        })
}

/// Extracts the error message from an `Operation`.
///
/// `ChainedInvoke` errors are deliberately excluded: they map to
/// `invoke_error_message` (see `operation_to_record`), matching the JSON
/// path in `parse_single_operation`.
fn extract_error_message(op: &Operation) -> Option<String> {
    op.step_details
        .as_ref()
        .and_then(|s| s.error.as_ref())
        .and_then(|e| e.error_message.clone())
        .or_else(|| {
            op.context_details
                .as_ref()
                .and_then(|c| c.error.as_ref())
                .and_then(|e| e.error_message.clone())
        })
        .or_else(|| {
            op.callback_details
                .as_ref()
                .and_then(|cb| cb.error.as_ref())
                .and_then(|e| e.error_message.clone())
        })
}

/// Extracts one field of the first error object present on the operation,
/// checking step, context, and callback details in that order (mirroring
/// [`extract_error_type`]).
fn extract_wire_error_field<T>(
    op: &Operation,
    read: impl Fn(&aws_sdk_lambda::types::ErrorObject) -> Option<T>,
) -> Option<T> {
    op.step_details
        .as_ref()
        .and_then(|s| s.error.as_ref())
        .and_then(&read)
        .or_else(|| {
            op.context_details
                .as_ref()
                .and_then(|c| c.error.as_ref())
                .and_then(&read)
        })
        .or_else(|| {
            op.callback_details
                .as_ref()
                .and_then(|cb| cb.error.as_ref())
                .and_then(&read)
        })
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Request mapping (aws-smithy-mocks) ──────────────────────────────
    //
    // These tests mock the Lambda SDK client at the HTTP layer and pin the
    // per-operation request mapping of `LambdaExecutionClient`: what the
    // client puts on the wire, how it maps responses back, and how the
    // final error (after the SDK's own retry) maps into `ClientError`.

    use aws_sdk_lambda::operation::checkpoint_durable_execution::{
        CheckpointDurableExecutionError, CheckpointDurableExecutionOutput,
    };
    use aws_sdk_lambda::operation::get_durable_execution_state::{
        GetDurableExecutionStateError, GetDurableExecutionStateOutput,
    };
    use aws_sdk_lambda::types::{CheckpointUpdatedExecutionState, OperationAction};
    use aws_smithy_mocks::{RuleMode, mock, mock_client};

    /// Builds a minimal `OperationUpdate` for request-mapping assertions.
    #[allow(clippy::expect_used)] // reason: test helper — all required fields are set
    fn make_update(id: &str, action: OperationAction) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Step)
            .action(action)
            .build()
            .expect("valid OperationUpdate")
    }

    /// `checkpoint` maps its arguments onto the `CheckpointDurableExecution`
    /// request — ARN, token, and every update — and maps the response's
    /// token, updated operations, and pagination marker back out.
    #[tokio::test]
    #[allow(clippy::expect_used)] // reason: test assertions
    async fn checkpoint_maps_request_and_response() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .match_requests(|req| {
                req.durable_execution_arn() == Some("arn:aws:lambda:us-east-1:1:function:f")
                    && req.checkpoint_token() == Some("tok-in")
                    && req.updates().len() == 2
                    && req.updates().first().map(OperationUpdate::id) == Some("op-1")
                    && req.updates().get(1).map(OperationUpdate::id) == Some("op-2")
            })
            .then_output(|| {
                CheckpointDurableExecutionOutput::builder()
                    .checkpoint_token("tok-out")
                    .new_execution_state(
                        CheckpointUpdatedExecutionState::builder()
                            .operations(make_step_op("op-1", r#""r1""#))
                            .next_marker("more-pages")
                            .build(),
                    )
                    .build()
            });
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let updates = vec![
            make_update("op-1", OperationAction::Start),
            make_update("op-2", OperationAction::Succeed),
        ];
        let output = client
            .checkpoint("arn:aws:lambda:us-east-1:1:function:f", "tok-in", updates)
            .await
            .expect("mocked checkpoint succeeds");

        assert_eq!(output.checkpoint_token, "tok-out");
        assert_eq!(output.updated_operations.len(), 1);
        assert_eq!(
            output.updated_operations.first().map(Operation::id),
            Some("op-1")
        );
        assert_eq!(output.next_marker.as_deref(), Some("more-pages"));
        assert_eq!(rule.num_calls(), 1);
    }

    /// A checkpoint response without a token is a protocol violation and
    /// maps to a non-retryable `ClientError`.
    #[tokio::test]
    #[allow(clippy::expect_used)] // reason: test assertions
    async fn checkpoint_without_token_is_non_retryable() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .then_output(|| CheckpointDurableExecutionOutput::builder().build());
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect_err("missing token must fail");
        assert!(!err.is_retryable());
        assert!(err.to_string().contains("no checkpoint token"));
    }

    /// A checkpoint call whose final outcome is a modeled parameter
    /// rejection — a deterministic client fault the SDK does not retry —
    /// classifies NON-retryable: re-invoking replays into the same
    /// rejection, so the recovery is a terminal `FAIL` write followed by
    /// failing the execution, not another lap.
    #[tokio::test]
    #[allow(clippy::expect_used)] // reason: test assertions
    async fn checkpoint_final_error_maps_to_non_retryable_client_error() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution).then_error(|| {
            CheckpointDurableExecutionError::InvalidParameterValueException(
                aws_sdk_lambda::types::error::InvalidParameterValueException::builder().build(),
            )
        });
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect_err("modeled error must fail the call");
        assert!(!err.is_retryable());
        assert_eq!(rule.num_calls(), 1, "a client fault is not retried");
    }

    /// Transient-failure retry belongs to the aws-sdk's standard retry and
    /// nothing else: a persistent 503 is attempted exactly the SDK's
    /// default 3 times — not 9, which the deleted hand-rolled outer loop
    /// used to multiply it to — and the exhausted call maps to a retryable
    /// `ClientError`.
    #[tokio::test]
    #[allow(clippy::expect_used)] // reason: test assertions
    async fn checkpoint_retry_is_the_sdk_standard_retry_alone() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .sequence()
            .http_status(503, None)
            .repeatedly()
            .build();
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect_err("persistent 503 must exhaust retries");
        assert!(err.is_retryable());
        assert_eq!(
            rule.num_calls(),
            3,
            "attempts must equal the SDK standard-retry budget — no outer loop multiplying them"
        );
    }

    /// `get_state` maps ARN and token onto the request, sends no marker on
    /// the first page, echoes the response's `next_marker` as the next
    /// request's `marker`, and concatenates the pages in order.
    #[tokio::test]
    #[allow(clippy::expect_used)] // reason: test assertions
    async fn get_state_paginates_with_marker() {
        let first_page = mock!(aws_sdk_lambda::Client::get_durable_execution_state)
            .match_requests(|req| {
                req.durable_execution_arn() == Some("arn:test")
                    && req.checkpoint_token() == Some("tok")
                    && req.marker().is_none()
            })
            .then_output(|| {
                GetDurableExecutionStateOutput::builder()
                    .operations(make_step_op("op-1", r#""p1""#))
                    .next_marker("page-2")
                    .build()
                    .expect("operations is set")
            });
        let second_page = mock!(aws_sdk_lambda::Client::get_durable_execution_state)
            .match_requests(|req| req.marker() == Some("page-2"))
            .then_output(|| {
                GetDurableExecutionStateOutput::builder()
                    .operations(make_step_op("op-2", r#""p2""#))
                    .build()
                    .expect("operations is set")
            });
        let sdk_client = mock_client!(
            aws_sdk_lambda,
            RuleMode::Sequential,
            [&first_page, &second_page]
        );
        let client = LambdaExecutionClient::new(sdk_client);

        let state = client
            .get_state("arn:test", "tok")
            .await
            .expect("paginated get_state succeeds");

        assert_eq!(state.operations.len(), 2);
        assert_eq!(state.operations.first().map(Operation::id), Some("op-1"));
        assert_eq!(state.operations.get(1).map(Operation::id), Some("op-2"));
        assert_eq!(first_page.num_calls(), 1);
        assert_eq!(second_page.num_calls(), 1);
    }

    /// A `get_state` failure maps into a retryable `ClientError` exactly
    /// like a checkpoint failure.
    #[tokio::test]
    #[allow(clippy::expect_used)] // reason: test assertions
    async fn get_state_final_error_maps_to_retryable_client_error() {
        let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_error(|| {
            GetDurableExecutionStateError::InvalidParameterValueException(
                aws_sdk_lambda::types::error::InvalidParameterValueException::builder().build(),
            )
        });
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .get_state("arn:test", "tok")
            .await
            .expect_err("modeled error must fail the call");
        assert!(err.is_retryable());
    }

    // ── Test double behavior ────────────────────────────────────────────

    #[tokio::test]
    async fn in_memory_client_returns_queued_responses_in_order() {
        let client = InMemoryExecutionClient::new(Vec::new());
        client.enqueue_checkpoint_response(TestResponse::RetryableError("throttled".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError("invalid".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::Success(Vec::new()));

        let first = client.checkpoint("arn:test", "token-0", Vec::new()).await;
        #[allow(clippy::unwrap_used)] // reason: test assertion — err verified above
        let first_err = first.unwrap_err();
        assert!(first_err.is_retryable());

        let second = client.checkpoint("arn:test", "token-0", Vec::new()).await;
        #[allow(clippy::unwrap_used)] // reason: test assertion — err verified above
        let second_err = second.unwrap_err();
        assert!(!second_err.is_retryable());

        let third = client.checkpoint("arn:test", "token-0", Vec::new()).await;
        assert!(third.is_ok());

        let count = *client
            .checkpoint_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn success_returns_checkpoint_output() {
        let client = InMemoryExecutionClient::new(Vec::new());
        let result = client.checkpoint("arn:test", "token-0", Vec::new()).await;

        assert!(result.is_ok());
        #[allow(clippy::unwrap_used)] // reason: test assertion — ok verified above
        let output = result.unwrap();
        assert!(output.checkpoint_token.starts_with("token-"));
    }

    // ── Checkpoint log population ───────────────────────────────────────

    #[test]
    fn checkpoint_log_from_operations_maps_statuses() {
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let ops = vec![
            Operation::builder()
                .id("op-1")
                .r#type(OperationType::Step)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .step_details(
                    aws_sdk_lambda::types::StepDetails::builder()
                        .attempt(1)
                        .result(r#""hello""#)
                        .build(),
                )
                .build()
                .unwrap(),
            Operation::builder()
                .id("op-2")
                .r#type(OperationType::Step)
                .status(aws_sdk_lambda::types::OperationStatus::Failed)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(1))
                .step_details(
                    aws_sdk_lambda::types::StepDetails::builder()
                        .attempt(2)
                        .error(
                            aws_sdk_lambda::types::ErrorObject::builder()
                                .error_type("StepError")
                                .error_message("oops")
                                .build(),
                        )
                        .build(),
                )
                .build()
                .unwrap(),
        ];

        let log = operations_to_checkpoint_log(&ops);

        let r1 = log.get("op-1");
        assert!(r1.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let r1 = r1.unwrap();
        assert_eq!(r1.status, CheckpointStatus::Succeeded);
        assert_eq!(r1.result.as_deref(), Some(r#""hello""#));
        assert_eq!(r1.attempt, 1);

        let r2 = log.get("op-2");
        assert!(r2.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let r2 = r2.unwrap();
        assert_eq!(r2.status, CheckpointStatus::Failed);
        assert_eq!(r2.error_type.as_deref(), Some("StepError"));
        assert_eq!(r2.error_message.as_deref(), Some("oops"));
        assert_eq!(r2.attempt, 2);
    }

    /// A paginated `ChainedInvoke` operation converts into the
    /// invoke-specific record fields (`invoke_result` /
    /// `invoke_error_*`) that invoke replay reads — matching the inline
    /// JSON conversion in `parse_single_operation` — and never leaks into
    /// the generic `result` / `error_*` fields.
    #[test]
    fn chained_invoke_maps_to_invoke_specific_fields() {
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let ops = vec![
            Operation::builder()
                .id("invoke-ok")
                .r#type(OperationType::ChainedInvoke)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .chained_invoke_details(
                    aws_sdk_lambda::types::ChainedInvokeDetails::builder()
                        .result(r#""invoke-payload""#)
                        .build(),
                )
                .build()
                .unwrap(),
            Operation::builder()
                .id("invoke-err")
                .r#type(OperationType::ChainedInvoke)
                .status(aws_sdk_lambda::types::OperationStatus::Failed)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(1))
                .chained_invoke_details(
                    aws_sdk_lambda::types::ChainedInvokeDetails::builder()
                        .error(
                            aws_sdk_lambda::types::ErrorObject::builder()
                                .error_type("InvokeError")
                                .error_message("downstream failed")
                                .build(),
                        )
                        .build(),
                )
                .build()
                .unwrap(),
        ];

        let log = operations_to_checkpoint_log(&ops);

        let ok = log.get("invoke-ok");
        assert!(ok.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let ok = ok.unwrap();
        assert_eq!(ok.invoke_result.as_deref(), Some(r#""invoke-payload""#));
        assert_eq!(
            ok.result, None,
            "invoke results must not leak into the generic result field"
        );

        let err = log.get("invoke-err");
        assert!(err.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let err = err.unwrap();
        assert_eq!(err.invoke_error_type.as_deref(), Some("InvokeError"));
        assert_eq!(
            err.invoke_error_message.as_deref(),
            Some("downstream failed")
        );
        assert_eq!(
            err.error_type, None,
            "invoke errors must not leak into the generic error fields"
        );
        assert_eq!(err.error_message, None);
    }

    /// A paginated child-context operation preserves
    /// `ContextDetails.ReplayChildren`, which large child contexts require
    /// for correct replay — matching the inline JSON conversion.
    #[test]
    fn context_replay_children_is_preserved() {
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let ops = vec![
            Operation::builder()
                .id("child-replay")
                .r#type(OperationType::Context)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .context_details(
                    aws_sdk_lambda::types::ContextDetails::builder()
                        .replay_children(true)
                        .build(),
                )
                .build()
                .unwrap(),
            Operation::builder()
                .id("child-inline")
                .r#type(OperationType::Context)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(1))
                .context_details(
                    aws_sdk_lambda::types::ContextDetails::builder()
                        .result(r#""child-result""#)
                        .build(),
                )
                .build()
                .unwrap(),
        ];

        let log = operations_to_checkpoint_log(&ops);

        let replayed = log.get("child-replay");
        assert!(replayed.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let replayed = replayed.unwrap();
        assert!(
            replayed.replay_children,
            "ReplayChildren must survive the typed-state conversion"
        );

        let inline = log.get("child-inline");
        assert!(inline.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let inline = inline.unwrap();
        assert!(!inline.replay_children);
        assert_eq!(inline.result.as_deref(), Some(r#""child-result""#));
    }

    /// Execution-type operations are filtered from the typed-state
    /// conversion, matching `parse_inline_operations` in the JSON path.
    #[test]
    fn execution_operations_are_filtered_from_log() {
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let ops = vec![
            Operation::builder()
                .id("exec-0")
                .r#type(OperationType::Execution)
                .status(aws_sdk_lambda::types::OperationStatus::Started)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .build()
                .unwrap(),
            Operation::builder()
                .id("step-1")
                .r#type(OperationType::Step)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(1))
                .build()
                .unwrap(),
        ];

        let log = operations_to_checkpoint_log(&ops);

        assert!(
            log.get("exec-0").is_none(),
            "Execution operations must be filtered like the JSON path does"
        );
        assert!(log.get("step-1").is_some());
    }

    // ── Get state ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn get_state_returns_preloaded_operations() {
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let ops = vec![
            Operation::builder()
                .id("op-1")
                .r#type(OperationType::Step)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .build()
                .unwrap(),
        ];

        let client = InMemoryExecutionClient::new(ops);
        let result = client.get_state("arn:test", "token-0").await;

        assert!(result.is_ok());
        #[allow(clippy::unwrap_used)] // reason: test assertion — ok verified above
        let output = result.unwrap();
        assert_eq!(output.operations.len(), 1);
        assert_eq!(output.operations.first().map(Operation::id), Some("op-1"));
    }

    // ── Merge updated operations into log ───────────────────────────────

    #[test]
    fn merge_operations_inserts_callback_id_into_existing_log() {
        // Simulate: log starts with a STARTED callback (no callback_id yet).
        let log = CheckpointLog::from_records(vec![(
            "op-cb".to_owned(),
            CheckpointRecord {
                id: "op-cb".to_owned(),
                status: CheckpointStatus::Started,
                result: None,
                error_type: None,
                error_message: None,
                error_data: None,
                stack_trace: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: None,
                op_type: None,
                sub_type: None,
                op_name: None,
            },
        )]);

        // Backend returns the operation with callback_id assigned.
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let updated_ops = vec![
            Operation::builder()
                .id("op-cb")
                .r#type(OperationType::Callback)
                .status(aws_sdk_lambda::types::OperationStatus::Pending)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .callback_details(
                    aws_sdk_lambda::types::CallbackDetails::builder()
                        .callback_id("assigned-cb-id-42")
                        .build(),
                )
                .build()
                .unwrap(),
        ];

        merge_operations_into_log(&log, &updated_ops);

        let record = log.get("op-cb");
        assert!(record.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let record = record.unwrap();
        assert_eq!(record.callback_id.as_deref(), Some("assigned-cb-id-42"));
        assert_eq!(record.status, CheckpointStatus::Pending);
    }

    #[test]
    fn merge_operations_adds_new_record_to_log() {
        let log = CheckpointLog::empty();

        // Backend returns a new operation not previously in the log.
        #[allow(clippy::unwrap_used)] // reason: test — builder is infallible for valid input
        let updated_ops = vec![
            Operation::builder()
                .id("new-op")
                .r#type(OperationType::Step)
                .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .step_details(
                    aws_sdk_lambda::types::StepDetails::builder()
                        .result(r#""merged result""#)
                        .build(),
                )
                .build()
                .unwrap(),
        ];

        merge_operations_into_log(&log, &updated_ops);

        let record = log.get("new-op");
        assert!(record.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let record = record.unwrap();
        assert_eq!(record.status, CheckpointStatus::Succeeded);
        assert_eq!(record.result.as_deref(), Some(r#""merged result""#));
    }

    // ── Pagination tests ────────────────────────────────────────────────

    /// Helper: builds a Step operation with a result for pagination tests.
    #[allow(clippy::unwrap_used)] // reason: test helper
    fn make_step_op(id: &str, result: &str) -> Operation {
        Operation::builder()
            .id(id)
            .r#type(OperationType::Step)
            .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .step_details(
                aws_sdk_lambda::types::StepDetails::builder()
                    .result(result)
                    .build(),
            )
            .build()
            .unwrap()
    }

    /// Tests that `get_state` in `InMemoryExecutionClient` returns all
    /// pre-loaded operations (simulating a paginated full fetch).
    #[tokio::test]
    async fn in_memory_get_state_returns_all_operations() {
        let ops = vec![
            make_step_op("step-1", "\"page1\""),
            make_step_op("step-2", "\"page2\""),
            make_step_op("step-3", "\"page3\""),
        ];
        let client = InMemoryExecutionClient::new(ops);

        let state = client.get_state("arn:test", "token-0").await;
        assert!(state.is_ok());
        #[allow(clippy::unwrap_used)]
        let state = state.unwrap();
        assert_eq!(state.operations.len(), 3);
    }

    /// Tests that a paginated checkpoint response (with `next_marker`)
    /// triggers the caller to call `get_state`. This test verifies the
    /// `SuccessPaginated` variant works correctly.
    #[tokio::test]
    async fn checkpoint_with_marker_returns_paginated_response() {
        let all_ops = vec![
            make_step_op("step-1", "\"r1\""),
            make_step_op("step-2", "\"r2\""),
            make_step_op("step-3", "\"r3\""),
        ];
        let client = InMemoryExecutionClient::new(all_ops);

        // Enqueue a checkpoint response that indicates pagination.
        let page1_ops = vec![make_step_op("step-1", "\"r1\"")];
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            page1_ops,
            "marker-page-2".to_owned(),
        ));

        let result = client.checkpoint("arn:test", "token-0", vec![]).await;
        assert!(result.is_ok());
        #[allow(clippy::unwrap_used)]
        let output = result.unwrap();
        assert_eq!(output.updated_operations.len(), 1);
        assert_eq!(output.next_marker, Some("marker-page-2".to_owned()));

        // Caller should then call get_state to get all operations.
        let full_state = client.get_state("arn:test", &output.checkpoint_token).await;
        assert!(full_state.is_ok());
        #[allow(clippy::unwrap_used)]
        let full_state = full_state.unwrap();
        assert_eq!(full_state.operations.len(), 3);
    }

    /// Tests that `operations_to_checkpoint_log` correctly converts a
    /// multi-page set of operations into a complete checkpoint log.
    #[test]
    fn operations_to_checkpoint_log_handles_multi_page_operations() {
        let ops = vec![
            make_step_op("step-1", "\"result-1\""),
            make_step_op("step-2", "\"result-2\""),
            make_step_op("step-3", "\"result-3\""),
        ];

        let log = operations_to_checkpoint_log(&ops);

        // All three operations should be in the log.
        assert!(log.get("step-1").is_some());
        assert!(log.get("step-2").is_some());
        assert!(log.get("step-3").is_some());

        #[allow(clippy::unwrap_used)]
        let r1 = log.get("step-1").unwrap();
        assert_eq!(r1.result.as_deref(), Some("\"result-1\""));
        assert_eq!(r1.status, CheckpointStatus::Succeeded);

        #[allow(clippy::unwrap_used)]
        let r3 = log.get("step-3").unwrap();
        assert_eq!(r3.result.as_deref(), Some("\"result-3\""));
    }

    // ── Checkpoint error classification ─────────────────────────────────

    use aws_sdk_lambda::error::SdkError;
    use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError as CkptErr;

    fn classify<E: Into<CkptErr>>(err: E) -> ClientError {
        classify_checkpoint_error(&SdkError::service_error(err.into(), ()))
    }

    fn invalid_parameter(message: Option<&str>) -> CkptErr {
        let mut builder = aws_sdk_lambda::types::error::InvalidParameterValueException::builder();
        if let Some(m) = message {
            builder = builder.message(m);
        }
        CkptErr::InvalidParameterValueException(builder.build())
    }

    #[test]
    fn throttling_and_5xx_classify_retryable() {
        let throttled = classify(CkptErr::TooManyRequestsException(
            aws_sdk_lambda::types::error::TooManyRequestsException::builder().build(),
        ));
        assert!(throttled.is_retryable());

        let server = classify(CkptErr::ServiceException(
            aws_sdk_lambda::types::error::ServiceException::builder().build(),
        ));
        assert!(server.is_retryable());
    }

    #[test]
    fn invalid_parameter_classifies_non_retryable() {
        assert!(!classify(invalid_parameter(Some("payload too large"))).is_retryable());
        // No message at all: no stale-token evidence, permanent rejection.
        assert!(!classify(invalid_parameter(None)).is_retryable());
    }

    /// The stale-token carve-out: `InvalidParameterValueException` whose
    /// message starts with `Invalid Checkpoint Token` resolves on
    /// re-invocation (the service issues a fresh token), so it is
    /// retryable. Prefix match only — a stale-token mention elsewhere in
    /// the message does not qualify.
    #[test]
    fn stale_token_rejection_classifies_retryable() {
        let stale = classify(invalid_parameter(Some(
            "Invalid Checkpoint Token: token has been superseded",
        )));
        assert!(stale.is_retryable());

        let not_prefix = classify(invalid_parameter(Some(
            "field X rejected (not an Invalid Checkpoint Token case)",
        )));
        assert!(!not_prefix.is_retryable());
    }

    #[test]
    fn kms_misconfiguration_classifies_non_retryable() {
        let kms = classify(CkptErr::KmsAccessDeniedException(
            aws_sdk_lambda::types::error::KmsAccessDeniedException::builder().build(),
        ));
        assert!(!kms.is_retryable());
    }

    /// Unknown checkpoint API errors default to RETRYABLE. Under the #43
    /// model a non-retryable classification fails the execution, so a
    /// novel transient error variant must not kill executions by default
    /// (the pre-#43 non-retryable default relied on the old model, where
    /// the surfaced error only failed the invocation — r3726032098).
    #[test]
    fn unknown_error_classifies_retryable() {
        let unknown = classify(CkptErr::unhandled("some new service error"));
        assert!(unknown.is_retryable());
    }

    #[test]
    fn transport_failures_classify_retryable() {
        let timeout: SdkError<CkptErr, ()> = SdkError::timeout_error("connect timed out");
        assert!(classify_checkpoint_error(&timeout).is_retryable());
    }

    /// Canary: pins the variant set of `CheckpointDurableExecutionError`,
    /// the upstream enum [`classify_checkpoint_error`] matches with a
    /// wildcard arm.
    ///
    /// The enum is `#[non_exhaustive]`, so the classifier's `match` cannot
    /// reject new variants at compile time — a variant added by an
    /// aws-sdk-lambda upgrade would silently take the unknown→retryable
    /// default. This test parses the enum out of the dependency's own
    /// source (the exact version Cargo.lock pins, in the local cargo
    /// registry) and fails when the variant set changes, forcing the
    /// classification to be revisited instead of defaulted. The sibling
    /// enum audit for `OperationStatus` lands with #45.
    #[test]
    fn checkpoint_error_variant_canary() {
        const CLASSIFIED_VARIANTS: &[&str] = &[
            "InvalidParameterValueException",
            "KmsAccessDeniedException",
            "KmsDisabledException",
            "KmsInvalidStateException",
            "KmsNotFoundException",
            "ServiceException",
            "TooManyRequestsException",
            "Unhandled",
        ];

        let source = read_sdk_source("src/operation/checkpoint_durable_execution.rs");
        let mut actual = extract_enum_variants(&source, "CheckpointDurableExecutionError");
        actual.sort_unstable();

        let mut expected: Vec<String> = CLASSIFIED_VARIANTS
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        expected.sort_unstable();

        assert_eq!(
            actual, expected,
            "aws-sdk-lambda's CheckpointDurableExecutionError variant set changed. \
             Reclassify the new/removed variants in classify_checkpoint_error \
             (src/client.rs), then update this canary's pinned list."
        );
    }

    /// Reads a source file out of the aws-sdk-lambda version Cargo.lock
    /// pins, from the local cargo registry (already extracted — the test
    /// binary that runs this compiled against it).
    #[allow(clippy::expect_used)] // reason: test helper assertions
    fn read_sdk_source(relative: &str) -> String {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let lockfile = std::fs::read_to_string(manifest_dir.join("Cargo.lock"))
            .expect("Cargo.lock readable at the workspace root");
        let version = lockfile
            .split("[[package]]")
            .find(|block| block.contains("name = \"aws-sdk-lambda\""))
            .and_then(|block| {
                block
                    .lines()
                    .find_map(|line| line.trim().strip_prefix("version = \""))
                    .map(|rest| rest.trim_end_matches('"').to_owned())
            })
            .expect("aws-sdk-lambda pinned in Cargo.lock");

        let cargo_home = std::env::var_os("CARGO_HOME").map_or_else(
            || {
                let home = std::env::var_os("HOME").expect("HOME set");
                std::path::PathBuf::from(home).join(".cargo")
            },
            std::path::PathBuf::from,
        );
        let registry_src = cargo_home.join("registry").join("src");
        let crate_dir_name = format!("aws-sdk-lambda-{version}");
        let crate_dir = std::fs::read_dir(&registry_src)
            .expect("cargo registry src directory readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path().join(&crate_dir_name))
            .find(|candidate| candidate.is_dir())
            .expect("pinned aws-sdk-lambda version extracted in the cargo registry");

        std::fs::read_to_string(crate_dir.join(relative)).expect("SDK source file readable")
    }

    /// Extracts the variant identifiers of a top-level `pub enum` from
    /// generated SDK source. The generated layout is stable: variants sit
    /// at one indentation level, as `Identifier(...)` or bare
    /// `Identifier,`.
    #[allow(clippy::panic, clippy::indexing_slicing, clippy::map_unwrap_or)]
    // reason: test helper assertions over generated-source text
    fn extract_enum_variants(source: &str, enum_name: &str) -> Vec<String> {
        let header = format!("pub enum {enum_name} {{");
        let start = source
            .find(&header)
            .unwrap_or_else(|| panic!("enum {enum_name} present in SDK source"));
        let body_start = start + header.len();
        let body_end = source[body_start..]
            .find("\n}")
            .map(|offset| body_start + offset)
            .unwrap_or_else(|| panic!("enum {enum_name} body terminated"));
        let body = &source[body_start..body_end];

        body.lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                // Variant lines start with an uppercase identifier followed
                // directly by `(` (tuple payload) or `,` (unit variant).
                // Doc comments, attributes (including multi-line
                // deprecation notes), and continuation lines do not.
                let first = trimmed.chars().next()?;
                if !first.is_ascii_uppercase() {
                    return None;
                }
                let ident: String = trimmed
                    .chars()
                    .take_while(char::is_ascii_alphanumeric)
                    .collect();
                let rest = trimmed.get(ident.len()..)?;
                if ident.is_empty() || !(rest.starts_with('(') || rest.starts_with(',')) {
                    return None;
                }
                Some(ident)
            })
            .collect()
    }
}
