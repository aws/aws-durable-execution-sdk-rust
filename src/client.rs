//! Execution client abstraction for checkpoint operations.
//!
//! This module provides the internal trait that the engine calls to
//! checkpoint operation results and retrieve execution state. The
//! production implementation is backed by `aws-sdk-lambda`.

use std::pin::Pin;
use std::sync::Mutex;
use std::time::Duration;

use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError;
use aws_sdk_lambda::operation::get_durable_execution_state::GetDurableExecutionStateError;
use aws_sdk_lambda::types::{Operation, OperationType, OperationUpdate};

use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};

// ────────────────────────────────────────────────────────────────────────────
// Error Classification
// ────────────────────────────────────────────────────────────────────────────

/// Whether a checkpoint failure should be retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetryClassification {
    /// Transient failure — retry with backoff.
    Retryable,
    /// Permanent failure — do not retry.
    NonRetryable,
}

/// Classifies a checkpoint `SdkError` into a retry decision.
///
/// Classification rules:
/// 1. `TooManyRequestsException` → retryable (throttling).
/// 2. `ServiceException` → retryable (server-side 5xx).
/// 3. Timeout/dispatch/network errors → retryable (transient).
/// 4. `InvalidParameterValueException`, KMS errors → non-retryable (client fault).
/// 5. Unknown structured errors → non-retryable. This default is deliberate:
///    treating an unrecognized service error as retryable risks a retry storm
///    against an error class we know nothing about, and a wrong non-retryable
///    is cushioned because the durable service re-invokes the execution
///    regardless, giving the call another chance on the next invocation.
pub(crate) fn classify_checkpoint_error(
    err: &aws_sdk_lambda::error::SdkError<CheckpointDurableExecutionError>,
) -> RetryClassification {
    match err {
        aws_sdk_lambda::error::SdkError::ServiceError(service_err) => {
            match service_err.err() {
                CheckpointDurableExecutionError::TooManyRequestsException(_)
                | CheckpointDurableExecutionError::ServiceException(_) => {
                    RetryClassification::Retryable
                }
                // Client faults: invalid params, KMS issues.
                CheckpointDurableExecutionError::InvalidParameterValueException(_)
                | CheckpointDurableExecutionError::KmsAccessDeniedException(_)
                | CheckpointDurableExecutionError::KmsDisabledException(_)
                | CheckpointDurableExecutionError::KmsInvalidStateException(_)
                | CheckpointDurableExecutionError::KmsNotFoundException(_) => {
                    RetryClassification::NonRetryable
                }
                // Unknown service errors default to non-retryable
                // deliberately (see rule 5 in the doc above): no retry storm
                // on an unrecognized error class, and the durable service's
                // re-invocation cushions a wrong non-retryable.
                _ => RetryClassification::NonRetryable,
            }
        }
        // Timeout, dispatch, and response errors are transient.
        aws_sdk_lambda::error::SdkError::TimeoutError(_)
        | aws_sdk_lambda::error::SdkError::DispatchFailure(_)
        | aws_sdk_lambda::error::SdkError::ResponseError(_) => RetryClassification::Retryable,
        // Construction failures are non-retryable.
        aws_sdk_lambda::error::SdkError::ConstructionFailure(_) => {
            RetryClassification::NonRetryable
        }
        _ => RetryClassification::NonRetryable,
    }
}

/// Classifies a `GetDurableExecutionState` error into a retry decision.
pub(crate) fn classify_get_state_error(
    err: &aws_sdk_lambda::error::SdkError<GetDurableExecutionStateError>,
) -> RetryClassification {
    match err {
        aws_sdk_lambda::error::SdkError::ServiceError(service_err) => match service_err.err() {
            GetDurableExecutionStateError::TooManyRequestsException(_)
            | GetDurableExecutionStateError::ServiceException(_) => RetryClassification::Retryable,
            GetDurableExecutionStateError::InvalidParameterValueException(_)
            | GetDurableExecutionStateError::KmsAccessDeniedException(_)
            | GetDurableExecutionStateError::KmsDisabledException(_)
            | GetDurableExecutionStateError::KmsInvalidStateException(_)
            | GetDurableExecutionStateError::KmsNotFoundException(_) => {
                RetryClassification::NonRetryable
            }
            // Unknown service errors default to non-retryable deliberately —
            // same rationale as [`classify_checkpoint_error`] rule 5.
            _ => RetryClassification::NonRetryable,
        },
        aws_sdk_lambda::error::SdkError::TimeoutError(_)
        | aws_sdk_lambda::error::SdkError::DispatchFailure(_)
        | aws_sdk_lambda::error::SdkError::ResponseError(_) => RetryClassification::Retryable,
        aws_sdk_lambda::error::SdkError::ConstructionFailure(_) => {
            RetryClassification::NonRetryable
        }
        _ => RetryClassification::NonRetryable,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Retry Parameters (3 attempts, 100ms base, 2s max)
// ────────────────────────────────────────────────────────────────────────────

/// Maximum number of attempts for a retryable checkpoint call.
const MAX_ATTEMPTS: u32 = 3;

/// Initial backoff delay before the first retry.
const BASE_DELAY: Duration = Duration::from_millis(100);

/// Maximum backoff delay cap.
const MAX_DELAY: Duration = Duration::from_secs(2);

/// Computes exponential backoff for the given zero-based attempt index.
/// `attempt` 0 → `BASE_DELAY`, 1 → 2×`BASE_DELAY`, capped at `MAX_DELAY`.
fn backoff_delay(attempt: u32) -> Duration {
    let delay = BASE_DELAY.saturating_mul(2_u32.saturating_pow(attempt));
    if delay > MAX_DELAY { MAX_DELAY } else { delay }
}

// ────────────────────────────────────────────────────────────────────────────
// `ExecutionClient` Trait
// ────────────────────────────────────────────────────────────────────────────

/// Checkpoint client error type wrapping the underlying cause.
#[derive(Debug)]
pub(crate) struct ClientError {
    message: String,
    retryable: bool,
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "durable: client: {}", self.message)
    }
}

impl std::error::Error for ClientError {}

impl ClientError {
    /// Whether this error is classified as retryable.
    pub(crate) fn retryable(&self) -> bool {
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

    fn from_retryable(message: String) -> Self {
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
    /// Implementations MUST handle retry internally for retryable errors.
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
            let mut last_err: Option<ClientError> = None;

            for attempt in 0..MAX_ATTEMPTS {
                let result = self
                    .client
                    .checkpoint_durable_execution()
                    .durable_execution_arn(&arn)
                    .checkpoint_token(&token)
                    .set_updates(Some(updates.clone()))
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
                        return Ok(CheckpointOutput {
                            checkpoint_token: new_token,
                            updated_operations: updated_ops,
                            next_marker,
                        });
                    }
                    Err(err) => {
                        let classification = classify_checkpoint_error(&err);
                        if classification == RetryClassification::NonRetryable {
                            return Err(ClientError::non_retryable(format!("{err}")));
                        }
                        last_err = Some(ClientError::from_retryable(format!("{err}")));
                        if attempt < MAX_ATTEMPTS - 1 {
                            tokio::time::sleep(backoff_delay(attempt)).await;
                        }
                    }
                }
            }
            Err(last_err.unwrap_or_else(|| {
                ClientError::from_retryable("retry attempts exhausted".to_owned())
            }))
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
                let mut last_err: Option<ClientError> = None;
                let mut page_result = None;

                for attempt in 0..MAX_ATTEMPTS {
                    let mut builder = self
                        .client
                        .get_durable_execution_state()
                        .durable_execution_arn(&arn)
                        .checkpoint_token(&token);

                    if let Some(ref m) = marker {
                        builder = builder.marker(m.as_str());
                    }

                    match builder.send().await {
                        Ok(output) => {
                            page_result = Some(output);
                            break;
                        }
                        Err(err) => {
                            let classification = classify_get_state_error(&err);
                            if classification == RetryClassification::NonRetryable {
                                return Err(ClientError::non_retryable(format!("{err}")));
                            }
                            last_err = Some(ClientError::from_retryable(format!("{err}")));
                            if attempt < MAX_ATTEMPTS - 1 {
                                tokio::time::sleep(backoff_delay(attempt)).await;
                            }
                        }
                    }
                }

                let Some(output) = page_result else {
                    return Err(last_err.unwrap_or_else(|| {
                        ClientError::from_retryable("retry attempts exhausted".to_owned())
                    }));
                };

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
// In-Memory Test Double
// ────────────────────────────────────────────────────────────────────────────

/// Injection point for controlling test double behavior per call.
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
    /// Token counter for generating unique tokens.
    token_counter: Mutex<u32>,
    /// All operation updates received across checkpoint calls.
    recorded_updates: Mutex<Vec<OperationUpdate>>,
}

impl InMemoryExecutionClient {
    /// Creates a new test double with the given pre-loaded state.
    pub(crate) fn new(state_operations: Vec<Operation>) -> Self {
        Self {
            state_operations: Mutex::new(state_operations),
            checkpoint_responses: Mutex::new(Vec::new()),
            checkpoint_call_count: Mutex::new(0),
            get_state_call_count: Mutex::new(0),
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

    /// Returns all operation updates recorded across checkpoint calls.
    pub(crate) fn recorded_updates(&self) -> Vec<OperationUpdate> {
        self.recorded_updates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

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

// ────────────────────────────────────────────────────────────────────────────
// Retry Wrapper
// ────────────────────────────────────────────────────────────────────────────

/// Performs a checkpoint call with internal retry-with-backoff.
///
/// The `InMemoryExecutionClient` returns one response per call (no
/// internal retry), so this wrapper provides the retry loop for testing
/// and production uniformly.
pub(crate) async fn checkpoint_with_retry(
    client: &dyn ExecutionClient,
    execution_arn: &str,
    checkpoint_token: &str,
    updates: Vec<OperationUpdate>,
) -> Result<CheckpointOutput, ClientError> {
    let mut last_err: Option<ClientError> = None;

    for attempt in 0..MAX_ATTEMPTS {
        match client
            .checkpoint(execution_arn, checkpoint_token, updates.clone())
            .await
        {
            Ok(output) => return Ok(output),
            Err(err) => {
                if !err.retryable() {
                    return Err(err);
                }
                last_err = Some(err);
                if attempt < MAX_ATTEMPTS - 1 {
                    tokio::time::sleep(backoff_delay(attempt)).await;
                }
            }
        }
    }

    Err(last_err
        .unwrap_or_else(|| ClientError::from_retryable("retry attempts exhausted".to_owned())))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a minimal HTTP response for `SdkError::service_error`.
    #[allow(clippy::expect_used)] // reason: test helper — valid status codes will never fail
    fn test_http_response(
        status: u16,
    ) -> aws_smithy_runtime_api::client::orchestrator::HttpResponse {
        aws_smithy_runtime_api::http::Response::new(
            aws_smithy_runtime_api::http::StatusCode::try_from(status)
                .expect("valid test status code"),
            aws_smithy_types::body::SdkBody::empty(),
        )
    }

    // ── Classification tests ────────────────────────────────────────────

    #[test]
    fn classify_too_many_requests_is_retryable() {
        let inner = CheckpointDurableExecutionError::TooManyRequestsException(
            aws_sdk_lambda::types::error::TooManyRequestsException::builder().build(),
        );
        let err = aws_sdk_lambda::error::SdkError::service_error(inner, test_http_response(429));
        assert_eq!(
            classify_checkpoint_error(&err),
            RetryClassification::Retryable
        );
    }

    #[test]
    fn classify_service_exception_is_retryable() {
        let inner = CheckpointDurableExecutionError::ServiceException(
            aws_sdk_lambda::types::error::ServiceException::builder().build(),
        );
        let err = aws_sdk_lambda::error::SdkError::service_error(inner, test_http_response(500));
        assert_eq!(
            classify_checkpoint_error(&err),
            RetryClassification::Retryable
        );
    }

    #[test]
    fn classify_invalid_parameter_is_non_retryable() {
        let inner = CheckpointDurableExecutionError::InvalidParameterValueException(
            aws_sdk_lambda::types::error::InvalidParameterValueException::builder().build(),
        );
        let err = aws_sdk_lambda::error::SdkError::service_error(inner, test_http_response(400));
        assert_eq!(
            classify_checkpoint_error(&err),
            RetryClassification::NonRetryable
        );
    }

    #[test]
    fn classify_kms_access_denied_is_non_retryable() {
        let inner = CheckpointDurableExecutionError::KmsAccessDeniedException(
            aws_sdk_lambda::types::error::KmsAccessDeniedException::builder().build(),
        );
        let err = aws_sdk_lambda::error::SdkError::service_error(inner, test_http_response(403));
        assert_eq!(
            classify_checkpoint_error(&err),
            RetryClassification::NonRetryable
        );
    }

    #[test]
    fn classify_timeout_is_retryable() {
        let err: aws_sdk_lambda::error::SdkError<CheckpointDurableExecutionError> =
            aws_sdk_lambda::error::SdkError::timeout_error(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "timed out",
            ));
        assert_eq!(
            classify_checkpoint_error(&err),
            RetryClassification::Retryable
        );
    }

    #[test]
    fn classify_construction_failure_is_non_retryable() {
        let err: aws_sdk_lambda::error::SdkError<CheckpointDurableExecutionError> =
            aws_sdk_lambda::error::SdkError::construction_failure("bad input");
        assert_eq!(
            classify_checkpoint_error(&err),
            RetryClassification::NonRetryable
        );
    }

    // ── Backoff computation ─────────────────────────────────────────────

    #[test]
    fn backoff_delays_are_exponential_and_capped() {
        assert_eq!(backoff_delay(0), Duration::from_millis(100));
        assert_eq!(backoff_delay(1), Duration::from_millis(200));
        assert_eq!(backoff_delay(2), Duration::from_millis(400));
        assert_eq!(backoff_delay(10), MAX_DELAY);
        assert_eq!(backoff_delay(100), MAX_DELAY);
    }

    // ── Retry loop with test double ─────────────────────────────────────

    #[tokio::test]
    async fn retry_succeeds_after_transient_failures() {
        tokio::time::pause();

        let client = InMemoryExecutionClient::new(Vec::new());
        client.enqueue_checkpoint_response(TestResponse::RetryableError("throttled".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::RetryableError("server error".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::Success(Vec::new()));

        let result = checkpoint_with_retry(&client, "arn:test", "token-0", Vec::new()).await;

        assert!(result.is_ok());
        let count = *client
            .checkpoint_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn retry_gives_up_after_max_attempts() {
        tokio::time::pause();

        let client = InMemoryExecutionClient::new(Vec::new());
        client.enqueue_checkpoint_response(TestResponse::RetryableError("t1".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::RetryableError("t2".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::RetryableError("t3".to_owned()));

        let result = checkpoint_with_retry(&client, "arn:test", "token-0", Vec::new()).await;

        assert!(result.is_err());
        #[allow(clippy::unwrap_used)] // reason: test assertion — err verified above
        let err = result.unwrap_err();
        assert!(err.retryable());
        let count = *client
            .checkpoint_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn non_retryable_error_surfaces_immediately() {
        let client = InMemoryExecutionClient::new(Vec::new());
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError("invalid".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::Success(Vec::new()));

        let result = checkpoint_with_retry(&client, "arn:test", "token-0", Vec::new()).await;

        assert!(result.is_err());
        #[allow(clippy::unwrap_used)] // reason: test assertion — err verified above
        let err = result.unwrap_err();
        assert!(!err.retryable());
        let count = *client
            .checkpoint_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn success_returns_checkpoint_output() {
        let client = InMemoryExecutionClient::new(Vec::new());
        let result = checkpoint_with_retry(&client, "arn:test", "token-0", Vec::new()).await;

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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
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
}
