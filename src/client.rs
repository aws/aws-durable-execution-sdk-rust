//! The checkpoint transport: the trait the SDK calls to checkpoint
//! operation updates and read execution state, its error type, and the
//! default transport backed by `aws-sdk-lambda`.

use std::pin::Pin;
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use aws_sdk_lambda::types::{Operation, OperationType, OperationUpdate};

use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};

/// The recovery scope of an [`ExecutionClientError`].
///
/// The scope decides what the SDK does with a transport failure. See the
/// [`ExecutionClient`] contract for each case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionClientErrorScope {
    /// The invocation fails; the service re-invokes and replay resumes.
    Invocation,
    /// The execution fails.
    Execution,
}

/// The terminal outcome a transport returns: a scope, a message and,
/// when the transport has one, the error that caused it.
///
/// [`source()`](std::error::Error::source) returns the error passed to
/// [`with_source`](Self::with_source).
///
/// A transport puts the detail a log needs into the message. The built-in
/// [`LambdaExecutionClient`] keeps the full `DisplayErrorContext` text of
/// the AWS SDK error in the message, and passes the final `SdkError` as the
/// source.
///
/// The name of this type is not a wire `ErrorType`, and its `Display` text
/// is not a wire `ErrorMessage`. An [`Execution`](ExecutionClientErrorScope::Execution)-scoped
/// error that ends the bootstrap state read records the AWS error code when
/// the Lambda transport supplied it, and `DurableExecutionClientError`
/// otherwise. A recorded message is the AWS error's message when the Lambda
/// transport supplied one, and the message passed to [`new`](Self::new) or
/// [`with_source`](Self::with_source) otherwise.
///
/// `Clone` because a coalesced checkpoint batch publishes one result to
/// every operation that contributed updates to it. The source is held in an
/// `Arc` for that reason.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecutionClientError {
    scope: ExecutionClientErrorScope,
    message: String,
    source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

impl std::fmt::Display for ExecutionClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "durable: client: {}", self.message)
    }
}

impl std::error::Error for ExecutionClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|s| s as &(dyn std::error::Error + 'static))
    }
}

impl ExecutionClientError {
    /// Creates an error with the given scope and message, and no source.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::{ExecutionClientError, ExecutionClientErrorScope};
    ///
    /// let err = ExecutionClientError::new(ExecutionClientErrorScope::Execution, "rejected");
    /// assert_eq!(err.scope(), ExecutionClientErrorScope::Execution);
    /// assert_eq!(err.to_string(), "durable: client: rejected");
    /// ```
    #[must_use]
    pub fn new(scope: ExecutionClientErrorScope, message: impl Into<String>) -> Self {
        Self {
            scope,
            message: message.into(),
            source: None,
        }
    }

    /// Creates an error with the given scope and message, keeping `source`
    /// as the error that caused it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error as _;
    /// use aws_durable_execution_sdk::{ExecutionClientError, ExecutionClientErrorScope};
    ///
    /// let cause = std::io::Error::other("connection reset");
    /// let err = ExecutionClientError::with_source(
    ///     ExecutionClientErrorScope::Invocation,
    ///     "checkpoint failed",
    ///     cause,
    /// );
    /// assert!(err.source().is_some());
    /// ```
    #[must_use]
    pub fn with_source(
        scope: ExecutionClientErrorScope,
        message: impl Into<String>,
        source: impl Into<crate::BoxError>,
    ) -> Self {
        Self {
            scope,
            message: message.into(),
            source: Some(Arc::from(source.into())),
        }
    }

    /// The recovery scope of this error.
    pub fn scope(&self) -> ExecutionClientErrorScope {
        self.scope
    }

    /// Whether this error fails only the invocation: `true` exactly for
    /// [`ExecutionClientErrorScope::Invocation`]. `false` means a permanent
    /// rejection: the terminal-`FAIL`-then-fail-the-execution path.
    pub(crate) fn is_retryable(&self) -> bool {
        self.scope == ExecutionClientErrorScope::Invocation
    }

    /// The error's message, without the `durable: client: ` prefix that
    /// `Display` adds.
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    /// Creates a non-retryable error from an owned message. Only test
    /// doubles call it: the Lambda transport builds every error with a
    /// source.
    #[cfg(test)]
    pub(crate) fn non_retryable(message: String) -> Self {
        Self::new(ExecutionClientErrorScope::Execution, message)
    }

    /// Creates a non-retryable error from a string slice (convenience).
    pub(crate) fn new_non_retryable(message: &str) -> Self {
        Self::new(ExecutionClientErrorScope::Execution, message)
    }

    /// Creates an `Invocation`-scoped error from an owned message. Only
    /// test doubles call it: the Lambda transport builds every error with a
    /// source.
    #[cfg(any(test, feature = "test-util"))]
    pub(crate) fn from_retryable(message: String) -> Self {
        Self::new(ExecutionClientErrorScope::Invocation, message)
    }
}

/// The message the SDK records for a transport error: the service's
/// message when the Lambda transport supplied an AWS error, and the
/// transport's own message otherwise.
///
/// It reads, in this order:
///
/// 1. The `message()` of the AWS error, when the source is the final
///    `SdkError` of `CheckpointDurableExecution` or
///    `GetDurableExecutionState` and that error has a message. JS reads
///    `awsError.message` the same way.
/// 2. Otherwise [`ExecutionClientError::message`], without the
///    `durable: client: ` prefix of `Display`, as JS reads the `message` of
///    the error the transport threw.
pub(crate) fn client_error_record_message(err: &ExecutionClientError) -> String {
    use aws_sdk_lambda::error::ProvideErrorMetadata;
    use std::error::Error as _;

    if let Some(source) = err.source() {
        if let Some(sdk) = source.downcast_ref::<CheckpointSdkError>()
            && let Some(m) = sdk.message()
        {
            return m.to_owned();
        }
        if let Some(sdk) = source.downcast_ref::<GetStateSdkError>()
            && let Some(m) = sdk.message()
        {
            return m.to_owned();
        }
    }
    err.message().to_owned()
}

/// The wire `ErrorType` of the bootstrap `FAILED` record for `err`.
///
/// When the source is the final `SdkError` of `GetDurableExecutionState`
/// and it carries an AWS error code, the type is that code, as JS records
/// the AWS error's `name`. Otherwise the type is
/// [`EXECUTION_CLIENT_ERROR_TYPE`](crate::error::EXECUTION_CLIENT_ERROR_TYPE),
/// as JS records for its `DurableExecutionClientError`.
pub(crate) fn bootstrap_error_type(err: &ExecutionClientError) -> String {
    use aws_sdk_lambda::error::ProvideErrorMetadata;
    use std::error::Error as _;

    err.source()
        .and_then(|s| s.downcast_ref::<GetStateSdkError>())
        .and_then(ProvideErrorMetadata::code)
        .unwrap_or(crate::error::EXECUTION_CLIENT_ERROR_TYPE)
        .to_owned()
}

/// The final error of a `CheckpointDurableExecution` call.
type CheckpointSdkError = aws_sdk_lambda::error::SdkError<
    aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError,
    aws_sdk_lambda::config::http::HttpResponse,
>;

/// The final error of a `GetDurableExecutionState` call.
type GetStateSdkError = aws_sdk_lambda::error::SdkError<
    aws_sdk_lambda::operation::get_durable_execution_state::GetDurableExecutionStateError,
    aws_sdk_lambda::config::http::HttpResponse,
>;

/// The output of a successful [`ExecutionClient::checkpoint`] call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CheckpointOutput {
    /// The new checkpoint token for subsequent calls. `None` when the
    /// service returned no token.
    pub(crate) checkpoint_token: Option<String>,
    /// Updated operations from the backend (may be empty).
    pub(crate) updated_operations: Vec<Operation>,
    /// Pagination marker: if present, more operations are available via
    /// `get_state` and the caller must paginate.
    pub(crate) next_marker: Option<String>,
}

impl CheckpointOutput {
    /// Creates a checkpoint output.
    ///
    /// `checkpoint_token` is the token for the next request, or `None` when
    /// the service returned none. `next_marker` is the pagination marker of
    /// the returned state, or `None` when `updated_operations` is complete.
    #[must_use]
    pub fn new(
        checkpoint_token: Option<String>,
        updated_operations: Vec<Operation>,
        next_marker: Option<String>,
    ) -> Self {
        Self {
            checkpoint_token,
            updated_operations,
            next_marker,
        }
    }

    /// The token for the next request. An empty token reads as `None`.
    pub fn checkpoint_token(&self) -> Option<&str> {
        self.checkpoint_token.as_deref().filter(|t| !t.is_empty())
    }
}

/// One page of execution state, returned by [`ExecutionClient::get_state`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GetStateOutput {
    /// The operations of this page.
    pub(crate) operations: Vec<Operation>,
    /// The marker of the next page, `None` on the last page.
    pub(crate) next_marker: Option<String>,
}

impl GetStateOutput {
    /// Creates one page of execution state. `next_marker` is the marker of
    /// the next page, or `None` on the last page.
    #[must_use]
    pub fn new(operations: Vec<Operation>, next_marker: Option<String>) -> Self {
        Self {
            operations,
            next_marker,
        }
    }

    /// The marker of the next page. An empty marker reads as `None`.
    pub fn next_marker(&self) -> Option<&str> {
        self.next_marker.as_deref().filter(|m| !m.is_empty())
    }
}

/// The transport for the two durable execution service calls.
///
/// The SDK uses [`LambdaExecutionClient`] unless
/// [`OptionsBuilder::execution_client`](crate::OptionsBuilder::execution_client)
/// supplies another implementation. The trait is object-safe, so a
/// transport is passed as `Arc<dyn ExecutionClient>`.
///
/// # Contract
///
/// 1. [`checkpoint`](Self::checkpoint) sends `updates` in order in one
///    request. It returns the new token, the operations the backend
///    returned, and the pagination marker of the returned state.
/// 2. The returned token is `None` when the service accepted the request
///    and returned no token for the next one. An empty token is reported
///    as `None`.
/// 3. [`get_state`](Self::get_state) returns one page of the execution's
///    operations and the marker of the next page, or `None` on the last
///    page. The SDK follows the marker.
/// 4. Transient retry is the transport's job. An error the method returns
///    is final for the invocation.
/// 5. An error's [`scope`](ExecutionClientError::scope) decides the
///    outcome:
///    - [`Invocation`](ExecutionClientErrorScope::Invocation) fails the
///      invocation; the service re-invokes and replay resumes.
///    - [`Execution`](ExecutionClientErrorScope::Execution) from
///      `checkpoint` records a terminal `FAIL` for the operation and fails
///      the execution. The operation's terminal `FAIL` has `ErrorType`
///      `CheckpointFailedError`, the record that replay reads if the
///      invocation runs again. The execution's `FAILED` record has
///      `ErrorType` `CheckpointUnrecoverableExecutionError` and
///      `ErrorMessage` `[Unrecoverable Execution] Checkpoint failed: {m}`.
///      `{m}` is the AWS error's message when the Lambda transport supplied
///      one, and the error's own message otherwise.
///    - `Execution` from the bootstrap `get_state` returns a `FAILED`
///      response. Its `ErrorType` is the AWS error code when the Lambda
///      transport supplied one, and `DurableExecutionClientError`
///      otherwise. Its `ErrorMessage` is `{m}` as above.
pub trait ExecutionClient: Send + Sync {
    /// Checkpoints operation updates and returns the next token.
    fn checkpoint(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>;

    /// Reads one page of the execution state. `None` reads the first page,
    /// and `Some(m)` reads the page after marker `m`.
    fn get_state(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
        marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>;
}

/// The page size of a `GetDurableExecutionState` request, as in JS.
const GET_STATE_MAX_ITEMS: i32 = 1000;

/// The transport the SDK uses when `execution_client` is not set. Wrap it
/// to decorate the default.
///
/// Retry is delegated entirely to the wrapped SDK client's own retry
/// strategy (standard retry by default: jittered exponential backoff, 3
/// attempts). This matches the sibling JS and Python SDKs, which rely on
/// their AWS SDKs' built-in retry. A call that exhausts the SDK's retries
/// fails the invocation; the durable execution service re-invokes the
/// handler, which is the recovery path: longer in-process retry would
/// only delay it. Callers who need a different retry policy supply a
/// preconfigured client.
///
/// Every final error carries the final `SdkError` as its
/// [`source`](std::error::Error::source).
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use aws_durable_execution_sdk::{
///     ExecutionClient, LambdaClient, LambdaExecutionClient, Options, SdkConfig,
/// };
///
/// let client = LambdaClient::new(&SdkConfig::builder().build());
/// let transport: Arc<dyn ExecutionClient> = Arc::new(LambdaExecutionClient::new(client));
/// let options = Options::builder()
///     .execution_client(transport)
///     .build()
///     .expect("valid config");
/// # drop(options);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct LambdaExecutionClient {
    client: aws_sdk_lambda::Client,
}

impl std::fmt::Debug for LambdaExecutionClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LambdaExecutionClient")
    }
}

impl LambdaExecutionClient {
    /// Creates a transport that calls the service through `client`.
    #[must_use]
    pub fn new(client: crate::LambdaClient) -> Self {
        Self { client }
    }
}

impl ExecutionClient for LambdaExecutionClient {
    fn checkpoint(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
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
                    let new_token = output.checkpoint_token.filter(|t| !t.is_empty());
                    let (updated_ops, next_marker) = match output.new_execution_state {
                        Some(state) => (
                            state.operations.unwrap_or_default(),
                            state.next_marker.filter(|m| !m.is_empty()),
                        ),
                        None => (Vec::new(), None),
                    };
                    Ok(CheckpointOutput::new(new_token, updated_ops, next_marker))
                }
                // The SDK's standard retry has already retried everything
                // transient; the final error is classified into a recovery
                // scope (invocation vs execution): see
                // `classify_checkpoint_error`.
                Err(err) => Err(classify_checkpoint_error(err)),
            }
        })
    }

    /// Sends one `GetDurableExecutionState` request with `MaxItems` 1000.
    /// The four KMS errors are `Execution`-scoped, and every other final
    /// error, a plain 4xx included, is `Invocation`-scoped, as in JS.
    fn get_state(
        &self,
        execution_arn: &str,
        checkpoint_token: &str,
        marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
        let arn = execution_arn.to_owned();
        let token = checkpoint_token.to_owned();
        let marker = marker.map(str::to_owned);
        Box::pin(async move {
            // As with `checkpoint`, the SDK's standard retry owns the
            // transient-failure retry; the final error is classified by
            // `classify_get_state_error`.
            let output = self
                .client
                .get_durable_execution_state()
                .durable_execution_arn(&arn)
                .checkpoint_token(&token)
                .set_marker(marker)
                .max_items(GET_STATE_MAX_ITEMS)
                .send()
                .await
                .map_err(classify_get_state_error)?;

            Ok(GetStateOutput::new(
                output.operations,
                output.next_marker.filter(|m| !m.is_empty()),
            ))
        })
    }
}

/// Reads the whole execution state: calls [`ExecutionClient::get_state`]
/// with no marker, then with each returned marker until
/// [`GetStateOutput::next_marker`] is `None`, and concatenates the pages.
pub(crate) async fn get_full_state(
    client: &dyn ExecutionClient,
    execution_arn: &str,
    checkpoint_token: &str,
) -> Result<Vec<Operation>, ExecutionClientError> {
    let mut operations = Vec::new();
    let mut marker: Option<String> = None;
    loop {
        let page = client
            .get_state(execution_arn, checkpoint_token, marker.as_deref())
            .await?;
        let next = page.next_marker().map(str::to_owned);
        operations.extend(page.operations);
        match next {
            Some(m) => marker = Some(m),
            None => return Ok(operations),
        }
    }
}

/// The message prefix the service puts on an `InvalidParameterValueException`
/// that rejects a stale checkpoint token.
///
/// A stale token resolves on re-invocation (the service issues a fresh one),
/// so this specific rejection is classified retryable: the carve-out the
/// Java SDK's `DurableApiErrorClassifier` applies.
const STALE_TOKEN_MESSAGE_PREFIX: &str = "Invalid checkpoint token";

/// Classifies a final `CheckpointDurableExecution` error into a recovery
/// scope, carried as [`ExecutionClientError::scope`]. The final error is
/// kept as the source.
///
/// By the time this runs, the aws-sdk's standard retry has exhausted its
/// transport-level attempts, so the question is no longer "try again now"
/// but *which recovery converges*:
///
/// - **Retryable**: fail the invocation; the durable service re-invokes
///   and replay resumes from the recorded state. This is the same recovery
///   as an interruption. Applied to transport-level failures (dispatch,
///   timeout, a malformed response without a 4xx status), throttling (any
///   429), 5xx, and the stale-token carve-out
///   ([`STALE_TOKEN_MESSAGE_PREFIX`]).
/// - **Non-retryable**: the service permanently rejected this write (for
///   example an oversized payload, a KMS misconfiguration, or any other
///   4xx except 429). Re-invoking re-runs the body and hits
///   the same rejection, so the caller persists a small terminal `FAIL`
///   for the operation and fails the execution instead.
///
/// After the modeled variants, the classifier reads the HTTP status of the
/// raw response, as JS reads `$metadata.httpStatusCode`. A status from 400
/// to 499 other than 429 is non-retryable: the service rejected the
/// request, so the same write fails the same way on every invocation. This
/// covers an unmodeled service error, such as a 403
/// `AccessDeniedException` when the role lacks
/// `lambda:CheckpointDurableExecution`, and a response the SDK could not
/// parse. Every other status, and an error without a raw response
/// (dispatch failure, timeout, construction failure), is retryable.
///
/// **An unmodeled error is retryable unless its status is a 4xx other
/// than 429.** The original classification
/// (pre-#43) defaulted unknown to non-retryable, with the rationale, from
/// the r3726032098 review thread, that the surfaced error only failed the
/// invocation and the service re-invoked anyway, so the default was
/// harmless. That rationale belongs to the old model: under this model a
/// non-retryable classification fails the *execution*, so a novel
/// transient error variant defaulting to non-retryable would kill
/// executions. A novel transient error arrives as 429, 5xx or a transport
/// failure, and those stay retryable; a 4xx other than 429 is a permanent
/// rejection, which JS also fails the execution on. Java and Python
/// default unknown to retryable for the same reason. The
/// `checkpoint_error_variant_canary` test pins the variant set this
/// function classifies explicitly, so a new upstream variant fails CI for
/// reclassification instead of falling silently into the status rule.
fn classify_checkpoint_error(err: CheckpointSdkError) -> ExecutionClientError {
    let message = format!("{}", aws_sdk_lambda::error::DisplayErrorContext(&err));
    let scope = checkpoint_error_scope(&err);
    ExecutionClientError::with_source(scope, message, err)
}

/// The recovery scope of a final `CheckpointDurableExecution` error: see
/// [`classify_checkpoint_error`].
fn checkpoint_error_scope(err: &CheckpointSdkError) -> ExecutionClientErrorScope {
    use ExecutionClientErrorScope::{Execution, Invocation};
    use aws_sdk_lambda::error::SdkError;
    use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError as E;

    if let SdkError::ServiceError(service_err) = err {
        match service_err.err() {
            // Throttling and service-side 5xx: transient by definition.
            E::TooManyRequestsException(_) | E::ServiceException(_) => return Invocation,
            // A parameter rejection is deterministic, the same write fails
            // the same way on every lap, EXCEPT the stale-token rejection,
            // which a re-invocation resolves with a fresh token.
            E::InvalidParameterValueException(inner) => {
                return if inner
                    .message()
                    .is_some_and(|m| m.starts_with(STALE_TOKEN_MESSAGE_PREFIX))
                {
                    Invocation
                } else {
                    Execution
                };
            }
            // KMS misconfiguration on the function: deterministic until an
            // operator fixes the key, so re-invoking loops without progress.
            E::KmsAccessDeniedException(_)
            | E::KmsDisabledException(_)
            | E::KmsInvalidStateException(_)
            | E::KmsNotFoundException(_) => return Execution,
            // Unmodeled errors fall through to the status rule. The variant
            // canary test keeps this arm from silently absorbing new modeled
            // variants.
            _ => {}
        }
    }

    // The status rule: see the function docs. Errors without a raw response
    // (dispatch failure, timeout, construction failure) are channel-level:
    // nothing was rejected.
    match err.raw_response().map(|r| r.status().as_u16()) {
        Some(status) if (400..500).contains(&status) && status != 429 => Execution,
        _ => Invocation,
    }
}

/// Classifies a final `GetDurableExecutionState` error into a recovery
/// scope, carried as [`ExecutionClientError::scope`]. The final error is
/// kept as the source.
///
/// The four KMS errors are non-retryable: the function's key is
/// misconfigured, so every re-invocation fails the same way. Every other
/// error is retryable, a plain 4xx included, because JS rethrows every
/// other error of its bootstrap state read and so fails only the
/// invocation. The message is the `DisplayErrorContext` text, so it names
/// the exception.
fn classify_get_state_error(err: GetStateSdkError) -> ExecutionClientError {
    use aws_sdk_lambda::operation::get_durable_execution_state::GetDurableExecutionStateError as E;

    let message = format!("{}", aws_sdk_lambda::error::DisplayErrorContext(&err));
    let scope = match err.as_service_error() {
        Some(
            E::KmsAccessDeniedException(_)
            | E::KmsDisabledException(_)
            | E::KmsInvalidStateException(_)
            | E::KmsNotFoundException(_),
        ) => ExecutionClientErrorScope::Execution,
        _ => ExecutionClientErrorScope::Invocation,
    };
    ExecutionClientError::with_source(scope, message, err)
}

/// An HTTP error response as the service sends it: the AWS error code in
/// the `x-amzn-errortype` header and the message in the JSON body. The
/// generated deserializer reads both into the error's metadata, which a
/// modeled error built in a test does not carry.
#[cfg(test)]
pub(crate) fn mock_aws_error_response(
    status: u16,
    code: &str,
    message: &str,
) -> aws_sdk_lambda::config::http::HttpResponse {
    let body = serde_json::json!({ "Type": "User", "message": message }).to_string();
    let mut response = aws_sdk_lambda::config::http::HttpResponse::new(
        aws_smithy_runtime_api::http::StatusCode::try_from(status).expect("valid HTTP status"),
        aws_smithy_types::body::SdkBody::from(body),
    );
    response
        .headers_mut()
        .insert("x-amzn-errortype", code.to_owned());
    response
        .headers_mut()
        .insert("content-type", "application/json");
    response
}

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
/// Sufficient for unit tests, not the full `test-util` `LocalRunner`.
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
    /// Counter for backend-style callback ID assignment on Callback START.
    callback_counter: Mutex<u32>,
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
            callback_counter: Mutex::new(0),
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
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
        {
            let mut count = self
                .checkpoint_call_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *count += 1;
        }

        // Mirror the backend contract: the Callback START response carries
        // the backend-assigned callback ID (issue #47). Assign one per
        // Callback START update so the default success path behaves like
        // the real service; explicitly enqueued responses stay
        // authoritative and receive no auto-assigned operations.
        let auto_ops: Vec<Operation> = updates
            .iter()
            .filter(|u| {
                u.r#type() == &OperationType::Callback
                    && u.action() == &aws_sdk_lambda::types::OperationAction::Start
            })
            .map(|u| {
                let mut counter = self
                    .callback_counter
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *counter += 1;
                Operation::builder()
                    .id(u.id())
                    .r#type(OperationType::Callback)
                    .status(aws_sdk_lambda::types::OperationStatus::Pending)
                    .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                    .callback_details(
                        aws_sdk_lambda::types::CallbackDetails::builder()
                            .callback_id(format!("in-mem-cb-{counter}"))
                            .build(),
                    )
                    .build()
                    .expect("all required Operation fields set")
            })
            .collect();

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
                Some(TestResponse::RetryableError(msg)) => {
                    Err(ExecutionClientError::from_retryable(msg))
                }
                Some(TestResponse::NonRetryableError(msg)) => {
                    Err(ExecutionClientError::non_retryable(msg))
                }
                Some(TestResponse::Success(ops)) => {
                    let mut counter = self
                        .token_counter
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *counter += 1;
                    Ok(CheckpointOutput {
                        checkpoint_token: Some(format!("token-{counter}")),
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
                        checkpoint_token: Some(format!("token-{counter}")),
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
                        checkpoint_token: Some(format!("token-{counter}")),
                        updated_operations: auto_ops,
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
        _marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
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
            return Box::pin(async move { Err(ExecutionClientError::non_retryable(message)) });
        }

        let ops = self
            .state_operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();

        Box::pin(async move { Ok(GetStateOutput::new(ops, None)) })
    }
}

/// Converts a single SDK `Operation` into a `(wire_id, CheckpointRecord)` pair.
fn operation_to_record(op: &Operation) -> (String, CheckpointRecord) {
    let attempt = op.step_details.as_ref().map_or(0, |s| {
        #[expect(clippy::cast_sign_loss)] // reason: attempt is non-negative from backend
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
        // ChainedInvoke payloads live in their own record fields: mirroring
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
/// Keyed by wire ID (the operation's `id` field from the service). Only
/// tests build a log this way: bootstrap merges the fetched pages into the
/// inline log in [`resolve_bootstrap_log`].
#[cfg(test)]
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

/// Resolves the bootstrap checkpoint log and the execution input, following
/// the pagination marker.
///
/// This is the single production decision point for bootstrap pagination.
/// `first_page` is the log parsed from the inline
/// `InitialExecutionState`, and `next_marker` is its `NextMarker`.
/// `first_page_input` is `None` when the inline page holds no EXECUTION
/// operation, and `Some(input)` when it holds one, with `input` that
/// operation's `InputPayload`.
///
/// - Without a marker the inline page is the whole state. The function
///   returns `first_page` and the inline input unchanged, and calls nothing.
/// - With a marker the function keeps `first_page`. It calls
///   [`ExecutionClient::get_state`] with the inline marker, merges each
///   returned page into the log, and calls again with each returned marker
///   until a page has none. It never reads the first page again.
///
/// The returned input is the inline input when the inline page holds the
/// EXECUTION operation. Otherwise it is the `InputPayload` of the first
/// EXECUTION operation of the fetched pages, in page order, or `None`
/// when no page holds one. The caller parses it after this function
/// returns, so an input on a later page reaches the handler.
pub(crate) async fn resolve_bootstrap_log(
    client: &dyn ExecutionClient,
    execution_arn: &str,
    checkpoint_token: &str,
    first_page: CheckpointLog,
    first_page_input: Option<Option<String>>,
    next_marker: Option<&str>,
) -> Result<(CheckpointLog, Option<String>), ExecutionClientError> {
    let mut input = first_page_input;
    let mut marker = next_marker.map(str::to_owned);
    while let Some(current) = marker {
        let page = client
            .get_state(execution_arn, checkpoint_token, Some(&current))
            .await?;
        marker = page.next_marker().map(str::to_owned);
        // The checkpoint log skips EXECUTION operations, so the input is
        // read from the raw page before the page is merged.
        if input.is_none() {
            input = page
                .operations
                .iter()
                .find(|op| op.r#type == OperationType::Execution)
                .map(|op| {
                    op.execution_details
                        .as_ref()
                        .and_then(|d| d.input_payload.clone())
                });
        }
        for op in page
            .operations
            .iter()
            .filter(|op| op.r#type != OperationType::Execution)
        {
            let (wire_id, record) = operation_to_record(op);
            first_page.insert(wire_id, record);
        }
    }
    Ok((first_page, input.flatten()))
}

/// Maps SDK `OperationStatus` to internal `CheckpointStatus`.
///
/// Every modeled variant maps explicitly; anything else, the generated
/// enum's unknown-variant carrier, or a variant added by an
/// aws-sdk-lambda upgrade before this list is revisited, maps to
/// [`CheckpointStatus::Unknown`] carrying the raw wire value, which replay
/// refuses to interpret (issue #45). The `operation_status_variant_canary`
/// test pins the modeled variant set, so a new upstream variant fails CI
/// for explicit classification instead of shipping through `Unknown`.
fn sdk_status_to_checkpoint_status(
    status: &aws_sdk_lambda::types::OperationStatus,
) -> CheckpointStatus {
    match status {
        aws_sdk_lambda::types::OperationStatus::Started => CheckpointStatus::Started,
        aws_sdk_lambda::types::OperationStatus::Pending => CheckpointStatus::Pending,
        aws_sdk_lambda::types::OperationStatus::Ready => CheckpointStatus::Ready,
        aws_sdk_lambda::types::OperationStatus::Succeeded => CheckpointStatus::Succeeded,
        aws_sdk_lambda::types::OperationStatus::Failed => CheckpointStatus::Failed,
        aws_sdk_lambda::types::OperationStatus::Cancelled => CheckpointStatus::Cancelled,
        aws_sdk_lambda::types::OperationStatus::TimedOut => CheckpointStatus::TimedOut,
        aws_sdk_lambda::types::OperationStatus::Stopped => CheckpointStatus::Stopped,
        other => CheckpointStatus::Unknown(other.as_str().to_owned()),
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

#[cfg(test)]
mod tests {
    use super::*;

    // Request mapping (aws-smithy-mocks)
    //
    // These tests mock the Lambda SDK client at the HTTP layer and pin the
    // per-operation request mapping of `LambdaExecutionClient`: what the
    // client puts on the wire, how it maps responses back, and how the
    // final error (after the SDK's own retry) maps into `ExecutionClientError`.

    use aws_sdk_lambda::operation::checkpoint_durable_execution::{
        CheckpointDurableExecutionError, CheckpointDurableExecutionOutput,
    };
    use aws_sdk_lambda::operation::get_durable_execution_state::{
        GetDurableExecutionStateError, GetDurableExecutionStateOutput,
    };
    use aws_sdk_lambda::types::{CheckpointUpdatedExecutionState, OperationAction};
    use aws_smithy_mocks::{RuleMode, mock, mock_client};

    /// Builds a minimal `OperationUpdate` for request-mapping assertions.
    fn make_update(id: &str, action: OperationAction) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Step)
            .action(action)
            .build()
            .expect("valid OperationUpdate")
    }

    /// `checkpoint` maps its arguments onto the `CheckpointDurableExecution`
    /// request, ARN, token, and every update, and maps the response's
    /// token, updated operations, and pagination marker back out.
    #[tokio::test]
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

        assert_eq!(output.checkpoint_token(), Some("tok-out"));
        assert_eq!(output.updated_operations.len(), 1);
        assert_eq!(
            output.updated_operations.first().map(Operation::id),
            Some("op-1")
        );
        assert_eq!(output.next_marker.as_deref(), Some("more-pages"));
        assert_eq!(rule.num_calls(), 1);
    }

    /// The wire ID of the first root operation: the SHA-256 of `"1"`.
    const FIRST_ROOT_WIRE_ID: &str =
        "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b";

    /// Drives `wrap` over `sdk_client` with a one-step handler whose step
    /// the initial state already records as `STARTED`, so the step writes
    /// no START and its first request is its SUCCEED. Returns the response.
    async fn invoke_started_step(sdk_client: aws_sdk_lambda::Client) -> serde_json::Value {
        let service = crate::wrap(
            |_event: serde_json::Value, ctx: crate::DurableContext| async move {
                let value: u32 = ctx.step(|_| async { Ok(7_u32) }).name("one").await?;
                Ok::<u32, crate::BoxError>(value)
            },
            crate::Options::builder()
                .execution_client(Arc::new(LambdaExecutionClient::new(sdk_client)))
                .build()
                .expect("options build"),
        );
        let event = lambda_runtime::LambdaEvent::new(
            crate::InvocationPayload(serde_json::json!({
                "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
                "CheckpointToken": "token-abc",
                "InitialExecutionState": {
                    "Operations": [
                        {
                            "Id": "root",
                            "Type": "Execution",
                            "Status": "STARTED",
                            "ExecutionDetails": { "InputPayload": "{}" }
                        },
                        {
                            "Id": FIRST_ROOT_WIRE_ID,
                            "Type": "Step",
                            "SubType": "Step",
                            "Name": "one",
                            "Status": "STARTED",
                            "StepDetails": { "Attempt": 1 }
                        }
                    ]
                }
            })),
            lambda_runtime::Context::default(),
        );
        service(event)
            .await
            .map(|response| response.0)
            .expect("a tokenless response fails the execution, not the invocation")
    }

    /// Every mocked checkpoint response has no `CheckpointToken`. The
    /// transport reports `None`, and `DurableContext` maps it to the
    /// non-retryable error the transport returned before, which fails the
    /// execution after the step's terminal `FAIL`.
    #[tokio::test]
    async fn checkpoint_output_without_token_is_non_retryable_until_withdrawal_lands() {
        use std::sync::{Arc, Mutex as StdMutex};

        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .then_output(|| CheckpointDurableExecutionOutput::builder().build());
        let client = LambdaExecutionClient::new(mock_client!(aws_sdk_lambda, [&rule]));
        let output = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect("a tokenless response is Ok");
        assert_eq!(output.checkpoint_token(), None);

        // Capture each request's updates, so the terminal FAIL can be read
        // from the second request.
        let requests: Arc<StdMutex<Vec<Vec<OperationUpdate>>>> = Arc::default();
        let seen = Arc::clone(&requests);
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .match_requests(move |req| {
                seen.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(req.updates().to_vec());
                true
            })
            .then_output(|| CheckpointDurableExecutionOutput::builder().build());
        let response =
            invoke_started_step(mock_client!(aws_sdk_lambda, RuleMode::MatchAny, [&rule])).await;

        assert_eq!(
            response.get("Status").and_then(serde_json::Value::as_str),
            Some("FAILED"),
            "unexpected response: {response}"
        );
        let error = response
            .get("Error")
            .expect("a FAILED envelope has an Error");
        assert_eq!(
            error.get("ErrorType").and_then(serde_json::Value::as_str),
            Some("CheckpointUnrecoverableExecutionError")
        );
        assert_eq!(
            error
                .get("ErrorMessage")
                .and_then(serde_json::Value::as_str),
            Some(
                "[Unrecoverable Execution] Checkpoint failed: backend returned no checkpoint token"
            )
        );

        let requests = requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(requests.len(), 2, "{requests:?}");
        let fail = requests
            .get(1)
            .and_then(|second| second.iter().find(|u| u.action() == &OperationAction::Fail))
            .expect("the second request carries the step's terminal FAIL");
        assert_eq!(fail.id(), FIRST_ROOT_WIRE_ID);
        assert_eq!(fail.r#type(), &OperationType::Step);
        assert_eq!(
            fail.error().and_then(|e| e.error_type()),
            Some(crate::error::CHECKPOINT_FAILED_ERROR_TYPE)
        );
    }

    /// `ExecutionClientError::new` keeps the scope and the message, has no
    /// source, and `is_retryable` derives from the scope.
    #[test]
    fn execution_client_error_new_keeps_scope_and_message() {
        use std::error::Error as _;

        let err = ExecutionClientError::new(ExecutionClientErrorScope::Execution, "x");
        assert_eq!(err.scope(), ExecutionClientErrorScope::Execution);
        assert!(!err.is_retryable());
        assert!(err.source().is_none());
        assert_eq!(err.to_string(), "durable: client: x");

        let err = ExecutionClientError::new(ExecutionClientErrorScope::Invocation, "x");
        assert_eq!(err.scope(), ExecutionClientErrorScope::Invocation);
        assert!(err.is_retryable());
    }

    /// The Lambda transport keeps the final `SdkError` as the source, and
    /// keeps today's `DisplayErrorContext` text as the message.
    #[tokio::test]
    async fn lambda_transport_error_source_downcasts_to_sdk_error() {
        use std::error::Error as _;

        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .sequence()
            .http_status(403, None)
            .build();
        let client = LambdaExecutionClient::new(mock_client!(aws_sdk_lambda, [&rule]));
        let err = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect_err("a 403 fails the call");

        let source = err.source().expect("the transport keeps the SdkError");
        let sdk = source
            .downcast_ref::<CheckpointSdkError>()
            .expect("the source is the final SdkError");
        let text = format!("{}", aws_sdk_lambda::error::DisplayErrorContext(sdk));
        assert_eq!(err.message(), text);
        assert_eq!(format!("{err:#}"), format!("durable: client: {text}"));
        assert_eq!(err.scope(), ExecutionClientErrorScope::Execution);
    }

    /// A checkpoint call whose final outcome is a modeled parameter
    /// rejection, a deterministic client fault the SDK does not retry,
    /// classifies NON-retryable: re-invoking replays into the same
    /// rejection, so the recovery is a terminal `FAIL` write followed by
    /// failing the execution, not another lap.
    #[tokio::test]
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

    /// The backend reports stale checkpoint tokens as an
    /// `InvalidParameterValueException` whose message starts with
    /// `Invalid checkpoint token`; the production client must classify
    /// that modeled 4xx as retryable so the invocation fails and durable
    /// execution can resume with a fresh token.
    #[tokio::test]
    async fn checkpoint_stale_token_error_maps_to_retryable_client_error() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution).then_error(|| {
            CheckpointDurableExecutionError::InvalidParameterValueException(
                aws_sdk_lambda::types::error::InvalidParameterValueException::builder()
                    .message("Invalid checkpoint token: token has been superseded")
                    .build(),
            )
        });
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect_err("stale checkpoint token must fail the call");
        assert!(err.is_retryable());
        assert_eq!(rule.num_calls(), 1, "a client fault is not retried");
    }

    /// Transient-failure retry belongs to the aws-sdk's standard retry and
    /// nothing else: a persistent 503 is attempted exactly the SDK's
    /// default 3 times, not 9, which the deleted hand-rolled outer loop
    /// used to multiply it to, and the exhausted call maps to a retryable
    /// `ExecutionClientError`.
    #[tokio::test]
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

    /// `get_state` sends one request with the given marker and `MaxItems`
    /// 1000, and returns that page and its `NextMarker`.
    #[tokio::test]
    async fn get_state_returns_one_page_with_marker() {
        let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state)
            .match_requests(|req| {
                req.durable_execution_arn() == Some("arn:test")
                    && req.checkpoint_token() == Some("tok")
                    && req.marker() == Some("m-1")
                    && req.max_items() == Some(1000)
            })
            .then_output(|| {
                GetDurableExecutionStateOutput::builder()
                    .operations(make_step_op("op-1", r#""p1""#))
                    .operations(make_step_op("op-2", r#""p2""#))
                    .next_marker("m-2")
                    .build()
                    .expect("operations is set")
            });
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let page = client
            .get_state("arn:test", "tok", Some("m-1"))
            .await
            .expect("get_state succeeds");

        assert_eq!(rule.num_calls(), 1);
        assert_eq!(page.operations.len(), 2);
        assert_eq!(page.operations.first().map(Operation::id), Some("op-1"));
        assert_eq!(page.operations.get(1).map(Operation::id), Some("op-2"));
        assert_eq!(page.next_marker(), Some("m-2"));
    }

    /// A transport that returns three pages, the last without a marker,
    /// and records the marker of every call.
    struct ThreePageClient {
        markers: Mutex<Vec<Option<String>>>,
    }

    impl ExecutionClient for ThreePageClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(CheckpointOutput::new(Some("t".to_owned()), vec![], None)) })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            self.markers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(marker.map(str::to_owned));
            let page = match marker {
                None => GetStateOutput::new(vec![make_step_op("a", "1")], Some("p-2".to_owned())),
                Some("p-2") => {
                    GetStateOutput::new(vec![make_step_op("b", "2")], Some("p-3".to_owned()))
                }
                Some(_) => GetStateOutput::new(vec![make_step_op("c", "3")], None),
            };
            Box::pin(async move { Ok(page) })
        }
    }

    /// The whole-state helper follows each marker until the last page and
    /// concatenates the pages in order.
    #[tokio::test]
    async fn sdk_loops_get_state_until_marker_empty() {
        let client = ThreePageClient {
            markers: Mutex::new(Vec::new()),
        };
        let operations = get_full_state(&client, "arn:test", "tok")
            .await
            .expect("whole state");

        let ids: Vec<&str> = operations.iter().map(Operation::id).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        let markers = client
            .markers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(
            markers,
            [None, Some("p-2".to_owned()), Some("p-3".to_owned())]
        );
    }

    /// A transport that serves page 2 for marker `m-1` and page 3 for
    /// marker `m-2`, and records the marker of every call. A call without
    /// a marker, a re-read of the first page, gets a step `s0` that the
    /// test then finds in the log.
    struct InlineContinuationClient {
        markers: Mutex<Vec<Option<String>>>,
    }

    impl ExecutionClient for InlineContinuationClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(CheckpointOutput::new(Some("t".to_owned()), vec![], None)) })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            self.markers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(marker.map(str::to_owned));
            let page = match marker {
                Some("m-1") => {
                    GetStateOutput::new(vec![make_step_op("s2", "2")], Some("m-2".to_owned()))
                }
                Some("m-2") => GetStateOutput::new(vec![make_step_op("s3", "3")], None),
                _ => GetStateOutput::new(vec![make_step_op("s0", "0")], None),
            };
            Box::pin(async move { Ok(page) })
        }
    }

    /// Bootstrap keeps the inline page and continues from its marker: it
    /// never re-reads the first page, and the merged log holds the inline
    /// step and the steps of both fetched pages.
    #[tokio::test]
    async fn bootstrap_keeps_inline_page_and_continues_from_marker() {
        let client = InlineContinuationClient {
            markers: Mutex::new(Vec::new()),
        };
        let execution = Operation::builder()
            .id("execution")
            .r#type(OperationType::Execution)
            .status(aws_sdk_lambda::types::OperationStatus::Started)
            .start_timestamp(aws_sdk_lambda::primitives::DateTime::from_secs(0))
            .execution_details(
                aws_sdk_lambda::types::ExecutionDetails::builder()
                    .input_payload("{}")
                    .build(),
            )
            .build()
            .expect("id, type, status and start timestamp are set");
        let first_page = operations_to_checkpoint_log(&[execution, make_step_op("s1", "1")]);

        let (log, input) = resolve_bootstrap_log(
            &client,
            "arn:test",
            "tok",
            first_page,
            Some(Some("{}".to_owned())),
            Some("m-1"),
        )
        .await
        .expect("bootstrap succeeds");

        let markers = client
            .markers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(markers, [Some("m-1".to_owned()), Some("m-2".to_owned())]);
        for id in ["s1", "s2", "s3"] {
            assert!(log.get(id).is_some(), "{id} must be in the merged log");
        }
        assert!(
            log.get("s0").is_none(),
            "the first page must not be re-read"
        );
        assert_eq!(input.as_deref(), Some("{}"));
    }

    /// Builds an EXECUTION operation with the given `InputPayload`, or with
    /// none when `input` is `None`.
    fn make_execution_op(id: &str, input: Option<&str>) -> Operation {
        let mut details = aws_sdk_lambda::types::ExecutionDetails::builder();
        if let Some(input) = input {
            details = details.input_payload(input);
        }
        Operation::builder()
            .id(id)
            .r#type(OperationType::Execution)
            .status(aws_sdk_lambda::types::OperationStatus::Started)
            .start_timestamp(aws_sdk_lambda::primitives::DateTime::from_secs(0))
            .execution_details(details.build())
            .build()
            .expect("id, type, status and start timestamp are set")
    }

    /// A transport that serves a fixed page for each marker and records the
    /// marker of every call. A marker it does not know gets an empty last
    /// page.
    struct MarkerPagesClient {
        pages: Vec<(&'static str, Vec<Operation>, Option<&'static str>)>,
        markers: Mutex<Vec<Option<String>>>,
    }

    impl ExecutionClient for MarkerPagesClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(CheckpointOutput::new(Some("t".to_owned()), vec![], None)) })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            self.markers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(marker.map(str::to_owned));
            let page = self
                .pages
                .iter()
                .find(|(key, _, _)| Some(*key) == marker)
                .map_or_else(
                    || GetStateOutput::new(vec![], None),
                    |(_, ops, next)| GetStateOutput::new(ops.clone(), next.map(str::to_owned)),
                );
            Box::pin(async move { Ok(page) })
        }
    }

    /// Resolves a bootstrap whose inline page holds no operation and has
    /// marker `m-1`, against the given fetched pages. Returns the input and
    /// the recorded markers.
    async fn resolve_input_from_pages(
        pages: Vec<(&'static str, Vec<Operation>, Option<&'static str>)>,
    ) -> (Option<String>, Vec<Option<String>>) {
        let client = MarkerPagesClient {
            pages,
            markers: Mutex::new(Vec::new()),
        };
        let (_, input) = resolve_bootstrap_log(
            &client,
            "arn:test",
            "tok",
            operations_to_checkpoint_log(&[]),
            None,
            Some("m-1"),
        )
        .await
        .expect("bootstrap succeeds");
        let markers = client
            .markers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        (input, markers)
    }

    /// When the inline page holds no EXECUTION operation, the input comes
    /// from the first EXECUTION operation of the fetched pages, in page
    /// order. The search continues past a fetched page that holds none.
    /// The first EXECUTION operation decides the input even when its
    /// `InputPayload` is absent, so a later page's payload does not replace
    /// it.
    #[tokio::test]
    async fn bootstrap_input_comes_from_first_fetched_execution_operation() {
        let (input, markers) = resolve_input_from_pages(vec![
            ("m-1", vec![make_step_op("s1", "1")], Some("m-2")),
            (
                "m-2",
                vec![make_execution_op("execution", Some(r#"{"count":1}"#))],
                Some("m-3"),
            ),
            (
                "m-3",
                vec![make_execution_op("late", Some(r#"{"late":1}"#))],
                None,
            ),
        ])
        .await;
        assert_eq!(input.as_deref(), Some(r#"{"count":1}"#));
        assert_eq!(
            markers,
            [
                Some("m-1".to_owned()),
                Some("m-2".to_owned()),
                Some("m-3".to_owned())
            ]
        );

        let (input, markers) = resolve_input_from_pages(vec![
            ("m-1", vec![make_step_op("s1", "1")], Some("m-2")),
            ("m-2", vec![make_execution_op("execution", None)], Some("m-3")),
            (
                "m-3",
                vec![make_execution_op("late", Some(r#"{"late":1}"#))],
                None,
            ),
        ])
        .await;
        assert_eq!(input, None, "an absent first payload must not be replaced");
        assert_eq!(markers.len(), 3, "pagination still reaches the last page");
    }

    /// The accessors of a transport's output read an empty token or an
    /// empty marker as `None`, so a custom transport that passes the
    /// service's empty string through behaves like an absent value.
    #[test]
    fn output_accessors_read_empty_values_as_none() {
        let output = CheckpointOutput::new(Some(String::new()), vec![], None);
        assert_eq!(output.checkpoint_token(), None);
        let page = GetStateOutput::new(vec![], Some(String::new()));
        assert_eq!(page.next_marker(), None);
    }

    /// A transport whose first page carries an empty marker.
    struct EmptyMarkerClient {
        calls: Mutex<u32>,
    }

    impl ExecutionClient for EmptyMarkerClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(CheckpointOutput::new(Some("t".to_owned()), vec![], None)) })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            *self
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += 1;
            let page = GetStateOutput::new(vec![make_step_op("a", "1")], Some(String::new()));
            Box::pin(async move { Ok(page) })
        }
    }

    /// The whole-state helper treats an empty marker as the last page: it
    /// makes one call and returns that page.
    #[tokio::test]
    async fn sdk_stops_paginating_on_empty_marker() {
        let client = EmptyMarkerClient {
            calls: Mutex::new(0),
        };
        let operations = get_full_state(&client, "arn:test", "tok")
            .await
            .expect("whole state");

        let ids: Vec<&str> = operations.iter().map(Operation::id).collect();
        assert_eq!(ids, ["a"]);
        assert_eq!(
            *client
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            1
        );
    }

    /// `checkpoint` stores an empty `CheckpointToken` and an empty
    /// `NextMarker` as `None`. `DurableContext` reads the marker field
    /// directly to decide whether to paginate, so the transport itself must
    /// normalize it.
    #[tokio::test]
    async fn checkpoint_maps_empty_token_and_marker_to_none() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution).then_output(|| {
            CheckpointDurableExecutionOutput::builder()
                .checkpoint_token("")
                .new_execution_state(
                    CheckpointUpdatedExecutionState::builder()
                        .operations(make_step_op("op-1", r#""r1""#))
                        .next_marker("")
                        .build(),
                )
                .build()
        });
        let client = LambdaExecutionClient::new(mock_client!(aws_sdk_lambda, [&rule]));

        let output = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect("mocked checkpoint succeeds");

        assert_eq!(output.checkpoint_token, None);
        assert_eq!(output.next_marker, None);
        assert_eq!(output.updated_operations.len(), 1);
    }

    /// `get_state` stores an empty `NextMarker` as `None`.
    #[tokio::test]
    async fn get_state_maps_empty_marker_to_none() {
        let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_output(|| {
            GetDurableExecutionStateOutput::builder()
                .operations(make_step_op("op-1", r#""p1""#))
                .next_marker("")
                .build()
                .expect("operations is set")
        });
        let client = LambdaExecutionClient::new(mock_client!(aws_sdk_lambda, [&rule]));

        let page = client
            .get_state("arn:test", "tok", None)
            .await
            .expect("get_state succeeds");

        assert_eq!(page.next_marker, None);
        assert_eq!(page.operations.len(), 1);
    }

    /// A `get_state` failure maps into a retryable `ExecutionClientError` exactly
    /// like a checkpoint failure.
    #[tokio::test]
    async fn get_state_final_error_maps_to_retryable_client_error() {
        let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_error(|| {
            GetDurableExecutionStateError::InvalidParameterValueException(
                aws_sdk_lambda::types::error::InvalidParameterValueException::builder().build(),
            )
        });
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .get_state("arn:test", "tok", None)
            .await
            .expect_err("modeled error must fail the call");
        assert!(err.is_retryable());
    }

    /// An unmodeled 403 on a checkpoint, such as an `AccessDeniedException`
    /// when the role lacks `lambda:CheckpointDurableExecution`, fails the
    /// execution. Standard retry does not retry it.
    #[tokio::test]
    async fn checkpoint_unmodeled_403_fails_the_execution() {
        let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .sequence()
            .http_status(403, None)
            .repeatedly()
            .build();
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .checkpoint("arn:test", "tok", Vec::new())
            .await
            .expect_err("a 403 must fail the call");
        assert!(!err.is_retryable());
        assert_eq!(rule.num_calls(), 1, "a 403 is not retried");
    }

    /// The four KMS errors of `GetDurableExecutionState` are a function
    /// misconfiguration that a re-invocation cannot fix, so each one is
    /// non-retryable, and the message names the exception.
    #[tokio::test]
    async fn get_state_kms_errors_classify_non_retryable() {
        use aws_sdk_lambda::types::error::{
            KmsAccessDeniedException, KmsDisabledException, KmsInvalidStateException,
            KmsNotFoundException,
        };
        type Make = fn() -> GetDurableExecutionStateError;
        let cases: [(Make, &str); 4] = [
            (
                || {
                    GetDurableExecutionStateError::KmsAccessDeniedException(
                        KmsAccessDeniedException::builder().build(),
                    )
                },
                "KMSAccessDeniedException",
            ),
            (
                || {
                    GetDurableExecutionStateError::KmsDisabledException(
                        KmsDisabledException::builder().build(),
                    )
                },
                "KMSDisabledException",
            ),
            (
                || {
                    GetDurableExecutionStateError::KmsInvalidStateException(
                        KmsInvalidStateException::builder().build(),
                    )
                },
                "KMSInvalidStateException",
            ),
            (
                || {
                    GetDurableExecutionStateError::KmsNotFoundException(
                        KmsNotFoundException::builder().build(),
                    )
                },
                "KMSNotFoundException",
            ),
        ];
        for (make, wire_name) in cases {
            let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_error(make);
            let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
            let client = LambdaExecutionClient::new(sdk_client);

            let err = client
                .get_state("arn:test", "tok", None)
                .await
                .expect_err("a KMS error must fail the call");
            assert!(!err.is_retryable(), "{wire_name} must be non-retryable");
            assert!(
                err.to_string().contains(wire_name),
                "message must name {wire_name}, got: {err}"
            );
        }
    }

    /// A plain 4xx on the state read stays retryable, as in JS, which
    /// rethrows it and so fails only the invocation.
    #[tokio::test]
    async fn get_state_plain_4xx_stays_retryable() {
        let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state)
            .sequence()
            .http_status(403, None)
            .repeatedly()
            .build();
        let sdk_client = mock_client!(aws_sdk_lambda, [&rule]);
        let client = LambdaExecutionClient::new(sdk_client);

        let err = client
            .get_state("arn:test", "tok", None)
            .await
            .expect_err("a 403 must fail the call");
        assert!(err.is_retryable());
    }

    // Test double behavior

    #[tokio::test]
    async fn in_memory_client_returns_queued_responses_in_order() {
        let client = InMemoryExecutionClient::new(Vec::new());
        client.enqueue_checkpoint_response(TestResponse::RetryableError("throttled".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError("invalid".to_owned()));
        client.enqueue_checkpoint_response(TestResponse::Success(Vec::new()));

        let first = client.checkpoint("arn:test", "token-0", Vec::new()).await;
        let first_err = first.unwrap_err();
        assert!(first_err.is_retryable());

        let second = client.checkpoint("arn:test", "token-0", Vec::new()).await;
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
        let output = result.unwrap();
        assert!(
            output
                .checkpoint_token()
                .is_some_and(|t| t.starts_with("token-"))
        );
    }

    // Checkpoint log population

    #[test]
    fn checkpoint_log_from_operations_maps_statuses() {
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
        let r1 = r1.unwrap();
        assert_eq!(r1.status, CheckpointStatus::Succeeded);
        assert_eq!(r1.result.as_deref(), Some(r#""hello""#));
        assert_eq!(r1.attempt, 1);

        let r2 = log.get("op-2");
        assert!(r2.is_some());
        let r2 = r2.unwrap();
        assert_eq!(r2.status, CheckpointStatus::Failed);
        assert_eq!(r2.error_type.as_deref(), Some("StepError"));
        assert_eq!(r2.error_message.as_deref(), Some("oops"));
        assert_eq!(r2.attempt, 2);
    }

    /// A paginated `ChainedInvoke` operation converts into the
    /// invoke-specific record fields (`invoke_result` /
    /// `invoke_error_*`) that invoke replay reads, matching the inline
    /// JSON conversion in `parse_single_operation`, and never leaks into
    /// the generic `result` / `error_*` fields.
    #[test]
    fn chained_invoke_maps_to_invoke_specific_fields() {
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
        let ok = ok.unwrap();
        assert_eq!(ok.invoke_result.as_deref(), Some(r#""invoke-payload""#));
        assert_eq!(
            ok.result, None,
            "invoke results must not leak into the generic result field"
        );

        let err = log.get("invoke-err");
        assert!(err.is_some());
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
    /// for correct replay: matching the inline JSON conversion.
    #[test]
    fn context_replay_children_is_preserved() {
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
        let replayed = replayed.unwrap();
        assert!(
            replayed.replay_children,
            "ReplayChildren must survive the typed-state conversion"
        );

        let inline = log.get("child-inline");
        assert!(inline.is_some());
        let inline = inline.unwrap();
        assert!(!inline.replay_children);
        assert_eq!(inline.result.as_deref(), Some(r#""child-result""#));
    }

    /// Execution-type operations are filtered from the typed-state
    /// conversion, matching `parse_inline_operations` in the JSON path.
    #[test]
    fn execution_operations_are_filtered_from_log() {
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

    // Get state

    #[tokio::test]
    async fn get_state_returns_preloaded_operations() {
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
        let result = client.get_state("arn:test", "token-0", None).await;

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.operations.len(), 1);
        assert_eq!(output.operations.first().map(Operation::id), Some("op-1"));
    }

    // Merge updated operations into log

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
        let record = record.unwrap();
        assert_eq!(record.callback_id.as_deref(), Some("assigned-cb-id-42"));
        assert_eq!(record.status, CheckpointStatus::Pending);
    }

    #[test]
    fn merge_operations_adds_new_record_to_log() {
        let log = CheckpointLog::empty();

        // Backend returns a new operation not previously in the log.
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
        let record = record.unwrap();
        assert_eq!(record.status, CheckpointStatus::Succeeded);
        assert_eq!(record.result.as_deref(), Some(r#""merged result""#));
    }

    // Pagination tests

    /// Helper: builds a Step operation with a result for pagination tests.
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

        let state = client.get_state("arn:test", "token-0", None).await;
        assert!(state.is_ok());
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
        let output = result.unwrap();
        assert_eq!(output.updated_operations.len(), 1);
        assert_eq!(output.next_marker, Some("marker-page-2".to_owned()));

        // Caller should then call get_state to get all operations.
        let full_state = client
            .get_state(
                "arn:test",
                output.checkpoint_token().unwrap_or_default(),
                None,
            )
            .await;
        assert!(full_state.is_ok());
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

        let r1 = log.get("step-1").unwrap();
        assert_eq!(r1.result.as_deref(), Some("\"result-1\""));
        assert_eq!(r1.status, CheckpointStatus::Succeeded);

        let r3 = log.get("step-3").unwrap();
        assert_eq!(r3.result.as_deref(), Some("\"result-3\""));
    }

    // Checkpoint error classification

    use aws_sdk_lambda::config::http::HttpResponse;
    use aws_sdk_lambda::error::SdkError;
    use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionError as CkptErr;
    use aws_smithy_runtime_api::http::StatusCode;
    use aws_smithy_types::body::SdkBody;

    /// A raw HTTP response with the given status and an empty body.
    fn raw_response(status: u16) -> HttpResponse {
        HttpResponse::new(
            StatusCode::try_from(status).expect("valid HTTP status"),
            SdkBody::empty(),
        )
    }

    /// Classifies `err` as a service error whose raw response has `status`.
    fn classify<E: Into<CkptErr>>(err: E, status: u16) -> ExecutionClientError {
        classify_checkpoint_error(SdkError::service_error(err.into(), raw_response(status)))
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
        let throttled = classify(
            CkptErr::TooManyRequestsException(
                aws_sdk_lambda::types::error::TooManyRequestsException::builder().build(),
            ),
            429,
        );
        assert!(throttled.is_retryable());

        let server = classify(
            CkptErr::ServiceException(
                aws_sdk_lambda::types::error::ServiceException::builder().build(),
            ),
            500,
        );
        assert!(server.is_retryable());
    }

    #[test]
    fn invalid_parameter_classifies_non_retryable() {
        assert!(!classify(invalid_parameter(Some("payload too large")), 400).is_retryable());
        // No message at all: no stale-token evidence, permanent rejection.
        assert!(!classify(invalid_parameter(None), 400).is_retryable());
    }

    /// The stale-token carve-out: `InvalidParameterValueException` whose
    /// message starts with `Invalid checkpoint token` resolves on
    /// re-invocation (the service issues a fresh token), so it is
    /// retryable. Prefix match only: a stale-token mention elsewhere in
    /// the message does not qualify. The carve-out precedes the status
    /// rule, so the 400 status does not make it non-retryable.
    #[test]
    fn stale_token_rejection_classifies_retryable() {
        let stale = classify(
            invalid_parameter(Some("Invalid checkpoint token: token has been superseded")),
            400,
        );
        assert!(stale.is_retryable());

        let not_prefix = classify(
            invalid_parameter(Some(
                "field X rejected (not an Invalid checkpoint token case)",
            )),
            400,
        );
        assert!(!not_prefix.is_retryable());

        let wrong_case = classify(
            invalid_parameter(Some("Invalid Checkpoint Token: old non-service spelling")),
            400,
        );
        assert!(!wrong_case.is_retryable());
    }

    #[test]
    fn kms_misconfiguration_classifies_non_retryable() {
        let kms = classify(
            CkptErr::KmsAccessDeniedException(
                aws_sdk_lambda::types::error::KmsAccessDeniedException::builder().build(),
            ),
            502,
        );
        assert!(!kms.is_retryable());
    }

    /// An unmodeled 4xx other than 429 is the service rejecting the
    /// request, so the same write fails the same way on every invocation:
    /// it fails the execution, as in JS.
    #[test]
    fn unhandled_4xx_classifies_non_retryable() {
        for status in [400, 403, 404, 409, 413] {
            let err = classify(CkptErr::unhandled("unmodeled rejection"), status);
            assert!(!err.is_retryable(), "status {status} must be non-retryable");
        }
    }

    /// An unmodeled 429 is throttling, so it stays retryable.
    #[test]
    fn unhandled_429_classifies_retryable() {
        assert!(classify(CkptErr::unhandled("throttled"), 429).is_retryable());
    }

    /// Unmodeled 5xx errors stay RETRYABLE. Under the #43 model a
    /// non-retryable classification fails the execution, so a novel
    /// transient error must not kill executions by default (the pre-#43
    /// non-retryable default relied on the old model, where the surfaced
    /// error only failed the invocation: r3726032098).
    #[test]
    fn unhandled_5xx_classifies_retryable() {
        for status in [500, 502, 503] {
            let err = classify(CkptErr::unhandled("some new service error"), status);
            assert!(err.is_retryable(), "status {status} must be retryable");
        }
    }

    /// A 4xx response whose body the SDK could not parse is a service
    /// rejection too, so it fails the execution.
    #[test]
    fn response_error_4xx_classifies_non_retryable() {
        let err: SdkError<CkptErr, HttpResponse> =
            SdkError::response_error("unparseable body", raw_response(403));
        assert!(!classify_checkpoint_error(err).is_retryable());
    }

    /// An unparseable 200 response is a channel fault, not a rejection.
    #[test]
    fn response_error_200_classifies_retryable() {
        let err: SdkError<CkptErr, HttpResponse> =
            SdkError::response_error("unparseable body", raw_response(200));
        assert!(classify_checkpoint_error(err).is_retryable());
    }

    #[test]
    fn transport_failures_classify_retryable() {
        let timeout: SdkError<CkptErr, HttpResponse> = SdkError::timeout_error("connect timed out");
        assert!(classify_checkpoint_error(timeout).is_retryable());
    }

    /// Canary: pins the variant set of `CheckpointDurableExecutionError`,
    /// the upstream enum [`classify_checkpoint_error`] matches with a
    /// wildcard arm.
    ///
    /// The enum is `#[non_exhaustive]`, so the classifier's `match` cannot
    /// reject new variants at compile time: a variant added by an
    /// aws-sdk-lambda upgrade would silently take the unknown→retryable
    /// default. This test parses the enum out of the dependency's own
    /// source (the exact version Cargo.lock pins, in the local cargo
    /// registry) and fails when the variant set changes, forcing the
    /// classification to be revisited instead of defaulted. The sibling
    /// audit for the `types` enums lives in
    /// `upstream_type_enum_variant_canary` (issue #45).
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

    /// Canary: pins the variant set of every upstream `types` enum the SDK
    /// matches with a wildcard arm: `OperationStatus`, `OperationType`,
    /// `OperationAction`, `ExecutionStatus`, and `EventType` (issue #45,
    /// extending the #43 pattern above).
    ///
    /// All five enums are `#[non_exhaustive]`, so a variant added by an
    /// aws-sdk-lambda upgrade would silently take each match's wildcard
    /// default: for `OperationStatus` that now means
    /// `CheckpointStatus::Unknown`, which fails replay rather than
    /// misclassifying, but the canary still forces the variant to be mapped
    /// explicitly instead of shipping through the fallback. The pinned
    /// lists include the generated enums' own `Unknown` carrier variant.
    #[test]
    #[expect(clippy::too_many_lines)] // reason: table-driven canary pins five enums' variant lists
    fn upstream_type_enum_variant_canary() {
        struct Pin {
            file: &'static str,
            enum_name: &'static str,
            variants: &'static [&'static str],
            wildcard_sites: &'static str,
        }
        let pins = [
            Pin {
                file: "src/types/_operation_status.rs",
                enum_name: "OperationStatus",
                variants: &[
                    "Cancelled",
                    "Failed",
                    "Pending",
                    "Ready",
                    "Started",
                    "Stopped",
                    "Succeeded",
                    "TimedOut",
                    "Unknown",
                ],
                wildcard_sites: "sdk_status_to_checkpoint_status (src/client.rs); \
                                 operation_status_str and operation_status_wire_str \
                                 (src/test_util.rs)",
            },
            Pin {
                file: "src/types/_operation_type.rs",
                enum_name: "OperationType",
                variants: &[
                    "Callback",
                    "ChainedInvoke",
                    "Context",
                    "Execution",
                    "Step",
                    "Unknown",
                    "Wait",
                ],
                wildcard_sites: "operation_type_to_string (src/client.rs); \
                                 operation_type_str (src/tracing_layer.rs); \
                                 operation_type_str (src/test_util.rs)",
            },
            Pin {
                file: "src/types/_operation_action.rs",
                enum_name: "OperationAction",
                variants: &["Cancel", "Fail", "Retry", "Start", "Succeed", "Unknown"],
                wildcard_sites: "terminalize_unwritten_outcomes (src/context.rs); \
                                 PendingTransitionEvent::emit (src/tracing_layer.rs); \
                                 BackendState::apply_update (src/test_util.rs); \
                                 child_starts_before_first_terminal \
                                 (src/map_parallel.rs tests)",
            },
            Pin {
                file: "src/types/_execution_status.rs",
                enum_name: "ExecutionStatus",
                variants: &[
                    "Failed",
                    "Running",
                    "Stopped",
                    "Succeeded",
                    "TimedOut",
                    "Unknown",
                ],
                wildcard_sites: "execution_status_str (src/test_util.rs)",
            },
            Pin {
                file: "src/types/_event_type.rs",
                enum_name: "EventType",
                variants: &[
                    "CallbackFailed",
                    "CallbackStarted",
                    "CallbackSucceeded",
                    "CallbackTimedOut",
                    "ChainedInvokeFailed",
                    "ChainedInvokeStarted",
                    "ChainedInvokeStopped",
                    "ChainedInvokeSucceeded",
                    "ChainedInvokeTimedOut",
                    "ContextFailed",
                    "ContextStarted",
                    "ContextSucceeded",
                    "ExecutionFailed",
                    "ExecutionStarted",
                    "ExecutionStopped",
                    "ExecutionSucceeded",
                    "ExecutionTimedOut",
                    "InvocationCompleted",
                    "StepFailed",
                    "StepStarted",
                    "StepSucceeded",
                    "Unknown",
                    "WaitCancelled",
                    "WaitStarted",
                    "WaitSucceeded",
                ],
                wildcard_sites: "apply_event_to_op and event_op_type (src/test_util.rs)",
            },
        ];

        for pin in pins {
            let source = read_sdk_source(pin.file);
            let mut actual = extract_enum_variants(&source, pin.enum_name);
            actual.sort_unstable();

            let mut expected: Vec<String> = pin.variants.iter().map(|s| (*s).to_owned()).collect();
            expected.sort_unstable();

            assert_eq!(
                actual, expected,
                "aws-sdk-lambda's {} variant set changed. Classify the new/removed \
                 variants explicitly at every wildcard match site — {} — then update \
                 this canary's pinned list.",
                pin.enum_name, pin.wildcard_sites
            );
        }
    }

    /// A modeled `OperationStatus` maps to its explicit `CheckpointStatus`;
    /// anything the SDK build does not model maps to
    /// `CheckpointStatus::Unknown` carrying the raw wire value, never to a
    /// guessed known status (the pre-#45 wildcard mapped it to `Started`,
    /// silently treating a possibly-terminal operation as re-runnable).
    #[test]
    fn unknown_operation_status_maps_to_unknown_carrying_raw() {
        use aws_sdk_lambda::types::OperationStatus;

        let novel = OperationStatus::from("PAUSED");
        assert_eq!(
            sdk_status_to_checkpoint_status(&novel),
            CheckpointStatus::Unknown("PAUSED".to_owned())
        );

        let modeled = [
            (OperationStatus::Started, CheckpointStatus::Started),
            (OperationStatus::Pending, CheckpointStatus::Pending),
            (OperationStatus::Ready, CheckpointStatus::Ready),
            (OperationStatus::Succeeded, CheckpointStatus::Succeeded),
            (OperationStatus::Failed, CheckpointStatus::Failed),
            (OperationStatus::Cancelled, CheckpointStatus::Cancelled),
            (OperationStatus::TimedOut, CheckpointStatus::TimedOut),
            (OperationStatus::Stopped, CheckpointStatus::Stopped),
        ];
        for (sdk, expected) in modeled {
            assert_eq!(
                sdk_status_to_checkpoint_status(&sdk),
                expected,
                "modeled variant {sdk:?} must map explicitly"
            );
        }
    }

    /// Reads a source file out of the aws-sdk-lambda version Cargo.lock
    /// pins, from the local cargo registry (already extracted: the test
    /// binary that runs this compiled against it).
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
    #[expect(clippy::panic, clippy::map_unwrap_or)]
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
