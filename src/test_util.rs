//! In-process local testing runner (`test-util` feature).
//!
//! [`LocalRunner`] drives a durable handler to completion entirely in
//! memory: it runs the handler through the same driver and operation
//! machinery the production runtime uses, backed by an internal in-memory
//! execution client instead of the Lambda checkpoint API. When the handler
//! suspends, the runner advances the simulated backend (timers fire, retry
//! delays elapse, callbacks are delivered) and re-invokes, exactly as the
//! real service would, until the execution reaches a terminal outcome.
//!
//! The runner records every checkpointed operation so a test can assert on
//! the execution history via [`TestResult::operations`] and
//! [`TestOperation`].
//!
//! [`CloudRunner`] is the real-AWS counterpart: it invokes a deployed durable
//! function, polls the execution to a terminal state, and folds the recorded
//! service history into the SAME [`TestResult`] / [`TestOperation`] types, so a
//! test written against [`LocalRunner`] can be re-pointed at real AWS with only
//! the runner swapped.
//!
//! # Backend fidelity
//!
//! The in-memory client reproduces the behaviours the SDK depends on,
//! traced to this crate's own engine:
//!
//! - Checkpoint responses return whole-record operation updates that the
//!   SDK merges into its in-memory log (`client::merge_operations_into_log`
//!   → `CheckpointLog::insert` whole-record overwrite). A `Start` update
//!   carries no payload, so the merged record clears any carried result —
//!   the exact condition the `wait_for_condition` read-before-`Start` fix
//!   guards against.
//! - Timers (`wait`) and retry delays are modelled as state transitions
//!   between invocations rather than wall-clock sleeps, so there is no
//!   real time in the runner or its tests.
//!
//! # Runner divergences
//!
//! [`LocalRunner`] and [`CloudRunner`] return the same types, and a test can
//! be re-pointed between them by swapping the runner. The two agree on
//! operation type, status, result, error, [`TestOperation::attempt`] (both
//! report completed retries), and success/failure disposition. A few
//! behaviours still differ because one runner observes an in-memory model and
//! the other observes the recorded execution history; assert around these
//! when a test targets both runners:
//!
//! - **Open-callback status.** A created but not-yet-resolved callback reads
//!   `"Pending"` under [`LocalRunner`] and `"Started"` under [`CloudRunner`]
//!   (the recorded history has no distinct callback-pending status). Once the
//!   callback resolves, both report `"Succeeded"` / `"Failed"` / `"TimedOut"`.
//! - **Truncated results.** When a result exceeds the recorded-history inline
//!   size limit, [`CloudRunner`] reports [`TestOperation::result`] as `None`
//!   (the payload is withheld); [`LocalRunner`] always holds the full
//!   in-memory result.
//! - **Timed-out / stopped executions.** [`CloudRunner`] maps a terminal
//!   `TimedOut` or `Stopped` execution to [`TestResult::is_failure`] with the
//!   status name (or the recorded error) as the error type; [`LocalRunner`]
//!   has no execution-level timed-out or stopped disposition (it reports
//!   success, failure, or suspended).
//! - **Transient operation statuses.** [`LocalRunner`] may surface the
//!   intermediate `"Ready"` status it uses to model an elapsed retry delay;
//!   [`CloudRunner`]'s status vocabulary comes from the event stream and does
//!   not include `"Ready"`.
//!
//! # Examples
//!
//! ```
//! use aws_durable_execution_sdk_rust as durable;
//! use durable::test_util::LocalRunner;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let result = LocalRunner::new()
//!     .run(
//!         |event: i32, ctx: durable::DurableContext| async move {
//!             let doubled = ctx.step(move |_| async move { Ok(event * 2) }).await?;
//!             Ok::<_, durable::BoxError>(doubled)
//!         },
//!         21_i32,
//!     )
//!     .await;
//!
//! assert!(result.is_success());
//! assert_eq!(result.output(), Some(&42));
//! # }
//! ```

use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;

use aws_sdk_lambda::primitives::{Blob, DateTime};
use aws_sdk_lambda::types::{
    CallbackDetails, ChainedInvokeDetails, ContextDetails, ErrorObject, Event, EventType,
    ExecutionStatus, InvocationType, Operation, OperationAction, OperationStatus, OperationType,
    OperationUpdate, StepDetails,
};

use crate::BoxError;
use crate::client::{
    CheckpointOutput, ClientError, ExecutionClient, GetStateOutput, operations_to_checkpoint_log,
};
use crate::context::DurableContext;
use crate::driver::{InvocationOutcome, drive_invocation};
use crate::error::{OperationError, OperationErrorKind};

/// Default cap on the number of invocations the runner will drive before
/// declaring the execution stuck. Generous enough for deep timer/retry
/// chains, low enough to fail a non-terminating handler (e.g. a
/// `wait_for_condition` that never advances) instead of looping forever.
const DEFAULT_MAX_INVOCATIONS: usize = 100;

// ────────────────────────────────────────────────────────────────────────────
// TestOperation
// ────────────────────────────────────────────────────────────────────────────

/// A single checkpointed operation recorded during a [`LocalRunner`] run.
///
/// Returned in order by [`TestResult::operations`]. All accessors return
/// borrowed views; construct these only through the runner.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             ctx.step(|_| async { Ok(1_i32) }).name("only").await?;
///             Ok::<_, durable::BoxError>(())
///         },
///         (),
///     )
///     .await;
///
/// let ops = result.operations();
/// assert_eq!(ops.len(), 1);
/// assert_eq!(ops[0].op_type(), "Step");
/// assert!(ops[0].succeeded());
/// assert_eq!(ops[0].name(), Some("only"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TestOperation {
    id: String,
    op_type: String,
    sub_type: Option<String>,
    status: String,
    result: Option<String>,
    error_type: Option<String>,
    error_message: Option<String>,
    attempt: u32,
    name: Option<String>,
}

impl TestOperation {
    /// The operation's wire ID (the SHA-256 hex of its positional path).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The operation type, e.g. `"Step"`, `"Wait"`, `"Context"`,
    /// `"Callback"`.
    #[must_use]
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// The operation sub-type, e.g. `"Step"`, `"Wait"`, `"Map"`,
    /// `"WaitForCondition"`, or `None` if the operation carried none.
    #[must_use]
    pub fn sub_type(&self) -> Option<&str> {
        self.sub_type.as_deref()
    }

    /// The terminal-or-current status, e.g. `"Succeeded"`, `"Failed"`,
    /// `"Pending"`, `"TimedOut"`.
    #[must_use]
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Returns `true` if the operation completed successfully.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.status == "Succeeded"
    }

    /// Returns `true` if the operation ended in a failed status.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.status == "Failed"
    }

    /// The serialized result payload, if the operation stored one.
    #[must_use]
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }

    /// The recorded error type, if the operation failed.
    #[must_use]
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The recorded error message, if the operation failed.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The number of completed retries recorded for this operation
    /// (`0` when never retried). [`LocalRunner`] and [`CloudRunner`] report
    /// this value identically.
    #[must_use]
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// The operation's caller-assigned name, if any.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TestResult
// ────────────────────────────────────────────────────────────────────────────

/// Terminal disposition of a [`LocalRunner`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Succeeded,
    Failed,
    Suspended,
}

/// The outcome of a [`LocalRunner::run`], including the deserialized handler
/// output (on success) and the recorded operation history.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), _ctx: durable::DurableContext| async move {
///             Ok::<_, durable::BoxError>("hi".to_owned())
///         },
///         (),
///     )
///     .await;
///
/// assert!(result.is_success());
/// assert_eq!(result.output().map(String::as_str), Some("hi"));
/// assert_eq!(result.invocation_count(), 1);
/// # }
/// ```
#[derive(Debug)]
pub struct TestResult<O> {
    disposition: Disposition,
    output: Option<O>,
    error_type: Option<String>,
    error_message: Option<String>,
    operations: Vec<TestOperation>,
    invocations: usize,
}

impl<O> TestResult<O> {
    /// Returns `true` if the handler completed successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.disposition == Disposition::Succeeded
    }

    /// Returns `true` if the handler returned an error.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.disposition == Disposition::Failed
    }

    /// Returns `true` if the handler suspended and never reached a terminal
    /// outcome within the invocation budget (e.g. it awaits a callback the
    /// test never delivered, or it never terminates).
    #[must_use]
    pub fn is_suspended(&self) -> bool {
        self.disposition == Disposition::Suspended
    }

    /// The deserialized handler output on success, or `None` otherwise.
    #[must_use]
    pub fn output(&self) -> Option<&O> {
        self.output.as_ref()
    }

    /// Consumes the result and returns the owned handler output on success.
    #[must_use]
    pub fn into_output(self) -> Option<O> {
        self.output
    }

    /// The wire error type on failure (e.g. `"StepError"`), or `None`.
    #[must_use]
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The wire error message on failure, or `None`.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The recorded operation history, in first-checkpoint order.
    #[must_use]
    pub fn operations(&self) -> &[TestOperation] {
        &self.operations
    }

    /// The number of invocations that drove this execution.
    ///
    /// For [`LocalRunner`] this is the count of simulated invocations; for
    /// [`CloudRunner`] it is the number of recorded `InvocationCompleted`
    /// events (falling back to the poll count when none are recorded).
    #[must_use]
    pub fn invocation_count(&self) -> usize {
        self.invocations
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LocalRunner
// ────────────────────────────────────────────────────────────────────────────

/// A queued external outcome for a pending callback.
#[derive(Debug, Clone)]
enum CallbackOutcome {
    Success(String),
    Timeout,
}

/// Drives a durable handler to completion in memory, simulating the backend.
///
/// Construct with [`LocalRunner::new`], optionally queue callback outcomes
/// with [`callback_success`](Self::callback_success) /
/// [`callback_timeout`](Self::callback_timeout), then drive a handler with
/// [`run`](Self::run).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() {
/// // Deliver a value to the first callback the handler awaits.
/// let runner = LocalRunner::new().callback_success(&true);
/// let result = runner
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             let cb = ctx.create_callback::<bool>().await?;
///             let approved = cb.result().await?;
///             Ok::<_, durable::BoxError>(approved)
///         },
///         (),
///     )
///     .await;
///
/// assert_eq!(result.output(), Some(&true));
/// # }
/// ```
#[derive(Debug)]
pub struct LocalRunner {
    max_invocations: usize,
    callback_outcomes: Vec<CallbackOutcome>,
}

impl Default for LocalRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalRunner {
    /// Creates a runner with default settings and no queued callback
    /// outcomes.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_invocations: DEFAULT_MAX_INVOCATIONS,
            callback_outcomes: Vec::new(),
        }
    }

    /// Sets the maximum number of invocations the runner will drive before
    /// returning a suspended result.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().max_invocations(10);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn max_invocations(mut self, max: usize) -> Self {
        self.max_invocations = max.max(1);
        self
    }

    /// Queues a successful outcome for the next callback the handler awaits.
    ///
    /// The value is serialized as the callback payload and delivered when
    /// the callback becomes pending, in the order queued.
    ///
    /// # Panics
    ///
    /// Never panics: a value that fails to serialize is queued as an empty
    /// payload, which surfaces as a deserialization error to the handler
    /// rather than aborting the test.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().callback_success(&"approved");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn callback_success<T: Serialize>(mut self, value: &T) -> Self {
        let payload = serde_json::to_string(value).unwrap_or_default();
        self.callback_outcomes
            .push(CallbackOutcome::Success(payload));
        self
    }

    /// Queues a timeout outcome for the next callback the handler awaits.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().callback_timeout();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn callback_timeout(mut self) -> Self {
        self.callback_outcomes.push(CallbackOutcome::Timeout);
        self
    }

    /// Drives `handler` to a terminal outcome, feeding it `event`.
    ///
    /// The handler is invoked once per simulated invocation. The event is
    /// serialized once and a fresh copy is deserialized for each invocation,
    /// mirroring the way the service re-delivers the input payload on every
    /// re-invocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust as durable;
    /// use durable::test_util::LocalRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = LocalRunner::new()
    ///     .run(
    ///         |n: u32, ctx: durable::DurableContext| async move {
    ///             let v = ctx.step(move |_| async move { Ok(n + 1) }).await?;
    ///             Ok::<_, durable::BoxError>(v)
    ///         },
    ///         41_u32,
    ///     )
    ///     .await;
    /// assert_eq!(result.output(), Some(&42));
    /// # }
    /// ```
    #[allow(clippy::too_many_lines)] // reason: the invoke → drive → advance loop reads better as one flow
    pub async fn run<E, O, F, Fut>(&self, handler: F, event: E) -> TestResult<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let backend = Arc::new(Backend::new(self.callback_outcomes.clone()));
        let client: Arc<dyn ExecutionClient> = Arc::clone(&backend) as Arc<dyn ExecutionClient>;

        // Serialize the event once; deserialize a fresh copy per invocation.
        let event_json = match serde_json::to_string(&event) {
            Ok(json) => json,
            Err(e) => {
                return TestResult {
                    disposition: Disposition::Failed,
                    output: None,
                    error_type: Some("SerializationFailed".to_owned()),
                    error_message: Some(format!("serialize event: {e}")),
                    operations: Vec::new(),
                    invocations: 0,
                };
            }
        };

        let mut invocations = 0_usize;

        loop {
            invocations += 1;
            if invocations > self.max_invocations {
                return TestResult {
                    disposition: Disposition::Suspended,
                    output: None,
                    error_type: None,
                    error_message: Some(format!(
                        "execution did not terminate within {} invocations",
                        self.max_invocations
                    )),
                    operations: backend.snapshot_operations(),
                    invocations: invocations - 1,
                };
            }

            let ops = backend.build_operations();
            let checkpoint_log = Arc::new(operations_to_checkpoint_log(&ops));
            let token = backend.current_token();

            let ctx = DurableContext::new_root_with_client(
                String::from("arn:aws:lambda:us-west-2:000000000000:function:local-test"),
                lambda_runtime::Context::default(),
                checkpoint_log,
                Arc::clone(&client),
                token,
            );
            let signal = ctx.suspension_signal().clone();

            // Deserialize a fresh event for this invocation.
            let event_inst: E = match serde_json::from_str(&event_json) {
                Ok(v) => v,
                Err(e) => {
                    return TestResult {
                        disposition: Disposition::Failed,
                        output: None,
                        error_type: Some("SerializationFailed".to_owned()),
                        error_message: Some(format!("deserialize event: {e}")),
                        operations: backend.snapshot_operations(),
                        invocations,
                    };
                }
            };

            let handler_ref = &handler;
            let outcome = drive_invocation(
                async move {
                    match handler_ref(event_inst, ctx).await {
                        Ok(value) => serde_json::to_string(&value)
                            .map_err(|e| ("HandlerError".to_owned(), e.to_string())),
                        Err(e) => Ok(wire_error_from_box_error(e))
                            .map_or_else(|never: (String, String)| Err(never), Err),
                    }
                },
                signal,
            )
            .await;

            match outcome {
                InvocationOutcome::Complete(serialized) => {
                    let output = serde_json::from_str::<O>(&serialized).ok();
                    let (error_type, error_message, disposition) = if output.is_some() {
                        (None, None, Disposition::Succeeded)
                    } else {
                        (
                            Some("SerializationFailed".to_owned()),
                            Some("could not deserialize handler output".to_owned()),
                            Disposition::Failed,
                        )
                    };
                    return TestResult {
                        disposition,
                        output,
                        error_type,
                        error_message,
                        operations: backend.snapshot_operations(),
                        invocations,
                    };
                }
                InvocationOutcome::Failed {
                    error_type,
                    error_message,
                } => {
                    return TestResult {
                        disposition: Disposition::Failed,
                        output: None,
                        error_type: Some(error_type),
                        error_message: Some(error_message),
                        operations: backend.snapshot_operations(),
                        invocations,
                    };
                }
                InvocationOutcome::Pending => {
                    // Advance the simulated backend (timers, retries,
                    // callbacks). If nothing can advance, the execution is
                    // genuinely stuck.
                    if !backend.advance() {
                        return TestResult {
                            disposition: Disposition::Suspended,
                            output: None,
                            error_type: None,
                            error_message: Some(
                                "execution suspended with no pending timer, retry, or callback \
                                 to advance (missing callback outcome?)"
                                    .to_owned(),
                            ),
                            operations: backend.snapshot_operations(),
                            invocations,
                        };
                    }
                }
            }
        }
    }
}

/// Extracts a `(wire_error_type, message)` pair from a boxed handler error.
fn wire_error_from_box_error(err: BoxError) -> (String, String) {
    match err.downcast::<OperationError>() {
        Ok(op_err) => (operation_error_type(&op_err), op_err.to_string()),
        Err(other) => ("HandlerError".to_owned(), other.to_string()),
    }
}

/// Maps an `OperationError` kind to its wire error type name.
fn operation_error_type(err: &OperationError) -> String {
    match err.kind() {
        OperationErrorKind::Step(_) => "StepError",
        OperationErrorKind::Invoke(_) => "InvokeError",
        OperationErrorKind::Callback(_) => "CallbackError",
        OperationErrorKind::ChildContext(_) => "ChildContextError",
        OperationErrorKind::WaitForCondition(_) => "WaitForConditionError",
        OperationErrorKind::Combinator(_) => "PromiseCombinatorError",
    }
    .to_owned()
}

// ────────────────────────────────────────────────────────────────────────────
// CloudRunner (real-AWS runner returning the same TestResult/TestOperation)
// ────────────────────────────────────────────────────────────────────────────

/// Default seconds between execution-status polls.
const DEFAULT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Default cap on status polls before the runner reports the execution as
/// still running (suspended). 150 × 2 s ≈ 5 minutes.
const DEFAULT_MAX_POLL_ATTEMPTS: usize = 150;

/// `error_type` used on a [`TestResult`] when a `CloudRunner` infrastructure
/// step fails (invoke error, missing execution ARN, poll-budget exhaustion,
/// or history retrieval). Distinct from a durable execution `FAILED`, whose
/// `error_type` carries the service-recorded error.
const CLOUD_RUNNER_ERROR: &str = "CloudRunnerError";

/// Drives a durable function against **real AWS** and returns the same
/// [`TestResult`] / [`TestOperation`] types [`LocalRunner`] returns.
///
/// `CloudRunner` invokes a deployed durable function (`InvocationType=Event`),
/// polls [`GetDurableExecution`] until the execution reaches a terminal state,
/// then folds the recorded [`GetDurableExecutionHistory`] event stream into
/// per-operation [`TestOperation`]s. Because both runners return the same
/// types, a test can be re-pointed from in-memory to real AWS by swapping the
/// runner.
///
/// The AWS client is built internally from the ambient credentials and region
/// (the same way the production entry point builds it), so no AWS type appears
/// in any public signature. Configure the region explicitly with
/// [`region`](Self::region) when the environment does not supply one.
///
/// Infrastructure failures (invoke error, no execution ARN, poll-budget
/// exhaustion, deserialization) surface through the returned [`TestResult`]:
/// [`is_failure`](TestResult::is_failure) with `error_type() ==
/// "CloudRunnerError"`, or [`is_suspended`](TestResult::is_suspended) when the
/// execution is still running at the poll budget.
///
/// [`GetDurableExecution`]: https://docs.aws.amazon.com/lambda/latest/api/API_GetDurableExecution.html
/// [`GetDurableExecutionHistory`]: https://docs.aws.amazon.com/lambda/latest/api/API_GetDurableExecutionHistory.html
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::CloudRunner;
/// use std::time::Duration;
///
/// # #[tokio::main]
/// # async fn main() {
/// let result = CloudRunner::new("my-durable-function")
///     .region("us-west-2")
///     .poll_interval(Duration::from_secs(3))
///     .run::<_, i32>(21_i32)
///     .await;
///
/// if result.is_success() {
///     assert_eq!(result.output(), Some(&42));
///     for op in result.operations() {
///         println!("{} {} {}", op.id(), op.op_type(), op.status());
///     }
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CloudRunner {
    function_name: String,
    region: Option<String>,
    qualifier: String,
    poll_interval: std::time::Duration,
    max_poll_attempts: usize,
}

impl CloudRunner {
    /// Creates a runner targeting the named deployed durable function.
    ///
    /// `function_name` may be a function name or a fully-qualified ARN. The
    /// qualifier defaults to `$LATEST` (durable functions require a qualified
    /// invocation); override it with [`qualifier`](Self::qualifier).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("order-processor");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            region: None,
            qualifier: String::from("$LATEST"),
            poll_interval: DEFAULT_POLL_INTERVAL,
            max_poll_attempts: DEFAULT_MAX_POLL_ATTEMPTS,
        }
    }

    /// Sets the AWS region used to build the Lambda client.
    ///
    /// When unset, the region is resolved from the ambient environment
    /// (`AWS_REGION`, profile, or instance metadata).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("f").region("us-west-2");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the invocation qualifier (function version or alias).
    ///
    /// Defaults to `$LATEST`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("f").qualifier("PROD");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn qualifier(mut self, qualifier: impl Into<String>) -> Self {
        self.qualifier = qualifier.into();
        self
    }

    /// Sets the delay between execution-status polls.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    /// use std::time::Duration;
    ///
    /// let runner = CloudRunner::new("f").poll_interval(Duration::from_secs(5));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn poll_interval(mut self, interval: std::time::Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the maximum number of status polls before the runner reports the
    /// execution as still running (a suspended [`TestResult`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("f").max_poll_attempts(30);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn max_poll_attempts(mut self, max: usize) -> Self {
        self.max_poll_attempts = max.max(1);
        self
    }

    /// Invokes the durable function with `event`, polls to a terminal state,
    /// and returns the folded [`TestResult`].
    ///
    /// The event is serialized to JSON and delivered as the invocation
    /// payload. The handler output (on success) is deserialized into `O`.
    ///
    /// This method never panics: every AWS or serialization failure is
    /// reported through the returned [`TestResult`] (see [`CloudRunner`]).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = CloudRunner::new("echo-fn")
    ///     .run::<_, String>("hello".to_owned())
    ///     .await;
    /// assert!(result.is_success() || result.is_failure() || result.is_suspended());
    /// # }
    /// ```
    #[allow(clippy::too_many_lines)] // reason: the invoke → poll → history → fold flow reads better as one sequence
    pub async fn run<E, O>(&self, event: E) -> TestResult<O>
    where
        E: Serialize,
        O: DeserializeOwned,
    {
        let payload = match serde_json::to_vec(&event) {
            Ok(bytes) => bytes,
            Err(e) => return cloud_failure(format!("serialize event: {e}")),
        };

        let client = self.build_client().await;

        // 1. Invoke (async) — start the durable execution and capture its ARN.
        let invoke_result = client
            .invoke()
            .function_name(&self.function_name)
            .invocation_type(InvocationType::Event)
            .qualifier(&self.qualifier)
            .payload(Blob::new(payload))
            .send()
            .await;
        let invoke_output = match invoke_result {
            Ok(out) => out,
            Err(e) => return cloud_failure(format!("invoke {}: {e}", self.function_name)),
        };
        let Some(execution_arn) = invoke_output.durable_execution_arn().map(ToOwned::to_owned)
        else {
            return cloud_failure(
                "invoke response carried no DurableExecutionArn (is the target a durable \
                 function invoked with a qualifier?)"
                    .to_owned(),
            );
        };

        // 2. Poll GetDurableExecution until terminal or the budget is spent.
        let mut polls = 0_usize;
        let terminal = loop {
            if polls >= self.max_poll_attempts {
                let operations = self.fetch_operations(&client, &execution_arn).await;
                return TestResult {
                    disposition: Disposition::Suspended,
                    output: None,
                    error_type: None,
                    error_message: Some(format!(
                        "execution {execution_arn} still running after {polls} polls"
                    )),
                    operations,
                    invocations: polls,
                };
            }
            polls += 1;
            tokio::time::sleep(self.poll_interval).await;
            match client
                .get_durable_execution()
                .durable_execution_arn(&execution_arn)
                .include_execution_data(true)
                .send()
                .await
            {
                Ok(out) if matches!(out.status(), ExecutionStatus::Running) => {}
                Ok(out) => break out,
                Err(e) => return cloud_failure(format!("get_durable_execution: {e}")),
            }
        };

        // 3. Retrieve the recorded history and fold it into operations.
        let (operations, invocation_events) =
            match self.fetch_history(&client, &execution_arn).await {
                Ok(events) => (fold_history(&events), invocations_from_history(&events)),
                Err(msg) => return cloud_failure(msg),
            };
        let invocations = if invocation_events == 0 {
            polls
        } else {
            invocation_events
        };

        // 4. Map the terminal status into a TestResult disposition.
        match terminal.status() {
            ExecutionStatus::Succeeded => match deserialize_output::<O>(terminal.result()) {
                Ok(output) => TestResult {
                    disposition: Disposition::Succeeded,
                    output: Some(output),
                    error_type: None,
                    error_message: None,
                    operations,
                    invocations,
                },
                Err(msg) => TestResult {
                    disposition: Disposition::Failed,
                    output: None,
                    error_type: Some("SerializationFailed".to_owned()),
                    error_message: Some(msg),
                    operations,
                    invocations,
                },
            },
            other => {
                let status_name = execution_status_str(other);
                let (error_type, error_message) = terminal.error().map_or_else(
                    || {
                        (
                            status_name.to_owned(),
                            format!("execution ended with status {status_name}"),
                        )
                    },
                    |err| {
                        (
                            err.error_type().unwrap_or(status_name).to_owned(),
                            err.error_message().unwrap_or_default().to_owned(),
                        )
                    },
                );
                TestResult {
                    disposition: Disposition::Failed,
                    output: None,
                    error_type: Some(error_type),
                    error_message: Some(error_message),
                    operations,
                    invocations,
                }
            }
        }
    }

    /// Builds the Lambda client from the ambient config, honoring an explicit
    /// region override.
    async fn build_client(&self) -> aws_sdk_lambda::Client {
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = &self.region {
            loader = loader.region(aws_sdk_lambda::config::Region::new(region.clone()));
        }
        let config = loader.load().await;
        aws_sdk_lambda::Client::new(&config)
    }

    /// Best-effort history fetch used on the suspended (budget-exhausted) path.
    async fn fetch_operations(
        &self,
        client: &aws_sdk_lambda::Client,
        execution_arn: &str,
    ) -> Vec<TestOperation> {
        (self.fetch_history(client, execution_arn).await)
            .map(|events| fold_history(&events))
            .unwrap_or_default()
    }

    /// Retrieves the full paginated execution history.
    async fn fetch_history(
        &self,
        client: &aws_sdk_lambda::Client,
        execution_arn: &str,
    ) -> Result<Vec<Event>, String> {
        let mut events = Vec::new();
        let mut marker: Option<String> = None;
        loop {
            let mut builder = client
                .get_durable_execution_history()
                .durable_execution_arn(execution_arn)
                .include_execution_data(true);
            if let Some(m) = &marker {
                builder = builder.marker(m);
            }
            match builder.send().await {
                Ok(out) => {
                    events.extend(out.events().iter().cloned());
                    match out.next_marker() {
                        Some(m) if !m.is_empty() => marker = Some(m.to_owned()),
                        _ => break,
                    }
                }
                Err(e) => return Err(format!("get_durable_execution_history: {e}")),
            }
        }
        Ok(events)
    }
}

/// Builds a failed [`TestResult`] for a `CloudRunner` infrastructure error.
fn cloud_failure<O>(message: String) -> TestResult<O> {
    TestResult {
        disposition: Disposition::Failed,
        output: None,
        error_type: Some(CLOUD_RUNNER_ERROR.to_owned()),
        error_message: Some(message),
        operations: Vec::new(),
        invocations: 0,
    }
}

/// Deserializes the handler output from the `GetDurableExecution` result
/// string, tolerating a double-encoded (JSON-string-of-JSON) payload.
fn deserialize_output<O: DeserializeOwned>(result: Option<&str>) -> Result<O, String> {
    let raw = result.unwrap_or("null");
    match serde_json::from_str::<O>(raw) {
        Ok(value) => Ok(value),
        Err(first) => match serde_json::from_str::<String>(raw) {
            Ok(inner) => serde_json::from_str::<O>(&inner)
                .map_err(|e| format!("deserialize handler output: {e}")),
            Err(_) => Err(format!("deserialize handler output: {first}")),
        },
    }
}

/// Folds a chronological execution-event stream into per-operation
/// [`TestOperation`]s in first-appearance order.
///
/// Only operation-scoped events (`Step*`, `Wait*`, `Context*`, `Callback*`,
/// `ChainedInvoke*`) with an operation id are folded; execution-level and
/// `InvocationCompleted` events are ignored here (the execution outcome comes
/// from `GetDurableExecution`, the invocation count from
/// [`invocations_from_history`]). Later transitions overwrite earlier ones, so
/// the recorded `status` is the operation's terminal-or-latest state.
fn fold_history(events: &[Event]) -> Vec<TestOperation> {
    use std::collections::HashMap;

    let mut order: Vec<String> = Vec::new();
    let mut acc: HashMap<String, TestOperation> = HashMap::new();

    for event in events {
        let Some(op_type) = event_family_op_type(event.event_type()) else {
            continue;
        };
        let Some(id) = event.id() else { continue };

        let op = acc.entry(id.to_owned()).or_insert_with(|| {
            order.push(id.to_owned());
            TestOperation {
                id: id.to_owned(),
                op_type: op_type.to_owned(),
                sub_type: None,
                status: String::from("Started"),
                result: None,
                error_type: None,
                error_message: None,
                attempt: 0,
                name: None,
            }
        });

        if op.sub_type.is_none() {
            op.sub_type = event.sub_type().map(ToOwned::to_owned);
        }
        if op.name.is_none() {
            op.name = event.name().map(ToOwned::to_owned);
        }
        apply_event_transition(op, event);
    }

    order.iter().filter_map(|id| acc.get(id).cloned()).collect()
}

/// Applies a single event's transition (status, result, error, attempt) to the
/// accumulating operation record.
fn apply_event_transition(op: &mut TestOperation, event: &Event) {
    match event.event_type() {
        Some(
            EventType::StepStarted
            | EventType::WaitStarted
            | EventType::ContextStarted
            | EventType::CallbackStarted
            | EventType::ChainedInvokeStarted,
        ) => op.status = String::from("Started"),
        Some(EventType::StepSucceeded) => {
            op.status = String::from("Succeeded");
            if let Some(details) = event.step_succeeded_details() {
                op.result = event_result_payload(details.result());
                set_attempt(op, details.retry_details());
            }
        }
        Some(EventType::StepFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.step_failed_details() {
                set_error(op, details.error());
                set_attempt(op, details.retry_details());
            }
        }
        Some(EventType::WaitSucceeded) => op.status = String::from("Succeeded"),
        Some(EventType::WaitCancelled) => op.status = String::from("Cancelled"),
        Some(EventType::ContextSucceeded) => {
            op.status = String::from("Succeeded");
            op.result = event
                .context_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::ContextFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.context_failed_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::CallbackSucceeded) => {
            op.status = String::from("Succeeded");
            op.result = event
                .callback_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::CallbackFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.callback_failed_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::CallbackTimedOut) => {
            op.status = String::from("TimedOut");
            if let Some(details) = event.callback_timed_out_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::ChainedInvokeSucceeded) => {
            op.status = String::from("Succeeded");
            op.result = event
                .chained_invoke_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::ChainedInvokeFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.chained_invoke_failed_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::ChainedInvokeTimedOut) => op.status = String::from("TimedOut"),
        Some(EventType::ChainedInvokeStopped) => op.status = String::from("Stopped"),
        _ => {}
    }
}

/// Extracts the JSON payload string from an optional `EventResult`.
fn event_result_payload(result: Option<&aws_sdk_lambda::types::EventResult>) -> Option<String> {
    result
        .and_then(aws_sdk_lambda::types::EventResult::payload)
        .map(ToOwned::to_owned)
}

/// Records the error type/message from an optional `EventError` onto the op.
fn set_error(op: &mut TestOperation, error: Option<&aws_sdk_lambda::types::EventError>) {
    if let Some(obj) = error.and_then(aws_sdk_lambda::types::EventError::payload) {
        op.error_type = obj.error_type().map(ToOwned::to_owned);
        op.error_message = obj.error_message().map(ToOwned::to_owned);
    }
}

/// Records the number of completed retries from an optional `RetryDetails`
/// onto the op.
///
/// `RetryDetails.current_attempt` is 1-based (a first-try success reports `1`),
/// while [`TestOperation::attempt`] reports completed retries (a first-try
/// success reports `0`). The two are reconciled here so both runners agree.
/// `saturating_sub` guards the conversion: a non-positive `current_attempt`
/// maps to `0` without underflow.
fn set_attempt(op: &mut TestOperation, retry: Option<&aws_sdk_lambda::types::RetryDetails>) {
    if let Some(details) = retry {
        let one_based = u32::try_from(details.current_attempt().max(0)).unwrap_or(0);
        op.attempt = one_based.saturating_sub(1);
    }
}

/// Maps an event type to the operation-type string [`LocalRunner`] uses, or
/// `None` for execution-level / invocation-level events that are not folded
/// into operations.
fn event_family_op_type(event_type: Option<&EventType>) -> Option<&'static str> {
    match event_type? {
        EventType::StepStarted | EventType::StepSucceeded | EventType::StepFailed => Some("Step"),
        EventType::WaitStarted | EventType::WaitSucceeded | EventType::WaitCancelled => {
            Some("Wait")
        }
        EventType::ContextStarted | EventType::ContextSucceeded | EventType::ContextFailed => {
            Some("Context")
        }
        EventType::CallbackStarted
        | EventType::CallbackSucceeded
        | EventType::CallbackFailed
        | EventType::CallbackTimedOut => Some("Callback"),
        EventType::ChainedInvokeStarted
        | EventType::ChainedInvokeSucceeded
        | EventType::ChainedInvokeFailed
        | EventType::ChainedInvokeTimedOut
        | EventType::ChainedInvokeStopped => Some("ChainedInvoke"),
        _ => None,
    }
}

/// Counts `InvocationCompleted` events — the number of times the service
/// invoked the function to drive the execution.
fn invocations_from_history(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e.event_type(), Some(EventType::InvocationCompleted)))
        .count()
}

/// Human-readable execution status for a failed/terminal disposition.
fn execution_status_str(status: &ExecutionStatus) -> &'static str {
    match *status {
        ExecutionStatus::Succeeded => "Succeeded",
        ExecutionStatus::Failed => "Failed",
        ExecutionStatus::TimedOut => "TimedOut",
        ExecutionStatus::Stopped => "Stopped",
        ExecutionStatus::Running => "Running",
        _ => "Unknown",
    }
}

// ────────────────────────────────────────────────────────────────────────────
// In-memory backend (internal — implements the pub(crate) ExecutionClient)
// ────────────────────────────────────────────────────────────────────────────

/// A stored operation record inside the in-memory backend.
#[derive(Debug, Clone)]
struct StoredOp {
    id: String,
    op_type: OperationType,
    sub_type: Option<String>,
    status: OperationStatus,
    name: Option<String>,
    result: Option<String>,
    error_type: Option<String>,
    error_message: Option<String>,
    attempt: i32,
    callback_id: Option<String>,
    wait_seconds: Option<i32>,
    retry_delay: Option<i32>,
    callback_delivered: bool,
}

/// Mutable state of the in-memory backend, guarded by a single mutex.
#[derive(Debug)]
struct BackendState {
    /// Operations in first-checkpoint order.
    ops: Vec<StoredOp>,
    token_counter: u64,
    callback_counter: u64,
    /// Queued external callback outcomes, delivered FIFO.
    callback_outcomes: Vec<CallbackOutcome>,
}

/// The in-memory execution client. Shared (via `Arc`) between the runner and
/// the durable context. Debug-printable and `Send + Sync` as the
/// `ExecutionClient` trait requires.
#[derive(Debug)]
struct Backend {
    state: Mutex<BackendState>,
}

impl Backend {
    fn new(callback_outcomes: Vec<CallbackOutcome>) -> Self {
        Self {
            state: Mutex::new(BackendState {
                ops: Vec::new(),
                token_counter: 0,
                callback_counter: 0,
                callback_outcomes,
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BackendState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The current checkpoint token string.
    fn current_token(&self) -> String {
        let state = self.lock();
        format!("local-token-{}", state.token_counter)
    }

    /// Builds `Operation` records for the current state (used to seed the
    /// per-invocation checkpoint log).
    fn build_operations(&self) -> Vec<Operation> {
        let state = self.lock();
        state.ops.iter().filter_map(build_operation).collect()
    }

    /// Snapshots the recorded operations as public [`TestOperation`]s.
    fn snapshot_operations(&self) -> Vec<TestOperation> {
        let state = self.lock();
        state
            .ops
            .iter()
            .map(|op| TestOperation {
                id: op.id.clone(),
                op_type: operation_type_str(&op.op_type).to_owned(),
                sub_type: op.sub_type.clone(),
                status: operation_status_str(&op.status).to_owned(),
                result: op.result.clone(),
                error_type: op.error_type.clone(),
                error_message: op.error_message.clone(),
                attempt: u32::try_from(op.attempt.max(0)).unwrap_or(0),
                name: op.name.clone(),
            })
            .collect()
    }

    /// Advances the simulated backend between invocations: fire elapsed
    /// timers, elapse retry delays, and deliver or time out pending
    /// callbacks. Returns `true` if any operation was advanced.
    ///
    /// Backend-behaviour anchors (this crate's own replay logic):
    /// - `wait`: a `Started` wait becomes `Succeeded` once its timer
    ///   elapses (`src/wait.rs` replay: `Succeeded` returns immediately,
    ///   `Started` suspends).
    /// - `step`/`wait_for_condition` retry: a `Pending` retry becomes
    ///   `Ready` once its delay elapses, so the next invocation re-executes
    ///   the body (`src/step.rs` and `src/wait_for_condition.rs` replay:
    ///   `Pending` suspends, `Ready` falls through to live execution).
    /// - callback: a `Pending` callback resolves to `Succeeded` (with a
    ///   payload) or `TimedOut` (`src/callback.rs` replay).
    fn advance(&self) -> bool {
        let mut state = self.lock();
        let mut progressed = false;

        // Timers and retry delays.
        for op in &mut state.ops {
            match op.op_type {
                OperationType::Wait if op.status == OperationStatus::Started => {
                    op.status = OperationStatus::Succeeded;
                    progressed = true;
                }
                OperationType::Step
                    if op.status == OperationStatus::Pending && op.retry_delay.is_some() =>
                {
                    // Retry timer elapsed: make the op re-executable.
                    op.status = OperationStatus::Ready;
                    op.retry_delay = None;
                    progressed = true;
                }
                _ => {}
            }
        }

        // Callback delivery (FIFO across queued outcomes).
        // Split the borrow so we can pop the outcome queue while iterating.
        let BackendState {
            ops,
            callback_outcomes,
            ..
        } = &mut *state;
        for op in ops.iter_mut() {
            let is_pending_callback = op.op_type == OperationType::Callback
                && !op.callback_delivered
                && matches!(
                    op.status,
                    OperationStatus::Pending | OperationStatus::Started
                );
            if !is_pending_callback {
                continue;
            }
            if callback_outcomes.is_empty() {
                continue;
            }
            let outcome = callback_outcomes.remove(0);
            match outcome {
                CallbackOutcome::Success(payload) => {
                    op.status = OperationStatus::Succeeded;
                    op.result = Some(payload);
                }
                CallbackOutcome::Timeout => {
                    op.status = OperationStatus::TimedOut;
                }
            }
            op.callback_delivered = true;
            progressed = true;
        }

        progressed
    }
}

impl BackendState {
    /// Finds a stored op by wire ID.
    fn find_mut(&mut self, id: &str) -> Option<&mut StoredOp> {
        self.ops.iter_mut().find(|op| op.id == id)
    }

    /// Applies a single `OperationUpdate` to the store, returning the
    /// resulting stored op (cloned) for the checkpoint response.
    fn apply_update(&mut self, update: &OperationUpdate) -> Option<StoredOp> {
        let id = update.id().to_owned();
        let op_type = update.r#type().clone();
        let sub_type = update.sub_type().map(ToOwned::to_owned);
        let action = update.action().clone();
        let name = update.name().map(ToOwned::to_owned);
        let payload = update.payload().map(ToOwned::to_owned);
        let (err_type, err_msg) = update.error().map_or((None, None), |e| {
            (
                e.error_type().map(ToOwned::to_owned),
                e.error_message().map(ToOwned::to_owned),
            )
        });
        let wait_seconds = update
            .wait_options()
            .and_then(aws_sdk_lambda::types::WaitOptions::wait_seconds);
        let retry_delay = update
            .step_options()
            .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds);

        // Ensure the op exists.
        if self.find_mut(&id).is_none() {
            self.ops.push(StoredOp {
                id: id.clone(),
                op_type: op_type.clone(),
                sub_type: sub_type.clone(),
                status: OperationStatus::Started,
                name: name.clone(),
                result: None,
                error_type: None,
                error_message: None,
                attempt: 0,
                callback_id: None,
                wait_seconds: None,
                retry_delay: None,
                callback_delivered: false,
            });
        }

        // Decide callback-id assignment before taking the mutable op borrow
        // (assigning self.callback_counter later would conflict with it).
        let needs_callback_id = matches!(action, OperationAction::Start)
            && op_type == OperationType::Callback
            && self
                .ops
                .iter()
                .find(|o| o.id == id)
                .is_some_and(|o| o.callback_id.is_none());
        let new_callback_id = if needs_callback_id {
            self.callback_counter += 1;
            Some(format!("local-callback-{}", self.callback_counter))
        } else {
            None
        };

        let op = self.find_mut(&id)?;

        // Keep metadata current.
        if op.sub_type.is_none() {
            op.sub_type = sub_type;
        }
        if name.is_some() {
            op.name = name;
        }

        match action {
            OperationAction::Start => {
                // A Start carries no payload — clear any prior result so the
                // whole-record merge in the SDK clobbers the carried value
                // (the wait_for_condition read-before-Start invariant).
                op.result = None;
                op.error_type = None;
                op.error_message = None;
                if op.op_type == OperationType::Wait {
                    op.status = OperationStatus::Started;
                    op.wait_seconds = wait_seconds;
                } else if op.op_type == OperationType::Callback {
                    // Callbacks are pending until an external outcome; the
                    // backend assigns the callback id here.
                    op.status = OperationStatus::Pending;
                    if let Some(cb) = new_callback_id {
                        op.callback_id = Some(cb);
                    }
                } else {
                    op.status = OperationStatus::Started;
                }
            }
            OperationAction::Succeed => {
                op.status = OperationStatus::Succeeded;
                op.result = payload;
            }
            OperationAction::Fail => {
                op.status = OperationStatus::Failed;
                op.error_type = err_type;
                op.error_message = err_msg;
            }
            OperationAction::Retry => {
                op.status = OperationStatus::Pending;
                op.attempt += 1;
                op.retry_delay = retry_delay.or(Some(1));
                op.error_type = err_type;
                op.error_message = err_msg;
                // wait_for_condition carries per-attempt state in the payload.
                if payload.is_some() {
                    op.result = payload;
                }
            }
            _ => {}
        }

        Some(op.clone())
    }
}

impl ExecutionClient for Backend {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>>
    {
        let updated_ops: Vec<Operation> = {
            let mut state = self.lock();
            state.token_counter += 1;
            let mut out = Vec::with_capacity(updates.len());
            for update in &updates {
                if let Some(stored) = state.apply_update(update)
                    && let Some(op) = build_operation(&stored)
                {
                    out.push(op);
                }
            }
            out
        };
        let token = self.current_token();
        Box::pin(async move {
            Ok(CheckpointOutput {
                checkpoint_token: token,
                updated_operations: updated_ops,
            })
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
    {
        let operations = self.build_operations();
        Box::pin(async move { Ok(GetStateOutput { operations }) })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StoredOp → aws_sdk_lambda::types::Operation
// ────────────────────────────────────────────────────────────────────────────

/// Builds an SDK `Operation` from a stored op, placing result/error in the
/// details bucket the SDK reads for that operation type.
fn build_operation(stored: &StoredOp) -> Option<Operation> {
    let error_obj = if stored.error_type.is_some() || stored.error_message.is_some() {
        let mut eb = ErrorObject::builder();
        if let Some(t) = &stored.error_type {
            eb = eb.error_type(t);
        }
        if let Some(m) = &stored.error_message {
            eb = eb.error_message(m);
        }
        Some(eb.build())
    } else {
        None
    };

    let mut builder = Operation::builder()
        .id(&stored.id)
        .r#type(stored.op_type.clone())
        .status(stored.status.clone())
        .start_timestamp(DateTime::from_secs(0));

    match stored.op_type {
        OperationType::Step => {
            let mut sd = StepDetails::builder().attempt(stored.attempt);
            if let Some(r) = &stored.result {
                sd = sd.result(r);
            }
            if let Some(e) = error_obj.clone() {
                sd = sd.error(e);
            }
            builder = builder.step_details(sd.build());
        }
        OperationType::Context => {
            let mut cd = ContextDetails::builder();
            if let Some(r) = &stored.result {
                cd = cd.result(r);
            }
            if let Some(e) = error_obj.clone() {
                cd = cd.error(e);
            }
            builder = builder.context_details(cd.build());
        }
        OperationType::Callback => {
            let mut cb = CallbackDetails::builder();
            if let Some(id) = &stored.callback_id {
                cb = cb.callback_id(id);
            }
            if let Some(r) = &stored.result {
                cb = cb.result(r);
            }
            if let Some(e) = error_obj.clone() {
                cb = cb.error(e);
            }
            builder = builder.callback_details(cb.build());
        }
        OperationType::ChainedInvoke => {
            let mut ci = ChainedInvokeDetails::builder();
            if let Some(r) = &stored.result {
                ci = ci.result(r);
            }
            if let Some(e) = error_obj.clone() {
                ci = ci.error(e);
            }
            builder = builder.chained_invoke_details(ci.build());
        }
        _ => {}
    }

    builder.build().ok()
}

/// Human-readable operation type for [`TestOperation::op_type`].
fn operation_type_str(t: &OperationType) -> &'static str {
    match *t {
        OperationType::Step => "Step",
        OperationType::Wait => "Wait",
        OperationType::Context => "Context",
        OperationType::Callback => "Callback",
        OperationType::ChainedInvoke => "ChainedInvoke",
        OperationType::Execution => "Execution",
        _ => "Unknown",
    }
}

/// Human-readable operation status for [`TestOperation::status`].
fn operation_status_str(s: &OperationStatus) -> &'static str {
    match *s {
        OperationStatus::Started => "Started",
        OperationStatus::Pending => "Pending",
        OperationStatus::Ready => "Ready",
        OperationStatus::Succeeded => "Succeeded",
        OperationStatus::Failed => "Failed",
        OperationStatus::Cancelled => "Cancelled",
        OperationStatus::TimedOut => "TimedOut",
        OperationStatus::Stopped => "Stopped",
        _ => "Unknown",
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
#[allow(clippy::indexing_slicing)] // reason: test assertions index known-length op vectors
mod tests {
    use super::*;
    use crate::{CompletionConfig, RetryDecision, RetryStrategy, StepError};
    use serde::{Deserialize, Serialize};
    use std::sync::atomic::{AtomicU32, Ordering};

    // 1. step success ────────────────────────────────────────────────────

    #[tokio::test]
    async fn step_success_returns_output_and_records_op() {
        let result = LocalRunner::new()
            .run(
                |n: i32, ctx: DurableContext| async move {
                    let v = ctx
                        .step(move |_| async move { Ok(n + 1) })
                        .name("inc")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                41_i32,
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&42));
        assert_eq!(result.invocation_count(), 1);
        assert_eq!(result.operations().len(), 1);
        assert!(result.operations()[0].succeeded());
        assert_eq!(result.operations()[0].name(), Some("inc"));
    }

    // 1. step failure ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn step_failure_propagates_and_records_failed_op() {
        // A strategy that never retries so the failure is permanent.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let strategy: RetryStrategy =
                        Box::new(|_e: &StepError, _a: u32| RetryDecision::Stop);
                    let v: i32 = ctx
                        .step(|_| async { Err("boom".into()) })
                        .retry_strategy(strategy)
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("StepError"));
        assert!(result.operations().iter().any(TestOperation::failed));
    }

    // 1. step retry (retry strategy honored) ──────────────────────────────

    #[tokio::test]
    async fn step_retry_succeeds_on_third_attempt() {
        // Fails on attempts 1 and 2, succeeds on attempt 3. Each retry
        // suspends and the runner elapses the retry delay before re-invoking.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let attempts_seen = Arc::clone(&attempts_seen);
                        let strategy: RetryStrategy = Box::new(|_e: &StepError, attempt: u32| {
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: std::time::Duration::from_secs(1),
                                }
                            }
                        });
                        let v: i32 = ctx
                            .step(move |sc| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    if sc.attempt() < 3 {
                                        Err(format!("attempt {} failed", sc.attempt()).into())
                                    } else {
                                        Ok(99)
                                    }
                                }
                            })
                            .name("flaky")
                            .retry_strategy(strategy)
                            .await?;
                        Ok::<_, BoxError>(v)
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&99));
        // Body executed exactly 3 times (attempts 1, 2, 3).
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        // The step took multiple invocations (one per retry suspension).
        assert!(result.invocation_count() >= 3);
        let step = result
            .operations()
            .iter()
            .find(|o| o.name() == Some("flaky"))
            .expect("flaky step recorded");
        assert!(step.succeeded());
        assert_eq!(step.attempt(), 2, "backend recorded 2 completed retries");
    }

    // 2. wait then resume across invocations ──────────────────────────────

    #[tokio::test]
    async fn wait_resumes_across_invocations() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let before = ctx
                        .step(|_| async { Ok("before".to_owned()) })
                        .name("before")
                        .await?;
                    ctx.wait(std::time::Duration::from_mins(5))
                        .name("nap")
                        .await?;
                    let after = ctx
                        .step(|_| async { Ok("after".to_owned()) })
                        .name("after")
                        .await?;
                    Ok::<_, BoxError>(format!("{before}-{after}"))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("before-after"));
        // The wait forced at least a second invocation.
        assert!(result.invocation_count() >= 2);
        let wait_op = result
            .operations()
            .iter()
            .find(|o| o.op_type() == "Wait")
            .expect("wait op recorded");
        assert!(wait_op.succeeded());
    }

    // 3. wait_for_condition state carry (exercises the merge-back clobber) ─

    #[derive(Clone, Serialize, Deserialize)]
    struct Counter {
        count: u32,
    }

    #[tokio::test]
    async fn wait_for_condition_carries_state_across_attempts() {
        // The check increments the carried count each attempt; the strategy
        // completes once count reaches 3. If the carried state were clobbered
        // to initial_state each attempt (the pre-554ac90 bug), count would
        // never advance and the runner would exhaust its invocation budget.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let strategy: crate::WaitStrategyFn<Counter> =
                        Box::new(|state: Counter, _attempt: u32| {
                            if state.count >= 3 {
                                crate::WaitDecision::complete()
                            } else {
                                crate::WaitDecision::continue_with(std::time::Duration::from_secs(
                                    1,
                                ))
                            }
                        });
                    let final_state = ctx
                        .wait_for_condition(
                            |_sc, state: Counter| async move {
                                Ok(Counter {
                                    count: state.count + 1,
                                })
                            },
                            Counter { count: 0 },
                        )
                        .name("poll")
                        .wait_strategy_fn(strategy)
                        .await?;
                    Ok::<_, BoxError>(final_state.count)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&3),
            "carried state must advance across attempts (merge-back clobber must not reset it)"
        );
    }

    // 4. child contexts ───────────────────────────────────────────────────

    #[tokio::test]
    async fn child_context_returns_nested_result() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let v = ctx
                        .run_in_child_context(|child| async move {
                            let a = child.step(|_| async { Ok(10) }).await?;
                            let b = child.step(|_| async { Ok(20) }).await?;
                            Ok(a + b)
                        })
                        .name("branch")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&30));
        // Parent context op + two child steps recorded.
        assert!(result.operations().iter().any(|o| o.op_type() == "Context"));
        let step_count = result
            .operations()
            .iter()
            .filter(|o| o.op_type() == "Step")
            .count();
        assert_eq!(step_count, 2);
    }

    // 5. parallel and map basics ──────────────────────────────────────────

    #[tokio::test]
    async fn parallel_runs_branches_and_collects_results() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let branches = vec![
                        crate::Branch::new("a", |c: DurableContext| async move {
                            let v = c.step(|_| async { Ok(1) }).await?;
                            Ok(v)
                        }),
                        crate::Branch::new("b", |c: DurableContext| async move {
                            let v = c.step(|_| async { Ok(2) }).await?;
                            Ok(v)
                        }),
                    ];
                    let results: Vec<i32> = ctx.parallel(branches).name("fan").await?;
                    Ok::<_, BoxError>(results.into_iter().sum::<i32>())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&3));
    }

    #[tokio::test]
    async fn map_applies_over_items() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let items = vec![1_i32, 2, 3];
                    let out: Vec<i32> = ctx
                        .map(items, |child, item, _idx| async move {
                            let scaled = child.step(move |_| async move { Ok(item * 10) }).await?;
                            Ok(scaled)
                        })
                        .name("scale")
                        .max_concurrency(2)
                        .completion(CompletionConfig::default())
                        .await?;
                    Ok::<_, BoxError>(out.into_iter().sum::<i32>())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&60));
    }

    // An arbitrary user error returned as BoxError from a child closure
    // surfaces with its message intact through the operation's error type,
    // with no error-conversion ceremony at the boundary.
    #[tokio::test]
    async fn child_closure_boxerror_message_propagates() {
        #[derive(Debug)]
        struct CustomBoundaryError;
        impl std::fmt::Display for CustomBoundaryError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "custom-boundary-failure")
            }
        }
        impl std::error::Error for CustomBoundaryError {}

        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let v: i32 = ctx
                        .run_in_child_context(|_child| async move {
                            // Plain `?`-free early return of an arbitrary error;
                            // no map_err, no ChildFnError.
                            Err(CustomBoundaryError)?;
                            Ok(0)
                        })
                        .name("boundary")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(
            !result.is_success(),
            "child failure must surface as an error"
        );
        let msg = result.error_message().unwrap_or_default();
        assert!(
            msg.contains("custom-boundary-failure"),
            "user error message must survive the boundary: {msg}"
        );
    }

    // 6. callback success and callback timeout ────────────────────────────

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Approval {
        approved: bool,
    }

    #[tokio::test]
    async fn callback_success_delivers_payload() {
        let result = LocalRunner::new()
            .callback_success(&Approval { approved: true })
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<Approval>().name("approval").await?;
                    let approval = cb.result().await?;
                    Ok::<_, BoxError>(approval.approved)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&true));
        assert!(result.invocation_count() >= 2);
    }

    #[tokio::test]
    async fn callback_timeout_surfaces_error() {
        let result = LocalRunner::new()
            .callback_timeout()
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<Approval>().name("approval").await?;
                    let approval = cb.result().await?;
                    Ok::<_, BoxError>(approval.approved)
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("CallbackError"));
        let cb_op = result
            .operations()
            .iter()
            .find(|o| o.op_type() == "Callback")
            .expect("callback op recorded");
        assert_eq!(cb_op.status(), "TimedOut");
    }

    /// A callback result created via the public `ctx.create_callback` API
    /// composes with a real step future inside `try_join_all`. This exercises
    /// the full builder→callback→combinator path through replay, ensuring
    /// regressions in the public surface are caught.
    #[tokio::test]
    async fn callback_result_composes_with_step_in_combinator() {
        let result = LocalRunner::new()
            .callback_success(&"from-callback".to_owned())
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().name("approval").await?;

                    let step_future = ctx
                        .step(|_| async { Ok("from-step".to_owned()) })
                        .name("work")
                        .spawn();

                    let results = ctx.try_join_all([cb.result(), step_future]).await?;

                    Ok::<_, BoxError>(results)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler should succeed: {:?}",
            result.error_message()
        );
        let output = result.output().expect("should have output");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "from-callback");
        assert_eq!(output[1], "from-step");
        // At least 2 invocations: first suspends on the callback, second
        // replays with the settled outcome and completes.
        assert!(result.invocation_count() >= 2);
    }

    // 7. replay determinism ───────────────────────────────────────────────

    #[tokio::test]
    async fn replay_is_deterministic() {
        // The same handler + event yields identical operation IDs and
        // results across two independent runs.
        let make_run = || async {
            LocalRunner::new()
                .run(
                    |(), ctx: DurableContext| async move {
                        let a = ctx.step(|_| async { Ok(1) }).name("a").await?;
                        ctx.wait(std::time::Duration::from_mins(1))
                            .name("w")
                            .await?;
                        let b = ctx.step(|_| async { Ok(2) }).name("b").await?;
                        Ok::<_, BoxError>(a + b)
                    },
                    (),
                )
                .await
        };

        let first = make_run().await;
        let second = make_run().await;

        assert!(first.is_success());
        assert!(second.is_success());
        assert_eq!(first.output(), second.output());

        let ids_first: Vec<&str> = first.operations().iter().map(TestOperation::id).collect();
        let ids_second: Vec<&str> = second.operations().iter().map(TestOperation::id).collect();
        assert_eq!(ids_first, ids_second, "operation IDs must be deterministic");

        let results_first: Vec<Option<&str>> = first
            .operations()
            .iter()
            .map(TestOperation::result)
            .collect();
        let results_second: Vec<Option<&str>> = second
            .operations()
            .iter()
            .map(TestOperation::result)
            .collect();
        assert_eq!(results_first, results_second);
    }

    // ── CloudRunner pure-mapping tests (no AWS) ──────────────────────────

    use aws_sdk_lambda::types::{
        CallbackTimedOutDetails, EventError, EventResult, RetryDetails, StepFailedDetails,
        StepSucceededDetails,
    };

    #[test]
    fn fold_history_folds_step_lifecycle() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("op-1")
                .name("charge")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("op-1")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("\"ok\"").build())
                        .retry_details(RetryDetails::builder().current_attempt(0).build())
                        .build(),
                )
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type(), "Step");
        assert_eq!(ops[0].status(), "Succeeded");
        assert_eq!(ops[0].result(), Some("\"ok\""));
        assert_eq!(ops[0].name(), Some("charge"));
        assert_eq!(ops[0].id(), "op-1");
        assert!(ops[0].succeeded());
    }

    #[test]
    fn fold_history_records_retry_attempt() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepFailed)
                .id("s")
                .event_id(2)
                .step_failed_details(
                    StepFailedDetails::builder()
                        .error(
                            EventError::builder()
                                .payload(
                                    ErrorObject::builder()
                                        .error_type("StepError")
                                        .error_message("boom")
                                        .build(),
                                )
                                .build(),
                        )
                        .retry_details(RetryDetails::builder().current_attempt(1).build())
                        .build(),
                )
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(3)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("99").build())
                        .retry_details(RetryDetails::builder().current_attempt(2).build())
                        .build(),
                )
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].status(), "Succeeded");
        assert_eq!(
            ops[0].attempt(),
            1,
            "current_attempt 2 normalizes to 1 completed retry"
        );
        assert_eq!(ops[0].result(), Some("99"));
    }

    #[test]
    fn fold_history_normalizes_attempt_to_completed_retries() {
        // A first-try success reports current_attempt=1, which must normalize
        // to 0 completed retries so LocalRunner and CloudRunner agree.
        let first_try = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("\"ok\"").build())
                        .retry_details(RetryDetails::builder().current_attempt(1).build())
                        .build(),
                )
                .build(),
        ];
        assert_eq!(
            fold_history(&first_try)[0].attempt(),
            0,
            "first-try success (current_attempt=1) is 0 completed retries"
        );

        // An unexpected current_attempt=0 must not underflow or panic.
        let zero = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("\"ok\"").build())
                        .retry_details(RetryDetails::builder().current_attempt(0).build())
                        .build(),
                )
                .build(),
        ];
        assert_eq!(
            fold_history(&zero)[0].attempt(),
            0,
            "non-positive current_attempt saturates to 0"
        );
    }

    #[test]
    fn fold_history_maps_callback_timeout() {
        let events = vec![
            Event::builder()
                .event_type(EventType::CallbackStarted)
                .id("cb")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::CallbackTimedOut)
                .id("cb")
                .event_id(2)
                .callback_timed_out_details(
                    CallbackTimedOutDetails::builder()
                        .error(
                            EventError::builder()
                                .payload(
                                    ErrorObject::builder()
                                        .error_type("CallbackError")
                                        .error_message("timed out")
                                        .build(),
                                )
                                .build(),
                        )
                        .build(),
                )
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type(), "Callback");
        assert_eq!(ops[0].status(), "TimedOut");
        assert_eq!(ops[0].error_type(), Some("CallbackError"));
    }

    #[test]
    fn fold_history_skips_execution_and_invocation_events() {
        let events = vec![
            Event::builder()
                .event_type(EventType::ExecutionStarted)
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::InvocationCompleted)
                .event_id(2)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("op")
                .event_id(3)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("1").build())
                        .build(),
                )
                .build(),
            Event::builder()
                .event_type(EventType::InvocationCompleted)
                .event_id(4)
                .build(),
            Event::builder()
                .event_type(EventType::ExecutionSucceeded)
                .event_id(5)
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1, "only the operation-scoped event is folded");
        assert_eq!(ops[0].op_type(), "Step");
        assert_eq!(invocations_from_history(&events), 2);
    }

    #[test]
    fn fold_history_preserves_first_appearance_order() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("first")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::WaitStarted)
                .id("second")
                .event_id(2)
                .build(),
            Event::builder()
                .event_type(EventType::WaitSucceeded)
                .id("second")
                .event_id(3)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("first")
                .event_id(4)
                .step_succeeded_details(StepSucceededDetails::builder().build())
                .build(),
        ];
        let ops = fold_history(&events);
        let ids: Vec<&str> = ops.iter().map(TestOperation::id).collect();
        assert_eq!(ids, vec!["first", "second"]);
        assert_eq!(ops[1].op_type(), "Wait");
        assert_eq!(ops[1].status(), "Succeeded");
    }

    #[test]
    fn deserialize_output_handles_plain_missing_and_double_encoded() {
        assert_eq!(deserialize_output::<i32>(Some("42")).unwrap(), 42);
        assert_eq!(deserialize_output::<()>(None).unwrap(), ());
        // Double-encoded: a JSON string whose contents are the JSON value.
        assert_eq!(deserialize_output::<i32>(Some("\"7\"")).unwrap(), 7);
        assert!(deserialize_output::<i32>(Some("not json")).is_err());
    }

    #[test]
    fn execution_status_and_family_maps() {
        assert_eq!(execution_status_str(&ExecutionStatus::TimedOut), "TimedOut");
        assert_eq!(execution_status_str(&ExecutionStatus::Stopped), "Stopped");
        assert_eq!(
            event_family_op_type(Some(&EventType::ChainedInvokeSucceeded)),
            Some("ChainedInvoke")
        );
        assert_eq!(
            event_family_op_type(Some(&EventType::ExecutionStarted)),
            None
        );
        assert_eq!(event_family_op_type(None), None);
    }

    #[test]
    fn cloud_failure_is_reported_as_failure() {
        let result: TestResult<i32> = cloud_failure("boom".to_owned());
        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("CloudRunnerError"));
        assert_eq!(result.output(), None);
    }
}
