//! In-process local testing runner (`test-util` feature).
//!
//! [`LocalRunner`] drives a durable handler to completion entirely in
//! memory: each simulated invocation goes through the **same service
//! function production uses**: the one [`wrap`](crate::wrap) builds:
//! fed a synthesized invocation envelope and backed by an internal
//! in-memory execution client instead of the Lambda checkpoint API.
//! Envelope parsing, bootstrap pagination, the suspension driver, wire
//! error mapping, and the response envelope are therefore the production
//! code paths, not a parallel reimplementation. When the handler
//! suspends, the runner advances the simulated backend (timers fire,
//! retry delays elapse, callbacks are delivered) and re-invokes, exactly
//! as the real service would, until the execution reaches a terminal
//! outcome.
//!
//! The runner also reproduces the **production task topology**: the
//! service future is awaited inline (under the caller's `block_on`, as
//! `lambda_runtime` does), never on a spawned task, so
//! `tokio::task::try_id()` is `None` at context-creation time and the
//! task-ownership guard behaves exactly as it does in deployment.
//!
//! By default the in-memory backend serves execution state in **multiple
//! pages** (2+ whenever the recorded history is large enough), matching
//! the paginating service; [`LocalRunner::single_page`] opts into the
//! single-page special case.
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
//!   carries no payload, so the merged record clears any carried result:
//!   the exact condition the `wait_for_condition` read-before-`Start` fix
//!   guards against.
//! - The backend keeps a virtual clock in whole seconds. It starts at 0
//!   for each run. A wait is due at the clock time it started plus its
//!   duration, and a retry is due at the clock time it was recorded plus
//!   its delay. Between invocations, each advance moves the clock to the
//!   earliest pending wait or retry end and fires every wait and retry due
//!   by then. So waits and retries end in time order, as on the service.
//! - Queued callback replies arrive before the clock moves. An advance that
//!   delivers a callback reply fires no timer.
//! - A chained invoke runs its registered target between invocations,
//!   before the virtual clock moves (see [`LocalRunner::register_function`]
//!   and [`LocalRunner::register_durable_function`]).
//! - Callback replies are queued before the run. The runner runs no
//!   callback timeout or heartbeat timer; [`CallbackReply::timeout`] and
//!   [`CallbackReply::heartbeat_timeout`] deliver the records those timers
//!   produce.
//! - Time never moves while an invocation runs. So a local run can use
//!   more invocations than the same execution uses on the service.
//! - [`LocalRunner::checkpoint_latency`] delays each checkpoint response in
//!   tokio time. It does not move the virtual clock.
//! - The backend stores each operation's `ParentId` and the `ReplayChildren`
//!   option of a completed context. It omits the children of a completed
//!   context from the invocation input and from `GetDurableExecutionState`,
//!   as the service does, unless the context recorded `ReplayChildren`.
//!   Checkpoint responses and [`TestResult::operations`] keep every record.
//! - The backend rejects a checkpoint that exceeds a service size limit or
//!   a delay range, with the service's message. It does not check the rest
//!   of the service's update model: ID and name patterns, parent existence
//!   and action order.
//! - Each invocation input carries `UpdatedOperationIds`: the operations
//!   the backend changed between invocations, such as a fired wait, a
//!   retry that became ready, or a callback that received a reply.
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
//! use aws_durable_execution_sdk as durable;
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

use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;

use aws_sdk_lambda::primitives::{Blob, DateTime};
use aws_sdk_lambda::types::{
    CallbackDetails, ChainedInvokeDetails, ContextDetails, ContextOptions, ErrorObject, Event,
    EventType, ExecutionStatus, InvocationType, Operation, OperationAction, OperationStatus,
    OperationType, OperationUpdate, StepDetails,
};

use crate::BoxError;
use crate::client::{CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput};
use crate::context::DurableContext;

/// Default cap on the number of invocations the runner will drive before
/// declaring the execution stuck. Generous enough for deep timer/retry
/// chains, low enough to fail a non-terminating handler (e.g. a
/// `wait_for_condition` that never advances) instead of looping forever.
const DEFAULT_MAX_INVOCATIONS: usize = 100;

/// Default page size for simulated execution-state pagination. `1` maximizes
/// fidelity pressure: any history of two or more operations is served as 2+
/// pages (inline first page plus a `get_state` fetch), so every non-trivial
/// test exercises the bootstrap and checkpoint pagination paths by default:
/// the dimension where a missing-pagination defect was previously untestable
/// by construction.
const DEFAULT_STATE_PAGE_SIZE: usize = 1;

/// A single checkpointed operation recorded during a [`LocalRunner`] run.
///
/// Returned in order by [`TestResult::operations`]. All accessors return
/// borrowed views; construct these only through the runner.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk as durable;
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
#[non_exhaustive]
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
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The operation type, e.g. `"Step"`, `"Wait"`, `"Context"`,
    /// `"Callback"`.
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// The operation sub-type, e.g. `"Step"`, `"Wait"`, `"Map"`,
    /// `"WaitForCondition"`, or `None` if the operation carried none.
    pub fn sub_type(&self) -> Option<&str> {
        self.sub_type.as_deref()
    }

    /// The terminal-or-current status, e.g. `"Succeeded"`, `"Failed"`,
    /// `"Pending"`, `"TimedOut"`.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Returns `true` if the operation completed successfully.
    pub fn succeeded(&self) -> bool {
        self.status == "Succeeded"
    }

    /// Returns `true` if the operation ended in a failed status.
    pub fn failed(&self) -> bool {
        self.status == "Failed"
    }

    /// The serialized result payload, if the operation stored one.
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }

    /// The recorded error type, if the operation failed.
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The recorded error message, if the operation failed.
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The number of completed retries recorded for this operation
    /// (`0` when never retried). [`LocalRunner`] and [`CloudRunner`] report
    /// this value identically.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// The operation's caller-assigned name, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

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
/// use aws_durable_execution_sdk as durable;
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
#[non_exhaustive]
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
    pub fn is_success(&self) -> bool {
        self.disposition == Disposition::Succeeded
    }

    /// Returns `true` if the handler returned an error.
    pub fn is_failure(&self) -> bool {
        self.disposition == Disposition::Failed
    }

    /// Returns `true` if the handler suspended and never reached a terminal
    /// outcome within the invocation budget (e.g. it awaits a callback the
    /// test never delivered, or it never terminates).
    pub fn is_suspended(&self) -> bool {
        self.disposition == Disposition::Suspended
    }

    /// The deserialized handler output on success, or `None` otherwise.
    pub fn output(&self) -> Option<&O> {
        self.output.as_ref()
    }

    /// Consumes the result and returns the owned handler output on success.
    #[must_use]
    pub fn into_output(self) -> Option<O> {
        self.output
    }

    /// The wire error type on failure (e.g. `"StepError"`), or `None`.
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The wire error message on failure, or `None`.
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The recorded operation history, in first-checkpoint order.
    pub fn operations(&self) -> &[TestOperation] {
        &self.operations
    }

    /// The number of invocations that drove this execution.
    ///
    /// For [`LocalRunner`] this is the count of simulated invocations; for
    /// [`CloudRunner`] it is the number of recorded `InvocationCompleted`
    /// events (falling back to the poll count when none are recorded).
    pub fn invocation_count(&self) -> usize {
        self.invocations
    }
}

/// One reply that the simulated backend delivers to a pending callback.
///
/// Queue a reply with [`LocalRunner::callback`], for the next pending
/// callback in order, or with [`LocalRunner::callback_for`], for the
/// callback or `wait_for_callback` with a given name. The runner delivers
/// queued replies between invocations, before it moves its virtual clock.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk as durable;
/// use durable::test_util::{CallbackFailure, CallbackReply, LocalRunner};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), durable::BoxError> {
/// let runner = LocalRunner::new()
///     .callback_for("approval", CallbackReply::success(&true)?)
///     .callback(CallbackReply::failure(CallbackFailure::new(
///         "Declined",
///         "the reviewer said no",
///     )));
/// let result = runner
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             let audit = ctx.create_callback::<bool>().name("audit").await?;
///             let approval = ctx.create_callback::<bool>().name("approval").await?;
///             let approved = approval.result().await?;
///             let audited = audit.result().await.is_ok();
///             Ok::<_, durable::BoxError>((approved, audited))
///         },
///         (),
///     )
///     .await;
///
/// assert_eq!(result.output(), Some(&(true, false)));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CallbackReply {
    kind: CallbackReplyKind,
}

/// The record a [`CallbackReply`] gives a callback.
#[derive(Debug, Clone)]
enum CallbackReplyKind {
    /// `Succeeded` with this payload.
    Success(String),
    /// `Failed` with the reported error fields.
    Failure(CallbackFailure),
    /// `TimedOut`, as a callback timeout records it.
    Timeout,
    /// `TimedOut`, as a missed heartbeat records it.
    HeartbeatTimeout,
}

/// The message the JS local checkpoint server records for a callback
/// timeout.
const CALLBACK_TIMEOUT_MESSAGE: &str = "Callback timed out";
/// The error type the SDK reads as a heartbeat timeout.
const CALLBACK_HEARTBEAT_ERROR_TYPE: &str = "Callback.Heartbeat";
/// The message the JS local checkpoint server records for a missed
/// heartbeat.
const CALLBACK_HEARTBEAT_MESSAGE: &str = "Callback timed out on heartbeat";

impl CallbackReply {
    /// A success whose payload is `value` encoded as JSON.
    ///
    /// # Errors
    ///
    /// Returns the encoding error when `value` fails to serialize.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackReply, LocalRunner};
    ///
    /// # fn main() -> Result<(), aws_durable_execution_sdk::BoxError> {
    /// let runner = LocalRunner::new().callback(CallbackReply::success(&"approved")?);
    /// # drop(runner);
    /// # Ok(())
    /// # }
    /// ```
    pub fn success<T: Serialize>(value: &T) -> Result<Self, BoxError> {
        let payload = serde_json::to_string(value)?;
        Ok(Self {
            kind: CallbackReplyKind::Success(payload),
        })
    }

    /// A success whose payload is `payload`, unchanged.
    ///
    /// The handler decodes the payload with the callback's serdes, so with
    /// the default JSON serdes the payload must be JSON.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackReply, LocalRunner};
    ///
    /// let runner = LocalRunner::new().callback(CallbackReply::raw(r#""approved""#));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn raw(payload: impl Into<String>) -> Self {
        Self {
            kind: CallbackReplyKind::Success(payload.into()),
        }
    }

    /// An external failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackFailure, CallbackReply, LocalRunner};
    ///
    /// let failure = CallbackFailure::new("Declined", "no").with_error_data(r#"{"code":7}"#);
    /// let runner = LocalRunner::new().callback(CallbackReply::failure(failure));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn failure(failure: CallbackFailure) -> Self {
        Self {
            kind: CallbackReplyKind::Failure(failure),
        }
    }

    /// A timeout, as the JS local checkpoint server records one.
    ///
    /// The callback becomes `TimedOut` with the message
    /// `Callback timed out`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackReply, LocalRunner};
    ///
    /// let runner = LocalRunner::new().callback(CallbackReply::timeout());
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn timeout() -> Self {
        Self {
            kind: CallbackReplyKind::Timeout,
        }
    }

    /// A heartbeat timeout, as the service records one.
    ///
    /// The callback becomes `TimedOut` with the error type
    /// `Callback.Heartbeat` and the message
    /// `Callback timed out on heartbeat`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackReply, LocalRunner};
    ///
    /// let runner = LocalRunner::new().callback(CallbackReply::heartbeat_timeout());
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn heartbeat_timeout() -> Self {
        Self {
            kind: CallbackReplyKind::HeartbeatTimeout,
        }
    }

    /// Writes this reply into a pending callback record, as the table in
    /// the `LocalRunner` callback docs states.
    fn deliver_to(self, op: &mut StoredOp) {
        op.result = None;
        op.error_type = None;
        op.error_message = None;
        op.error_data = None;
        op.stack_trace = Vec::new();
        match self.kind {
            CallbackReplyKind::Success(payload) => {
                op.status = OperationStatus::Succeeded;
                op.result = Some(payload);
            }
            CallbackReplyKind::Failure(failure) => {
                op.status = OperationStatus::Failed;
                op.error_type = failure.error_type;
                op.error_message = failure.error_message;
                op.error_data = failure.error_data;
                op.stack_trace = failure.stack_trace;
            }
            CallbackReplyKind::Timeout => {
                op.status = OperationStatus::TimedOut;
                op.error_message = Some(CALLBACK_TIMEOUT_MESSAGE.to_owned());
            }
            CallbackReplyKind::HeartbeatTimeout => {
                op.status = OperationStatus::TimedOut;
                op.error_type = Some(CALLBACK_HEARTBEAT_ERROR_TYPE.to_owned());
                op.error_message = Some(CALLBACK_HEARTBEAT_MESSAGE.to_owned());
            }
        }
        op.callback_delivered = true;
    }
}

/// The error an external system reports for a callback, as JS `ErrorObject`.
///
/// Each of the four fields is absent until it is set.
/// [`CallbackFailure::default`] has every field absent.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::test_util::CallbackFailure;
///
/// let failure = CallbackFailure::new("Declined", "no")
///     .with_error_data(r#"{"code":7}"#)
///     .with_stack_trace(vec!["at review".to_owned()]);
/// # drop(failure);
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CallbackFailure {
    error_type: Option<String>,
    error_message: Option<String>,
    error_data: Option<String>,
    stack_trace: Vec<String>,
}

impl CallbackFailure {
    /// A failure with this error type and message.
    #[must_use]
    pub fn new(error_type: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            error_type: Some(error_type.into()),
            error_message: Some(error_message.into()),
            ..Self::default()
        }
    }

    /// Sets the opaque error data.
    #[must_use]
    pub fn with_error_data(mut self, error_data: impl Into<String>) -> Self {
        self.error_data = Some(error_data.into());
        self
    }

    /// Sets the stack trace frames.
    #[must_use]
    pub fn with_stack_trace(mut self, frames: Vec<String>) -> Self {
        self.stack_trace = frames;
        self
    }
}

/// The callback replies a runner queues before a run.
#[derive(Debug, Clone, Default)]
struct QueuedReplies {
    /// Replies for the next pending callback, in queue order.
    fifo: std::collections::VecDeque<CallbackReply>,
    /// Replies for a named callback or `wait_for_callback`, in queue order.
    named: Vec<(String, CallbackReply)>,
}

/// A type-erased plain invoke target: it takes the invoke's input JSON and
/// returns the output JSON.
type PlainTarget = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>> + Send + Sync,
>;

/// A type-erased durable invoke target: it runs the handler as a nested
/// execution with the given runner settings and the invoke's input JSON.
type DurableTarget = Arc<
    dyn Fn(LocalRunner, String) -> Pin<Box<dyn Future<Output = NestedOutcome> + Send>>
        + Send
        + Sync,
>;

/// A target registered for `ctx.invoke` with
/// [`LocalRunner::register_function`] or
/// [`LocalRunner::register_durable_function`].
#[derive(Clone)]
enum RegisteredFunction {
    /// A plain function, run once inline.
    Plain(PlainTarget),
    /// A durable handler, run as a nested execution on a fresh backend.
    Durable(DurableTarget),
}

impl std::fmt::Debug for RegisteredFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plain(_) => f.write_str("RegisteredFunction::Plain"),
            Self::Durable(_) => f.write_str("RegisteredFunction::Durable"),
        }
    }
}

/// The outcome of an invoke target, as the caller's invoke record stores
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NestedOutcome {
    /// The target succeeded. The string is the raw result JSON; `None`
    /// records `"null"`.
    Succeeded(Option<String>),
    /// The target failed with this wire error: its type, message, error
    /// data and stack trace.
    Failed(crate::WireError),
    /// A nested durable execution suspended. The invoke stays `Started`.
    Suspended,
}

impl NestedOutcome {
    /// The outcome of a failed plain target, with the wire error of the
    /// production handler mapping.
    fn from_error(err: BoxError) -> Self {
        Self::Failed(crate::wire_error_from_box_error(err))
    }
}

/// Drives a durable handler to completion in memory, simulating the backend.
///
/// Construct with [`LocalRunner::new`], optionally pass the handler's
/// [`Options`](crate::Options) with [`options`](Self::options), queue
/// callback replies with [`callback`](Self::callback) and
/// [`callback_for`](Self::callback_for), and register the targets of
/// `ctx.invoke` with [`register_function`](Self::register_function) and
/// [`register_durable_function`](Self::register_durable_function), then
/// drive a handler with [`run`](Self::run).
///
/// # Callback replies
///
/// A pending callback is a callback record in `Pending` or `Started` that
/// has not received a reply. Between invocations the runner visits the
/// pending callbacks in first-checkpoint order. For each one it delivers
/// the first queued [`callback_for`](Self::callback_for) reply that names
/// it, else the front [`callback`](Self::callback) reply, else nothing. A
/// named reply names a callback when the callback has that name, or when
/// the callback's parent is a `wait_for_callback` with that name.
///
/// | Reply | Status | Stored fields |
/// | --- | --- | --- |
/// | [`CallbackReply::success`] | `Succeeded` | the result is the JSON of the value |
/// | [`CallbackReply::raw`] | `Succeeded` | the result is the payload |
/// | [`CallbackReply::failure`] | `Failed` | each field set in the [`CallbackFailure`] |
/// | [`CallbackReply::timeout`] | `TimedOut` | the message `Callback timed out` |
/// | [`CallbackReply::heartbeat_timeout`] | `TimedOut` | the type `Callback.Heartbeat` and the message `Callback timed out on heartbeat` |
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk as durable;
/// use durable::test_util::{CallbackReply, LocalRunner};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), durable::BoxError> {
/// // Deliver a value to the first callback the handler awaits.
/// let runner = LocalRunner::new().callback(CallbackReply::success(&true)?);
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
///
/// // Deliver a value to the `wait_for_callback` named "approve".
/// let runner = LocalRunner::new().callback_for("approve", CallbackReply::success(&7_u32)?);
/// let result = runner
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             let value = ctx
///                 .wait_for_callback::<u32, _, _>(|_step_ctx, _callback_id| async { Ok(()) })
///                 .name("approve")
///                 .await?;
///             Ok::<_, durable::BoxError>(value)
///         },
///         (),
///     )
///     .await;
///
/// assert_eq!(result.output(), Some(&7));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LocalRunner {
    max_invocations: usize,
    callback_replies: QueuedReplies,
    /// The targets of `ctx.invoke`, by exact function name (see
    /// [`register_function`](Self::register_function) and
    /// [`register_durable_function`](Self::register_durable_function)).
    functions: BTreeMap<String, RegisteredFunction>,
    /// Initial-state page size. When set, the synthesized invocation
    /// envelope carries at most this many history operations inline plus a
    /// pagination marker, so the context fetches the remainder via
    /// `get_state`: the paginating service is the DEFAULT
    /// ([`DEFAULT_STATE_PAGE_SIZE`]); [`LocalRunner::single_page`] opts
    /// into the single-page special case (`None`).
    initial_page_size: Option<usize>,
    /// Checkpoint-response page size. When set, the backend's checkpoint
    /// response includes `next_marker` once the total stored operations
    /// exceed this threshold, simulating a paginated checkpoint response
    /// (the default; see [`DEFAULT_STATE_PAGE_SIZE`]). This exercises the
    /// checkpoint pagination path in `DurableContext::checkpoint_updates`.
    checkpoint_page_size: Option<usize>,
    /// The [`Options`](crate::Options) each run passes to
    /// [`wrap`](crate::wrap), with the transport replaced by the in-memory
    /// backend. [`Options::default()`](crate::Options::default) unless set
    /// with [`options`](Self::options).
    options: crate::Options,
    /// Number of upcoming checkpoint calls the backend rejects with a
    /// non-retryable error (fault injection; see
    /// [`fail_next_checkpoints`](Self::fail_next_checkpoints)).
    checkpoint_failures: usize,
    /// Number of checkpoint calls to let through before the injected
    /// failures start (see
    /// [`fail_checkpoints_after`](Self::fail_checkpoints_after)).
    checkpoint_failure_skip: usize,
    /// Whether the injected checkpoint failures are retryable (invocation
    /// fault; the runner re-invokes) rather than non-retryable (terminal
    /// `FAIL` then execution failure).
    checkpoint_failures_retryable: bool,
    /// How long the backend waits before it applies and answers each
    /// checkpoint request (see
    /// [`checkpoint_latency`](Self::checkpoint_latency)). Zero by default.
    checkpoint_latency: std::time::Duration,
}

impl Default for LocalRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalRunner {
    /// Creates a runner with default settings and no queued callback
    /// replies.
    ///
    /// By default the simulated backend **paginates execution state**
    /// (both the initial invocation envelope and checkpoint responses
    /// split into 2+ pages once history is large enough), matching the
    /// real service. Use [`single_page`](Self::single_page) for the
    /// explicit single-page special case.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_invocations: DEFAULT_MAX_INVOCATIONS,
            callback_replies: QueuedReplies::default(),
            functions: BTreeMap::new(),
            initial_page_size: Some(DEFAULT_STATE_PAGE_SIZE),
            checkpoint_page_size: Some(DEFAULT_STATE_PAGE_SIZE),
            options: crate::Options::default(),
            checkpoint_failures: 0,
            checkpoint_failure_skip: 0,
            checkpoint_failures_retryable: false,
            checkpoint_latency: std::time::Duration::ZERO,
        }
    }

    /// Disables state pagination: the invocation envelope carries the full
    /// history inline and checkpoint responses never set a pagination
    /// marker.
    ///
    /// This is the **special case**: the real service paginates, and the
    /// runner does too by default. Reach for this only when a test
    /// specifically targets single-page behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().single_page();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn single_page(mut self) -> Self {
        self.initial_page_size = None;
        self.checkpoint_page_size = None;
        self
    }

    /// Sets the maximum number of invocations the runner will drive before
    /// returning a suspended result.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().max_invocations(10);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn max_invocations(mut self, max: usize) -> Self {
        self.max_invocations = max.max(1);
        self
    }

    /// Sets the initial-state page size for pagination testing.
    ///
    /// When set, the runner truncates the checkpoint log passed to each
    /// invocation to at most this many operations, simulating the service
    /// embedding only the first page in `InitialExecutionState`. The
    /// context then calls `get_state` to fetch the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().initial_page_size(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn initial_page_size(mut self, size: usize) -> Self {
        self.initial_page_size = Some(size.max(1));
        self
    }

    /// Sets the checkpoint-response page size for pagination testing.
    ///
    /// When set, the backend's checkpoint response includes a pagination
    /// marker once total stored operations exceed this threshold, forcing
    /// the context to paginate via `get_state`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().checkpoint_page_size(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn checkpoint_page_size(mut self, size: usize) -> Self {
        self.checkpoint_page_size = Some(size.max(1));
        self
    }

    /// Sets the [`Options`](crate::Options) every run passes to
    /// [`wrap`](crate::wrap).
    ///
    /// Every field applies as in production, except the transport.
    /// `sdk_config`, `lambda_client` and `execution_client` are ignored: the
    /// runner always uses its in-memory backend. So a test can pass the same
    /// `Options` its production handler uses.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::Options;
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// // Send each checkpoint in its own request.
    /// let runner = LocalRunner::new().options(
    ///     Options::builder()
    ///         .checkpoint_batching(false)
    ///         .build()
    ///         .expect("valid options"),
    /// );
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn options(mut self, options: crate::Options) -> Self {
        self.options = options;
        self
    }

    /// Makes the simulated backend reject the next `count` checkpoint
    /// calls with a non-retryable error, persisting nothing for them,
    /// exactly like a permanent service-side rejection.
    ///
    /// Under the #43 model a non-retryable checkpoint failure never
    /// reaches the handler: the SDK persists a small terminal `FAIL` for
    /// the operation (when user code already ran) and fails the
    /// execution. Use this to test that model, and to assert no
    /// record-transition lifecycle event is emitted for the rejected
    /// write (see [`crate::observability`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().fail_next_checkpoints(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn fail_next_checkpoints(mut self, count: usize) -> Self {
        self.checkpoint_failures = count;
        self
    }

    /// Like [`fail_next_checkpoints`](Self::fail_next_checkpoints), but
    /// lets the first `skip` checkpoint calls through before rejecting
    /// the following `count` non-retryably.
    ///
    /// Use it to reject a specific write: a step's live path writes START
    /// first and its outcome second, so `fail_checkpoints_after(1, 1)`
    /// rejects exactly the outcome write while the START (and everything
    /// after the failure window, such as the SDK's terminal `FAIL`
    /// record) persists.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().fail_checkpoints_after(1, 1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn fail_checkpoints_after(mut self, skip: usize, count: usize) -> Self {
        self.checkpoint_failure_skip = skip;
        self.checkpoint_failures = count;
        self.checkpoint_failures_retryable = false;
        self
    }

    /// Like [`fail_checkpoints_after`](Self::fail_checkpoints_after), but
    /// the injected failures are RETRYABLE: simulating a transient
    /// failure that exhausted the transport's own retries.
    ///
    /// Under the #43 model a retryable checkpoint failure fails the
    /// invocation with no further writes; the runner then re-invokes,
    /// exactly as the durable service does, and replay resumes from the
    /// recorded state.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().fail_checkpoints_after_retryable(1, 1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn fail_checkpoints_after_retryable(mut self, skip: usize, count: usize) -> Self {
        self.checkpoint_failure_skip = skip;
        self.checkpoint_failures = count;
        self.checkpoint_failures_retryable = true;
        self
    }

    /// Delays each checkpoint response, as JS `checkpointDelay`, to expose races.
    /// Not the SDK's coalescing window.
    ///
    /// The backend waits `latency` in tokio time before it applies and
    /// answers each checkpoint request. The default is zero, which answers
    /// at once. The latency does not move the backend's virtual clock, so
    /// waits and retry delays end at the same virtual times as without it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().checkpoint_latency(Duration::from_millis(50));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn checkpoint_latency(mut self, latency: std::time::Duration) -> Self {
        self.checkpoint_latency = latency;
        self
    }

    /// Queues `reply` for the next pending callback, in order.
    ///
    /// A pending callback that a [`callback_for`](Self::callback_for)
    /// reply names takes that reply instead. See
    /// [Callback replies](Self#callback-replies).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackReply, LocalRunner};
    ///
    /// let runner = LocalRunner::new()
    ///     .callback(CallbackReply::raw(r#""first""#))
    ///     .callback(CallbackReply::timeout());
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn callback(mut self, reply: CallbackReply) -> Self {
        self.callback_replies.fifo.push_back(reply);
        self
    }

    /// Queues `reply` for the pending callback named `name`, or for the
    /// callback of the `wait_for_callback` named `name`.
    ///
    /// When no pending callback matches by the end of the run, the
    /// suspended result's message names `name`. See
    /// [Callback replies](Self#callback-replies).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CallbackReply, LocalRunner};
    ///
    /// let runner = LocalRunner::new()
    ///     .callback_for("cb-a", CallbackReply::raw(r#""A""#))
    ///     .callback_for("cb-b", CallbackReply::raw(r#""B""#));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn callback_for(mut self, name: impl Into<String>, reply: CallbackReply) -> Self {
        self.callback_replies.named.push((name.into(), reply));
        self
    }

    /// Registers `function` as the plain target of `ctx.invoke(name, ..)`.
    ///
    /// The name matches exactly: no qualifier, alias or ARN handling. A
    /// later registration of either kind under the same name replaces this
    /// one.
    ///
    /// When an invocation suspends, the runner runs every pending invoke
    /// whose target is registered, once, in the order the invokes were
    /// first checkpointed, before the virtual clock moves. A plain target
    /// decodes its input from the invoke's payload, runs, and encodes its
    /// output:
    ///
    /// - `Ok(value)` records the invoke `Succeeded` with the JSON of
    ///   `value`. `Ok(())` records `null`.
    /// - `Err(err)` records the invoke `Failed` with the error fields a
    ///   failing handler records: a [`TypedError`](crate::TypedError) keeps
    ///   its type and error data, and any other error records the type
    ///   `Error`.
    /// - An input that does not decode, or an output that does not encode,
    ///   records the invoke `Failed` with the type `Error` and a message
    ///   that starts `deserialize input:` or `serialize output:`.
    ///
    /// The target runs once per invoke, whatever its outcome. A replay of
    /// the invoke reads the recorded outcome.
    ///
    /// An invoke of a name that no registration matches fails the run: the
    /// backend rejects the invoke's checkpoint, and the result's
    /// [`error_message`](TestResult::error_message) names the function.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk as durable;
    /// use durable::test_util::LocalRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = LocalRunner::new()
    ///     .register_function("double", |x: i32| async move { Ok(x * 2) })
    ///     .run(
    ///         |n: i32, ctx: durable::DurableContext| async move {
    ///             let doubled = ctx.invoke::<i32, _>("double", n).await?;
    ///             Ok::<_, durable::BoxError>(doubled)
    ///         },
    ///         21_i32,
    ///     )
    ///     .await;
    ///
    /// assert_eq!(result.output(), Some(&42));
    /// # }
    /// ```
    #[must_use]
    pub fn register_function<I, O, F, Fut>(mut self, name: impl Into<String>, function: F) -> Self
    where
        I: DeserializeOwned + Send + 'static,
        O: Serialize + Send + 'static,
        F: Fn(I) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    {
        let function = Arc::new(function);
        let target: PlainTarget = Arc::new(move |input: String| {
            let function = Arc::clone(&function);
            Box::pin(async move {
                let input: I = serde_json::from_str(&input)
                    .map_err(|e| -> BoxError { format!("deserialize input: {e}").into() })?;
                let output = function(input).await?;
                serde_json::to_string(&output)
                    .map_err(|e| -> BoxError { format!("serialize output: {e}").into() })
            })
        });
        self.functions
            .insert(name.into(), RegisteredFunction::Plain(target));
        self
    }

    /// Registers `handler` as the durable target of `ctx.invoke(name, ..)`.
    ///
    /// Name matching, replacement, and the time a target runs follow
    /// [`register_function`](Self::register_function). The handler runs as
    /// its own durable execution, driven to its end before the caller's
    /// next invocation:
    ///
    /// - The target runs once per invoke, whatever its outcome. A replay of
    ///   the invoke reads the recorded outcome.
    /// - The nested execution has a fresh backend and its own virtual
    ///   clock. It uses a copy of this runner's settings taken when the
    ///   caller's run starts: the [`Options`](crate::Options), the page
    ///   sizes, [`max_invocations`](Self::max_invocations) and
    ///   [`checkpoint_latency`](Self::checkpoint_latency). It has no
    ///   registered functions, no queued callback replies and no injected
    ///   checkpoint failures.
    /// - A nested execution that succeeds records the invoke `Succeeded`
    ///   with the handler's result, or `null` when the envelope carries
    ///   none. One that fails records the invoke `Failed` with the
    ///   handler's wire error: its type, message, error data and stack
    ///   trace. One that suspends, for example on a callback no reply
    ///   answers, leaves the invoke `Started`.
    /// - When the caller's run then ends suspended, because nothing can
    ///   advance, and at least one chained invoke is still `Started`, the
    ///   result's [`error_message`](TestResult::error_message) ends with
    ///   `; chained invokes still pending for: ` and the function names.
    ///   Each name is in double quotes, the names are separated by `, `,
    ///   they appear in the order the invokes were stored, and each name
    ///   appears once. This part comes after the part that names
    ///   unmatched queued callback replies. For example:
    ///   `; chained invokes still pending for: "approval", "audit"`.
    /// - The nested execution's invocations do not count in the caller's
    ///   [`invocation_count`](TestResult::invocation_count), and its
    ///   operations do not appear in the caller's
    ///   [`operations`](TestResult::operations).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use aws_durable_execution_sdk as durable;
    /// use durable::test_util::LocalRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = LocalRunner::new()
    ///     .register_durable_function("workflow", |n: u32, ctx: durable::DurableContext| async move {
    ///         let next = ctx.step(move |_| async move { Ok(n + 1) }).await?;
    ///         ctx.wait(Duration::from_secs(5)).await?;
    ///         Ok::<_, durable::BoxError>(next * 2)
    ///     })
    ///     .run(
    ///         |n: u32, ctx: durable::DurableContext| async move {
    ///             let out = ctx.invoke::<u32, _>("workflow", n).await?;
    ///             Ok::<_, durable::BoxError>(out)
    ///         },
    ///         1_u32,
    ///     )
    ///     .await;
    ///
    /// assert_eq!(result.output(), Some(&4));
    /// # }
    /// ```
    #[must_use]
    pub fn register_durable_function<E, O, F, Fut>(
        mut self,
        name: impl Into<String>,
        handler: F,
    ) -> Self
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let handler = Arc::new(handler);
        let target: DurableTarget = Arc::new(move |runner: LocalRunner, input: String| {
            let handler = Arc::clone(&handler);
            Box::pin(async move {
                let backend = Arc::new(runner.new_backend());
                let nested = move |event: E, ctx: DurableContext| handler(event, ctx);
                runner.drive(backend, nested, input).await.nested_outcome()
            })
        });
        self.functions
            .insert(name.into(), RegisteredFunction::Durable(target));
        self
    }

    /// The settings of a nested durable run: a copy of this runner with no
    /// registered functions, no queued callback replies and the fault
    /// injection of [`LocalRunner::new`].
    fn nested_settings(&self) -> Self {
        let defaults = Self::new();
        let mut nested = self.clone();
        nested.functions = defaults.functions;
        nested.callback_replies = defaults.callback_replies;
        nested.checkpoint_failures = defaults.checkpoint_failures;
        nested.checkpoint_failure_skip = defaults.checkpoint_failure_skip;
        nested.checkpoint_failures_retryable = defaults.checkpoint_failures_retryable;
        nested
    }

    /// Drives `handler` to a terminal outcome, feeding it `event`.
    ///
    /// Each simulated invocation goes through the **production service
    /// function** ([`wrap`](crate::wrap)'s body) fed a synthesized
    /// invocation envelope, so envelope parsing, bootstrap pagination, the
    /// suspension driver, and wire error mapping are the exact code
    /// production runs. The service future is awaited inline, the same
    /// task topology `lambda_runtime` produces under `block_on`, so
    /// task-ownership behavior matches deployment.
    ///
    /// The event is serialized once and embedded in each invocation's
    /// envelope, mirroring the way the service re-delivers the input
    /// payload on every re-invocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk as durable;
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
    pub async fn run<E, O, F, Fut>(&self, handler: F, event: E) -> TestResult<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        // Serialize the event once; the envelope re-delivers it per
        // invocation.
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
        self.run_on_backend(Arc::new(self.new_backend()), handler, event_json)
            .await
    }

    /// A fresh backend configured from this runner's settings: queued
    /// callback replies, checkpoint page size, registered function names,
    /// injected checkpoint failures and checkpoint latency.
    fn new_backend(&self) -> Backend {
        let mut backend = Backend::new(self.callback_replies.clone(), self.checkpoint_page_size);
        backend.registered_functions = self.functions.keys().cloned().collect();
        backend.checkpoint_failures_remaining.store(
            self.checkpoint_failures,
            std::sync::atomic::Ordering::SeqCst,
        );
        backend.checkpoint_failures_skip.store(
            self.checkpoint_failure_skip,
            std::sync::atomic::Ordering::SeqCst,
        );
        backend.checkpoint_failures_retryable.store(
            self.checkpoint_failures_retryable,
            std::sync::atomic::Ordering::SeqCst,
        );
        backend.checkpoint_latency = self.checkpoint_latency;
        backend
    }

    /// The invoke → parse → advance loop over an externally supplied
    /// backend, with the event as JSON. Internal seam: unit tests inject a
    /// backend they retain a handle to, so they can assert on
    /// transport-level facts (e.g. how many `get_state` fetches pagination
    /// forced).
    async fn run_on_backend<E, O, F, Fut>(
        &self,
        backend: Arc<Backend>,
        handler: F,
        event_json: String,
    ) -> TestResult<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        self.drive(backend, handler, event_json)
            .await
            .into_test_result()
    }

    /// Drives one execution on `backend` and returns its end with the
    /// handler output still raw JSON. [`run_on_backend`](Self::run_on_backend)
    /// decodes the output; a nested durable run keeps it raw.
    async fn drive<E, O, F, Fut>(
        &self,
        backend: Arc<Backend>,
        handler: F,
        event_json: String,
    ) -> RawRun
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let client: Arc<dyn ExecutionClient> = Arc::clone(&backend) as Arc<dyn ExecutionClient>;

        // The handler runs behind the SAME service function production
        // registers with the Lambda runtime, with the caller's `Options`:
        // only the transport is faked. The three transport fields are set
        // directly, so `build()`'s check that rejects `execution_client`
        // together with `lambda_client` does not apply.
        let mut options = self.options.clone();
        options.sdk_config = None;
        options.lambda_client = None;
        options.execution_client = Some(client);
        let service = crate::wrap(handler, options);

        // A nested durable run uses the settings this run starts with.
        let nested = self.nested_settings();

        let mut invocations = 0_usize;
        // The most recent invocation fault (Lambda runtime error), kept so
        // an execution that never terminates because every invocation
        // faults reports the fault rather than a generic budget message.
        let mut last_invocation_fault: Option<String> = None;

        loop {
            invocations += 1;
            if invocations > self.max_invocations {
                return RawRun {
                    end: RawEnd::Suspended {
                        error_type: last_invocation_fault
                            .is_some()
                            .then(|| "RuntimeError".to_owned()),
                        message: last_invocation_fault.map_or_else(
                            || {
                                format!(
                                    "execution did not terminate within {} invocations",
                                    self.max_invocations
                                )
                            },
                            |fault| {
                                format!(
                                    "execution did not terminate within {} invocations; \
                                     last invocation fault: {fault}",
                                    self.max_invocations
                                )
                            },
                        ),
                    },
                    operations: backend.snapshot_operations(),
                    invocations: invocations - 1,
                };
            }

            let payload = build_envelope(&backend, &event_json, self.initial_page_size);
            let lambda_event = lambda_runtime::LambdaEvent::new(
                crate::InvocationPayload(payload),
                lambda_runtime::Context::default(),
            );

            // Await the service future INLINE, never on a spawned task.
            // This reproduces the production topology (`lambda_runtime`
            // awaits the handler under `block_on`), so
            // `tokio::task::try_id()` is `None` at context-creation time
            // and the task-ownership guard behaves as it does deployed.
            let response = match service(lambda_event).await {
                Ok(envelope) => envelope.0,
                Err(e) => {
                    // The invocation itself failed (a Lambda runtime
                    // error), e.g. a retryable checkpoint failure that
                    // exhausted transport retries (issue #43), or a
                    // bootstrap failure. The durable service re-invokes
                    // on an invocation fault, so the runner does too;
                    // `max_invocations` bounds a persistent fault.
                    last_invocation_fault = Some(e.to_string());
                    continue;
                }
            };

            let end = match response.get("Status").and_then(serde_json::Value::as_str) {
                Some("SUCCEEDED") => RawEnd::Succeeded(
                    response
                        .get("Result")
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned),
                ),
                Some("FAILED") => RawEnd::Failed(envelope_wire_error(&response)),
                Some("PENDING") => {
                    // Pending invokes run before the clock moves. When one
                    // of them changed its record, the next invocation
                    // starts without an advance.
                    if self.run_pending_invokes(&backend, &nested).await {
                        continue;
                    }
                    // Advance the simulated backend (timers, retries,
                    // callbacks). If nothing can advance, the execution is
                    // genuinely stuck.
                    if backend.advance() {
                        continue;
                    }
                    RawEnd::Suspended {
                        error_type: None,
                        message: suspended_message(
                            &backend.unmatched_reply_names(),
                            &backend.pending_invoke_names(),
                        ),
                    }
                }
                other => RawEnd::Failed(crate::WireError::new(
                    Some("RuntimeError"),
                    Some(format!("unexpected response envelope status: {other:?}")),
                )),
            };
            return RawRun {
                end,
                operations: backend.snapshot_operations(),
                invocations,
            };
        }
    }

    /// Runs each pending invoke with a registered target once, inline, in
    /// stored order, and records its outcome. Returns whether at least one
    /// record changed.
    ///
    /// The backend lock is not held while a target runs: the pending
    /// invokes are collected first, then each target is awaited.
    async fn run_pending_invokes(&self, backend: &Backend, nested: &Self) -> bool {
        let mut changed = false;
        for (id, target, input) in backend.take_pending_invokes(&self.functions) {
            let outcome = match target {
                RegisteredFunction::Plain(function) => match function(input).await {
                    Ok(json) => NestedOutcome::Succeeded(Some(json)),
                    Err(err) => NestedOutcome::from_error(err),
                },
                RegisteredFunction::Durable(handler) => handler(nested.clone(), input).await,
            };
            changed |= backend.record_invoke_outcome(&id, outcome);
        }
        changed
    }
}

/// How one driven execution ended, with the handler output still raw JSON.
#[derive(Debug)]
struct RawRun {
    end: RawEnd,
    operations: Vec<TestOperation>,
    invocations: usize,
}

/// The end of a driven execution (see [`RawRun`]).
#[derive(Debug)]
enum RawEnd {
    /// A `SUCCEEDED` envelope, with its raw `Result` string.
    Succeeded(Option<String>),
    /// A `FAILED` envelope, with the wire error of its `Error` object. An
    /// envelope with an unexpected status ends here too, with the type
    /// `RuntimeError`.
    Failed(crate::WireError),
    /// The execution did not terminate.
    Suspended {
        error_type: Option<String>,
        message: String,
    },
}

impl RawRun {
    /// The [`TestResult`] of a top-level run: the output decoded into `O`.
    fn into_test_result<O: DeserializeOwned>(self) -> TestResult<O> {
        let Self {
            end,
            operations,
            invocations,
        } = self;
        let (disposition, output, error_type, error_message) = match end {
            RawEnd::Succeeded(raw) => {
                let output = serde_json::from_str::<O>(raw.as_deref().unwrap_or("null")).ok();
                if output.is_some() {
                    (Disposition::Succeeded, output, None, None)
                } else {
                    (
                        Disposition::Failed,
                        None,
                        Some("SerializationFailed".to_owned()),
                        Some("could not deserialize handler output".to_owned()),
                    )
                }
            }
            RawEnd::Failed(wire) => (
                Disposition::Failed,
                None,
                Some(wire.error_type().unwrap_or("Error").to_owned()),
                Some(wire.error_message().unwrap_or_default().to_owned()),
            ),
            RawEnd::Suspended {
                error_type,
                message,
            } => (Disposition::Suspended, None, error_type, Some(message)),
        };
        TestResult {
            disposition,
            output,
            error_type,
            error_message,
            operations,
            invocations,
        }
    }

    /// The outcome a nested durable run records on the caller's invoke.
    fn nested_outcome(self) -> NestedOutcome {
        match self.end {
            RawEnd::Succeeded(raw) => NestedOutcome::Succeeded(raw),
            RawEnd::Failed(wire) => NestedOutcome::Failed(wire),
            RawEnd::Suspended { .. } => NestedOutcome::Suspended,
        }
    }
}

/// The wire error of a `FAILED` envelope's `Error` object: its
/// `ErrorType`, `ErrorMessage`, `ErrorData` and `StackTrace`.
fn envelope_wire_error(response: &serde_json::Value) -> crate::WireError {
    let error = response.get("Error");
    let field = |key: &str| {
        error
            .and_then(|e| e.get(key))
            .and_then(serde_json::Value::as_str)
    };
    crate::WireError::new(field("ErrorType"), field("ErrorMessage"))
        .with_error_data(field("ErrorData"))
        .with_stack_trace(error.and_then(crate::parse_stack_trace).unwrap_or_default())
}

/// The execution ARN the local runner stamps on every synthesized envelope.
const LOCAL_EXECUTION_ARN: &str = "arn:aws:lambda:us-west-2:000000000000:function:local-test";

/// Synthesizes the durable invocation envelope for one simulated
/// invocation, in the exact wire shape the service delivers and
/// [`wrap`](crate::wrap) parses: `DurableExecutionArn`, `CheckpointToken`,
/// and `InitialExecutionState.Operations` with the customer input embedded
/// in the leading `Execution` operation's `ExecutionDetails.InputPayload`.
/// `UpdatedOperationIds` lists the records the backend changed between
/// invocations, and building the envelope clears that list (see
/// [`Backend::take_updated_ids`]).
///
/// When `initial_page_size` is set and the recorded history exceeds it,
/// only the first page of history rides inline and
/// `InitialExecutionState.NextMarker` signals the truncation: the
/// production bootstrap path then fetches the remainder via `get_state`,
/// exactly as it does against the paginating service.
fn build_envelope(
    backend: &Backend,
    event_json: &str,
    initial_page_size: Option<usize>,
) -> serde_json::Value {
    let history = backend.envelope_history();
    let (page, next_marker) = match initial_page_size {
        Some(size) if history.len() > size => (
            history.get(..size).unwrap_or(&history).to_vec(),
            Some(format!("initial-marker-{}", history.len())),
        ),
        _ => (history, None),
    };

    let mut operations = Vec::with_capacity(page.len() + 1);
    operations.push(serde_json::json!({
        "Id": "execution",
        "Type": "EXECUTION",
        "Status": "STARTED",
        "ExecutionDetails": { "InputPayload": event_json }
    }));
    operations.extend(page);

    let initial_state = match next_marker {
        Some(marker) => serde_json::json!({
            "Operations": operations,
            "NextMarker": marker,
        }),
        None => serde_json::json!({ "Operations": operations }),
    };

    serde_json::json!({
        "DurableExecutionArn": LOCAL_EXECUTION_ARN,
        "CheckpointToken": backend.current_token(),
        "InitialExecutionState": initial_state,
        "UpdatedOperationIds": backend.take_updated_ids(),
    })
}

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
/// use aws_durable_execution_sdk as durable;
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
#[non_exhaustive]
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
    /// use aws_durable_execution_sdk::test_util::CloudRunner;
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
    /// use aws_durable_execution_sdk::test_util::CloudRunner;
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
    /// use aws_durable_execution_sdk::test_util::CloudRunner;
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
    /// use aws_durable_execution_sdk::test_util::CloudRunner;
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
    /// use aws_durable_execution_sdk::test_util::CloudRunner;
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
    /// use aws_durable_execution_sdk::test_util::CloudRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = CloudRunner::new("echo-fn")
    ///     .run::<_, String>("hello".to_owned())
    ///     .await;
    /// assert!(result.is_success() || result.is_failure() || result.is_suspended());
    /// # }
    /// ```
    #[expect(clippy::too_many_lines)] // reason: the invoke → poll → history → fold flow reads better as one sequence
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

        // 1. Invoke (async): start the durable execution and capture its ARN.
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

/// Counts `InvocationCompleted` events: the number of times the service
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
    /// Opaque error payload recorded with a failure, round-tripped
    /// verbatim like the production backend does.
    error_data: Option<String>,
    /// Recorded stack trace frames, round-tripped verbatim.
    stack_trace: Vec<String>,
    attempt: i32,
    callback_id: Option<String>,
    wait_seconds: Option<i32>,
    /// The virtual time, in seconds, at which a started wait ends or a
    /// pending retry may run again. Set by a WAIT `START` and a STEP
    /// `RETRY`. Firing the operation keeps the value.
    due_at: Option<u64>,
    callback_delivered: bool,
    /// The `ParentId` of the update that created the record. Later updates
    /// do not change it.
    parent_id: Option<String>,
    /// The `ReplayChildren` option of the update that closed a context.
    /// A close without the option records `false`.
    replay_children: bool,
    /// The function name of a chained invoke, from its `START`.
    invoke_function: Option<String>,
    /// The input payload of a chained invoke, from its `START`.
    invoke_input: Option<String>,
    /// Whether the runner already ran the invoke's target. A target runs
    /// once per invoke, whatever its outcome.
    invoke_dispatched: bool,
}

/// A fresh `Started` record for an operation seen for the first time; the
/// caller merges name/sub-type metadata and the action's outcome fields.
fn new_started_op(
    id: String,
    op_type: OperationType,
    sub_type: Option<String>,
    parent_id: Option<String>,
) -> StoredOp {
    StoredOp {
        id,
        op_type,
        sub_type,
        status: OperationStatus::Started,
        name: None,
        result: None,
        error_type: None,
        error_message: None,
        error_data: None,
        stack_trace: Vec::new(),
        attempt: 0,
        callback_id: None,
        wait_seconds: None,
        due_at: None,
        callback_delivered: false,
        parent_id,
        replay_children: false,
        invoke_function: None,
        invoke_input: None,
        invoke_dispatched: false,
    }
}

/// Decrements `counter` by one if it is above zero, and returns whether it
/// did.
///
/// `AtomicUsize::fetch_update` does this in one call. Rust 1.99 deprecates
/// it, and CI denies warnings. Its replacement, `try_update`, is unstable
/// on the 1.94.1 MSRV. So this function uses a `compare_exchange` loop,
/// which compiles without warnings on both toolchains.
fn atomic_decrement_if_positive(counter: &std::sync::atomic::AtomicUsize) -> bool {
    let mut current = counter.load(std::sync::atomic::Ordering::SeqCst);
    while current > 0 {
        match counter.compare_exchange(
            current,
            current - 1,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        ) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
    false
}

/// Mutable state of the in-memory backend, guarded by a single mutex.
#[derive(Debug)]
struct BackendState {
    /// Operations in first-checkpoint order.
    ops: Vec<StoredOp>,
    token_counter: u64,
    callback_counter: u64,
    /// Queued external callback replies (see [`Backend::advance`]).
    callback_replies: QueuedReplies,
    /// The virtual time in seconds. It starts at 0 and moves only in
    /// [`Backend::advance`], between invocations.
    now: u64,
    /// The IDs of the records [`Backend::advance`] changed since the last
    /// envelope, in change order. The next envelope sends them as
    /// `UpdatedOperationIds` and clears them (see
    /// [`Backend::take_updated_ids`]). A checkpoint call never adds to it.
    updated_ids: Vec<String>,
}

/// The in-memory execution client. Shared (via `Arc`) between the runner and
/// the durable context. Debug-printable and `Send + Sync` as the
/// `ExecutionClient` trait requires.
#[derive(Debug)]
struct Backend {
    state: Mutex<BackendState>,
    /// When set, the checkpoint response will include `next_marker` when
    /// the number of updated operations exceeds this value, simulating a
    /// paginated checkpoint response.
    checkpoint_page_size: Option<usize>,
    /// Counts `get_state` calls: lets tests assert that pagination
    /// actually forced a state fetch (or, under `single_page`, that none
    /// occurred).
    get_state_calls: std::sync::atomic::AtomicUsize,
    /// Counts `checkpoint` calls: lets tests assert that checkpoint
    /// coalescing (`checkpoint_delay`) actually reduced the number of
    /// writes.
    checkpoint_calls: std::sync::atomic::AtomicUsize,
    /// Number of upcoming `checkpoint` calls to reject, before touching
    /// any state (fault injection; see
    /// [`LocalRunner::fail_next_checkpoints`] and
    /// [`LocalRunner::fail_checkpoints_after`]).
    checkpoint_failures_remaining: std::sync::atomic::AtomicUsize,
    /// Number of `checkpoint` calls to let through before the injected
    /// failures begin.
    checkpoint_failures_skip: std::sync::atomic::AtomicUsize,
    /// Whether injected checkpoint failures are retryable (see
    /// [`LocalRunner::fail_checkpoints_after_retryable`]).
    checkpoint_failures_retryable: std::sync::atomic::AtomicBool,
    /// How long each `checkpoint` call waits, in tokio time, before it
    /// applies and answers the request (see
    /// [`LocalRunner::checkpoint_latency`]).
    checkpoint_latency: std::time::Duration,
    /// The names registered on the runner as invoke targets. A chained
    /// invoke `START` for any other name is rejected.
    registered_functions: BTreeSet<String>,
    /// Per-call checkpoint plan (in-crate test seam): while non-empty,
    /// each `checkpoint` call pops and obeys the front entry instead of
    /// the counter-based injection (see
    /// [`Backend::plan_checkpoint_calls`]).
    #[cfg(test)]
    checkpoint_plan: Mutex<std::collections::VecDeque<PlannedCheckpoint>>,
}

/// One planned behavior for an upcoming [`Backend`] `checkpoint` call
/// (in-crate test seam; see [`Backend::plan_checkpoint_calls`]).
#[cfg(test)]
#[derive(Debug)]
enum PlannedCheckpoint {
    /// The call proceeds normally against the simulated store.
    Pass,
    /// The call is rejected with a non-retryable error, persisting
    /// nothing. When `gate` is set, the rejection is held in flight until
    /// the test adds a permit to the semaphore, so the test can order
    /// events (e.g. dropping a contributor) against the in-flight write
    /// deterministically.
    FailNonRetryable {
        gate: Option<Arc<tokio::sync::Semaphore>>,
    },
}

impl Backend {
    fn new(callback_replies: QueuedReplies, checkpoint_page_size: Option<usize>) -> Self {
        Self {
            state: Mutex::new(BackendState {
                ops: Vec::new(),
                token_counter: 0,
                callback_counter: 0,
                callback_replies,
                now: 0,
                updated_ids: Vec::new(),
            }),
            checkpoint_page_size,
            get_state_calls: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_calls: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_remaining: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_skip: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_retryable: std::sync::atomic::AtomicBool::new(false),
            checkpoint_latency: std::time::Duration::ZERO,
            registered_functions: BTreeSet::new(),
            #[cfg(test)]
            checkpoint_plan: Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Queues a per-call plan for upcoming `checkpoint` calls (in-crate
    /// test seam). While entries remain, each call pops and obeys the
    /// front entry, pass, fail retryably, or fail non-retryably (with an
    /// optional in-flight gate), instead of the counter-based injection.
    /// Once the plan is exhausted, calls fall back to the counters.
    #[cfg(test)]
    fn plan_checkpoint_calls(&self, plan: Vec<PlannedCheckpoint>) {
        self.checkpoint_plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(plan);
    }

    /// Pops the next planned checkpoint behavior, if a plan is queued.
    #[cfg(test)]
    fn next_planned_checkpoint(&self) -> Option<PlannedCheckpoint> {
        self.checkpoint_plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
    }

    /// The number of `get_state` calls the SDK has made against this
    /// backend.
    #[cfg(test)]
    fn get_state_call_count(&self) -> usize {
        self.get_state_calls
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The number of `checkpoint` calls the SDK has made against this
    /// backend.
    #[cfg(test)]
    fn checkpoint_call_count(&self) -> usize {
        self.checkpoint_calls
            .load(std::sync::atomic::Ordering::SeqCst)
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

    /// Returns the IDs [`Backend::advance`] changed since the last call, in
    /// change order, and leaves the list empty.
    fn take_updated_ids(&self) -> Vec<String> {
        std::mem::take(&mut self.lock().updated_ids)
    }

    /// Builds `Operation` records for the current state (used to seed the
    /// per-invocation checkpoint log).
    fn build_operations(&self) -> Vec<Operation> {
        let state = self.lock();
        state.ops.iter().filter_map(build_operation).collect()
    }

    /// Renders the recorded history as envelope-wire JSON operation
    /// objects: the exact shape `parse_inline_operations` reads from
    /// `InitialExecutionState.Operations`.
    ///
    /// The history omits the records the service omits from the invocation
    /// input (see [`prune_replay_children`]).
    fn envelope_history(&self) -> Vec<serde_json::Value> {
        let state = self.lock();
        prune_replay_children(&state.ops)
            .into_iter()
            .map(stored_op_envelope_json)
            .collect()
    }

    /// Builds the `Operation` records `GetDurableExecutionState` returns:
    /// the current state without the records the service omits (see
    /// [`prune_replay_children`]).
    fn build_pruned_operations(&self) -> Vec<Operation> {
        let state = self.lock();
        prune_replay_children(&state.ops)
            .into_iter()
            .filter_map(build_operation)
            .collect()
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

    /// Advances the simulated backend between invocations. Returns `true`
    /// if any operation was advanced.
    ///
    /// The steps run in this order:
    ///
    /// 1. Queued callback replies go to pending callbacks (see
    ///    [`take_callback_deliveries`]). If at least one was delivered, the
    ///    advance ends here: the clock does not move and no timer fires. A
    ///    callback reply the service already holds arrives before any later
    ///    timer.
    /// 2. Otherwise the clock moves to the earliest end among timed
    ///    operations (see [`timed_due_at`]). With no timed operation, the
    ///    advance returns `false`.
    /// 3. Every timed operation due by the new time fires: a wait becomes
    ///    `Succeeded` and a pending retry becomes `Ready`, so the next
    ///    invocation re-executes the body.
    ///
    /// The ID of each record whose status changes is appended to
    /// `BackendState::updated_ids`. An advance changes each record at most
    /// once, so the list has no duplicates.
    ///
    /// Backend-behaviour anchors (this crate's own replay logic):
    /// - `wait`: `src/wait.rs` replay returns at once for `Succeeded` and
    ///   suspends for `Started`.
    /// - `step`/`wait_for_condition` retry: `src/step.rs` and
    ///   `src/wait_for_condition.rs` replay suspend for `Pending` and fall
    ///   through to live execution for `Ready`.
    /// - callback: a `Started` callback resolves to `Succeeded` (with a
    ///   payload), `Failed` or `TimedOut` (`src/callback.rs` replay).
    fn advance(&self) -> bool {
        let mut state = self.lock();
        // Split the borrow so the reply queues and the records can be
        // changed in the same pass.
        let BackendState {
            ops,
            callback_replies,
            now,
            updated_ids,
            ..
        } = &mut *state;

        // A named reply can match through the parent record, so the
        // matching pass reads `ops` while the delivery pass writes it.
        let deliveries = take_callback_deliveries(ops, callback_replies);
        if deliveries.iter().any(Option::is_some) {
            // `deliveries` holds one entry per record, in record order, so
            // zipping pairs each reply with the record it matched.
            for (op, reply) in ops.iter_mut().zip(deliveries) {
                if let Some(reply) = reply {
                    reply.deliver_to(op);
                    updated_ids.push(op.id.clone());
                }
            }
            return true;
        }

        let Some(next) = ops.iter().filter_map(timed_due_at).min() else {
            return false;
        };
        *now = next;
        for op in ops.iter_mut() {
            if timed_due_at(op).is_some_and(|due| due <= next) {
                op.status = if op.op_type == OperationType::Wait {
                    OperationStatus::Succeeded
                } else {
                    OperationStatus::Ready
                };
                updated_ids.push(op.id.clone());
            }
        }
        true
    }

    /// The names of the queued named replies that no callback took, in
    /// queue order, each name once.
    fn unmatched_reply_names(&self) -> Vec<String> {
        let state = self.lock();
        let mut names: Vec<String> = Vec::new();
        for (name, _) in &state.callback_replies.named {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        names
    }

    /// Marks every `Started` chained invoke whose target is registered and
    /// has not run as run, and returns its ID, target and input (the empty
    /// string when absent), in stored order.
    fn take_pending_invokes(
        &self,
        functions: &BTreeMap<String, RegisteredFunction>,
    ) -> Vec<(String, RegisteredFunction, String)> {
        let mut state = self.lock();
        state
            .ops
            .iter_mut()
            .filter(|op| {
                op.op_type == OperationType::ChainedInvoke
                    && op.status == OperationStatus::Started
                    && !op.invoke_dispatched
            })
            .filter_map(|op| {
                let target = functions.get(op.invoke_function.as_deref()?)?.clone();
                op.invoke_dispatched = true;
                Some((
                    op.id.clone(),
                    target,
                    op.invoke_input.clone().unwrap_or_default(),
                ))
            })
            .collect()
    }

    /// Records `outcome` on the chained invoke `id`, and adds `id` to the
    /// updated IDs. Returns whether the record changed: a suspended nested
    /// run leaves it `Started`.
    fn record_invoke_outcome(&self, id: &str, outcome: NestedOutcome) -> bool {
        let mut state = self.lock();
        let BackendState {
            ops, updated_ids, ..
        } = &mut *state;
        // The runner took `id` from the stored records, and the backend
        // never removes a record. No checkpoint reaches this backend while
        // a target runs, because the caller's handler is not running. So
        // the record is still stored.
        #[expect(clippy::expect_used)]
        let op = ops
            .iter_mut()
            .find(|op| op.id == id)
            .expect("a dispatched invoke record stays stored");
        match outcome {
            NestedOutcome::Succeeded(result) => {
                op.status = OperationStatus::Succeeded;
                op.result = Some(result.unwrap_or_else(|| "null".to_owned()));
            }
            NestedOutcome::Failed(wire) => {
                op.status = OperationStatus::Failed;
                op.error_type = wire.error_type().map(ToOwned::to_owned);
                op.error_message = wire.error_message().map(ToOwned::to_owned);
                op.error_data = wire.error_data().map(ToOwned::to_owned);
                op.stack_trace = wire.stack_trace().to_vec();
            }
            NestedOutcome::Suspended => return false,
        }
        updated_ids.push(op.id.clone());
        true
    }

    /// The function names of the `Started` chained invokes, in stored
    /// order, each name once.
    fn pending_invoke_names(&self) -> Vec<String> {
        let state = self.lock();
        let mut names: Vec<String> = Vec::new();
        for op in &state.ops {
            if op.op_type == OperationType::ChainedInvoke
                && op.status == OperationStatus::Started
                && let Some(name) = &op.invoke_function
                && !names.contains(name)
            {
                names.push(name.clone());
            }
        }
        names
    }
}

/// The message of a suspended result. It names every queued named reply
/// that no pending callback matched, so a test sees a misspelled or wrong
/// name. Then it names the function of every chained invoke still
/// pending.
fn suspended_message(unmatched_names: &[String], pending_invokes: &[String]) -> String {
    let quote = |names: &[String]| {
        names
            .iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut message = "execution suspended with no pending timer, retry, or callback \
                       to advance (missing callback outcome?)"
        .to_owned();
    if !unmatched_names.is_empty() {
        message.push_str("; no pending callback matched the queued replies for: ");
        message.push_str(&quote(unmatched_names));
    }
    if !pending_invokes.is_empty() {
        message.push_str("; chained invokes still pending for: ");
        message.push_str(&quote(pending_invokes));
    }
    message
}

/// Whether `op` is a pending callback: a callback record in `Pending` or
/// `Started` that has not received a reply.
fn is_pending_callback(op: &StoredOp) -> bool {
    op.op_type == OperationType::Callback
        && !op.callback_delivered
        && matches!(
            op.status,
            OperationStatus::Pending | OperationStatus::Started
        )
}

/// Whether a named reply for `name` matches the callback `op`.
///
/// It matches when the callback has the name, or when the callback's parent
/// is a stored `wait_for_callback` context with the name. A plain child
/// context's name does not match: the JS test runner reads the callback of
/// a context only for the `WAIT_FOR_CALLBACK` sub-type.
fn reply_name_matches(ops: &[StoredOp], op: &StoredOp, name: &str) -> bool {
    if op.name.as_deref() == Some(name) {
        return true;
    }
    let Some(parent_id) = op.parent_id.as_deref() else {
        return false;
    };
    ops.iter().any(|parent| {
        parent.id == parent_id
            && parent.op_type == OperationType::Context
            && parent.sub_type.as_deref() == Some(crate::callback::WFCB_SUB_TYPE)
            && parent.name.as_deref() == Some(name)
    })
}

/// Removes from `replies` the reply each pending callback receives, and
/// returns one entry per record of `ops`, in the same order: the reply the
/// record receives, or `None`.
///
/// The pending callbacks are visited in stored order. Each one takes the
/// first queued named reply that matches it (see [`reply_name_matches`]);
/// else the front FIFO reply; else nothing.
fn take_callback_deliveries(
    ops: &[StoredOp],
    replies: &mut QueuedReplies,
) -> Vec<Option<CallbackReply>> {
    ops.iter()
        .map(|op| {
            if !is_pending_callback(op) {
                return None;
            }
            let named = replies
                .named
                .iter()
                .position(|(name, _)| reply_name_matches(ops, op, name));
            match named {
                Some(position) => Some(replies.named.remove(position).1),
                None => replies.fifo.pop_front(),
            }
        })
        .collect()
}

/// The virtual time at which `op` ends, if it is timed: a wait in `Started`
/// or a step (or `wait_for_condition` check) in `Pending` with a due time.
/// Firing changes the status, which removes the operation from the timed
/// set.
fn timed_due_at(op: &StoredOp) -> Option<u64> {
    match (&op.op_type, &op.status) {
        (OperationType::Wait, OperationStatus::Started)
        | (OperationType::Step, OperationStatus::Pending) => op.due_at,
        _ => None,
    }
}

/// Converts a wire duration in seconds to the clock's unit. A missing or
/// negative value counts as 0, as in the JS and Python checkpoint servers.
fn whole_seconds(value: Option<i32>) -> u64 {
    u64::from(value.unwrap_or(0).max(0).unsigned_abs())
}

impl BackendState {
    /// Finds a stored op by wire ID.
    fn find_mut(&mut self, id: &str) -> Option<&mut StoredOp> {
        self.ops.iter_mut().find(|op| op.id == id)
    }

    /// Applies a single `OperationUpdate` to the store, returning the
    /// resulting stored op (cloned) for the checkpoint response.
    #[expect(clippy::too_many_lines)] // reason: one match over the update actions reads better as one flow
    fn apply_update(&mut self, update: &OperationUpdate) -> Option<StoredOp> {
        let id = update.id().to_owned();
        let op_type = update.r#type().clone();
        let sub_type = update.sub_type().map(ToOwned::to_owned);
        let action = update.action().clone();
        let name = update.name().map(ToOwned::to_owned);
        let payload = update.payload().map(ToOwned::to_owned);
        let (err_type, err_msg, err_data, err_trace) =
            update.error().map_or((None, None, None, Vec::new()), |e| {
                (
                    e.error_type().map(ToOwned::to_owned),
                    e.error_message().map(ToOwned::to_owned),
                    e.error_data().map(ToOwned::to_owned),
                    e.stack_trace().to_vec(),
                )
            });
        let wait_seconds = update
            .wait_options()
            .and_then(aws_sdk_lambda::types::WaitOptions::wait_seconds);
        let retry_delay = update
            .step_options()
            .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds);
        let parent_id = update.parent_id().map(ToOwned::to_owned);
        let invoke_function = update
            .chained_invoke_options()
            .map(|options| options.function_name().to_owned());
        let replay_children = update
            .context_options()
            .and_then(ContextOptions::replay_children)
            .unwrap_or(false);
        // Read the clock before the mutable op borrow below.
        let now = self.now;

        // Ensure the op exists. The parent comes from the creating update
        // only, as on the service.
        if self.find_mut(&id).is_none() {
            self.ops.push(new_started_op(
                id.clone(),
                op_type.clone(),
                sub_type.clone(),
                parent_id,
            ));
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
                // A Start carries no payload: clear any prior result so the
                // whole-record merge in the SDK clobbers the carried value
                // (the wait_for_condition read-before-Start invariant).
                op.result = None;
                op.error_type = None;
                op.error_message = None;
                op.error_data = None;
                op.stack_trace = Vec::new();
                if op.op_type == OperationType::Wait {
                    op.status = OperationStatus::Started;
                    op.wait_seconds = wait_seconds;
                    op.due_at = Some(now.saturating_add(whole_seconds(wait_seconds)));
                } else if op.op_type == OperationType::Callback {
                    // The service creates a callback in `Started` and keeps
                    // it there until an external outcome. The backend
                    // assigns the callback id here.
                    op.status = OperationStatus::Started;
                    if let Some(cb) = new_callback_id {
                        op.callback_id = Some(cb);
                    }
                } else {
                    op.status = OperationStatus::Started;
                    if op.op_type == OperationType::ChainedInvoke {
                        op.invoke_function = invoke_function;
                        op.invoke_input = payload;
                    }
                }
            }
            OperationAction::Succeed => {
                op.status = OperationStatus::Succeeded;
                op.result = payload;
                if op.op_type == OperationType::Context {
                    op.replay_children = replay_children;
                }
            }
            OperationAction::Fail => {
                op.status = OperationStatus::Failed;
                if op.op_type == OperationType::Context {
                    op.replay_children = replay_children;
                }
                op.error_type = err_type;
                op.error_message = err_msg;
                op.error_data = err_data;
                op.stack_trace = err_trace;
            }
            OperationAction::Retry => {
                op.status = OperationStatus::Pending;
                op.attempt += 1;
                op.due_at = Some(now.saturating_add(whole_seconds(retry_delay)));
                op.error_type = err_type;
                op.error_message = err_msg;
                op.error_data = err_data;
                op.stack_trace = err_trace;
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

impl Backend {
    /// Applies and answers one checkpoint request with no latency.
    fn checkpoint_now<'a>(
        &'a self,
        updates: &[OperationUpdate],
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + 'a>>
    {
        self.checkpoint_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // In-crate test seam: a per-call plan (see
        // [`Backend::plan_checkpoint_calls`]) takes precedence over the
        // counter-based injection below while entries remain.
        #[cfg(test)]
        if let Some(planned) = self.next_planned_checkpoint() {
            match planned {
                PlannedCheckpoint::Pass => {}
                PlannedCheckpoint::FailNonRetryable { gate } => {
                    return Box::pin(async move {
                        if let Some(gate) = gate {
                            // Hold the failing write in flight until the
                            // test releases it: lets a test drop a
                            // contributor deterministically BEFORE the
                            // failure publishes.
                            let permit = gate.acquire().await;
                            drop(permit);
                        }
                        Err(ExecutionClientError::new_non_retryable(
                            "planned non-retryable checkpoint failure (test plan)",
                        ))
                    });
                }
            }
        }
        // Injected fault (`LocalRunner::fail_next_checkpoints` /
        // `fail_checkpoints_after[_retryable]`): after letting `skip`
        // calls through, reject BEFORE touching any state, so the
        // rejected write persists nothing, exactly like a service-side
        // rejection.
        let skipped = atomic_decrement_if_positive(&self.checkpoint_failures_skip);
        if !skipped && atomic_decrement_if_positive(&self.checkpoint_failures_remaining) {
            let retryable = self
                .checkpoint_failures_retryable
                .load(std::sync::atomic::Ordering::SeqCst);
            return Box::pin(async move {
                if retryable {
                    Err(ExecutionClientError::from_retryable(
                        "injected retryable checkpoint failure \
                         (LocalRunner::fail_checkpoints_after_retryable)"
                            .to_owned(),
                    ))
                } else {
                    Err(ExecutionClientError::new_non_retryable(
                        "injected checkpoint failure (LocalRunner::fail_next_checkpoints)",
                    ))
                }
            });
        }
        // The service rejects a request that breaks a size limit or a
        // delay range before it changes any state. So the check runs before
        // the state lock, and a rejected request stores nothing and keeps
        // the token.
        if let Err(message) = validate_service_limits(updates) {
            return Box::pin(async move { Err(ExecutionClientError::new_non_retryable(&message)) });
        }
        // An invoke of a name the runner has no target for can never
        // complete. So the backend rejects it before any state change, and
        // the run fails with a message naming the function.
        if let Some(name) = updates
            .iter()
            .filter(|update| {
                update.r#type() == &OperationType::ChainedInvoke
                    && update.action() == &OperationAction::Start
            })
            .filter_map(OperationUpdate::chained_invoke_options)
            .map(aws_sdk_lambda::types::ChainedInvokeOptions::function_name)
            .find(|name| !self.registered_functions.contains(*name))
        {
            let message = format!(
                "No function found for function name \"{name}\". Register it on the LocalRunner."
            );
            return Box::pin(async move { Err(ExecutionClientError::new_non_retryable(&message)) });
        }
        let updated_ops: Vec<Operation> = {
            let mut state = self.lock();
            state.token_counter += 1;
            let mut out = Vec::with_capacity(updates.len());
            for update in updates {
                if let Some(stored) = state.apply_update(update)
                    && let Some(op) = build_operation(&stored)
                {
                    out.push(op);
                }
            }
            out
        };
        let token = self.current_token();

        // If a checkpoint page size is configured and the total stored
        // operations exceed it, simulate a genuinely paginated response:
        // return only the FIRST PAGE of the full execution state (mirroring
        // the service's NewExecutionState) and set next_marker. Operations
        // beyond the page, including the ones this call just updated when
        // they fall outside it, are only observable through the follow-up
        // get_state fetch, so a caller that ignores the marker genuinely
        // misses state.
        let (response_ops, next_marker) = if let Some(page_size) = self.checkpoint_page_size {
            let all_ops = self.build_operations();
            if all_ops.len() > page_size {
                let first_page: Vec<Operation> =
                    all_ops.get(..page_size).unwrap_or(&all_ops).to_vec();
                let marker = format!("page-marker-{}", all_ops.len());
                (first_page, Some(marker))
            } else {
                (updated_ops, None)
            }
        } else {
            (updated_ops, None)
        };

        Box::pin(async move {
            Ok(CheckpointOutput {
                checkpoint_token: Some(token),
                updated_operations: response_ops,
                next_marker,
            })
        })
    }
}

impl ExecutionClient for Backend {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
    {
        // `tokio::time::sleep` panics on a runtime built without the time
        // driver, and the SDK itself needs no timer. So the default zero
        // latency answers without creating a sleep.
        let latency = self.checkpoint_latency;
        if latency.is_zero() {
            return self.checkpoint_now(&updates);
        }
        Box::pin(async move {
            tokio::time::sleep(latency).await;
            self.checkpoint_now(&updates).await
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        _marker: Option<&str>,
    ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
    {
        self.get_state_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let operations = self.build_pruned_operations();
        Box::pin(async move { Ok(GetStateOutput::new(operations, None)) })
    }
}

/// Whether `stored` carries any error field. A callback failure may set
/// only some of the four fields, so any one of them makes an error object.
fn has_error_fields(stored: &StoredOp) -> bool {
    stored.error_type.is_some()
        || stored.error_message.is_some()
        || stored.error_data.is_some()
        || !stored.stack_trace.is_empty()
}

/// Builds an SDK `Operation` from a stored op, placing result/error in the
/// details bucket the SDK reads for that operation type.
fn build_operation(stored: &StoredOp) -> Option<Operation> {
    let error_obj = if has_error_fields(stored) {
        let mut eb = ErrorObject::builder();
        if let Some(t) = &stored.error_type {
            eb = eb.error_type(t);
        }
        if let Some(m) = &stored.error_message {
            eb = eb.error_message(m);
        }
        if let Some(d) = &stored.error_data {
            eb = eb.error_data(d);
        }
        for frame in &stored.stack_trace {
            eb = eb.stack_trace(frame);
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

    if let Some(ref st) = stored.sub_type {
        builder = builder.sub_type(st);
    }
    if let Some(ref n) = stored.name {
        builder = builder.name(n);
    }
    if let Some(ref parent) = stored.parent_id {
        builder = builder.parent_id(parent);
    }

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
            if stored.replay_children {
                cd = cd.replay_children(true);
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

/// Renders a stored op as an envelope-wire JSON operation object, placing
/// result/error in the details bucket `parse_single_operation` reads for
/// that operation type. Mirrors [`build_operation`] field-for-field so the
/// inline envelope page and the `get_state` fetch describe identical
/// records.
fn stored_op_envelope_json(stored: &StoredOp) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("Id".to_owned(), serde_json::json!(stored.id));
    obj.insert(
        "Type".to_owned(),
        serde_json::json!(stored.op_type.as_str()),
    );
    obj.insert(
        "Status".to_owned(),
        serde_json::json!(operation_status_wire_str(&stored.status)),
    );
    if let Some(st) = &stored.sub_type {
        obj.insert("SubType".to_owned(), serde_json::json!(st));
    }
    if let Some(n) = &stored.name {
        obj.insert("Name".to_owned(), serde_json::json!(n));
    }
    if let Some(parent) = &stored.parent_id {
        obj.insert("ParentId".to_owned(), serde_json::json!(parent));
    }

    let error_obj = has_error_fields(stored).then(|| {
        let mut err = serde_json::Map::new();
        if let Some(t) = &stored.error_type {
            err.insert("ErrorType".to_owned(), serde_json::json!(t));
        }
        if let Some(m) = &stored.error_message {
            err.insert("ErrorMessage".to_owned(), serde_json::json!(m));
        }
        if let Some(d) = &stored.error_data {
            err.insert("ErrorData".to_owned(), serde_json::json!(d));
        }
        if !stored.stack_trace.is_empty() {
            err.insert(
                "StackTrace".to_owned(),
                serde_json::json!(stored.stack_trace),
            );
        }
        serde_json::Value::Object(err)
    });

    let mut details = serde_json::Map::new();
    if let Some(r) = &stored.result {
        details.insert("Result".to_owned(), serde_json::json!(r));
    }
    if let Some(e) = error_obj {
        details.insert("Error".to_owned(), e);
    }

    match stored.op_type {
        OperationType::Step => {
            details.insert("Attempt".to_owned(), serde_json::json!(stored.attempt));
            obj.insert("StepDetails".to_owned(), serde_json::Value::Object(details));
        }
        OperationType::Context => {
            if stored.replay_children {
                details.insert("ReplayChildren".to_owned(), serde_json::json!(true));
            }
            obj.insert(
                "ContextDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationType::Callback => {
            if let Some(id) = &stored.callback_id {
                details.insert("CallbackId".to_owned(), serde_json::json!(id));
            }
            obj.insert(
                "CallbackDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationType::ChainedInvoke => {
            obj.insert(
                "ChainedInvokeDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        _ => {}
    }

    serde_json::Value::Object(obj)
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

/// Wire (`UPPER_CASE`) operation status for the envelope JSON that
/// `parse_single_operation` reads.
///
/// The envelope must speak the service's wire dialect exactly: the real
/// backend sends `TIMED_OUT` (never `TIMEDOUT`), and the production parser
/// treats any other spelling as an unrecognized status (issue #45), so the
/// local runner rendering `PascalCase` here would fail replay.
fn operation_status_wire_str(s: &OperationStatus) -> &'static str {
    match *s {
        OperationStatus::Started => "STARTED",
        OperationStatus::Pending => "PENDING",
        OperationStatus::Ready => "READY",
        OperationStatus::Succeeded => "SUCCEEDED",
        OperationStatus::Failed => "FAILED",
        OperationStatus::Cancelled => "CANCELLED",
        OperationStatus::TimedOut => "TIMED_OUT",
        OperationStatus::Stopped => "STOPPED",
        _ => "UNKNOWN",
    }
}

/// Returns the stored records the service sends in the invocation input and
/// in `GetDurableExecutionState`, in stored order.
///
/// The service omits every descendant of a completed context that did not
/// record `ReplayChildren`, because the context's result replaces them on
/// replay. The rules are:
///
/// 1. A record without a parent is kept.
/// 2. Otherwise the walk goes up the parents. A parent that is not stored
///    keeps the record. A parent that is a `Succeeded` or `Failed` context
///    without `ReplayChildren` omits the record. Any other parent continues
///    the walk. So a started context under a pruning ancestor is omitted
///    with its children.
/// 3. A crafted `ParentId` can form a cycle. So the walk takes at most
///    `ops.len()` steps, and a record whose walk reaches the bound is kept.
///
/// Checkpoint responses are not pruned, because the service does not prune
/// them.
fn prune_replay_children(ops: &[StoredOp]) -> Vec<&StoredOp> {
    let find = |id: &str| ops.iter().find(|op| op.id == id);
    let omitted = |op: &StoredOp| {
        let mut parent_id = op.parent_id.as_deref();
        for _ in 0..ops.len() {
            let Some(parent) = parent_id.and_then(find) else {
                return false;
            };
            if parent.op_type == OperationType::Context
                && matches!(
                    parent.status,
                    OperationStatus::Succeeded | OperationStatus::Failed
                )
                && !parent.replay_children
            {
                return true;
            }
            parent_id = parent.parent_id.as_deref();
        }
        false
    };
    ops.iter().filter(|op| !omitted(op)).collect()
}

/// The smallest delay, in seconds, the service accepts for
/// `WaitOptions.WaitSeconds` and `StepOptions.NextAttemptDelaySeconds`.
const MIN_DELAY_SECONDS: i32 = 1;

/// The largest delay, in seconds, the service accepts for the same fields.
const MAX_DELAY_SECONDS: i32 = 31_622_400;

/// The most delay violations one rejection message lists.
const MAX_REPORTED_VIOLATIONS: usize = 10;

/// The largest rendering of an update that carries an error.
const ERROR_OBJECT_LIMIT: usize = 262_144;

/// Checks a checkpoint request against the service limits, and returns the
/// service's rejection message for the first failed check.
///
/// The check has two passes, in the order the service runs them.
///
/// 1. The model pass checks every delay field of every update against the
///    range 1 to 31 622 400. It collects every violation and reports at
///    most 10 of them in one message. When it finds any violation, the
///    second pass does not run.
/// 2. The size pass visits the updates in request order and stops at the
///    first violation. An update that carries an error is measured as a
///    whole (see [`render_update`]), as the JS checkpoint server measures
///    it. Then a payload is measured in bytes against the limit of its
///    operation type (see [`payload_limit`]).
fn validate_service_limits(updates: &[OperationUpdate]) -> Result<(), String> {
    let mut entries = Vec::new();
    for (index, update) in updates.iter().enumerate() {
        let pos = index + 1;
        let delays = [
            (
                "waitOptions.waitSeconds",
                update
                    .wait_options()
                    .and_then(aws_sdk_lambda::types::WaitOptions::wait_seconds),
            ),
            (
                "stepOptions.nextAttemptDelaySeconds",
                update
                    .step_options()
                    .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds),
            ),
        ];
        for (path, value) in delays {
            let Some(value) = value else { continue };
            let constraint = if value < MIN_DELAY_SECONDS {
                "greater than or equal to 1"
            } else if value > MAX_DELAY_SECONDS {
                "less than or equal to 31622400"
            } else {
                continue;
            };
            entries.push(format!(
                "Value '{value}' at 'updates.{pos}.member.{path}' failed to satisfy \
                 constraint: Member must have value {constraint}"
            ));
        }
    }
    if !entries.is_empty() {
        entries.truncate(MAX_REPORTED_VIOLATIONS);
        let count = if entries.len() == 1 {
            "1 validation error detected".to_owned()
        } else {
            format!("{} validation errors detected", entries.len())
        };
        return Err(format!("{count}: {}", entries.join("; ")));
    }

    for update in updates {
        if update.error().is_some() && render_update(update).len() > ERROR_OBJECT_LIMIT {
            return Err("Error object size must be less than or equal to 262144 bytes.".to_owned());
        }
        if let (Some(payload), Some((limit, what))) =
            (update.payload(), payload_limit(update.r#type()))
            && payload.len() > limit
        {
            return Err(format!(
                "{what} size must be less than or equal to {limit} bytes."
            ));
        }
    }
    Ok(())
}

/// The payload limit in bytes for an operation type, and the name the
/// service's message uses for that payload. WAIT and CALLBACK payloads
/// have no limit.
fn payload_limit(op_type: &OperationType) -> Option<(usize, &'static str)> {
    match op_type {
        OperationType::Step => Some((262_144, "STEP output payload")),
        OperationType::Context => Some((262_144, "CONTEXT output payload")),
        OperationType::ChainedInvoke => Some((1_048_576, "CHAINED_INVOKE input payload")),
        OperationType::Execution => Some((6_291_456, "Execution output payload")),
        _ => None,
    }
}

/// Renders `update` as the JSON object the service model describes, with
/// every absent member omitted and every enum as its wire string.
///
/// `OperationUpdate` has no serializer, so the object is built from its
/// getters. The size pass of [`validate_service_limits`] measures this
/// rendering for an update that carries an error.
fn render_update(update: &OperationUpdate) -> String {
    use serde_json::{Map, Value};

    fn insert<T: Into<Value>>(map: &mut Map<String, Value>, key: &str, value: Option<T>) {
        if let Some(value) = value {
            map.insert(key.to_owned(), value.into());
        }
    }

    // A present error or options object stays in the rendering even when
    // all its members are absent, as `JSON.stringify` keeps an empty object.
    let mut out = Map::new();
    insert(&mut out, "Id", Some(update.id()));
    insert(&mut out, "ParentId", update.parent_id());
    insert(&mut out, "Name", update.name());
    insert(&mut out, "Type", Some(update.r#type().as_str()));
    insert(&mut out, "SubType", update.sub_type());
    insert(&mut out, "Action", Some(update.action().as_str()));
    insert(&mut out, "Payload", update.payload());
    if let Some(error) = update.error() {
        let mut map = Map::new();
        insert(&mut map, "ErrorType", error.error_type());
        insert(&mut map, "ErrorMessage", error.error_message());
        insert(&mut map, "ErrorData", error.error_data());
        insert(&mut map, "StackTrace", error.stack_trace.clone());
        out.insert("Error".to_owned(), Value::Object(map));
    }
    if let Some(options) = update.context_options() {
        let mut map = Map::new();
        insert(&mut map, "ReplayChildren", options.replay_children());
        out.insert("ContextOptions".to_owned(), Value::Object(map));
    }
    if let Some(options) = update.step_options() {
        let mut map = Map::new();
        insert(
            &mut map,
            "NextAttemptDelaySeconds",
            options.next_attempt_delay_seconds(),
        );
        out.insert("StepOptions".to_owned(), Value::Object(map));
    }
    if let Some(options) = update.wait_options() {
        let mut map = Map::new();
        insert(&mut map, "WaitSeconds", options.wait_seconds());
        out.insert("WaitOptions".to_owned(), Value::Object(map));
    }
    if let Some(options) = update.callback_options() {
        let mut map = Map::new();
        // The model types these fields as `i32` with 0 meaning "not
        // specified", and the wire serializer omits a 0. So the rendering
        // omits a 0 too.
        let nonzero = |value: i32| (value != 0).then_some(value);
        insert(
            &mut map,
            "TimeoutSeconds",
            nonzero(options.timeout_seconds()),
        );
        insert(
            &mut map,
            "HeartbeatTimeoutSeconds",
            nonzero(options.heartbeat_timeout_seconds()),
        );
        out.insert("CallbackOptions".to_owned(), Value::Object(map));
    }
    if let Some(options) = update.chained_invoke_options() {
        let mut map = Map::new();
        insert(&mut map, "FunctionName", Some(options.function_name()));
        insert(&mut map, "TenantId", options.tenant_id());
        out.insert("ChainedInvokeOptions".to_owned(), Value::Object(map));
    }
    Value::Object(out).to_string()
}

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
// reason: test assertions index known-length op vectors
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/future.rs helpers.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::builders::map_parallel::CompletionConfig;
    use crate::{RetryDecision, StepError};
    use serde::{Deserialize, Serialize};
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn local_runner_failed_envelope_without_type_reads_error() {
        let response = serde_json::json!({
            "Status": "FAILED",
            "Error": { "ErrorMessage": "boom" },
        });
        let result: TestResult<()> = RawRun {
            end: RawEnd::Failed(envelope_wire_error(&response)),
            operations: Vec::new(),
            invocations: 1,
        }
        .into_test_result();
        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("Error"));
        assert_eq!(result.error_message(), Some("boom"));
    }

    // 1. step success

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

    // 1. step failure

    #[tokio::test]
    async fn step_failure_propagates_and_records_failed_op() {
        // A strategy that never retries so the failure is permanent.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let strategy = |_e: &StepError, _a: u32| RetryDecision::Stop;
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
        // The execution record re-records the step failure's own recorded
        // identity: a plain boxed error carries no user type, so the step
        // record's generic "Error" passes through the boundary unchanged
        // (the "StepError" registry name is only the fallback for an
        // operation error with no attached wire identity).
        assert_eq!(result.error_type(), Some("Error"));
        assert!(result.operations().iter().any(TestOperation::failed));
    }

    // 1. step retry via RetryStrategyConfig

    #[tokio::test]
    async fn step_retry_config_drives_retries_to_success() {
        // A config-based strategy (no hand-written closure): 3 total
        // attempts, 1s deterministic delays. The step fails on attempts 1
        // and 2 and succeeds on attempt 3, exercising the config through
        // real suspend/replay cycles.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let attempts_seen = Arc::clone(&attempts_seen);
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(3)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .max_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build();
                        let v: i32 = ctx
                            .step(move |sc| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    if sc.attempt() < 3 {
                                        Err(format!("attempt {} failed", sc.attempt()).into())
                                    } else {
                                        Ok(7)
                                    }
                                }
                            })
                            .name("flaky-config")
                            .retry_strategy_config(config)
                            .await?;
                        Ok::<_, BoxError>(v)
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&7));
        // Body executed exactly 3 times (attempts 1, 2, 3).
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert!(result.invocation_count() >= 3);
    }

    #[tokio::test]
    async fn step_retry_config_max_attempts_exhausts_and_fails() {
        // max_attempts(2): the body runs exactly twice, then the error
        // propagates instead of retrying a third time.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let attempts_seen = Arc::clone(&attempts_seen);
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(2)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build();
                        let v: i32 = ctx
                            .step(move |_| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    Err("always fails".into())
                                }
                            })
                            .name("always-fails")
                            .retry_strategy_config(config)
                            .await?;
                        Ok::<_, BoxError>(v)
                    }
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(result.operations().iter().any(TestOperation::failed));
    }

    // 1. step retry (retry strategy honored)

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
                        let strategy = |_e: &StepError, attempt: u32| {
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: std::time::Duration::from_secs(1),
                                }
                            }
                        };
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

    // 2. wait then resume across invocations

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

    // 3. wait_for_condition state carry (exercises the merge-back clobber)

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
                    let strategy = |state: Counter, _attempt: u32| {
                        if state.count >= 3 {
                            crate::builders::wait_for_condition::WaitDecision::complete()
                        } else {
                            crate::builders::wait_for_condition::WaitDecision::continue_with(
                                std::time::Duration::from_secs(1),
                            )
                        }
                    };
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

    #[tokio::test]
    async fn wait_strategy_config_completes_on_predicate() {
        // The config-derived strategy (issue #35) completes once its
        // completion predicate matches the state the check produced.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
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
                        .wait_strategy(
                            crate::builders::wait_for_condition::WaitStrategy::builder(
                                |state: &Counter| state.count >= 3,
                            )
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build(),
                        )
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
            "the operation must complete when the predicate first matches"
        );
    }

    #[tokio::test]
    async fn wait_strategy_config_exhausts_at_max_attempts() {
        // A predicate that never matches must exhaust at max_attempts with
        // a MaxChecksExceeded failure, not poll forever.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let final_state = ctx
                        .wait_for_condition(
                            |_sc, state: Counter| async move {
                                Ok(Counter {
                                    count: state.count + 1,
                                })
                            },
                            Counter { count: 0 },
                        )
                        .name("never-ready")
                        .wait_strategy(
                            crate::builders::wait_for_condition::WaitStrategy::builder(
                                |_state: &Counter| false,
                            )
                            .max_attempts(2)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build(),
                        )
                        .await?;
                    Ok::<_, BoxError>(final_state.count)
                },
                (),
            )
            .await;

        assert!(
            !result.is_success(),
            "an unmatched predicate must exhaust, not succeed"
        );
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.contains("max checks exceeded: 2"),
            "failure must name exhaustion with the attempt count: {message}"
        );
    }

    /// REGRESSION TEST: exhaustion keeps its `MaxChecksExceeded`
    /// classification across a terminal replay.
    ///
    /// The handler catches the exhaustion error and then suspends on a
    /// wait, so the next invocation replays the failed operation from its
    /// checkpoint. Both the live catch and the replayed catch must
    /// classify as `MaxChecksExceeded` with the same `checks()` value:
    /// the returned output comes from the final (replaying) invocation,
    /// so a replay misclassified as `CheckFailed` fails the execution.
    #[tokio::test]
    async fn wait_strategy_exhaustion_classification_survives_replay() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let outcome = ctx
                        .wait_for_condition(
                            |_sc, state: Counter| async move {
                                Ok(Counter {
                                    count: state.count + 1,
                                })
                            },
                            Counter { count: 0 },
                        )
                        .name("never-ready")
                        .wait_strategy(
                            crate::builders::wait_for_condition::WaitStrategy::builder(
                                |_state: &Counter| false,
                            )
                            .max_attempts(2)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build(),
                        )
                        .await;

                    let Err(err) = outcome else {
                        return Err("expected exhaustion".into());
                    };
                    let checks = match err.kind() {
                        crate::OperationErrorKind::WaitForCondition(wfc) => match wfc.kind() {
                            crate::WaitForConditionErrorKind::MaxChecksExceeded(details) => {
                                details.checks()
                            }
                            other => {
                                return Err(
                                    format!("expected MaxChecksExceeded, got: {other:?}").into()
                                );
                            }
                        },
                        other => {
                            return Err(
                                format!("expected WaitForCondition error, got: {other:?}").into()
                            );
                        }
                    };

                    // Suspend so a later invocation replays the failed
                    // operation as a frozen terminal outcome.
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    Ok::<_, BoxError>(checks)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&2),
            "the replayed error must report the same check count as the live one"
        );
        assert!(
            result.invocation_count() >= 3,
            "the exhaustion must have been replayed at least once (got {} invocations)",
            result.invocation_count()
        );
    }

    // 4. child contexts

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

    // 5. parallel and map basics

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

    /// Replay of a predicate-completed batch is stable: the execution
    /// suspends mid-batch (a timer inside an item), resumes on a later
    /// invocation, replays the already-settled item from its checkpoint
    /// record without re-running its body, and the custom predicate then
    /// ends the batch early with the `PREDICATE_MATCHED` reason.
    #[tokio::test]
    async fn map_completion_predicate_survives_replay() {
        let item0_runs = Arc::new(AtomicU32::new(0));
        let item0_runs_handle = Arc::clone(&item0_runs);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let item0_runs = Arc::clone(&item0_runs_handle);
                    async move {
                        let batch = ctx
                            .map(vec![10_u32, 20, 30], move |child, item, idx| {
                                let item0_runs = Arc::clone(&item0_runs);
                                async move {
                                    if idx == 1 {
                                        // Suspends the batch mid-run, forcing a
                                        // second invocation that replays item 0.
                                        child
                                            .wait(std::time::Duration::from_secs(1))
                                            .name("stall")
                                            .await?;
                                    }
                                    let value = child
                                        .step(move |_| async move {
                                            if idx == 0 {
                                                item0_runs.fetch_add(1, Ordering::SeqCst);
                                            }
                                            Ok(item)
                                        })
                                        .name("work")
                                        .await?;
                                    Ok(value)
                                }
                            })
                            .name("predicated")
                            .max_concurrency(1)
                            .completion(
                                CompletionConfig::builder()
                                    .completion_predicate(|stats| stats.settled() >= 2)
                                    .build()?,
                            )
                            .await_batch()
                            .await?;
                        Ok::<_, BoxError>(format!(
                            "{}:{}",
                            batch.reason.as_str(),
                            batch.items.len()
                        ))
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // The predicate fired after two settles; item 2 never started.
        assert_eq!(result.output(), Some(&"PREDICATE_MATCHED:2".to_owned()));
        // The timer forced at least one suspension, so the batch spanned
        // several invocations and item 0 was replayed at least once.
        assert!(
            result.invocation_count() >= 2,
            "expected a replay, got {} invocation(s)",
            result.invocation_count()
        );
        // Replay returned item 0's recorded result instead of re-running
        // its step body.
        assert_eq!(
            item0_runs.load(Ordering::SeqCst),
            1,
            "item 0's step body must run exactly once across replays"
        );
    }

    /// Completion-trigger evaluation is deterministic under replay for a
    /// CONCURRENT batch: recorded-terminal children feed the statistics
    /// inline, in input order, before any resumed (live) branch joins.
    ///
    /// Three items at `max_concurrency(2)`: items 0 and 2 succeed on the
    /// first invocation, item 1 parks on a timer and fails after resume.
    /// The order-sensitive predicate `failed() > succeeded()` must never
    /// fire: on the resumed invocation the two recorded successes are
    /// applied (in input order) before item 1's live failure joins, so the
    /// predicate always sees 2 succeeded before it sees 1 failed:
    /// regardless of how the scheduler orders the join events. Without
    /// canonical ordering, resumed item 1's failure could join before
    /// item 0's recorded success replays, the predicate would see
    /// 1 failed / 0 succeeded and stop the batch, and item 2, whose
    /// success is already in the checkpoint log, would be dropped from
    /// the result: a different operation history from identical recorded
    /// state.
    #[tokio::test]
    async fn map_completion_predicate_replay_order_is_canonical() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let batch = ctx
                        .map(vec![10_u32, 20, 30], |child, item, idx| async move {
                            if idx == 1 {
                                // Parks the branch, forcing a second
                                // invocation where items 0 and 2 are
                                // recorded-terminal while item 1 resumes
                                // live, and then fails.
                                child
                                    .wait(std::time::Duration::from_secs(1))
                                    .name("stall")
                                    .await?;
                                return Err("item 1 fails after resume".into());
                            }
                            Ok(item)
                        })
                        .name("order-sensitive")
                        .max_concurrency(2)
                        .completion(
                            CompletionConfig::builder()
                                // Keep the fixed failure trigger out of the
                                // way; only the predicate could stop early.
                                .tolerated_failure_count(10)
                                .completion_predicate(|stats| stats.failed() > stats.succeeded())
                                .build()?,
                        )
                        .await_batch()
                        .await?;
                    Ok::<_, BoxError>(format!(
                        "{}:{}:{}:{}",
                        batch.reason.as_str(),
                        batch.items.len(),
                        batch.success_count(),
                        batch.failure_count()
                    ))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // All three items are in the result, the lone failure never
        // outnumbered the recorded successes, and the batch ran to
        // completion.
        assert_eq!(result.output(), Some(&"ALL_COMPLETED:3:2:1".to_owned()));
        // The timer forced a suspension, so the second invocation really did
        // mix recorded-terminal children with a resumed live child.
        assert!(
            result.invocation_count() >= 2,
            "expected a replay, got {} invocation(s)",
            result.invocation_count()
        );
    }

    /// Regression for reversed LIVE settlement order (finding: replay
    /// ordering nondeterminism). Three items at `max_concurrency(3)`:
    /// item 1 succeeds BEFORE the deliberately delayed item 0 fails, and
    /// item 2 parks on a timer. The order-sensitive predicate
    /// `failed() > succeeded()` must make the same decision however the
    /// scheduler interleaves the live settlements: outcomes commit to the
    /// statistics strictly in input order, so the predicate's first
    /// evaluation is on the prefix `[item 0: failed]`, one failure, zero
    /// successes, and it fires, on the FIRST invocation, before any
    /// suspension. Under settlement-order evaluation the fresh run would
    /// instead see `[1 succeeded]` then `[1 succeeded, 1 failed]`, never
    /// fire, suspend on item 2, and then the resumed invocation, replaying
    /// recorded outcomes in input order, would fire where the original run
    /// did not: two runs of the same execution disagreeing (the exact
    /// reviewed defect).
    #[tokio::test]
    async fn map_predicate_deterministic_under_reversed_live_settlement() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_handle = Arc::clone(&gate);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let gate = Arc::clone(&gate_handle);
                    async move {
                        let batch = ctx
                            .map(vec![10_u32, 20, 30], move |child, item, idx| {
                                let gate = Arc::clone(&gate);
                                async move {
                                    match idx {
                                        0 => {
                                            // Settle strictly after item 1:
                                            // wait for its signal, then let
                                            // its join drain first.
                                            gate.notified().await;
                                            for _ in 0..16 {
                                                tokio::task::yield_now().await;
                                            }
                                            Err("item 0 fails last".into())
                                        }
                                        1 => {
                                            gate.notify_one();
                                            Ok(item)
                                        }
                                        _ => {
                                            // Parks; excluded once the
                                            // trigger fires.
                                            child
                                                .wait(std::time::Duration::from_secs(1))
                                                .name("stall")
                                                .await?;
                                            Ok(item)
                                        }
                                    }
                                }
                            })
                            .name("reversed-order")
                            .max_concurrency(3)
                            .completion(
                                CompletionConfig::builder()
                                    // Keep the fixed failure trigger out of
                                    // the way; only the predicate can stop
                                    // the batch.
                                    .tolerated_failure_count(10)
                                    .completion_predicate(|stats| {
                                        stats.failed() > stats.succeeded()
                                    })
                                    .build()?,
                            )
                            .await_batch()
                            .await?;
                        Ok::<_, BoxError>(format!(
                            "{}:{}:{}:{}",
                            batch.reason.as_str(),
                            batch.items.len(),
                            batch.success_count(),
                            batch.failure_count()
                        ))
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // The predicate fired at item 0's committed failure; items 0 and 1
        // both settled and are in the result, parked item 2 is excluded.
        assert_eq!(result.output(), Some(&"PREDICATE_MATCHED:2:1:1".to_owned()));
        // The trigger fired on the first invocation: the batch never
        // suspended, so the fresh run and any replay cannot disagree.
        assert_eq!(
            result.invocation_count(),
            1,
            "the predicate must fire before the batch suspends"
        );
    }

    /// Reversed live settlement order PLUS a genuine suspension: item 1
    /// succeeds before the delayed item 0 fails (both on the first
    /// invocation), item 2 parks and only succeeds after resume. The
    /// order-sensitive predicate stays false on every committed prefix:
    /// `[0: failed]`, `[0: failed, 1: succeeded]`, then after resume
    /// `[.., 2: succeeded]`, and those are the SAME prefixes the resumed
    /// invocation derives from the recorded outcomes, so the batch runs to
    /// completion with every item's outcome in the result, on both sides
    /// of the suspension boundary.
    #[tokio::test]
    async fn map_predicate_reversed_settlement_with_suspension_replays_stably() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_handle = Arc::clone(&gate);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let gate = Arc::clone(&gate_handle);
                    async move {
                        let batch = ctx
                            .map(vec![10_u32, 20, 30], move |child, item, idx| {
                                let gate = Arc::clone(&gate);
                                async move {
                                    match idx {
                                        0 => {
                                            gate.notified().await;
                                            for _ in 0..16 {
                                                tokio::task::yield_now().await;
                                            }
                                            Err("item 0 fails last".into())
                                        }
                                        1 => {
                                            gate.notify_one();
                                            Ok(item)
                                        }
                                        _ => {
                                            // Parks the batch mid-run,
                                            // forcing a resumed invocation
                                            // that replays items 0 and 1
                                            // from their records.
                                            child
                                                .wait(std::time::Duration::from_secs(1))
                                                .name("stall")
                                                .await?;
                                            Ok(item)
                                        }
                                    }
                                }
                            })
                            .name("reversed-then-suspend")
                            .max_concurrency(3)
                            .completion(
                                CompletionConfig::builder()
                                    .tolerated_failure_count(10)
                                    // Order-sensitive but never true here:
                                    // one failure can never outnumber the
                                    // successes by two.
                                    .completion_predicate(|stats| {
                                        stats.failed() > stats.succeeded() + 1
                                    })
                                    .build()?,
                            )
                            .await_batch()
                            .await?;
                        Ok::<_, BoxError>(format!(
                            "{}:{}:{}:{}",
                            batch.reason.as_str(),
                            batch.items.len(),
                            batch.success_count(),
                            batch.failure_count()
                        ))
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // No trigger fired on either side of the suspension: all three
        // outcomes, including recorded items replayed after the resume,
        // are in the result.
        assert_eq!(result.output(), Some(&"ALL_COMPLETED:3:2:1".to_owned()));
        // The timer forced a suspension, so the batch really did span a
        // replay of the reversed-order recorded outcomes.
        assert!(
            result.invocation_count() >= 2,
            "expected a replay, got {} invocation(s)",
            result.invocation_count()
        );
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

    // 6. callback replies

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Approval {
        approved: bool,
    }

    /// `success(&v)` in the issue's test table.
    fn success<T: Serialize>(value: &T) -> CallbackReply {
        CallbackReply::success(value).expect("encodes")
    }

    /// The single callback record of `result`.
    fn callback_record<O>(result: &TestResult<O>) -> &TestOperation {
        result
            .operations()
            .iter()
            .find(|o| o.op_type() == "Callback")
            .expect("callback op recorded")
    }

    #[tokio::test]
    async fn callback_reply_success_delivers_json() {
        let result = LocalRunner::new()
            .callback(success(&Approval { approved: true }))
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
    async fn callback_reply_timeout_surfaces_error() {
        let result = LocalRunner::new()
            .callback(CallbackReply::timeout())
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
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), "TimedOut");
        assert_eq!(cb_op.error_message(), Some("Callback timed out"));
    }

    #[tokio::test]
    async fn callback_reply_raw_is_not_encoded() {
        let result = LocalRunner::new()
            .callback(CallbackReply::raw("\"hi\""))
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().await?;
                    Ok::<_, BoxError>(cb.result().await?)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("hi"));
    }

    /// The fields a handler reads from a callback failure: whether the kind
    /// is an external failure, then the wire type, message and data.
    type FailureFields = (bool, Option<String>, Option<String>, Option<String>);

    #[tokio::test]
    async fn callback_reply_failure_reaches_create_callback() {
        let result = LocalRunner::new()
            .callback(CallbackReply::failure(
                CallbackFailure::new("Declined", "no").with_error_data("{\"code\":7}"),
            ))
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().name("cb").await?;
                    let Err(err) = cb.result().await else {
                        return Err::<FailureFields, BoxError>("callback succeeded".into());
                    };
                    let external = matches!(
                        err.kind(),
                        crate::OperationErrorKind::Callback(e)
                            if matches!(e.kind(), crate::CallbackErrorKind::ExternalFailure)
                    );
                    let wire = err.wire();
                    let field = |read: fn(&crate::WireError) -> Option<&str>| {
                        wire.and_then(read).map(ToOwned::to_owned)
                    };
                    Ok((
                        external,
                        field(crate::WireError::error_type),
                        field(crate::WireError::error_message),
                        field(crate::WireError::error_data),
                    ))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&(
                true,
                Some("Declined".to_owned()),
                Some("no".to_owned()),
                Some("{\"code\":7}".to_owned()),
            ))
        );
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), "Failed");
        assert_eq!(cb_op.error_type(), Some("Declined"));
        assert_eq!(cb_op.error_message(), Some("no"));
    }

    #[tokio::test]
    async fn callback_reply_failure_reaches_wait_for_callback() {
        let result = LocalRunner::new()
            .callback(CallbackReply::failure(CallbackFailure::new(
                "Declined", "no",
            )))
            .run(
                |(), ctx: DurableContext| async move {
                    let outcome = ctx
                        .wait_for_callback::<String, _, _>(|_step_ctx, _callback_id| async {
                            Ok(())
                        })
                        .name("approve")
                        .await;
                    let Err(err) = outcome else {
                        return Err::<bool, BoxError>("wait_for_callback succeeded".into());
                    };
                    Ok(matches!(
                        err.kind(),
                        crate::OperationErrorKind::Callback(e)
                            if matches!(e.kind(), crate::CallbackErrorKind::ExternalFailure)
                    ))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&true),
            "kind is Callback(ExternalFailure)"
        );
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), "Failed");
        assert_eq!(cb_op.error_type(), Some("Declined"));
    }

    #[tokio::test]
    async fn callback_for_targets_named_callbacks() {
        let result = LocalRunner::new()
            .callback_for("cb-a", CallbackReply::raw("\"A\""))
            .callback_for("cb-b", CallbackReply::raw("\"B\""))
            .run(
                |(), ctx: DurableContext| async move {
                    let b = ctx.create_callback::<String>().name("cb-b").await?;
                    let a = ctx.create_callback::<String>().name("cb-a").await?;
                    let a = a.result().await?;
                    let b = b.result().await?;
                    Ok::<_, BoxError>(format!("a={a} b={b}"))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("a=A b=B"));
    }

    #[tokio::test]
    async fn callback_for_targets_wait_for_callback_by_context_name() {
        let result = LocalRunner::new()
            .callback_for("second", success(&2_u32))
            .callback_for("first", success(&1_u32))
            .run(
                |(), ctx: DurableContext| async move {
                    let branches: Vec<crate::Branch<u32>> = ["first", "second"]
                        .into_iter()
                        .map(|name| {
                            crate::Branch::new(name, move |child: DurableContext| async move {
                                let value = child
                                    .wait_for_callback::<u32, _, _>(
                                        |_step_ctx, _callback_id| async { Ok(()) },
                                    )
                                    .name(name)
                                    .await?;
                                Ok(value)
                            })
                        })
                        .collect();
                    Ok::<_, BoxError>(ctx.parallel(branches).name("both").await?)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&vec![1, 2]));
    }

    #[tokio::test]
    async fn callback_for_is_delivered_before_fifo() {
        let result = LocalRunner::new()
            .callback(success(&"fifo"))
            .callback_for("named", success(&"named"))
            .run(
                |(), ctx: DurableContext| async move {
                    let named = ctx.create_callback::<String>().name("named").await?;
                    let other = ctx.create_callback::<String>().name("other").await?;
                    let a = named.result().await?;
                    let b = other.result().await?;
                    Ok::<_, BoxError>(format!("named={a} other={b}"))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output().map(String::as_str),
            Some("named=named other=fifo")
        );
    }

    #[tokio::test]
    async fn callback_for_ignores_plain_child_context_name() {
        let result = LocalRunner::new()
            .callback_for("box", success(&1_u32))
            .run(
                |(), ctx: DurableContext| async move {
                    let value = ctx
                        .run_in_child_context(|child| async move {
                            let cb = child.create_callback::<u32>().await?;
                            Ok(cb.result().await?)
                        })
                        .name("box")
                        .await?;
                    Ok::<_, BoxError>(value)
                },
                (),
            )
            .await;

        assert!(result.is_suspended(), "{:?}", result.error_message());
    }

    #[tokio::test]
    async fn callback_for_unused_name_is_reported() {
        let result = LocalRunner::new()
            .callback_for("missing", success(&1_u32))
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<u32>().name("cb").await?;
                    Ok::<_, BoxError>(cb.result().await?)
                },
                (),
            )
            .await;

        assert!(result.is_suspended(), "{:?}", result.error_message());
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.ends_with("; no pending callback matched the queued replies for: \"missing\""),
            "{message}"
        );
    }

    /// Rule 8 of the issue: the unused names appear in queue order, each
    /// once, and a run with no unused name keeps the old message.
    #[test]
    fn suspended_message_lists_unused_names_once_in_queue_order() {
        let backend = Backend::new(
            QueuedReplies {
                fifo: std::collections::VecDeque::new(),
                named: vec![
                    ("missing".to_owned(), success(&1_u32)),
                    ("other".to_owned(), success(&2_u32)),
                    ("missing".to_owned(), success(&3_u32)),
                ],
            },
            None,
        );
        assert_eq!(
            suspended_message(&backend.unmatched_reply_names(), &[]),
            "execution suspended with no pending timer, retry, or callback to advance \
             (missing callback outcome?); no pending callback matched the queued replies \
             for: \"missing\", \"other\""
        );
        assert_eq!(
            suspended_message(&[], &[]),
            "execution suspended with no pending timer, retry, or callback to advance \
             (missing callback outcome?)"
        );
    }

    #[tokio::test]
    async fn callback_reply_failure_carries_stack_trace() {
        let result = LocalRunner::new()
            .callback(CallbackReply::failure(
                CallbackFailure::new("Declined", "no")
                    .with_stack_trace(vec!["at a".into(), "at b".into()]),
            ))
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().name("cb").await?;
                    let Err(err) = cb.result().await else {
                        return Err::<Vec<String>, BoxError>("callback succeeded".into());
                    };
                    Ok(err
                        .wire()
                        .map(|wire| wire.stack_trace().to_vec())
                        .unwrap_or_default())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&vec!["at a".to_owned(), "at b".to_owned()])
        );
    }

    #[tokio::test]
    async fn callback_reply_default_failure_has_no_fields() {
        let result = LocalRunner::new()
            .callback(CallbackReply::failure(CallbackFailure::default()))
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().name("cb").await?;
                    Ok::<_, BoxError>(cb.result().await?)
                },
                (),
            )
            .await;

        assert!(result.is_failure(), "{:?}", result.error_message());
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), "Failed");
        assert_eq!(cb_op.error_type(), None);
        assert_eq!(cb_op.error_message(), None);
    }

    /// A failure that sets only the error data or only the stack trace
    /// still reaches the handler with that field (see `has_error_fields`).
    #[tokio::test]
    async fn callback_reply_failure_with_only_data_or_trace_keeps_it() {
        let read = |failure: CallbackFailure| async move {
            LocalRunner::new()
                .callback(CallbackReply::failure(failure))
                .run(
                    |(), ctx: DurableContext| async move {
                        let cb = ctx.create_callback::<String>().name("cb").await?;
                        let Err(err) = cb.result().await else {
                            return Err::<(Option<String>, Vec<String>), BoxError>(
                                "callback succeeded".into(),
                            );
                        };
                        let wire = err.wire();
                        Ok((
                            wire.and_then(crate::WireError::error_data)
                                .map(ToOwned::to_owned),
                            wire.map(|w| w.stack_trace().to_vec()).unwrap_or_default(),
                        ))
                    },
                    (),
                )
                .await
        };

        let data_only = read(CallbackFailure::default().with_error_data("d")).await;
        assert_eq!(
            data_only.output(),
            Some(&(Some("d".to_owned()), Vec::new()))
        );
        let trace_only =
            read(CallbackFailure::default().with_stack_trace(vec!["at a".to_owned()])).await;
        assert_eq!(trace_only.output(), Some(&(None, vec!["at a".to_owned()])));
    }

    #[tokio::test]
    async fn callback_reply_heartbeat_timeout_reaches_handler() {
        let result = LocalRunner::new()
            .callback(CallbackReply::heartbeat_timeout())
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx
                        .create_callback::<u32>()
                        .name("cb")
                        .heartbeat(std::time::Duration::from_secs(60))
                        .await?;
                    let Err(err) = cb.result().await else {
                        return Err::<bool, BoxError>("callback succeeded".into());
                    };
                    Ok(matches!(
                        err.kind(),
                        crate::OperationErrorKind::Callback(e)
                            if matches!(e.kind(), crate::CallbackErrorKind::HeartbeatTimedOut)
                    ))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&true));
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), "TimedOut");
        assert_eq!(cb_op.error_type(), Some("Callback.Heartbeat"));
        assert_eq!(
            cb_op.error_message(),
            Some("Callback timed out on heartbeat")
        );
    }

    /// A value whose `Serialize` impl always fails.
    struct Refuses;

    impl Serialize for Refuses {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("refuse"))
        }
    }

    #[test]
    fn callback_reply_success_returns_the_encoding_error() {
        let err = CallbackReply::success(&Refuses).expect_err("encoding fails");
        assert!(err.to_string().contains("refuse"), "{err}");
    }

    /// A callback result created via the public `ctx.create_callback` API
    /// composes with a real step future inside `try_join_all`. This exercises
    /// the full builder→callback→combinator path through replay, ensuring
    /// regressions in the public surface are caught.
    #[tokio::test]
    async fn callback_result_composes_with_step_in_combinator() {
        let result = LocalRunner::new()
            .callback(success(&"from-callback".to_owned()))
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

    // 7. replay determinism

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

    // CloudRunner pure-mapping tests (no AWS)

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
    fn fold_history_maps_callback_timed_out() {
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

    // Pagination tests

    /// A two-page initial state replays all operations: the runner calls
    /// `get_state` when the history exceeds `initial_page_size`, and the
    /// handler sees results from prior invocations without re-executing.
    #[tokio::test]
    async fn two_page_initial_state_replays_all_operations() {
        // Use a static counter to detect re-execution of the first step.
        static FIRST_STEP_EXECUTIONS: AtomicU32 = AtomicU32::new(0);
        FIRST_STEP_EXECUTIONS.store(0, Ordering::SeqCst);

        let result = LocalRunner::new()
            .initial_page_size(1) // Only 1 op per "page" in initial state
            .run(
                |_event: (), ctx: DurableContext| async move {
                    // First step: should only execute on the first invocation.
                    let a = ctx
                        .step(|_| async {
                            FIRST_STEP_EXECUTIONS.fetch_add(1, Ordering::SeqCst);
                            Ok(10_i32)
                        })
                        .name("first")
                        .await?;

                    // Wait triggers a re-invocation boundary.
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("pause")
                        .await?;

                    // Second step on re-invocation: initial state now has
                    // 2+ ops, exceeding page_size=1. The runner must fetch
                    // the full state via get_state so `a` replays correctly.
                    let b = ctx
                        .step(move |_| async move { Ok(a + 32) })
                        .name("second")
                        .await?;

                    Ok::<_, BoxError>(b)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&42));
        // The first step must execute exactly once (not re-executed on replay).
        assert_eq!(
            FIRST_STEP_EXECUTIONS.load(Ordering::SeqCst),
            1,
            "first step must execute exactly once; pagination must not cause re-execution"
        );
        // Should take 2 invocations: first runs both steps+wait, second replays
        // the wait (now elapsed) and runs the second step.
        assert!(
            result.invocation_count() >= 2,
            "expected at least 2 invocations"
        );
    }

    /// A two-page checkpoint response is fully consumed: when the backend
    /// returns `next_marker` in a checkpoint response, `checkpoint_updates`
    /// calls `get_state` to merge remaining operations, ensuring subsequent
    /// reads see all state.
    #[tokio::test]
    async fn two_page_checkpoint_response_is_fully_consumed() {
        // When checkpoint_page_size is 1, the second checkpoint response
        // (which sees 2+ stored ops) includes a marker, triggering
        // pagination in checkpoint_updates.
        let result = LocalRunner::new()
            .checkpoint_page_size(1) // Trigger pagination after 1 stored op
            .run(
                |_event: (), ctx: DurableContext| async move {
                    // First step: creates 1 stored op. Checkpoint response
                    // has no marker (1 op <= page_size=1 threshold).
                    let a = ctx.step(|_| async { Ok(10_i32) }).name("step-a").await?;

                    // Second step: creates 2nd stored op. The checkpoint
                    // response now returns next_marker (2 ops > 1), so
                    // checkpoint_updates must call get_state to merge all
                    // operations.
                    let b = ctx
                        .step(move |_| async move { Ok(a + 5) })
                        .name("step-b")
                        .await?;

                    // Third step proves subsequent operations see the full
                    // state, including operations merged via get_state.
                    let c = ctx
                        .step(move |_| async move { Ok(b + 27) })
                        .name("step-c")
                        .await?;

                    Ok::<_, BoxError>(c)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&42));
        assert_eq!(result.operations().len(), 3);
    }

    /// A truncated checkpoint page genuinely requires the second page: the
    /// backend returns only the FIRST stored operation in the callback
    /// Start checkpoint response (plus a marker), so the backend-assigned
    /// `callback_id` is observable only through the follow-up `get_state`
    /// merge in `checkpoint_updates`. If the marker were ignored, the
    /// handler would see an empty callback id and fail.
    #[tokio::test]
    async fn truncated_checkpoint_page_requires_second_page() {
        let result = LocalRunner::new()
            .checkpoint_page_size(1)
            .callback(success(&"approved".to_owned()))
            .run(
                |_event: (), ctx: DurableContext| async move {
                    // First stored operation: fills page 1.
                    let base = ctx.step(|_| async { Ok(1_i32) }).name("filler").await?;

                    // Second stored operation: the callback. Its Start
                    // checkpoint response is truncated to page 1 (the
                    // filler step), so the callback_id below arrives only
                    // via the paginated get_state merge.
                    let cb = ctx.create_callback::<String>().name("approval").await?;
                    if cb.id().is_empty() {
                        return Err::<String, BoxError>(
                            "callback id missing: second checkpoint page was not consumed".into(),
                        );
                    }

                    let approval = cb.result().await?;
                    Ok(format!("{base}:{approval}"))
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&"1:approved".to_owned()));
    }

    /// Combining both pagination modes: a handler with paginated initial
    /// state AND paginated checkpoint responses still runs correctly.
    #[tokio::test]
    async fn combined_initial_and_checkpoint_pagination() {
        let result = LocalRunner::new()
            .initial_page_size(1)
            .checkpoint_page_size(2)
            .run(
                |_event: (), ctx: DurableContext| async move {
                    let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("timer")
                        .await?;
                    let b = ctx
                        .step(move |_| async move { Ok(a + 1) })
                        .name("b")
                        .await?;
                    let c = ctx
                        .step(move |_| async move { Ok(b + 1) })
                        .name("c")
                        .await?;
                    Ok::<_, BoxError>(c)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&3));
    }

    /// Tests that the Backend with `checkpoint_page_size` returns a
    /// `next_marker` when operations exceed the page size, simulating
    /// a two-page checkpoint response. The caller (`DurableContext`)
    /// should then paginate via `get_state`.
    #[tokio::test]
    async fn backend_paginated_checkpoint_returns_marker() {
        let backend = Arc::new(Backend::new(QueuedReplies::default(), Some(1)));

        // Checkpoint two operations so we exceed page_size=1.
        let updates = vec![
            OperationUpdate::builder()
                .id("op-1")
                .r#type(OperationType::Step)
                .action(OperationAction::Start)
                .build()
                .unwrap(),
            OperationUpdate::builder()
                .id("op-2")
                .r#type(OperationType::Step)
                .action(OperationAction::Start)
                .build()
                .unwrap(),
        ];

        let result = backend.checkpoint("arn:test", "token-0", updates).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(
            output.next_marker.is_some(),
            "expected a next_marker when operations exceed page_size"
        );

        // Verify that get_state returns all operations (used for pagination).
        let full_state = backend.get_state("arn:test", "token-0", None).await;
        assert!(full_state.is_ok());
        let full_state = full_state.unwrap();
        assert_eq!(
            full_state.operations.len(),
            2,
            "get_state must return all operations for pagination"
        );
    }

    /// Tests that the shared production bootstrap helper
    /// (`resolve_bootstrap_log`) follows the pagination marker: given a
    /// truncated first page plus a marker, it fetches the complete state
    /// via `get_state`; without a marker it uses the first page as-is and
    /// never calls `get_state`.
    #[tokio::test]
    async fn two_page_bootstrap_replays_all_operations() {
        use crate::client::{
            InMemoryExecutionClient, operations_to_checkpoint_log, resolve_bootstrap_log,
        };

        // Full backend state: two steps. Page 1 delivers only step-1.
        let all_ops = vec![
            Operation::builder()
                .id("step-1")
                .r#type(OperationType::Step)
                .status(OperationStatus::Succeeded)
                .start_timestamp(DateTime::from_secs(0))
                .step_details(StepDetails::builder().result("\"page1-result\"").build())
                .build()
                .unwrap(),
            Operation::builder()
                .id("step-2")
                .r#type(OperationType::Step)
                .status(OperationStatus::Succeeded)
                .start_timestamp(DateTime::from_secs(0))
                .step_details(StepDetails::builder().result("\"page2-result\"").build())
                .build()
                .unwrap(),
        ];

        let first_page: Vec<Operation> = all_ops.get(..1).unwrap_or(&all_ops).to_vec();

        let client = Arc::new(InMemoryExecutionClient::new(all_ops));

        // Marker present: the helper must fetch the complete state.
        let log = resolve_bootstrap_log(
            client.as_ref(),
            "arn:test",
            "token",
            operations_to_checkpoint_log(&first_page),
            crate::client::PageInput::Execution(Some("{}".to_owned())),
            Some("marker-1"),
        )
        .await
        .unwrap()
        .0;

        assert!(
            log.get("step-1").is_some(),
            "step-1 from page 1 must be in the log"
        );
        assert!(
            log.get("step-2").is_some(),
            "step-2 from page 2 must be in the log"
        );

        let r1 = log.get("step-1").unwrap();
        assert_eq!(r1.result.as_deref(), Some("\"page1-result\""));

        let r2 = log.get("step-2").unwrap();
        assert_eq!(r2.result.as_deref(), Some("\"page2-result\""));

        let calls_after_marker = *client
            .get_state_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            calls_after_marker, 1,
            "marker must trigger exactly one get_state fetch"
        );

        // No marker: the first page is complete, no get_state call, and
        // the log holds only page-1 operations.
        let log = resolve_bootstrap_log(
            client.as_ref(),
            "arn:test",
            "token",
            operations_to_checkpoint_log(&first_page),
            crate::client::PageInput::Execution(Some("{}".to_owned())),
            None,
        )
        .await
        .unwrap()
        .0;

        assert!(log.get("step-1").is_some());
        assert!(
            log.get("step-2").is_none(),
            "without a marker the helper must not fetch further pages"
        );
        let calls_after_no_marker = *client
            .get_state_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            calls_after_no_marker, 1,
            "no-marker path must not call get_state again"
        );
    }

    // Harness fidelity: production task topology

    /// The runner awaits the handler INLINE under the caller's `block_on`
    /// (the production `lambda_runtime` topology), so the context is
    /// created where `tokio::task::try_id()` is `None`, not on a spawned
    /// task. This is the dimension that previously hid the ownership
    /// guard's production inertness.
    #[tokio::test]
    async fn runner_topology_matches_production_inline_block_on() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    // Same task visibility production gives the handler:
                    // awaited inline under block_on → no task ID.
                    let inline = tokio::task::try_id().is_none();
                    let v = ctx.step(|_| async { Ok(1_i32) }).await?;
                    Ok::<_, BoxError>((inline, v))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&(true, 1)),
            "handler must run inline (try_id() == None), matching the deployed topology"
        );
    }

    /// Under the production topology, a durable operation invoked from a
    /// bare user `tokio::spawn` (unblessed) is rejected by the ownership
    /// guard: the runner must reproduce that, not mask it.
    #[tokio::test]
    async fn runner_rejects_durable_ops_from_unblessed_spawned_task() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let foreign = ctx.clone();
                    let joined: Result<i32, String> = tokio::spawn(async move {
                        foreign
                            .step(|_| async { Ok(7_i32) })
                            .await
                            // `{:#}` flattens the frame and its chain.
                            .map_err(|e| format!("{e:#}"))
                    })
                    .await
                    .map_err(|e| e.to_string())?;
                    let v = joined.map_err(BoxError::from)?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(
            result.is_failure(),
            "unblessed tokio::spawn must fail the ownership check as it does in production"
        );
        let msg = result.error_message().unwrap_or_default();
        assert!(
            msg.contains("Use .spawn()"),
            "ownership rejection should carry the production guidance: {msg}"
        );
    }

    // Harness fidelity: state pagination is the DEFAULT

    /// By default the backend serves execution state in 2+ pages: a
    /// re-invocation with two or more recorded operations gets a truncated
    /// inline envelope page plus `NextMarker`, and checkpoint responses
    /// paginate: both force real `get_state` fetches through the
    /// production pagination paths.
    #[tokio::test]
    async fn default_state_pagination_forces_get_state_fetches() {
        let runner = LocalRunner::new();
        let backend = Arc::new(Backend::new(
            QueuedReplies::default(),
            runner.checkpoint_page_size,
        ));

        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                |(), ctx: DurableContext| async move {
                    let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    // Suspend so the next invocation bootstraps from a
                    // multi-operation (hence multi-page) history.
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("cooldown")
                        .await?;
                    let b = ctx
                        .step(move |_| async move { Ok(a + 1) })
                        .name("b")
                        .await?;
                    Ok::<_, BoxError>(b)
                },
                "null".to_owned(),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&2));
        assert!(
            result.invocation_count() >= 2,
            "wait must force a re-invocation"
        );
        assert!(
            backend.get_state_call_count() >= 2,
            "multi-page default must force get_state fetches for both the bootstrap \
             (envelope NextMarker) and checkpoint pagination paths; saw {}",
            backend.get_state_call_count()
        );
    }

    /// `single_page()` is the explicit special case: the full history rides
    /// inline in every envelope and no checkpoint response paginates, so
    /// the SDK never needs a `get_state` fetch.
    #[tokio::test]
    async fn single_page_never_fetches_state() {
        let runner = LocalRunner::new().single_page();
        let backend = Arc::new(Backend::new(
            QueuedReplies::default(),
            runner.checkpoint_page_size,
        ));

        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                |(), ctx: DurableContext| async move {
                    let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("cooldown")
                        .await?;
                    let b = ctx
                        .step(move |_| async move { Ok(a + 1) })
                        .name("b")
                        .await?;
                    Ok::<_, BoxError>(b)
                },
                "null".to_owned(),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            backend.get_state_call_count(),
            0,
            "single-page mode must never require a get_state fetch"
        );
    }

    // Checkpoint coalescing (`checkpoint_delay`)

    /// A handler with two concurrently-spawned steps, a suspension, and a
    /// post-resume step. `executions` counts step-body runs so replay
    /// fidelity is observable from outside.
    fn coalescing_probe_handler(
        executions: Arc<std::sync::atomic::AtomicUsize>,
    ) -> impl Fn((), DurableContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
    + Send
    + Sync
    + 'static {
        move |(), ctx: DurableContext| {
            let executions = Arc::clone(&executions);
            Box::pin(async move {
                let e1 = Arc::clone(&executions);
                let e2 = Arc::clone(&executions);
                let e3 = Arc::clone(&executions);

                // Two concurrent steps: their START checkpoints (and their
                // SUCCEED checkpoints) land within one coalescing window.
                let a = ctx
                    .step(move |_| {
                        let e = e1;
                        async move {
                            e.fetch_add(1, Ordering::SeqCst);
                            Ok(1_i32)
                        }
                    })
                    .name("a")
                    .spawn();
                let b = ctx
                    .step(move |_| {
                        let e = e2;
                        async move {
                            e.fetch_add(1, Ordering::SeqCst);
                            Ok(2_i32)
                        }
                    })
                    .name("b")
                    .spawn();
                let (ra, rb) = tokio::join!(a, b);
                let sum = ra? + rb?;

                // Suspension boundary: the coalesced SUCCEED checkpoints
                // above must have landed by now, or replay after resume
                // would re-execute the bodies.
                ctx.wait(std::time::Duration::from_secs(1))
                    .name("pause")
                    .await?;

                let c = ctx
                    .step(move |_| {
                        let e = e3;
                        async move {
                            e.fetch_add(1, Ordering::SeqCst);
                            Ok(sum + 39)
                        }
                    })
                    .name("c")
                    .await?;
                Ok::<_, BoxError>(c)
            })
        }
    }

    /// `checkpoint_delay` coalesces concurrent checkpoints into fewer API
    /// calls, and everything that must land before the suspension still
    /// does: the spawned step bodies run exactly once across the
    /// suspend/resume boundary, proving their coalesced checkpoints
    /// flushed before the invocation reported PENDING.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_delay_coalesces_and_flushes_before_suspension() {
        // Baseline: identical handler, no coalescing. Batching is on by
        // default, so the baseline opts out to stay the immediate-write
        // baseline.
        let baseline_execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let baseline_runner = LocalRunner::new().single_page().options(
            crate::Options::builder()
                .checkpoint_batching(false)
                .build()
                .expect("valid options"),
        );
        let baseline_backend = Arc::new(Backend::new(QueuedReplies::default(), None));
        let baseline = baseline_runner
            .run_on_backend(
                Arc::clone(&baseline_backend),
                coalescing_probe_handler(Arc::clone(&baseline_execs)),
                "null".to_owned(),
            )
            .await;
        assert!(baseline.is_success(), "{:?}", baseline.error_message());
        assert_eq!(baseline.output(), Some(&42));

        // Coalesced: same handler with a checkpoint delay window.
        let coalesced_execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let coalesced_runner = LocalRunner::new().single_page().options(
            crate::Options::builder()
                .checkpoint_delay(std::time::Duration::from_millis(50))
                .build()
                .expect("valid options"),
        );
        let coalesced_backend = Arc::new(Backend::new(QueuedReplies::default(), None));
        let coalesced = coalesced_runner
            .run_on_backend(
                Arc::clone(&coalesced_backend),
                coalescing_probe_handler(Arc::clone(&coalesced_execs)),
                "null".to_owned(),
            )
            .await;
        assert!(coalesced.is_success(), "{:?}", coalesced.error_message());
        assert_eq!(
            coalesced.output(),
            Some(&42),
            "coalescing must not change the handler's result"
        );

        // Coalescing must have merged the concurrent steps' checkpoints
        // into fewer API calls than the immediate-write baseline.
        let baseline_calls = baseline_backend.checkpoint_call_count();
        let coalesced_calls = coalesced_backend.checkpoint_call_count();
        assert!(
            coalesced_calls < baseline_calls,
            "expected fewer checkpoint calls with coalescing: \
             coalesced={coalesced_calls}, baseline={baseline_calls}"
        );

        // Flush-before-suspension: each step body ran exactly once. A
        // checkpoint held past the PENDING boundary would force the
        // resumed invocation to re-execute the body.
        assert_eq!(
            coalesced_execs.load(Ordering::SeqCst),
            3,
            "each step body must run exactly once across suspend/resume"
        );
        assert_eq!(
            baseline_execs.load(Ordering::SeqCst),
            3,
            "baseline sanity: each step body runs exactly once"
        );
    }

    /// A runner that sends each checkpoint in its own request.
    fn unbatched_runner() -> LocalRunner {
        LocalRunner::new().single_page().options(
            crate::Options::builder()
                .checkpoint_batching(false)
                .build()
                .expect("valid options"),
        )
    }

    /// Runs `handler` on a retained backend and returns the number of
    /// checkpoint calls the backend received.
    async fn checkpoint_calls<O, F, Fut>(runner: &LocalRunner, handler: F) -> usize
    where
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let backend = Arc::new(Backend::new(
            QueuedReplies::default(),
            runner.checkpoint_page_size,
        ));
        let result = runner
            .run_on_backend(Arc::clone(&backend), handler, "null".to_owned())
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        backend.checkpoint_call_count()
    }

    /// A runner built with `new()` batches checkpoints: concurrent steps
    /// send fewer checkpoint calls than with `checkpoint_batching(false)`.
    #[tokio::test(start_paused = true)]
    async fn new_runner_batches_by_default() {
        let execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let default_calls = checkpoint_calls(
            &LocalRunner::new().single_page(),
            coalescing_probe_handler(Arc::clone(&execs)),
        )
        .await;
        let unbatched_calls = checkpoint_calls(
            &unbatched_runner(),
            coalescing_probe_handler(Arc::clone(&execs)),
        )
        .await;
        assert!(
            default_calls < unbatched_calls,
            "the default runner must batch: default={default_calls}, \
             unbatched={unbatched_calls}"
        );
    }

    /// A sequential handler has no concurrent writes to batch, so it sends
    /// the same number of checkpoint calls with and without batching.
    #[tokio::test(start_paused = true)]
    async fn sequential_handler_sends_one_request_per_checkpoint() {
        let handler = |(), ctx: DurableContext| async move {
            let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
            let b = ctx
                .step(move |_| async move { Ok(a + 1) })
                .name("b")
                .await?;
            Ok::<_, BoxError>(b)
        };
        let default_calls = checkpoint_calls(&LocalRunner::new().single_page(), handler).await;
        let unbatched_calls = checkpoint_calls(&unbatched_runner(), handler).await;
        assert_eq!(
            default_calls, unbatched_calls,
            "a sequential handler must send one request per checkpoint"
        );
    }

    /// A transport that fails every call and counts how often it was used.
    struct CountingClient {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ExecutionClient for CountingClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(ExecutionClientError::new(
                    crate::client::ExecutionClientErrorScope::Execution,
                    "must not be called",
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
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(ExecutionClientError::new(
                    crate::client::ExecutionClientErrorScope::Execution,
                    "must not be called",
                ))
            })
        }
    }

    /// A one-step handler that returns 42.
    async fn one_step_handler((): (), ctx: DurableContext) -> Result<i32, BoxError> {
        Ok(ctx.step(|_| async { Ok(42_i32) }).name("only").await?)
    }

    /// The runner replaces the transport in the supplied `Options` with its
    /// in-memory backend, so a test can pass its production `Options`.
    #[tokio::test]
    async fn runner_ignores_the_transport_in_options() {
        // Case 1: an offline Lambda client. It has no credentials, so a run
        // that used it would fail.
        let conf = aws_sdk_lambda::config::Config::builder()
            .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
            .region(aws_sdk_lambda::config::Region::new("us-east-1"))
            .build();
        let lambda_options = crate::Options::builder()
            .lambda_client(aws_sdk_lambda::Client::from_conf(conf))
            .build()
            .expect("valid options");
        let result = LocalRunner::new()
            .options(lambda_options)
            .run(one_step_handler, ())
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&42));

        // Case 2: a caller-supplied execution client that fails every call.
        let counting = Arc::new(CountingClient {
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let client_options = crate::Options::builder()
            .execution_client(Arc::clone(&counting) as Arc<dyn ExecutionClient>)
            .build()
            .expect("valid options");
        let result = LocalRunner::new()
            .options(client_options)
            .run(one_step_handler, ())
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&42));
        assert_eq!(
            counting.calls.load(Ordering::SeqCst),
            0,
            "the runner must never call the supplied execution client"
        );
    }

    /// `checkpoint_delay` with a whole-handler completion: the terminal
    /// SUCCEEDED envelope is only reported after the end-of-invocation
    /// flush drains the buffer, so a sequential handler under a large
    /// delay window still completes with all operations recorded.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_delay_completes_sequential_handlers() {
        let result = LocalRunner::new()
            .single_page()
            .options(
                crate::Options::builder()
                    .checkpoint_delay(std::time::Duration::from_millis(20))
                    .build()
                    .expect("valid options"),
            )
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
        // The step's START and SUCCEED records both landed.
        assert!(
            result
                .operations()
                .iter()
                .any(|op| op.status == "Succeeded" && op.name.as_deref() == Some("inc")),
            "the step's coalesced checkpoints must be recorded: {:?}",
            result.operations()
        );
    }

    /// `checkpoint_batching` preserves handler semantics end-to-end: the
    /// same probe handler (concurrent steps, a suspension, a post-resume
    /// step) completes with the same output, and every step body runs
    /// exactly once across the suspend/resume boundary: proving the
    /// batched checkpoints flushed (and any in-flight batched write was
    /// awaited) before the invocation reported PENDING.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_batching_preserves_handler_semantics() {
        let execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend = Arc::new(Backend::new(QueuedReplies::default(), None));
        let result = LocalRunner::new()
            .single_page()
            .options(
                crate::Options::builder()
                    .checkpoint_batching(true)
                    .build()
                    .expect("valid options"),
            )
            .run_on_backend(
                Arc::clone(&backend),
                coalescing_probe_handler(Arc::clone(&execs)),
                "null".to_owned(),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&42),
            "batching must not change the handler's result"
        );
        assert_eq!(
            execs.load(Ordering::SeqCst),
            3,
            "each step body must run exactly once across suspend/resume — \
             a checkpoint held past the PENDING boundary would re-execute it"
        );
    }

    /// Review regression (issue #43): a failure RETAINED by a detached
    /// batch flush, its only contributor was dropped before the rejection
    /// published, must not be discarded, and the orphaned operation's
    /// unwritten outcome must be terminalized. The retained rejection here
    /// is NON-retryable (deterministic on every future invocation): were
    /// it dropped, the orphaned operation would stay `Started` and
    /// re-execute on every lap.
    ///
    /// With the coalescer's failure latch, the very next buffered write,
    /// the later live step's START, never reaches the backend: its
    /// flusher observes the latch under the writer lock, republishes the
    /// retained non-retryable error, and the live contributor's
    /// unrecoverable routing fails the execution. The end-of-invocation
    /// flush point then classifies the retained failures and persists the
    /// orphan's terminal FAIL. (The `Fault`-path retained-failure drain in
    /// the wrapper remains as defense in depth: with the latch, a
    /// non-retryable retained failure surfaces through the first
    /// subsequent contributor as here, before any retryable fault can.)
    #[tokio::test]
    async fn retained_nonretryable_failure_terminalizes_orphan_and_fails_execution() {
        let runner = LocalRunner::new().options(
            crate::Options::builder()
                .checkpoint_batching(true)
                .build()
                .expect("valid options"),
        );
        let backend = Arc::new(Backend::new(
            QueuedReplies::default(),
            runner.checkpoint_page_size,
        ));

        // Call plan:
        //   1. orphan START: passes.
        //   2. orphan SUCCEED (detached batch flush): held at the gate
        //      until the handler releases it, then rejected non-retryably.
        //      The handler drops the contributor while the write is held,
        //      so the rejection publishes to nobody and is only RETAINED
        //      (and latched).
        // The live step's START never reaches the backend (failure
        // latch), so the next real call is the orphan's terminal FAIL:
        // it falls past the exhausted plan and persists.
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        backend.plan_checkpoint_calls(vec![
            PlannedCheckpoint::Pass,
            PlannedCheckpoint::FailNonRetryable {
                gate: Some(Arc::clone(&gate)),
            },
        ]);

        let body_runs = Arc::new(AtomicU32::new(0));
        let body_runs_h = Arc::clone(&body_runs);
        let backend_h = Arc::clone(&backend);
        let gate_h = Arc::clone(&gate);

        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                move |(), ctx: DurableContext| {
                    let body_runs = Arc::clone(&body_runs_h);
                    let backend = Arc::clone(&backend_h);
                    let gate = Arc::clone(&gate_h);
                    async move {
                        let body_runs_step = Arc::clone(&body_runs);
                        let orphan = ctx
                            .step(move |_| {
                                let body_runs = Arc::clone(&body_runs_step);
                                async move {
                                    body_runs.fetch_add(1, Ordering::SeqCst);
                                    Ok("orphaned-outcome".to_owned())
                                }
                            })
                            .name("orphan")
                            .spawn();

                        // Wait until the orphan's SUCCEED write is IN
                        // FLIGHT (call 2 reached the backend, held at the
                        // gate), then drop the contributor and release the
                        // gate: the rejection now publishes to nobody and
                        // is only retained by the coalescer.
                        while backend.checkpoint_call_count() < 2 {
                            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        }
                        drop(orphan);
                        gate.add_permits(1);

                        // A later LIVE operation: its START write hits the
                        // failure latch, no backend call, and republishes
                        // the retained non-retryable rejection, whose
                        // unrecoverable routing fails the execution. Before
                        // the retention + latch, the retained failure above
                        // was silently discarded.
                        let live: String = ctx
                            .step(|_| async { Ok("live".to_owned()) })
                            .name("live")
                            .await?;
                        Ok::<_, BoxError>(live)
                    }
                },
                "null".to_owned(),
            )
            .await;

        assert_eq!(
            body_runs.load(Ordering::SeqCst),
            1,
            "the orphaned body must run exactly once — a discarded retained \
             rejection would leave it `Started` and re-execute it"
        );
        assert_eq!(
            result.invocation_count(),
            1,
            "the retained non-retryable rejection is deterministic: the \
             execution must die in this invocation, not re-invoke into the \
             same rejection"
        );
        assert!(
            result.is_failure(),
            "the execution must fail — the orphaned operation's record \
             would otherwise claim less than what executed; got {:?} / {:?}",
            result.error_type(),
            result.error_message()
        );
        assert_eq!(
            result.error_type(),
            Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
        );

        // The terminal FAIL persisted for the orphaned operation.
        let ops = result.operations();
        let orphan_op = ops
            .iter()
            .find(|op| op.name.as_deref() == Some("orphan"))
            .expect("the orphaned step's operation record exists");
        assert_eq!(
            orphan_op.status, "Failed",
            "a terminal FAIL must be recorded for the operation whose \
             retained outcome rejection was classified at the flush point"
        );
    }

    // Harness fidelity: production wire-error mapping

    /// The runner reports the SAME wire error message production reports.
    /// Every failure flattens through the one flattening site, so a child
    /// failure's wire message is the full chain, one frame per layer,
    /// with the raw child message as its final frame. (The old special
    /// case that reported only the raw message was the Display/wire
    /// asymmetry the error-model redesign removed.)
    #[tokio::test]
    async fn child_failure_wire_error_matches_production_raw_message() {
        let result: TestResult<i32> = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let v = ctx
                        .run_in_child_context(|_child| async move {
                            Err::<i32, BoxError>("boom-child".into())
                        })
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        // The execution record re-records the child-context record's own
        // recorded identity ("Error", the body error's fallback type for a
        // plain boxed error) rather than degrading it to the outer kind's
        // registry name: recorded identity passes through boundaries.
        assert_eq!(result.error_type(), Some("Error"));
        let msg = result.error_message().unwrap_or_default();
        assert!(
            msg.contains("boom-child"),
            "raw child message must survive as the chain's final frame: {msg}"
        );
        assert_eq!(
            msg, "operation error: child_context: child failed: boom-child",
            "wire message is the flattened chain, one frame per layer"
        );
    }

    // ---- Virtual clock (waits and retry delays end in time order) ----

    /// The shorter of two raced waits wins, because the clock moves to the
    /// earliest end and fires only what is due then.
    #[tokio::test]
    async fn virtual_clock_race_returns_shorter_wait() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let slow = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(std::time::Duration::from_secs(60)).await?;
                            Ok("slow".to_owned())
                        })
                        .name("slow")
                        .future();
                    let fast = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(std::time::Duration::from_secs(1)).await?;
                            Ok("fast".to_owned())
                        })
                        .name("fast")
                        .future();
                    Ok::<_, BoxError>(ctx.race([slow, fast]).await?)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("fast"));
        assert_eq!(result.invocation_count(), 2);
    }

    /// A retry delayed by 300 seconds loses a race against a 1 second wait.
    ///
    /// The runner sends each checkpoint in its own request, so the test
    /// fails on a backend that fires every timer in one advance. Such a
    /// backend fires the retry and the wait together. The next invocation
    /// re-runs the step body while the wait child replays, and with
    /// unbatched checkpoints the step's child settles first, so the race
    /// returns `"retried"`.
    #[tokio::test]
    async fn virtual_clock_orders_retry_delay_against_wait() {
        let result = LocalRunner::new()
            .options(
                crate::Options::builder()
                    .checkpoint_batching(false)
                    .build()
                    .expect("valid options"),
            )
            .run(
                |(), ctx: DurableContext| async move {
                    let retried = ctx
                        .run_in_child_context(|child| async move {
                            child
                                .step(|step_ctx| async move {
                                    if step_ctx.attempt() == 1 {
                                        return Err::<String, BoxError>("first attempt".into());
                                    }
                                    Ok("retried".to_owned())
                                })
                                .retry_strategy(|_err, _attempt| RetryDecision::Retry {
                                    delay: std::time::Duration::from_secs(300),
                                })
                                .await
                                .map_err(BoxError::from)
                        })
                        .future();
                    let waited = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(std::time::Duration::from_secs(1)).await?;
                            Ok("waited".to_owned())
                        })
                        .future();
                    Ok::<_, BoxError>(ctx.race([retried, waited]).await?)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("waited"));
        assert_eq!(result.invocation_count(), 2);
    }

    /// The clock orders timers by due time, not by duration: a 6 second
    /// wait started at 10 ends at 16, after a 15 second wait started at 0.
    #[tokio::test]
    async fn virtual_clock_orders_by_due_time_not_duration() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let a = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(std::time::Duration::from_secs(10)).await?;
                            child.wait(std::time::Duration::from_secs(6)).await?;
                            Ok("a".to_owned())
                        })
                        .name("a")
                        .future();
                    let b = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(std::time::Duration::from_secs(15)).await?;
                            Ok("b".to_owned())
                        })
                        .name("b")
                        .future();
                    Ok::<_, BoxError>(ctx.race([a, b]).await?)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("b"));
        assert_eq!(result.invocation_count(), 3);
    }

    /// Waits due at the same time fire in the same advance.
    #[tokio::test]
    async fn virtual_clock_fires_equal_waits_together() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let first = ctx.wait(std::time::Duration::from_secs(10)).spawn();
                    let second = ctx.wait(std::time::Duration::from_secs(10)).spawn();
                    let (first, second) = tokio::join!(first, second);
                    first?;
                    second?;
                    Ok("done".to_owned())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("done"));
        assert_eq!(result.invocation_count(), 2);
    }

    /// A queued callback reply arrives before the clock moves, so it wins
    /// a race against an hour-long wait, and the wait never fires.
    #[tokio::test]
    async fn virtual_clock_delivers_callback_before_moving_clock() {
        let result = LocalRunner::new()
            .callback(success(&"cb"))
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().await?;
                    let timeout = ctx
                        .run_in_child_context(|child| async move {
                            child
                                .wait(std::time::Duration::from_secs(3600))
                                .name("hour")
                                .await?;
                            Ok("timeout".to_owned())
                        })
                        .future();
                    Ok::<_, BoxError>(ctx.race([cb.result(), timeout]).await?)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("cb"));
        assert_eq!(result.invocation_count(), 2);
        let hour = result
            .operations()
            .iter()
            .find(|op| op.name() == Some("hour"))
            .expect("the hour wait is recorded");
        assert_eq!(hour.status(), "Started");
    }

    /// A WAIT `START` update with the given duration.
    fn wait_start(id: &str, wait_seconds: i32) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Wait)
            .action(OperationAction::Start)
            .wait_options(
                aws_sdk_lambda::types::WaitOptions::builder()
                    .wait_seconds(wait_seconds)
                    .build(),
            )
            .build()
            .expect("valid update")
    }

    /// A STEP `RETRY` update, with `NextAttemptDelaySeconds` when `delay`
    /// is set.
    fn step_retry(id: &str, delay: Option<i32>) -> OperationUpdate {
        let mut builder = OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Step)
            .action(OperationAction::Retry);
        if let Some(delay) = delay {
            builder = builder.step_options(
                aws_sdk_lambda::types::StepOptions::builder()
                    .next_attempt_delay_seconds(delay)
                    .build(),
            );
        }
        builder.build().expect("valid update")
    }

    /// A CALLBACK `START` update.
    fn callback_start(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Callback)
            .action(OperationAction::Start)
            .build()
            .expect("valid update")
    }

    /// A backend at virtual time `now`, seeded with `updates`.
    fn seeded_backend(
        callback_replies: Vec<CallbackReply>,
        now: u64,
        updates: &[OperationUpdate],
    ) -> Backend {
        let backend = Backend::new(
            QueuedReplies {
                fifo: callback_replies.into(),
                named: Vec::new(),
            },
            None,
        );
        {
            let mut state = backend.lock();
            state.now = now;
            for update in updates {
                state.apply_update(update).expect("update applies");
            }
        }
        backend
    }

    /// The status of stored op `id`, and the backend's virtual time.
    fn status_and_now(backend: &Backend, id: &str) -> (OperationStatus, u64) {
        let state = backend.lock();
        let op = state
            .ops
            .iter()
            .find(|op| op.id == id)
            .expect("op is stored");
        (op.status.clone(), state.now)
    }

    /// A callback `START` stores `Started`, but `advance` still settles a
    /// callback whose status is `Pending`. This test sets the status to
    /// `Pending` directly, because no update produces it.
    #[test]
    fn advance_settles_pending_callback() {
        let backend = seeded_backend(vec![CallbackReply::raw("\"x\"")], 0, &[callback_start("c")]);
        backend
            .lock()
            .ops
            .iter_mut()
            .find(|op| op.id == "c")
            .expect("op is stored")
            .status = OperationStatus::Pending;
        assert!(backend.advance());
        assert_eq!(
            status_and_now(&backend, "c"),
            (OperationStatus::Succeeded, 0)
        );
    }

    /// `advance` follows the virtual clock: one assertion block per row of
    /// the case table in the issue.
    #[test]
    fn advance_follows_virtual_clock() {
        // Rows 1 and 2: the earlier wait fires first, then the later one.
        let backend = seeded_backend(Vec::new(), 0, &[wait_start("a", 5), wait_start("b", 9)]);
        assert!(backend.advance(), "row 1");
        assert_eq!(
            status_and_now(&backend, "a"),
            (OperationStatus::Succeeded, 5)
        );
        assert_eq!(status_and_now(&backend, "b"), (OperationStatus::Started, 5));
        assert!(backend.advance(), "row 2");
        assert_eq!(
            status_and_now(&backend, "b"),
            (OperationStatus::Succeeded, 9)
        );

        // Row 3: a retry and a wait due at the same time fire together.
        let backend = seeded_backend(
            Vec::new(),
            0,
            &[step_retry("r", Some(3)), wait_start("w", 3)],
        );
        assert!(backend.advance(), "row 3");
        assert_eq!(status_and_now(&backend, "r"), (OperationStatus::Ready, 3));
        assert_eq!(
            status_and_now(&backend, "w"),
            (OperationStatus::Succeeded, 3)
        );

        // Row 4: a queued callback reply arrives and the clock stays put.
        let backend = seeded_backend(
            vec![CallbackReply::raw("\"x\"")],
            0,
            &[callback_start("c"), wait_start("w", 100)],
        );
        assert!(backend.advance(), "row 4");
        assert_eq!(
            status_and_now(&backend, "c"),
            (OperationStatus::Succeeded, 0)
        );
        assert_eq!(status_and_now(&backend, "w"), (OperationStatus::Started, 0));

        // Row 5: nothing to deliver and nothing timed.
        let backend = seeded_backend(Vec::new(), 0, &[callback_start("c")]);
        assert!(!backend.advance(), "row 5");
        assert_eq!(status_and_now(&backend, "c"), (OperationStatus::Started, 0));

        // Row 6: a zero-second wait is due at once.
        let backend = seeded_backend(Vec::new(), 7, &[wait_start("w", 0)]);
        assert!(backend.advance(), "row 6");
        assert_eq!(
            status_and_now(&backend, "w"),
            (OperationStatus::Succeeded, 7)
        );

        // Row 7: negative durations count as 0.
        let backend = seeded_backend(
            Vec::new(),
            7,
            &[wait_start("w", -5), step_retry("r", Some(-5))],
        );
        assert!(backend.advance(), "row 7");
        assert_eq!(
            status_and_now(&backend, "w"),
            (OperationStatus::Succeeded, 7)
        );
        assert_eq!(status_and_now(&backend, "r"), (OperationStatus::Ready, 7));

        // Row 8: a retry without `StepOptions` is due at once.
        let backend = seeded_backend(Vec::new(), 4, &[step_retry("r", None)]);
        assert!(backend.advance(), "row 8");
        assert_eq!(status_and_now(&backend, "r"), (OperationStatus::Ready, 4));
    }

    /// The statuses of the step records of `backend`.
    fn step_statuses(backend: &Backend) -> Vec<String> {
        backend
            .snapshot_operations()
            .iter()
            .filter(|op| op.op_type() == "Step")
            .map(|op| op.status().to_owned())
            .collect()
    }

    /// `checkpoint_latency` holds each checkpoint request for the latency
    /// before the backend applies it, and changes nothing else.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_latency_delays_each_checkpoint() {
        let unbatched = || {
            LocalRunner::new().options(
                crate::Options::builder()
                    .checkpoint_batching(false)
                    .build()
                    .expect("valid options"),
            )
        };
        let runner = unbatched().checkpoint_latency(std::time::Duration::from_millis(50));
        let backend = Arc::new(runner.new_backend());

        let steps = async {
            tokio::time::advance(std::time::Duration::from_millis(49)).await;
            assert!(step_statuses(&backend).is_empty(), "START not applied yet");
            tokio::time::advance(std::time::Duration::from_millis(1)).await;
            assert_eq!(step_statuses(&backend), ["Started"]);
            tokio::time::advance(std::time::Duration::from_millis(49)).await;
            assert_eq!(step_statuses(&backend), ["Started"]);
            tokio::time::advance(std::time::Duration::from_millis(1)).await;
            assert_eq!(step_statuses(&backend), ["Succeeded"]);
        };
        let (delayed, ()) = tokio::join!(
            runner.run_on_backend(Arc::clone(&backend), one_step_handler, "null".to_owned()),
            steps
        );

        let baseline = unbatched().run(one_step_handler, ()).await;
        assert!(delayed.is_success(), "{:?}", delayed.error_message());
        assert_eq!(delayed.output(), baseline.output());
        assert_eq!(
            format!("{:?}", delayed.operations()),
            format!("{:?}", baseline.operations())
        );
    }

    /// With the default zero latency the backend creates no tokio timer, so
    /// a runtime built without the time driver still runs the handler.
    /// `tokio::time::sleep` cannot run on such a runtime, and the run then
    /// never ends. So the run happens on its own thread, and the test fails
    /// when no result arrives within 10 seconds.
    #[test]
    fn zero_checkpoint_latency_needs_no_timer() {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime builds");
            let result = runtime.block_on(LocalRunner::new().run(one_step_handler, ()));
            // The receiver is gone only after the test already failed.
            let _ = sender.send((result.is_success(), result.output().copied()));
        });
        let outcome = receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the run ends without a time driver");
        assert_eq!(outcome, (true, Some(42)));
    }

    // Issue #90: `ParentId` and `ReplayChildren` in the backend, and pruning.

    /// A map whose results exceed the checkpoint size limit records the
    /// batch with `ReplayChildren`. The wait forces a replay, which must
    /// rebuild the result from the item children.
    #[tokio::test]
    async fn oversized_map_result_replays_with_replay_children() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let items = ctx
                        .map(vec![0_u32, 1], |child, _item, _idx| async move {
                            let big = child
                                .step(|_| async { Ok("x".repeat(150 * 1024)) })
                                .name("big")
                                .await?;
                            Ok(big)
                        })
                        .name("oversized")
                        .await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("pause")
                        .await?;
                    Ok::<_, BoxError>(items.iter().map(String::len).collect::<Vec<_>>())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&vec![153_600, 153_600]));
        assert_eq!(result.invocation_count(), 2);
    }

    /// A child context whose result exceeds the checkpoint size limit
    /// records `ReplayChildren`. The wait forces a replay, which must rerun
    /// the child from its recorded steps.
    #[tokio::test]
    async fn oversized_child_result_replays_with_replay_children() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let joined = ctx
                        .run_in_child_context(|child| async move {
                            let a = child
                                .step(|_| async { Ok("x".repeat(150 * 1024)) })
                                .name("a")
                                .await?;
                            let b = child
                                .step(|_| async { Ok("x".repeat(150 * 1024)) })
                                .name("b")
                                .await?;
                            Ok(format!("{a}{b}"))
                        })
                        .name("oversized")
                        .await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("pause")
                        .await?;
                    Ok::<_, BoxError>(joined.len())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&307_200));
    }

    /// The service creates a callback in `Started`, so an open callback
    /// reports `"Started"`, as under `CloudRunner`.
    #[tokio::test]
    async fn open_callback_reports_started() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().name("approval").await?;
                    let value = cb.result().await?;
                    Ok::<_, BoxError>(value)
                },
                (),
            )
            .await;

        assert!(result.is_suspended(), "{:?}", result.error_message());
        let callback = result
            .operations()
            .iter()
            .find(|op| op.op_type() == "Callback")
            .expect("the callback is recorded");
        assert_eq!(callback.status(), "Started");
    }

    /// One update for the pruning tests. `replay_children` sets the
    /// context option when given.
    fn tree_update(
        id: &str,
        op_type: OperationType,
        action: OperationAction,
        parent: Option<&str>,
        replay_children: Option<bool>,
    ) -> OperationUpdate {
        let is_fail = action == OperationAction::Fail;
        let mut builder = OperationUpdate::builder()
            .id(id)
            .r#type(op_type)
            .action(action);
        if let Some(parent) = parent {
            builder = builder.parent_id(parent);
        }
        if let Some(replay) = replay_children {
            builder =
                builder.context_options(ContextOptions::builder().replay_children(replay).build());
        }
        if is_fail {
            builder = builder.error(ErrorObject::builder().error_message("boom").build());
        }
        builder.build().unwrap()
    }

    fn ctx_op(
        id: &str,
        action: OperationAction,
        parent: Option<&str>,
        replay_children: Option<bool>,
    ) -> OperationUpdate {
        tree_update(id, OperationType::Context, action, parent, replay_children)
    }

    fn step_op(id: &str, action: OperationAction, parent: Option<&str>) -> OperationUpdate {
        tree_update(id, OperationType::Step, action, parent, None)
    }

    /// Applies `updates` to a fresh backend.
    fn tree_backend(updates: &[OperationUpdate]) -> Backend {
        let backend = Backend::new(QueuedReplies::default(), None);
        {
            let mut state = backend.lock();
            for update in updates {
                state.apply_update(update).expect("update applies");
            }
        }
        backend
    }

    fn envelope_ids(backend: &Backend) -> Vec<String> {
        backend
            .envelope_history()
            .iter()
            .map(|op| op["Id"].as_str().expect("Id is a string").to_owned())
            .collect()
    }

    async fn get_state_ops(backend: &Backend) -> Vec<Operation> {
        backend
            .get_state("arn:test", "token-0", None)
            .await
            .expect("get_state succeeds")
            .operations
    }

    /// Asserts that the envelope history and `get_state` hold exactly
    /// `kept`, and that `snapshot_operations` holds every stored record.
    async fn assert_pruned(updates: &[OperationUpdate], kept: &[&str], all: &[&str]) {
        let backend = tree_backend(updates);
        assert_eq!(envelope_ids(&backend), kept, "envelope history");
        let state_ids: Vec<String> = get_state_ops(&backend)
            .await
            .iter()
            .map(|op| op.id().to_owned())
            .collect();
        assert_eq!(state_ids, kept, "get_state");
        let snapshot_ids: Vec<String> = backend
            .snapshot_operations()
            .iter()
            .map(|op| op.id().to_owned())
            .collect();
        assert_eq!(snapshot_ids, all, "snapshot_operations");
    }

    #[tokio::test]
    async fn prune_replay_children_top_level_kept() {
        assert_pruned(
            &[step_op("S", OperationAction::Succeed, None)],
            &["S"],
            &["S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_completed_context_omits_child() {
        assert_pruned(
            &[
                ctx_op("C", OperationAction::Succeed, None, Some(false)),
                step_op("S", OperationAction::Succeed, Some("C")),
            ],
            &["C"],
            &["C", "S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_failed_context_omits_child() {
        assert_pruned(
            &[
                ctx_op("C", OperationAction::Fail, None, Some(false)),
                step_op("S", OperationAction::Fail, Some("C")),
            ],
            &["C"],
            &["C", "S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_started_context_keeps_child() {
        assert_pruned(
            &[
                ctx_op("C", OperationAction::Start, None, None),
                step_op("S", OperationAction::Succeed, Some("C")),
            ],
            &["C", "S"],
            &["C", "S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_replay_children_keeps_child() {
        assert_pruned(
            &[
                ctx_op("C", OperationAction::Succeed, None, Some(true)),
                step_op("S", OperationAction::Succeed, Some("C")),
            ],
            &["C", "S"],
            &["C", "S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_ancestor_omits_replay_children_context() {
        assert_pruned(
            &[
                ctx_op("A", OperationAction::Succeed, None, Some(false)),
                ctx_op("B", OperationAction::Succeed, Some("A"), Some(true)),
                step_op("S", OperationAction::Succeed, Some("B")),
            ],
            &["A"],
            &["A", "B", "S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_ancestor_omits_started_context() {
        assert_pruned(
            &[
                ctx_op("A", OperationAction::Succeed, None, Some(false)),
                ctx_op("B", OperationAction::Start, Some("A"), None),
                step_op("S", OperationAction::Succeed, Some("B")),
            ],
            &["A"],
            &["A", "B", "S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_missing_parent_kept() {
        assert_pruned(
            &[step_op("S", OperationAction::Succeed, Some("ghost"))],
            &["S"],
            &["S"],
        )
        .await;
    }

    #[tokio::test]
    async fn prune_replay_children_cyclic_parents_terminate() {
        assert_pruned(
            &[
                ctx_op("X", OperationAction::Start, Some("Y"), None),
                ctx_op("Y", OperationAction::Start, Some("X"), None),
            ],
            &["X", "Y"],
            &["X", "Y"],
        )
        .await;
    }

    /// The service does not prune checkpoint responses, and the SDK merges
    /// a response into its log. So the response keeps a child that the
    /// same call's context close would omit from the invocation input.
    #[tokio::test]
    async fn checkpoint_response_is_not_pruned() {
        let backend = Backend::new(QueuedReplies::default(), None);
        let output = backend
            .checkpoint(
                "arn:test",
                "token-0",
                vec![
                    step_op("S", OperationAction::Succeed, Some("C")),
                    ctx_op("C", OperationAction::Succeed, None, None),
                ],
            )
            .await
            .expect("checkpoint succeeds");
        let ids: Vec<&str> = output
            .updated_operations
            .iter()
            .map(Operation::id)
            .collect();
        assert_eq!(ids, ["S", "C"]);
    }

    /// A paginated checkpoint response returns the first page of the full
    /// state. That page is not pruned either. So with a page size of 1 the
    /// page holds the child `S`, which was stored first, and not `C`.
    #[tokio::test]
    async fn paginated_checkpoint_response_is_not_pruned() {
        let backend = Backend::new(QueuedReplies::default(), Some(1));
        let output = backend
            .checkpoint(
                "arn:test",
                "token-0",
                vec![
                    step_op("S", OperationAction::Succeed, Some("C")),
                    ctx_op("C", OperationAction::Succeed, None, None),
                ],
            )
            .await
            .expect("checkpoint succeeds");
        assert!(output.next_marker.is_some(), "the response is paginated");
        let ids: Vec<&str> = output
            .updated_operations
            .iter()
            .map(Operation::id)
            .collect();
        assert_eq!(ids, ["S"]);
    }

    /// A context closed with `FAIL` and `ReplayChildren: true` records the
    /// option. So both renderers carry it, and both keep the child.
    #[tokio::test]
    async fn failed_context_with_replay_children_keeps_child() {
        let updates = [
            ctx_op("C", OperationAction::Start, None, None),
            step_op("S", OperationAction::Fail, Some("C")),
            ctx_op("C", OperationAction::Fail, None, Some(true)),
        ];
        assert_pruned(&updates, &["C", "S"], &["C", "S"]).await;

        let backend = tree_backend(&updates);
        let history = backend.envelope_history();
        let context = history.iter().find(|op| op["Id"] == "C").expect("C");
        assert_eq!(context["Status"], "FAILED");
        assert_eq!(context["ContextDetails"]["ReplayChildren"], true);

        let ops = get_state_ops(&backend).await;
        let context = ops.iter().find(|op| op.id() == "C").expect("C");
        assert_eq!(context.status(), &OperationStatus::Failed);
        assert_eq!(
            context
                .context_details()
                .and_then(ContextDetails::replay_children),
            Some(true)
        );
    }

    #[tokio::test]
    async fn parent_id_and_replay_children_are_rendered() {
        let backend = tree_backend(&[
            ctx_op("C", OperationAction::Start, None, None),
            step_op("S", OperationAction::Succeed, Some("C")),
            ctx_op("C", OperationAction::Succeed, None, Some(true)),
        ]);
        let history = backend.envelope_history();
        let context = history.iter().find(|op| op["Id"] == "C").expect("C");
        let step = history.iter().find(|op| op["Id"] == "S").expect("S");
        assert_eq!(step["ParentId"], "C");
        assert!(context.get("ParentId").is_none());
        assert_eq!(context["ContextDetails"]["ReplayChildren"], true);

        let ops = get_state_ops(&backend).await;
        let context = ops.iter().find(|op| op.id() == "C").expect("C");
        let step = ops.iter().find(|op| op.id() == "S").expect("S");
        assert_eq!(step.parent_id(), Some("C"));
        assert_eq!(context.parent_id(), None);
        assert_eq!(
            context
                .context_details()
                .and_then(ContextDetails::replay_children),
            Some(true)
        );
    }

    #[tokio::test]
    async fn parent_id_is_set_once() {
        let backend = tree_backend(&[
            step_op("S", OperationAction::Start, Some("P")),
            step_op("S", OperationAction::Succeed, Some("Q")),
        ]);
        assert_eq!(backend.envelope_history()[0]["ParentId"], "P");
        assert_eq!(get_state_ops(&backend).await[0].parent_id(), Some("P"));
    }

    #[tokio::test]
    async fn replay_children_set_only_on_context_close() {
        let backend = tree_backend(&[
            ctx_op("C", OperationAction::Start, None, Some(true)),
            step_op("S", OperationAction::Succeed, Some("C")),
        ]);
        let context_has_replay_children = |history: &[serde_json::Value]| {
            history
                .iter()
                .find(|op| op["Id"] == "C")
                .expect("C")
                .get("ContextDetails")
                .and_then(|details| details.get("ReplayChildren"))
                .is_some()
        };
        // A START ignores the option.
        assert!(!context_has_replay_children(&backend.envelope_history()));
        assert_eq!(
            get_state_ops(&backend).await[0]
                .context_details()
                .and_then(ContextDetails::replay_children),
            None
        );

        // A close without the option records `false`, so the child goes.
        backend
            .lock()
            .apply_update(&ctx_op("C", OperationAction::Succeed, None, None))
            .expect("update applies");
        assert!(!context_has_replay_children(&backend.envelope_history()));
        assert_eq!(envelope_ids(&backend), ["C"]);
        let ops = get_state_ops(&backend).await;
        assert_eq!(ops.iter().map(Operation::id).collect::<Vec<_>>(), ["C"]);
        assert_eq!(
            ops[0]
                .context_details()
                .and_then(ContextDetails::replay_children),
            None
        );
    }

    #[test]
    fn envelope_uses_service_type_spelling() {
        let backend = tree_backend(&[
            tree_update("s", OperationType::Step, OperationAction::Start, None, None),
            tree_update("w", OperationType::Wait, OperationAction::Start, None, None),
            tree_update(
                "c",
                OperationType::Context,
                OperationAction::Start,
                None,
                None,
            ),
            tree_update(
                "b",
                OperationType::Callback,
                OperationAction::Start,
                None,
                None,
            ),
            tree_update(
                "i",
                OperationType::ChainedInvoke,
                OperationAction::Start,
                None,
                None,
            ),
        ]);
        let envelope = build_envelope(&backend, "{}", None);
        let ops = envelope["InitialExecutionState"]["Operations"]
            .as_array()
            .expect("Operations is an array");
        let types: Vec<&str> = ops
            .iter()
            .map(|op| op["Type"].as_str().expect("Type is a string"))
            .collect();
        assert_eq!(
            types,
            [
                "EXECUTION",
                "STEP",
                "WAIT",
                "CONTEXT",
                "CALLBACK",
                "CHAINED_INVOKE"
            ]
        );
        assert_eq!(ops[0]["Status"], "STARTED");
    }

    /// An update of `op_type` and `action` with id `id`, carrying `payload`
    /// when given (service-limit tests).
    fn limit_update(
        id: &str,
        op_type: OperationType,
        action: OperationAction,
        payload: Option<String>,
    ) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(op_type)
            .action(action)
            .set_payload(payload)
            .build()
            .expect("valid update")
    }

    /// A STEP `FAIL` update whose error message is `message`, carrying
    /// `payload` when given.
    fn step_fail_with(id: &str, message: String, payload: Option<String>) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Step)
            .action(OperationAction::Fail)
            .set_payload(payload)
            .error(ErrorObject::builder().error_message(message).build())
            .build()
            .expect("valid update")
    }

    /// The model-pass entry for the value `value` at update `pos` and
    /// field `path`.
    fn delay_entry(pos: usize, path: &str, value: i32) -> String {
        let constraint = if value < 1 {
            "greater than or equal to 1"
        } else {
            "less than or equal to 31622400"
        };
        format!(
            "Value '{value}' at 'updates.{pos}.member.{path}' failed to satisfy \
             constraint: Member must have value {constraint}"
        )
    }

    const WAIT_PATH: &str = "waitOptions.waitSeconds";
    const RETRY_PATH: &str = "stepOptions.nextAttemptDelaySeconds";

    /// The single-entry message of the model pass.
    fn one_error(entry: &str) -> String {
        format!("1 validation error detected: {entry}")
    }

    /// A STEP `FAIL` update with a 1 000-byte error and a payload that
    /// makes the whole rendering `total` bytes.
    fn step_fail_rendered_at(total: usize) -> OperationUpdate {
        let message = "e".repeat(1_000);
        let base = render_update(&step_fail_with("f", message.clone(), Some(String::new()))).len();
        let update = step_fail_with("f", message, Some("x".repeat(total - base)));
        assert_eq!(render_update(&update).len(), total);
        update
    }

    /// Sends `updates` in one checkpoint call to a fresh backend. With
    /// `expected` set, asserts the rejection message, that nothing was
    /// stored and that the token did not change. Without it, asserts the
    /// call succeeded and stored one record per distinct update ID.
    async fn check_limit_row(row: usize, updates: Vec<OperationUpdate>, expected: Option<String>) {
        let backend = seeded_backend(Vec::new(), 0, &[]);
        let token = backend.current_token();
        let ids: std::collections::HashSet<String> =
            updates.iter().map(|u| u.id().to_owned()).collect();
        let result =
            ExecutionClient::checkpoint(&backend, LOCAL_EXECUTION_ARN, &token, updates).await;
        let error = result.err().map(|err| err.to_string());
        if let Some(expected) = expected {
            assert_eq!(
                error,
                Some(format!("durable: client: {expected}")),
                "row {row}"
            );
            assert!(backend.snapshot_operations().is_empty(), "row {row}");
            assert_eq!(backend.current_token(), token, "row {row}");
        } else {
            assert_eq!(error, None, "row {row}");
            assert_eq!(backend.snapshot_operations().len(), ids.len(), "row {row}");
        }
    }

    /// The delay rows of the case table: rows 1 to 7 and 19 to 21.
    fn delay_rows() -> Vec<(usize, Vec<OperationUpdate>, Option<String>)> {
        let retry_with_wait = OperationUpdate::builder()
            .id("r")
            .r#type(OperationType::Step)
            .action(OperationAction::Retry)
            .wait_options(
                aws_sdk_lambda::types::WaitOptions::builder()
                    .wait_seconds(0)
                    .build(),
            )
            .step_options(
                aws_sdk_lambda::types::StepOptions::builder()
                    .next_attempt_delay_seconds(0)
                    .build(),
            )
            .build()
            .expect("valid update");
        let eleven: Vec<OperationUpdate> =
            (1..=11).map(|i| wait_start(&format!("w{i}"), 0)).collect();
        let ten: Vec<String> = (1..=10).map(|pos| delay_entry(pos, WAIT_PATH, 0)).collect();
        let step_start = limit_update("s", OperationType::Step, OperationAction::Start, None);
        let big_step = limit_update(
            "s",
            OperationType::Step,
            OperationAction::Succeed,
            Some("x".repeat(262_145)),
        );
        vec![
            (
                1,
                vec![wait_start("w", 0)],
                Some(one_error(&delay_entry(1, WAIT_PATH, 0))),
            ),
            (2, vec![wait_start("w", 1)], None),
            (3, vec![wait_start("w", 31_622_400)], None),
            (
                4,
                vec![wait_start("w", 31_622_401)],
                Some(one_error(&delay_entry(1, WAIT_PATH, 31_622_401))),
            ),
            (
                5,
                vec![step_retry("r", Some(0))],
                Some(one_error(&delay_entry(1, RETRY_PATH, 0))),
            ),
            (
                6,
                vec![step_start, wait_start("w", 0)],
                Some(one_error(&delay_entry(2, WAIT_PATH, 0))),
            ),
            (
                7,
                vec![big_step, wait_start("w", 0)],
                Some(one_error(&delay_entry(2, WAIT_PATH, 0))),
            ),
            (
                19,
                vec![wait_start("a", 0), wait_start("b", 31_622_401)],
                Some(format!(
                    "2 validation errors detected: {}; {}",
                    delay_entry(1, WAIT_PATH, 0),
                    delay_entry(2, WAIT_PATH, 31_622_401)
                )),
            ),
            (
                20,
                eleven,
                Some(format!("10 validation errors detected: {}", ten.join("; "))),
            ),
            (
                21,
                vec![retry_with_wait],
                Some(format!(
                    "2 validation errors detected: {}; {}",
                    delay_entry(1, WAIT_PATH, 0),
                    delay_entry(1, RETRY_PATH, 0)
                )),
            ),
        ]
    }

    /// The size rows of the case table: rows 8 to 18.
    fn size_rows() -> Vec<(usize, Vec<OperationUpdate>, Option<String>)> {
        let step = |payload: String| {
            vec![limit_update(
                "s",
                OperationType::Step,
                OperationAction::Succeed,
                Some(payload),
            )]
        };
        let step_message =
            "STEP output payload size must be less than or equal to 262144 bytes.".to_owned();
        let error_message =
            "Error object size must be less than or equal to 262144 bytes.".to_owned();
        let one = |op_type: OperationType, action: OperationAction, len: usize| {
            vec![limit_update("o", op_type, action, Some("x".repeat(len)))]
        };
        vec![
            (8, step("x".repeat(262_144)), None),
            (9, step("x".repeat(262_145)), Some(step_message.clone())),
            (10, step("é".repeat(131_073)), Some(step_message)),
            (
                11,
                one(OperationType::Context, OperationAction::Succeed, 262_145),
                Some("CONTEXT output payload size must be less than or equal to 262144 bytes.".to_owned()),
            ),
            (12, one(OperationType::ChainedInvoke, OperationAction::Start, 1_048_576), None),
            (
                13,
                one(OperationType::ChainedInvoke, OperationAction::Start, 1_048_577),
                Some(
                    "CHAINED_INVOKE input payload size must be less than or equal to 1048576 bytes."
                        .to_owned(),
                ),
            ),
            (
                14,
                one(OperationType::Execution, OperationAction::Succeed, 6_291_457),
                Some("Execution output payload size must be less than or equal to 6291456 bytes.".to_owned()),
            ),
            (
                15,
                vec![step_fail_with("f", "e".repeat(262_145), None)],
                Some(error_message.clone()),
            ),
            (16, one(OperationType::Callback, OperationAction::Start, 300_000), None),
            (17, vec![step_fail_rendered_at(262_145)], Some(error_message)),
            (18, vec![step_fail_rendered_at(262_144)], None),
        ]
    }

    /// The backend applies the service limits: one call per row of the
    /// case table in the issue.
    #[tokio::test]
    async fn service_limits_reject_each_case() {
        let mut rows = delay_rows();
        rows.extend(size_rows());
        rows.sort_by_key(|(row, _, _)| *row);
        assert_eq!(
            rows.iter().map(|(row, _, _)| *row).collect::<Vec<_>>(),
            (1..=21).collect::<Vec<_>>()
        );
        for (row, updates, expected) in rows {
            check_limit_row(row, updates, expected).await;
        }
    }

    /// Row 17: the error alone is under the limit, but the whole update is
    /// measured, so the update is rejected.
    #[tokio::test]
    async fn error_under_limit_whole_update_over_is_rejected() {
        check_limit_row(
            17,
            vec![step_fail_rendered_at(262_145)],
            Some("Error object size must be less than or equal to 262144 bytes.".to_owned()),
        )
        .await;
    }

    /// Row 18: a whole update of exactly the limit is accepted.
    #[tokio::test]
    async fn whole_update_at_limit_is_accepted() {
        check_limit_row(18, vec![step_fail_rendered_at(262_144)], None).await;
    }

    /// A step result over the STEP payload limit fails the execution with
    /// the service's message.
    #[tokio::test]
    async fn oversized_step_result_fails_execution() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let out = ctx.step(|_| async { Ok("x".repeat(300_000)) }).await?;
                    Ok::<_, BoxError>(out.len())
                },
                (),
            )
            .await;
        assert!(result.is_failure(), "{:?}", result.error_message());
        let message = result.error_message().unwrap_or_default();
        assert!(
            message
                .contains("STEP output payload size must be less than or equal to 262144 bytes."),
            "{message}"
        );
    }

    /// A step result whose JSON is exactly the STEP payload limit succeeds.
    #[tokio::test]
    async fn step_result_at_the_limit_succeeds() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let out = ctx.step(|_| async { Ok("x".repeat(262_142)) }).await?;
                    Ok::<_, BoxError>(out.len())
                },
                (),
            )
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
    }

    /// The rendering of rule 3.1 holds every present member under its wire
    /// name, keeps an empty options object, and omits absent members and a
    /// callback timeout of 0.
    #[test]
    fn render_update_renders_every_present_member() {
        let full = OperationUpdate::builder()
            .id("i")
            .parent_id("p")
            .name("n")
            .r#type(OperationType::Step)
            .sub_type("st")
            .action(OperationAction::Fail)
            .payload("pl")
            .error(
                ErrorObject::builder()
                    .error_type("et")
                    .error_message("em")
                    .error_data("ed")
                    .stack_trace("s1")
                    .build(),
            )
            .context_options(ContextOptions::builder().replay_children(true).build())
            .step_options(
                aws_sdk_lambda::types::StepOptions::builder()
                    .next_attempt_delay_seconds(2)
                    .build(),
            )
            .wait_options(
                aws_sdk_lambda::types::WaitOptions::builder()
                    .wait_seconds(3)
                    .build(),
            )
            .callback_options(
                aws_sdk_lambda::types::CallbackOptions::builder()
                    .timeout_seconds(4)
                    .heartbeat_timeout_seconds(5)
                    .build(),
            )
            .chained_invoke_options(
                aws_sdk_lambda::types::ChainedInvokeOptions::builder()
                    .function_name("fn")
                    .tenant_id("t")
                    .build()
                    .expect("valid options"),
            )
            .build()
            .expect("valid update");
        let rendered: serde_json::Value =
            serde_json::from_str(&render_update(&full)).expect("rendering is JSON");
        assert_eq!(
            rendered,
            serde_json::json!({
                "Id": "i", "ParentId": "p", "Name": "n", "Type": "STEP",
                "SubType": "st", "Action": "FAIL", "Payload": "pl",
                "Error": {
                    "ErrorType": "et", "ErrorMessage": "em",
                    "ErrorData": "ed", "StackTrace": ["s1"]
                },
                "ContextOptions": { "ReplayChildren": true },
                "StepOptions": { "NextAttemptDelaySeconds": 2 },
                "WaitOptions": { "WaitSeconds": 3 },
                "CallbackOptions": { "TimeoutSeconds": 4, "HeartbeatTimeoutSeconds": 5 },
                "ChainedInvokeOptions": { "FunctionName": "fn", "TenantId": "t" }
            })
        );

        let sparse = OperationUpdate::builder()
            .id("i")
            .r#type(OperationType::Callback)
            .action(OperationAction::Start)
            .error(ErrorObject::builder().build())
            .context_options(ContextOptions::builder().build())
            .step_options(aws_sdk_lambda::types::StepOptions::builder().build())
            .wait_options(aws_sdk_lambda::types::WaitOptions::builder().build())
            .callback_options(aws_sdk_lambda::types::CallbackOptions::builder().build())
            .chained_invoke_options(
                aws_sdk_lambda::types::ChainedInvokeOptions::builder()
                    .function_name("fn")
                    .build()
                    .expect("valid options"),
            )
            .build()
            .expect("valid update");
        let rendered: serde_json::Value =
            serde_json::from_str(&render_update(&sparse)).expect("rendering is JSON");
        assert_eq!(
            rendered,
            serde_json::json!({
                "Id": "i", "Type": "CALLBACK", "Action": "START", "Error": {},
                "ContextOptions": {}, "StepOptions": {}, "WaitOptions": {},
                "CallbackOptions": {}, "ChainedInvokeOptions": { "FunctionName": "fn" }
            })
        );
    }

    /// Sends `updates` in one checkpoint call, which must succeed.
    async fn checkpoint_ok(backend: &Backend, updates: Vec<OperationUpdate>) {
        let token = backend.current_token();
        let result =
            ExecutionClient::checkpoint(backend, LOCAL_EXECUTION_ARN, &token, updates).await;
        assert_eq!(result.err().map(|err| err.to_string()), None);
    }

    /// The `UpdatedOperationIds` of a freshly built envelope.
    fn envelope_updated_ids(backend: &Backend) -> serde_json::Value {
        build_envelope(backend, "{}", None)["UpdatedOperationIds"].clone()
    }

    #[tokio::test]
    async fn updated_ids_empty_on_first_envelope() {
        let backend = seeded_backend(Vec::new(), 0, &[]);
        checkpoint_ok(&backend, vec![wait_start("w", 5)]).await;
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!([]));
    }

    #[tokio::test]
    async fn updated_ids_list_fired_wait() {
        let backend = seeded_backend(Vec::new(), 0, &[]);
        checkpoint_ok(&backend, vec![wait_start("w", 5)]).await;
        assert!(backend.advance());
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!(["w"]));
    }

    #[tokio::test]
    async fn updated_ids_cleared_after_envelope() {
        let backend = seeded_backend(Vec::new(), 0, &[]);
        checkpoint_ok(&backend, vec![wait_start("w", 5)]).await;
        assert!(backend.advance());
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!(["w"]));
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!([]));
    }

    #[tokio::test]
    async fn updated_ids_list_readied_retry_and_delivered_callback() {
        let backend = seeded_backend(vec![CallbackReply::raw("\"ok\"")], 0, &[]);
        checkpoint_ok(
            &backend,
            vec![step_retry("s", Some(5)), callback_start("c")],
        )
        .await;
        assert!(backend.advance(), "delivers the reply");
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!(["c"]));
        assert!(backend.advance(), "fires the retry");
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!(["s"]));
    }

    #[tokio::test]
    async fn updated_ids_skip_checkpoint_changes() {
        let backend = seeded_backend(Vec::new(), 0, &[]);
        checkpoint_ok(
            &backend,
            vec![limit_update(
                "s",
                OperationType::Step,
                OperationAction::Start,
                None,
            )],
        )
        .await;
        checkpoint_ok(
            &backend,
            vec![limit_update(
                "s",
                OperationType::Step,
                OperationAction::Succeed,
                Some("1".to_owned()),
            )],
        )
        .await;
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!([]));
    }

    // Chained invokes against registered targets.

    /// The one chained invoke record of `result`.
    fn invoke_record<O>(result: &TestResult<O>) -> &TestOperation {
        let invokes: Vec<&TestOperation> = result
            .operations()
            .iter()
            .filter(|op| op.op_type() == "ChainedInvoke")
            .collect();
        assert_eq!(invokes.len(), 1, "one chained invoke record");
        invokes[0]
    }

    /// The wire type and message of an invoke error, as a handler sees it.
    fn wire_type_and_message(err: &crate::OperationError) -> (Option<String>, Option<String>) {
        let wire = err.wire();
        (
            wire.and_then(crate::WireError::error_type)
                .map(ToOwned::to_owned),
            wire.and_then(crate::WireError::error_message)
                .map(ToOwned::to_owned),
        )
    }

    #[tokio::test]
    async fn invoke_registered_function_succeeds() {
        let result = LocalRunner::new()
            .register_function("double", |x: i32| async move { Ok(x * 2) })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let doubled = ctx.invoke::<i32, _>("double", 21).await?;
                    Ok::<_, BoxError>(doubled)
                },
                (),
            )
            .await;
        assert_eq!(result.output(), Some(&42), "{:?}", result.error_message());
        let op = invoke_record(&result);
        assert_eq!(op.status(), "Succeeded");
        assert_eq!(op.result(), Some("42"));
        assert_eq!(result.invocation_count(), 2);
    }

    #[tokio::test]
    async fn invoke_registered_function_failure_keeps_typed_error() {
        let result = LocalRunner::new()
            .register_function("pay", |(): ()| async move {
                Err::<(), BoxError>(crate::TypedError::with_type("PaymentError", "declined").into())
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<(), _>("pay", ())
                        .await
                        .expect_err("the invoke fails");
                    Ok::<_, BoxError>(wire_type_and_message(&err).0)
                },
                (),
            )
            .await;
        assert_eq!(
            result.output(),
            Some(&Some("PaymentError".to_owned())),
            "{:?}",
            result.error_message()
        );
        let op = invoke_record(&result);
        assert_eq!(op.status(), "Failed");
        assert_eq!(op.error_type(), Some("PaymentError"));
    }

    #[tokio::test]
    async fn invoke_plain_error_records_error_type() {
        let result = LocalRunner::new()
            .register_function(
                "pay",
                |(): ()| async move { Err::<(), BoxError>("boom".into()) },
            )
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<(), _>("pay", ())
                        .await
                        .expect_err("the invoke fails");
                    Ok::<_, BoxError>(wire_type_and_message(&err))
                },
                (),
            )
            .await;
        assert_eq!(
            result.output(),
            Some(&(Some("Error".to_owned()), Some("boom".to_owned()))),
            "{:?}",
            result.error_message()
        );
    }

    #[tokio::test]
    async fn invoke_unit_result_records_null() {
        let result = LocalRunner::new()
            .register_function("notify", |_: i32| async move { Ok(()) })
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.invoke::<(), _>("notify", 1).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        let op = invoke_record(&result);
        assert_eq!(op.status(), "Succeeded");
        assert_eq!(op.result(), Some("null"));
    }

    #[tokio::test]
    async fn invoke_unregistered_function_fails_run() {
        let result = LocalRunner::new()
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.invoke::<(), _>("child-fn", ()).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await;
        assert!(result.is_failure(), "{:?}", result.error_message());
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.contains(
                "No function found for function name \"child-fn\". Register it on the LocalRunner."
            ),
            "{message}"
        );
    }

    #[tokio::test]
    async fn invoke_input_decode_failure_fails_invoke() {
        let result = LocalRunner::new()
            .register_function("len", |s: String| async move { Ok(s.len()) })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<usize, _>("len", 5_i32)
                        .await
                        .expect_err("the invoke fails");
                    Ok::<_, BoxError>(wire_type_and_message(&err))
                },
                (),
            )
            .await;
        let (error_type, message) = result.into_output().expect("the handler succeeds");
        assert_eq!(error_type.as_deref(), Some("Error"));
        let message = message.unwrap_or_default();
        assert!(message.starts_with("deserialize input:"), "{message}");
    }

    #[tokio::test]
    async fn invoke_output_encode_failure_fails_invoke() {
        let result = LocalRunner::new()
            .register_function(
                "map",
                |(): ()| async move { Ok(BTreeMap::from([((1, 2), 3)])) },
            )
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<serde_json::Value, _>("map", ())
                        .await
                        .expect_err("the invoke fails");
                    Ok::<_, BoxError>(wire_type_and_message(&err).1)
                },
                (),
            )
            .await;
        let message = result
            .into_output()
            .expect("the handler succeeds")
            .unwrap_or_default();
        assert!(message.starts_with("serialize output:"), "{message}");
    }

    #[tokio::test]
    async fn invoke_target_runs_once() {
        let calls = Arc::new(AtomicU32::new(0));
        let target_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .register_function("count", move |(): ()| {
                let calls = Arc::clone(&target_calls);
                async move { Ok(calls.fetch_add(1, Ordering::SeqCst) + 1) }
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let count = ctx.invoke::<u32, _>("count", ()).await?;
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    ctx.step(|_| async { Ok(()) }).await?;
                    Ok::<_, BoxError>(count)
                },
                (),
            )
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn invoke_completes_before_the_clock_moves() {
        let result = LocalRunner::new()
            .register_function("fast", |(): ()| async move { Ok("invoked") })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let waited = ctx
                        .run_in_child_context(|child| async move {
                            child.wait(std::time::Duration::from_secs(1)).await?;
                            Ok("waited".to_owned())
                        })
                        .future();
                    let invoked = ctx
                        .run_in_child_context(|child| async move {
                            Ok(child.invoke::<String, _>("fast", ()).await?)
                        })
                        .future();
                    let winner = ctx.race([waited, invoked]).await?;
                    Ok::<_, BoxError>(winner)
                },
                (),
            )
            .await;
        assert_eq!(
            result.output().map(String::as_str),
            Some("invoked"),
            "{:?}",
            result.error_message()
        );
    }

    #[tokio::test]
    async fn invoke_durable_function_steps_and_waits() {
        let result = LocalRunner::new()
            .register_durable_function("workflow", |n: u32, ctx: DurableContext| async move {
                let next = ctx.step(move |_| async move { Ok(n + 1) }).await?;
                ctx.wait(std::time::Duration::from_secs(5)).await?;
                let doubled = ctx.step(move |_| async move { Ok(next * 2) }).await?;
                Ok::<_, BoxError>(doubled)
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let out = ctx.invoke::<u32, _>("workflow", 1).await?;
                    Ok::<_, BoxError>(out)
                },
                (),
            )
            .await;
        assert_eq!(result.output(), Some(&4), "{:?}", result.error_message());
        let op = invoke_record(&result);
        assert_eq!(op.status(), "Succeeded");
        assert_eq!(op.result(), Some("4"));
        assert_eq!(result.invocation_count(), 2);
    }

    #[tokio::test]
    async fn invoke_durable_function_failure_keeps_type() {
        let result = LocalRunner::new()
            .register_durable_function("pay", |(): (), _ctx: DurableContext| async move {
                Err::<(), BoxError>(
                    crate::TypedError::with_type("PaymentError", "declined")
                        .with_error_data("{\"code\":7}")
                        .into(),
                )
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<(), _>("pay", ())
                        .await
                        .expect_err("the invoke fails");
                    let wire = err.wire();
                    Ok::<_, BoxError>((
                        wire.and_then(crate::WireError::error_type)
                            .map(ToOwned::to_owned),
                        wire.and_then(crate::WireError::error_data)
                            .map(ToOwned::to_owned),
                    ))
                },
                (),
            )
            .await;
        assert_eq!(
            result.output(),
            Some(&(
                Some("PaymentError".to_owned()),
                Some("{\"code\":7}".to_owned())
            )),
            "{:?}",
            result.error_message()
        );
        let op = invoke_record(&result);
        assert_eq!(op.status(), "Failed");
        let message = op.error_message().unwrap_or_default();
        assert!(message.contains("declined"), "{message}");
    }

    /// A durable target that waits twice, so its run needs three
    /// invocations.
    async fn slow_target((): (), ctx: DurableContext) -> Result<(), BoxError> {
        ctx.wait(std::time::Duration::from_secs(1)).await?;
        ctx.wait(std::time::Duration::from_secs(1)).await?;
        Ok(())
    }

    /// A caller that invokes `"slow"`.
    async fn invoke_slow((): (), ctx: DurableContext) -> Result<(), BoxError> {
        ctx.invoke::<(), _>("slow", ()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn nested_invocations_do_not_count() {
        let result = LocalRunner::new()
            .register_durable_function("slow", slow_target)
            .run(invoke_slow, ())
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(result.operations().len(), 1);
    }

    #[tokio::test]
    async fn durable_target_runs_once() {
        let calls = Arc::new(AtomicU32::new(0));
        let target_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .register_durable_function("count", move |(): (), _ctx: DurableContext| {
                target_calls.fetch_add(1, Ordering::SeqCst);
                async move { Ok::<_, BoxError>(()) }
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.invoke::<(), _>("count", ()).await?;
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.invocation_count(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn suspended_durable_target_leaves_invoke_started() {
        let result = LocalRunner::new()
            .register_durable_function("approval", |(): (), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<bool>().await?;
                let approved = cb.result().await?;
                Ok::<_, BoxError>(approved)
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let approved = ctx.invoke::<bool, _>("approval", ()).await?;
                    Ok::<_, BoxError>(approved)
                },
                (),
            )
            .await;
        assert!(result.is_suspended(), "{:?}", result.error_message());
        assert_eq!(invoke_record(&result).status(), "Started");
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.ends_with("; chained invokes still pending for: \"approval\""),
            "{message}"
        );
    }

    /// A suspended durable target stays `Started`, and the runner does not
    /// run it again when a later invocation leaves it pending.
    #[tokio::test]
    async fn suspended_durable_target_runs_once() {
        let calls = Arc::new(AtomicU32::new(0));
        let target_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .register_durable_function("approval", move |(): (), ctx: DurableContext| {
                target_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    let cb = ctx.create_callback::<bool>().await?;
                    let approved = cb.result().await?;
                    Ok::<_, BoxError>(approved)
                }
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let approval = ctx.invoke::<bool, _>("approval", ()).spawn();
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    let approved = approval.await?;
                    Ok::<_, BoxError>(approved)
                },
                (),
            )
            .await;
        assert!(result.is_suspended(), "{:?}", result.error_message());
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn durable_target_does_not_see_registered_functions() {
        let result = LocalRunner::new()
            .register_function("leaf", |(): ()| async move { Ok(1_u32) })
            .register_durable_function("mid", |(): (), ctx: DurableContext| async move {
                let leaf = ctx.invoke::<u32, _>("leaf", ()).await?;
                Ok::<_, BoxError>(leaf)
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<u32, _>("mid", ())
                        .await
                        .expect_err("the invoke fails");
                    Ok::<_, BoxError>(wire_type_and_message(&err).1)
                },
                (),
            )
            .await;
        let message = result
            .into_output()
            .expect("the handler succeeds")
            .unwrap_or_default();
        assert!(
            message.contains("No function found for function name \"leaf\""),
            "{message}"
        );
    }

    #[tokio::test]
    async fn durable_target_uses_outer_settings() {
        let result = LocalRunner::new()
            .max_invocations(2)
            .register_durable_function("slow", slow_target)
            .run(invoke_slow, ())
            .await;
        assert!(result.is_suspended(), "{:?}", result.error_message());
        assert_eq!(invoke_record(&result).status(), "Started");
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.ends_with("; chained invokes still pending for: \"slow\""),
            "{message}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn durable_target_uses_outer_checkpoint_latency() -> Result<(), BoxError> {
        let result = LocalRunner::new()
            .options(
                crate::Options::builder()
                    .checkpoint_batching(false)
                    .build()?,
            )
            .checkpoint_latency(std::time::Duration::from_millis(100))
            .register_durable_function("timed", |(): (), ctx: DurableContext| async move {
                let before = tokio::time::Instant::now();
                ctx.step(|_| async { Ok(()) }).name("s").await?;
                let elapsed = tokio::time::Instant::now().duration_since(before);
                Ok::<_, BoxError>(u64::try_from(elapsed.as_millis())?)
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let millis = ctx.invoke::<u64, _>("timed", ()).await?;
                    Ok::<_, BoxError>(millis)
                },
                (),
            )
            .await;
        assert_eq!(result.output(), Some(&200), "{:?}", result.error_message());
        Ok(())
    }

    #[tokio::test]
    async fn later_registration_replaces_earlier_kind() {
        let result = LocalRunner::new()
            .register_function("t", |(): ()| async move { Ok(1_u32) })
            .register_durable_function("t", |(): (), _ctx: DurableContext| async move {
                Ok::<_, BoxError>(2_u32)
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let out = ctx.invoke::<u32, _>("t", ()).await?;
                    Ok::<_, BoxError>(out)
                },
                (),
            )
            .await;
        assert_eq!(result.output(), Some(&2), "{:?}", result.error_message());
    }

    /// An error whose wire record carries a stack trace, as an external
    /// callback failure read back from the wire does.
    fn error_with_stack_trace() -> BoxError {
        let wire = crate::WireError::new(Some("VendorError"), Some("denied"))
            .with_error_data(Some("opaque"))
            .with_stack_trace(vec!["frame-1".to_owned(), "frame-2".to_owned()]);
        crate::OperationError::from_kind(crate::OperationErrorKind::Callback(
            crate::error::CallbackError::new(
                crate::CallbackErrorKind::ExternalFailure,
                Some(crate::error::ReplayedFailure::source_from(wire.clone())),
            ),
        ))
        .with_wire(wire)
        .into()
    }

    /// A caller that invokes `"fail"` and returns the invoke error's wire
    /// record.
    async fn invoke_fail_wire(
        (): (),
        ctx: DurableContext,
    ) -> Result<(Option<String>, Option<String>, Vec<String>), BoxError> {
        let err = ctx
            .invoke::<(), _>("fail", ())
            .await
            .expect_err("the invoke fails");
        let wire = err.wire().cloned().unwrap_or_default();
        Ok((
            wire.error_type().map(ToOwned::to_owned),
            wire.error_data().map(ToOwned::to_owned),
            wire.stack_trace().to_vec(),
        ))
    }

    fn expected_fail_wire() -> (Option<String>, Option<String>, Vec<String>) {
        (
            Some("VendorError".to_owned()),
            Some("opaque".to_owned()),
            vec!["frame-1".to_owned(), "frame-2".to_owned()],
        )
    }

    #[tokio::test]
    async fn invoke_plain_failure_keeps_stack_trace() {
        let result = LocalRunner::new()
            .register_function("fail", |(): ()| async move {
                Err::<(), BoxError>(error_with_stack_trace())
            })
            .run(invoke_fail_wire, ())
            .await;
        assert_eq!(
            result.output(),
            Some(&expected_fail_wire()),
            "{:?}",
            result.error_message()
        );
    }

    #[tokio::test]
    async fn invoke_durable_failure_keeps_stack_trace() {
        let result = LocalRunner::new()
            .register_durable_function("fail", |(): (), _ctx: DurableContext| async move {
                Err::<(), BoxError>(error_with_stack_trace())
            })
            .run(invoke_fail_wire, ())
            .await;
        assert_eq!(
            result.output(),
            Some(&expected_fail_wire()),
            "{:?}",
            result.error_message()
        );
    }

    #[test]
    fn envelope_wire_error_reads_every_field() {
        let response = serde_json::json!({
            "Status": "FAILED",
            "Error": {
                "ErrorType": "T",
                "ErrorMessage": "m",
                "ErrorData": "d",
                "StackTrace": ["a", "b"],
            },
        });
        assert_eq!(
            envelope_wire_error(&response),
            crate::WireError::new(Some("T"), Some("m"))
                .with_error_data(Some("d"))
                .with_stack_trace(vec!["a".to_owned(), "b".to_owned()])
        );
    }

    /// A chained invoke `START` for `function`, with ID `id`.
    fn invoke_start(id: &str, function: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::ChainedInvoke)
            .action(OperationAction::Start)
            .payload("1")
            .chained_invoke_options(
                aws_sdk_lambda::types::ChainedInvokeOptions::builder()
                    .function_name(function)
                    .build()
                    .expect("valid options"),
            )
            .build()
            .expect("valid update")
    }

    #[test]
    fn pending_invoke_names_list_each_name_once_in_stored_order() {
        let backend = seeded_backend(
            Vec::new(),
            0,
            &[
                invoke_start("i1", "b"),
                invoke_start("i2", "a"),
                invoke_start("i3", "b"),
                invoke_start("i4", "c"),
            ],
        );
        assert!(backend.record_invoke_outcome("i4", NestedOutcome::Succeeded(None)));
        assert_eq!(backend.pending_invoke_names(), ["b", "a"]);
    }

    #[test]
    fn invoke_completion_is_listed_in_updated_ids() {
        let backend = seeded_backend(Vec::new(), 0, &[invoke_start("i", "double")]);
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!([]));
        assert!(backend.record_invoke_outcome("i", NestedOutcome::Succeeded(Some("2".to_owned()))));
        assert_eq!(envelope_updated_ids(&backend), serde_json::json!(["i"]));
    }
}
