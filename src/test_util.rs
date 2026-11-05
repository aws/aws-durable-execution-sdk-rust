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
//! - `LocalRunner` moves time only between invocations. On the service, a
//!   wait or a condition check can end while other work runs, and the SDK
//!   continues it in the same invocation. Locally its record never changes
//!   during an invocation. So a local run can take more invocations than
//!   the same execution takes on the service.
//! - [`LocalRunner::checkpoint_latency`] delays each checkpoint response in
//!   tokio time. It does not move the virtual clock.
//! - The backend stores each operation's `ParentId` and the `ReplayChildren`
//!   option of a completed context. It omits the children of a completed
//!   context from the invocation input and from `GetDurableExecutionState`,
//!   as the service does, unless the context recorded `ReplayChildren`.
//!   Checkpoint responses and [`TestResult::operations`] keep every record.
//! - The backend rejects a checkpoint that exceeds a service size limit or
//!   a delay range, with the service's message.
//! - The backend rejects a chained-invoke `START` whose function name has
//!   no target registered on the runner, and an update whose operation
//!   type it does not know. Neither rejection changes any state.
//! - The backend also checks a checkpoint against the history rules of the
//!   JS SDK's test backend, and rejects it with that backend's message. It
//!   checks the whole request against the operations stored before it, and
//!   a rejected request changes no operation, no token and no history
//!   event. The rules are:
//!   - An update must not change the `Type`, `SubType`, `Name` or
//!     `ParentId` of a stored operation. A field the update omits passes.
//!   - The action must fit the type. `STEP` allows `START`, `FAIL`, `RETRY`
//!     and `SUCCEED`; `CONTEXT` allows `START`, `FAIL` and `SUCCEED`; `WAIT`
//!     allows `START` and `CANCEL`; `CALLBACK` and `CHAINED_INVOKE` allow only
//!     `START`; `EXECUTION` allows `SUCCEED` and `FAIL`.
//!   - The action must fit the stored status. A stored `STEP` may `START`
//!     only when `READY`, and may `FAIL`, `SUCCEED` or `RETRY` only when
//!     `STARTED` or `READY`. A stored `CONTEXT` cannot `START` again, and may
//!     close only when `STARTED`. A `WAIT` may `START` only when it is not
//!     stored, and `CANCEL` only when it is `STARTED`. A `CALLBACK` or
//!     `CHAINED_INVOKE` may `START` only when it is not stored.
//!   - A `ParentId` must name a stored `CONTEXT`, or a `CONTEXT` that an
//!     earlier update of the same request starts.
//!   - An ID may appear twice in one request only when the earlier update
//!     is a `STEP` or `CONTEXT` `START` and the later one is not a `START`.
//!
//!   It does not check the rest of the service's update model: the request
//!   model, the ID and name patterns, the options and payloads each action
//!   requires or forbids, and the count and position of `EXECUTION`
//!   updates.
//!
//!   The SDK treats a rejection as any other non-retryable checkpoint
//!   failure. So the execution fails with the error type
//!   `CheckpointUnrecoverableExecutionError`, and a test fails where the SDK
//!   would write a history that the JS SDK's test backend rejects.
//! - Each invocation input carries `UpdatedOperationIds`: the operations
//!   the backend changed between invocations, such as a fired wait, a
//!   retry that became ready, or a callback that received a reply.
//! - The backend records a history event for each operation update it
//!   accepts, for each wait, callback and chained invoke it settles, for
//!   each invocation, and for the start and the end of the execution (see
//!   [`TestResult::history_events`]). Invocation `n` has the request ID
//!   `local-invocation-{n}`.
//! - The backend reports operation start and end times as the run's start
//!   time plus virtual seconds. The invocation envelope carries them as
//!   epoch milliseconds, as the service does.
//!
//! # Runner divergences
//!
//! [`LocalRunner`] and [`CloudRunner`] return the same types, and a test can
//! be re-pointed between them by swapping the runner. The two agree on
//! operation type, status, result, error, parent, [`StepDetails::attempt`]
//! (both report completed retries), [`WaitDetails::duration`],
//! [`CallbackDetails::callback_id`], and the [`ExecutionStatus`] of a
//! succeeded or failed execution. A few behaviours still differ because one
//! runner observes an in-memory model and the other observes the recorded
//! execution history; assert around these when a test targets both runners:
//!
//! - **Truncated results.** When a result exceeds the recorded-history inline
//!   size limit, [`CloudRunner`] reports [`TestOperation::result`] as `None`
//!   (the payload is withheld); [`LocalRunner`] always holds the full
//!   in-memory result.
//! - **Timed-out / stopped executions.** [`CloudRunner`] reports a terminal
//!   `TimedOut` or `Stopped` execution as [`ExecutionStatus::TimedOut`] or
//!   [`ExecutionStatus::Stopped`], with the status name (or the recorded
//!   error) as the error type; [`LocalRunner`] has no execution timeout and
//!   no stop, so it reports only `Succeeded`, `Failed` or `Running`.
//! - **Transient operation statuses.** [`LocalRunner`] may surface the
//!   intermediate [`OperationStatus::Ready`] status it uses to model an
//!   elapsed retry delay; [`CloudRunner`]'s status vocabulary comes from the
//!   event stream and does not include [`OperationStatus::Ready`].
//! - **Replay children.** [`ContextDetails::replay_children`] is reported
//!   only by [`LocalRunner`]. No history event carries it, so under
//!   [`CloudRunner`] it is `false`.
//! - **Times.** [`LocalRunner`] times come from the virtual clock, starting
//!   at the time the run started: each reported time is the run's start time
//!   plus virtual seconds. [`CloudRunner`] times are the timestamps of the
//!   recorded history events.
//!
//! Runner failures (serialization, an unexpected envelope, an AWS error) are
//! returned as [`TestRunnerError`]. A [`TestResult`] always describes the
//! durable execution, including one that is still
//! [`Running`](ExecutionStatus::Running).
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
//!     .await.expect("runner");
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
    CallbackDetails as WireCallbackDetails, ChainedInvokeDetails,
    ContextDetails as WireContextDetails, ContextOptions, ErrorObject, Event, EventError,
    EventInput, EventResult, EventType, ExecutionStatus as WireExecutionStatus,
    InvocationCompletedDetails, InvocationType as WireInvocationType, Operation, OperationAction,
    OperationStatus as WireOperationStatus, OperationType, OperationUpdate, RetryDetails,
    StepDetails as WireStepDetails, builders::EventBuilder,
};

use crate::client::{CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput};
use crate::context::DurableContext;
use crate::{BoxError, OperationKind, OperationStatus};
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
///     .await.expect("runner");
///
/// let ops = result.operations();
/// assert_eq!(ops.len(), 1);
/// assert_eq!(ops[0].kind(), durable::OperationKind::Step);
/// assert!(ops[0].succeeded());
/// assert_eq!(ops[0].name(), Some("only"));
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TestOperation {
    id: String,
    kind: OperationKind,
    sub_type: Option<String>,
    status: OperationStatus,
    result: Option<String>,
    error: Option<crate::WireError>,
    name: Option<String>,
    parent_id: Option<String>,
    start_time: Option<std::time::SystemTime>,
    end_time: Option<std::time::SystemTime>,
    // The detail of every kind is stored, and `details()` returns the one
    // of `kind`. So a runner fills a kind's field without matching on the
    // kind first.
    step: StepDetails,
    wait: WaitDetails,
    callback: CallbackDetails,
    context: ContextDetails,
    /// The history events whose `id` is this operation's ID, in history
    /// order.
    events: Vec<Event>,
}

impl TestOperation {
    /// A record of `kind` with no result, error or detail, in status
    /// `Started`.
    fn started(id: String, kind: OperationKind) -> Self {
        Self {
            id,
            kind,
            sub_type: None,
            status: OperationStatus::Started,
            result: None,
            error: None,
            name: None,
            parent_id: None,
            start_time: None,
            end_time: None,
            step: StepDetails {
                attempt: 0,
                next_attempt_time: None,
            },
            wait: WaitDetails {
                duration: None,
                scheduled_end_time: None,
            },
            callback: CallbackDetails { callback_id: None },
            context: ContextDetails {
                replay_children: false,
            },
            events: Vec::new(),
        }
    }

    /// The operation's wire ID (the SHA-256 hex of its positional path).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The history events of this operation: the events of
    /// [`TestResult::history_events`] whose `id` is this operation's
    /// [`id`](Self::id), in history order.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// The operation kind, such as [`OperationKind::Step`] or
    /// [`OperationKind::Callback`].
    pub fn kind(&self) -> OperationKind {
        self.kind
    }

    /// The operation sub-type, e.g. `"Step"`, `"Wait"`, `"Map"`,
    /// `"WaitForCondition"`, or `None` if the operation carried none.
    ///
    /// The constants of [`OperationSubType`](crate::OperationSubType) hold
    /// the built-in values. A step, a callback, a child context or a
    /// `with_retry` block built with a `sub_type` setter records the custom
    /// value instead.
    pub fn sub_type(&self) -> Option<&str> {
        self.sub_type.as_deref()
    }

    /// The terminal-or-current status, such as
    /// [`OperationStatus::Succeeded`] or [`OperationStatus::TimedOut`].
    pub fn status(&self) -> OperationStatus {
        self.status
    }

    /// Returns `true` if the operation completed successfully.
    pub fn succeeded(&self) -> bool {
        self.status == OperationStatus::Succeeded
    }

    /// Returns `true` if the operation ended in a failed status.
    pub fn failed(&self) -> bool {
        self.status == OperationStatus::Failed
    }

    /// The serialized result payload, if the operation stored one.
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }

    /// The recorded error, if the operation stored one: its type, message,
    /// error data and stack trace.
    pub fn error(&self) -> Option<&crate::WireError> {
        self.error.as_ref()
    }

    /// The recorded error type, if the operation failed. Reads
    /// [`error`](Self::error).
    pub fn error_type(&self) -> Option<&str> {
        self.error.as_ref().and_then(crate::WireError::error_type)
    }

    /// The recorded error message, if the operation failed. Reads
    /// [`error`](Self::error).
    pub fn error_message(&self) -> Option<&str> {
        self.error
            .as_ref()
            .and_then(crate::WireError::error_message)
    }

    /// The operation's caller-assigned name, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The [`id`](Self::id) of the operation that contains this one, or
    /// `None` for a top-level operation.
    pub fn parent_id(&self) -> Option<&str> {
        self.parent_id.as_deref()
    }

    /// The time the operation started, or `None` when none was recorded.
    ///
    /// Under [`LocalRunner`] it is the virtual time the record was created,
    /// reported as the run's start time plus virtual seconds. Under
    /// [`CloudRunner`] it is the timestamp of the operation's first history
    /// event.
    pub fn start_time(&self) -> Option<std::time::SystemTime> {
        self.start_time
    }

    /// The time the operation ended, or `None` while it has not ended.
    ///
    /// Under [`LocalRunner`] it is the virtual time the status first became
    /// `Succeeded`, `Failed`, `TimedOut`, `Cancelled` or `Stopped`. Under
    /// [`CloudRunner`] it is the timestamp of the operation's terminal
    /// history event.
    pub fn end_time(&self) -> Option<std::time::SystemTime> {
        self.end_time
    }

    /// The detail of this operation's kind. The variant matches
    /// [`kind`](Self::kind): [`OperationKind::Step`] gives
    /// [`TestOperationDetails::Step`], and so on.
    pub fn details(&self) -> TestOperationDetails<'_> {
        match self.kind {
            OperationKind::Execution => TestOperationDetails::Execution,
            OperationKind::Context => TestOperationDetails::Context(&self.context),
            OperationKind::Step => TestOperationDetails::Step(&self.step),
            OperationKind::Wait => TestOperationDetails::Wait(&self.wait),
            OperationKind::Callback => TestOperationDetails::Callback(&self.callback),
            OperationKind::ChainedInvoke => TestOperationDetails::ChainedInvoke,
        }
    }
}

/// The detail of one [`TestOperation`], by kind. Returned by
/// [`TestOperation::details`].
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use aws_durable_execution_sdk as durable;
/// use durable::test_util::{LocalRunner, TestOperationDetails};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), durable::BoxError> {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             ctx.wait(Duration::from_secs(10)).name("pause").await?;
///             Ok::<_, durable::BoxError>(())
///         },
///         (),
///     )
///     .await?;
///
/// let wait = result.operation("pause").ok_or("no wait")?;
/// assert!(matches!(
///     wait.details(),
///     TestOperationDetails::Wait(w) if w.duration() == Some(Duration::from_secs(10))
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum TestOperationDetails<'a> {
    /// The execution itself.
    Execution,
    /// A child context.
    Context(&'a ContextDetails),
    /// A step.
    Step(&'a StepDetails),
    /// A wait.
    Wait(&'a WaitDetails),
    /// A callback.
    Callback(&'a CallbackDetails),
    /// A chained invoke.
    ChainedInvoke,
}

/// The detail of a step [`TestOperation`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StepDetails {
    attempt: u32,
    next_attempt_time: Option<std::time::SystemTime>,
}

impl StepDetails {
    /// The number of completed retries recorded for this step (`0` when
    /// never retried). [`LocalRunner`] and [`CloudRunner`] report this
    /// value identically.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// The time the step's next attempt was scheduled for, after a retry.
    /// `None` for a step that was never retried.
    ///
    /// Under [`LocalRunner`] it is the virtual time the latest retry delay
    /// ended. Under [`CloudRunner`] it is the timestamp of the latest
    /// `StepFailed` event that carries a retry delay, plus that delay.
    pub fn next_attempt_time(&self) -> Option<std::time::SystemTime> {
        self.next_attempt_time
    }
}

/// The detail of a wait [`TestOperation`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WaitDetails {
    duration: Option<std::time::Duration>,
    scheduled_end_time: Option<std::time::SystemTime>,
}

impl WaitDetails {
    /// The recorded wait duration, in whole seconds, or `None` when no
    /// duration was recorded.
    pub fn duration(&self) -> Option<std::time::Duration> {
        self.duration
    }

    /// The time the wait was scheduled to end, or `None` when none was
    /// recorded.
    ///
    /// Under [`LocalRunner`] it is the virtual time the wait started plus
    /// its duration. Under [`CloudRunner`] it is the scheduled end of the
    /// `WaitStarted` event.
    pub fn scheduled_end_time(&self) -> Option<std::time::SystemTime> {
        self.scheduled_end_time
    }
}

/// The detail of a callback [`TestOperation`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CallbackDetails {
    callback_id: Option<String>,
}

impl CallbackDetails {
    /// The callback ID the backend assigned, the ID the handler sees, or
    /// `None` when none was recorded.
    pub fn callback_id(&self) -> Option<&str> {
        self.callback_id.as_deref()
    }
}

/// The detail of a child context [`TestOperation`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ContextDetails {
    replay_children: bool,
}

impl ContextDetails {
    /// Whether the context closed with the `ReplayChildren` option. Only
    /// [`LocalRunner`] reports it; under [`CloudRunner`] it is `false`.
    pub fn replay_children(&self) -> bool {
        self.replay_children
    }
}

/// The status of the durable execution a [`TestResult`] describes, as the
/// service reports it.
///
/// [`Display`](std::fmt::Display) writes the variant name, such as
/// `TimedOut`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExecutionStatus {
    /// The execution has not ended.
    Running,
    /// The handler returned a value.
    Succeeded,
    /// The handler returned an error.
    Failed,
    /// The execution exceeded its execution timeout.
    TimedOut,
    /// The execution was stopped.
    Stopped,
}

impl std::fmt::Display for ExecutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Running => "Running",
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
            Self::TimedOut => "TimedOut",
            Self::Stopped => "Stopped",
        })
    }
}

impl ExecutionStatus {
    /// Converts the service's status. A status this enum does not list
    /// ended the execution in a way the runner cannot name, so it maps to
    /// `Failed`.
    fn from_wire(status: &WireExecutionStatus) -> Self {
        match status {
            WireExecutionStatus::Running => Self::Running,
            WireExecutionStatus::Succeeded => Self::Succeeded,
            WireExecutionStatus::TimedOut => Self::TimedOut,
            WireExecutionStatus::Stopped => Self::Stopped,
            _ => Self::Failed,
        }
    }
}

/// A failure of the test runner itself, not of the durable execution.
///
/// [`LocalRunner::run`] and [`CloudRunner::run`] return it when they cannot
/// describe the execution. A durable execution that fails is not a runner
/// error: it is an `Ok` [`TestResult`] with [`ExecutionStatus::Failed`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TestRunnerError {
    /// The event did not serialize to JSON.
    EventSerialization(Arc<serde_json::Error>),
    /// The handler output did not decode into the output type.
    OutputDeserialization(Arc<serde_json::Error>),
    /// A response envelope carried a status other than `SUCCEEDED`,
    /// `FAILED` and `PENDING`. The value is the status, or `None` when the
    /// envelope carried no string status.
    UnexpectedEnvelope(Option<String>),
    /// A Lambda API call failed.
    Aws(Arc<dyn std::error::Error + Send + Sync>),
    /// The invoke response carried no `DurableExecutionArn`.
    NoExecutionArn,
}

impl std::fmt::Display for TestRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EventSerialization(e) => {
                write!(f, "durable: test runner: event serialization failed: {e}")
            }
            Self::OutputDeserialization(e) => {
                write!(
                    f,
                    "durable: test runner: output deserialization failed: {e}"
                )
            }
            Self::UnexpectedEnvelope(status) => write!(
                f,
                "durable: test runner: unexpected response envelope status: {status:?}"
            ),
            Self::Aws(e) => write!(f, "durable: test runner: AWS call failed: {e}"),
            Self::NoExecutionArn => f.write_str(
                "durable: test runner: the invoke response carried no DurableExecutionArn",
            ),
        }
    }
}

impl std::error::Error for TestRunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EventSerialization(e) | Self::OutputDeserialization(e) => Some(e.as_ref()),
            Self::Aws(e) => Some(e.as_ref()),
            Self::UnexpectedEnvelope(_) | Self::NoExecutionArn => None,
        }
    }
}

/// The outcome of a [`LocalRunner::run`] or [`CloudRunner::run`]: the
/// execution status, the deserialized handler output (on success) and the
/// recorded operation history.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk as durable;
/// use durable::test_util::{ExecutionStatus, LocalRunner};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), durable::BoxError> {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), _ctx: durable::DurableContext| async move {
///             Ok::<_, durable::BoxError>("hi".to_owned())
///         },
///         (),
///     )
///     .await?;
///
/// assert_eq!(result.status(), ExecutionStatus::Succeeded);
/// assert!(result.is_success());
/// assert_eq!(result.output().map(String::as_str), Some("hi"));
/// assert_eq!(result.invocation_count(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct TestResult<O> {
    status: ExecutionStatus,
    output: Option<O>,
    error: Option<crate::WireError>,
    operations: Vec<TestOperation>,
    invocations: usize,
    history_events: Vec<Event>,
    invocation_records: Vec<TestInvocation>,
}

impl<O> TestResult<O> {
    /// A result whose operations, history events and invocation records
    /// come from `operations` and `history_events`: each operation gets the
    /// events with its ID, and each `InvocationCompleted` event becomes a
    /// [`TestInvocation`].
    fn with_history(
        status: ExecutionStatus,
        output: Option<O>,
        error: Option<crate::WireError>,
        mut operations: Vec<TestOperation>,
        invocations: usize,
        history_events: Vec<Event>,
    ) -> Self {
        for op in &mut operations {
            op.events = history_events
                .iter()
                .filter(|event| event.id() == Some(op.id.as_str()))
                .cloned()
                .collect();
        }
        let invocation_records = history_events
            .iter()
            .filter(|event| event.event_type() == Some(&EventType::InvocationCompleted))
            .map(TestInvocation::from_event)
            .collect();
        Self {
            status,
            output,
            error,
            operations,
            invocations,
            history_events,
            invocation_records,
        }
    }

    /// The status of the execution.
    ///
    /// Under [`LocalRunner`], a run that ends because nothing can advance,
    /// or because [`max_invocations`](LocalRunner::max_invocations) is
    /// reached, is [`ExecutionStatus::Running`]. Under [`CloudRunner`], the
    /// service status maps one to one, and a run that reaches a configured
    /// [`max_poll_attempts`](CloudRunner::max_poll_attempts) is
    /// [`ExecutionStatus::Running`].
    pub fn status(&self) -> ExecutionStatus {
        self.status
    }

    /// Returns `true` if the status is [`ExecutionStatus::Succeeded`].
    pub fn is_success(&self) -> bool {
        self.status == ExecutionStatus::Succeeded
    }

    /// Returns `true` if the status is [`ExecutionStatus::Failed`],
    /// [`ExecutionStatus::TimedOut`] or [`ExecutionStatus::Stopped`].
    pub fn is_failure(&self) -> bool {
        matches!(
            self.status,
            ExecutionStatus::Failed | ExecutionStatus::TimedOut | ExecutionStatus::Stopped
        )
    }

    /// Returns `true` if the status is [`ExecutionStatus::Running`]: the
    /// handler suspended and never reached a terminal outcome within the
    /// runner's budget (e.g. it awaits a callback the test never delivered,
    /// or it never terminates).
    pub fn is_suspended(&self) -> bool {
        self.status == ExecutionStatus::Running
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

    /// The execution's error: on failure, the recorded error with its type,
    /// message and error data; while running, the reason the run ended.
    /// `None` on success.
    pub fn error(&self) -> Option<&crate::WireError> {
        self.error.as_ref()
    }

    /// The wire error type on failure (e.g. `"StepError"`), or `None`.
    /// Reads [`error`](Self::error).
    pub fn error_type(&self) -> Option<&str> {
        self.error.as_ref().and_then(crate::WireError::error_type)
    }

    /// The wire error message on failure, or `None`. Reads
    /// [`error`](Self::error).
    pub fn error_message(&self) -> Option<&str> {
        self.error
            .as_ref()
            .and_then(crate::WireError::error_message)
    }

    /// The recorded operation history, in first-checkpoint order.
    pub fn operations(&self) -> &[TestOperation] {
        &self.operations
    }

    /// The first operation in [`operations`](Self::operations) order whose
    /// [`name`](TestOperation::name) is `name`.
    pub fn operation(&self, name: &str) -> Option<&TestOperation> {
        self.operations.iter().find(|op| op.name() == Some(name))
    }

    /// Every operation whose [`name`](TestOperation::name) is `name`, in
    /// [`operations`](Self::operations) order.
    pub fn operations_named<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = &'a TestOperation> + 'a {
        self.operations
            .iter()
            .filter(move |op| op.name() == Some(name))
    }

    /// Every operation whose [`parent_id`](TestOperation::parent_id) is the
    /// [`id`](TestOperation::id) of `parent`, in
    /// [`operations`](Self::operations) order.
    pub fn children<'a>(
        &'a self,
        parent: &'a TestOperation,
    ) -> impl Iterator<Item = &'a TestOperation> + 'a {
        self.operations
            .iter()
            .filter(move |op| op.parent_id() == Some(parent.id()))
    }

    /// The number of invocations that drove this execution.
    ///
    /// For [`LocalRunner`] this is the count of simulated invocations; for
    /// [`CloudRunner`] it is the number of recorded `InvocationCompleted`
    /// events (falling back to the poll count when none are recorded).
    pub fn invocation_count(&self) -> usize {
        self.invocations
    }

    /// The history events of the execution, in history order, as the
    /// service's `Event` model type.
    ///
    /// [`CloudRunner`] returns the events of `GetDurableExecutionHistory`
    /// unchanged. [`LocalRunner`] synthesizes the events. They follow the
    /// service's event types and order: one event for each operation update
    /// the backend accepts, for each wait, callback and chained invoke the
    /// backend settles, for each invocation (`InvocationCompleted`), and for
    /// the start and the end of the execution. The local events are a model
    /// of the service's history. The `history` conformance suite is the
    /// check against the service.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk as durable;
    /// use durable::lambda_types::EventType;
    /// use durable::test_util::LocalRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), durable::BoxError> {
    /// let result = LocalRunner::new()
    ///     .run(
    ///         |_e: (), ctx: durable::DurableContext| async move {
    ///             ctx.step(|_| async { Ok(1_i32) }).await?;
    ///             Ok::<_, durable::BoxError>(())
    ///         },
    ///         (),
    ///     )
    ///     .await?;
    ///
    /// let types: Vec<_> = result
    ///     .history_events()
    ///     .iter()
    ///     .filter_map(|event| event.event_type().cloned())
    ///     .collect();
    /// assert_eq!(
    ///     types,
    ///     [
    ///         EventType::ExecutionStarted,
    ///         EventType::StepStarted,
    ///         EventType::StepSucceeded,
    ///         EventType::InvocationCompleted,
    ///         EventType::ExecutionSucceeded,
    ///     ]
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn history_events(&self) -> &[Event] {
        &self.history_events
    }

    /// One record per handler invocation: one per `InvocationCompleted`
    /// event of [`history_events`](Self::history_events), in history order.
    ///
    /// Under [`LocalRunner`], invocation `n` (counting from 1) has the
    /// request ID `local-invocation-{n}`, and the list has
    /// [`invocation_count`](Self::invocation_count) records.
    pub fn invocations(&self) -> &[TestInvocation] {
        &self.invocation_records
    }

    /// The operations as a table with the columns `columns` enables.
    ///
    /// [`Display`](std::fmt::Display) for `TestResult` prints this table
    /// with every column. The output is:
    ///
    /// 1. `No operations found.` when [`operations`](Self::operations) is
    ///    empty.
    /// 2. Otherwise a header line with the enabled column names, in the
    ///    order `parentId`, `name`, `type`, `subType`, `status`,
    ///    `startTime`, `endTime`, `duration`, then one line per operation in
    ///    [`operations`](Self::operations) order.
    ///
    /// Cells are joined by ` | `, and every line ends with a newline. A time
    /// prints as ISO 8601 UTC with three fractional digits, as JS
    /// `toISOString` prints it. The duration is the end time minus the
    /// start time in whole milliseconds, such as `1000ms`. An absent value
    /// prints `-`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk as durable;
    /// use durable::test_util::{LocalRunner, PrintColumns};
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), durable::BoxError> {
    /// let result = LocalRunner::new()
    ///     .run(
    ///         |_e: (), ctx: durable::DurableContext| async move {
    ///             ctx.step(|_| async { Ok(1_i32) }).name("fetch").await?;
    ///             Ok::<_, durable::BoxError>(())
    ///         },
    ///         (),
    ///     )
    ///     .await?;
    ///
    /// let mut columns = PrintColumns::default();
    /// columns.parent_id = false;
    /// columns.sub_type = false;
    /// columns.start_time = false;
    /// columns.end_time = false;
    /// columns.duration = false;
    /// assert_eq!(
    ///     result.display_with(columns).to_string(),
    ///     "name | type | status\nfetch | STEP | SUCCEEDED\n"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn display_with(&self, columns: PrintColumns) -> impl std::fmt::Display + '_ {
        OperationTable {
            operations: &self.operations,
            columns,
        }
    }
}

impl<O> std::fmt::Display for TestResult<O> {
    /// Prints [`display_with`](Self::display_with) with every column.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_with(PrintColumns::default()).fmt(f)
    }
}

/// One handler invocation of an execution, read from its
/// `InvocationCompleted` history event.
///
/// Returned in order by [`TestResult::invocations`].
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), durable::BoxError> {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), _ctx: durable::DurableContext| async move {
///             Ok::<_, durable::BoxError>(())
///         },
///         (),
///     )
///     .await?;
///
/// let invocation = &result.invocations()[0];
/// assert_eq!(invocation.request_id(), Some("local-invocation-1"));
/// assert!(invocation.error().is_none());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TestInvocation {
    start_time: Option<std::time::SystemTime>,
    end_time: Option<std::time::SystemTime>,
    request_id: Option<String>,
    error: Option<crate::WireError>,
}

impl TestInvocation {
    /// The record of an `InvocationCompleted` event, with the fields of its
    /// `InvocationCompletedDetails`. An event without details gives a
    /// record with every field absent.
    fn from_event(event: &Event) -> Self {
        let details = event.invocation_completed_details();
        let time = |t: &DateTime| std::time::SystemTime::try_from(*t).ok();
        Self {
            start_time: details.and_then(|d| time(d.start_timestamp())),
            end_time: details.and_then(|d| time(d.end_timestamp())),
            request_id: details.map(|d| d.request_id().to_owned()),
            error: details
                .and_then(InvocationCompletedDetails::error)
                .and_then(EventError::payload)
                .map(wire_error_from_object),
        }
    }

    /// The time the invocation started, or `None` when it lies outside the
    /// range of `SystemTime`.
    pub fn start_time(&self) -> Option<std::time::SystemTime> {
        self.start_time
    }

    /// The time the invocation ended, or `None` when it lies outside the
    /// range of `SystemTime`.
    pub fn end_time(&self) -> Option<std::time::SystemTime> {
        self.end_time
    }

    /// The request ID of the invocation.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// The error the invocation ended with, or `None` when it returned a
    /// `SUCCEEDED` or `PENDING` result.
    ///
    /// Under [`LocalRunner`], an invocation fault (the invocation itself
    /// failed, for example on a retryable checkpoint error) has no error
    /// type and its message is the fault's text. A `FAILED` result has the
    /// result's error type, message, error data and stack trace.
    pub fn error(&self) -> Option<&crate::WireError> {
        self.error.as_ref()
    }
}

/// The columns [`TestResult::display_with`] prints. Each field enables one
/// column; [`PrintColumns::default`] enables every column, as JS
/// `TestResult.print` does.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::test_util::PrintColumns;
///
/// let mut columns = PrintColumns::default();
/// columns.duration = false;
/// assert!(columns.name);
/// ```
#[expect(clippy::struct_excessive_bools)] // reason: one flag per JS column, as JS's print config has
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PrintColumns {
    /// The `parentId` column: [`TestOperation::parent_id`].
    pub parent_id: bool,
    /// The `name` column: [`TestOperation::name`].
    pub name: bool,
    /// The `type` column: the [`wire_str`](OperationKind::wire_str) of
    /// [`TestOperation::kind`].
    pub kind: bool,
    /// The `subType` column: [`TestOperation::sub_type`].
    pub sub_type: bool,
    /// The `status` column: the [`wire_str`](OperationStatus::wire_str) of
    /// [`TestOperation::status`].
    pub status: bool,
    /// The `startTime` column: [`TestOperation::start_time`].
    pub start_time: bool,
    /// The `endTime` column: [`TestOperation::end_time`].
    pub end_time: bool,
    /// The `duration` column: the end time minus the start time.
    pub duration: bool,
}

impl Default for PrintColumns {
    /// Every column enabled.
    fn default() -> Self {
        Self {
            parent_id: true,
            name: true,
            kind: true,
            sub_type: true,
            status: true,
            start_time: true,
            end_time: true,
            duration: true,
        }
    }
}

/// The table [`TestResult::display_with`] returns.
struct OperationTable<'a> {
    operations: &'a [TestOperation],
    columns: PrintColumns,
}

impl std::fmt::Display for OperationTable<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.operations.is_empty() {
            return f.write_str("No operations found.\n");
        }
        let c = self.columns;
        let enabled = [
            (c.parent_id, "parentId"),
            (c.name, "name"),
            (c.kind, "type"),
            (c.sub_type, "subType"),
            (c.status, "status"),
            (c.start_time, "startTime"),
            (c.end_time, "endTime"),
            (c.duration, "duration"),
        ];
        let header: Vec<&str> = enabled
            .iter()
            .filter(|(on, _)| *on)
            .map(|(_, name)| *name)
            .collect();
        writeln!(f, "{}", header.join(" | "))?;
        for op in self.operations {
            let start = op.start_time.map(unix_millis);
            let end = op.end_time.map(unix_millis);
            let duration = duration_cell(start, end);
            let cells = [
                (c.parent_id, op.parent_id().unwrap_or("-").to_owned()),
                (c.name, op.name().unwrap_or("-").to_owned()),
                (c.kind, op.kind().wire_str().to_owned()),
                (c.sub_type, op.sub_type().unwrap_or("-").to_owned()),
                (c.status, op.status().wire_str().to_owned()),
                (c.start_time, start.map_or_else(dash, iso_millis)),
                (c.end_time, end.map_or_else(dash, iso_millis)),
                (c.duration, duration),
            ];
            let row: Vec<String> = cells
                .into_iter()
                .filter(|(on, _)| *on)
                .map(|(_, cell)| cell)
                .collect();
            writeln!(f, "{}", row.join(" | "))?;
        }
        Ok(())
    }
}

/// The duration cell: `end - start` in whole milliseconds, or `-` unless
/// both times exist.
///
/// Each time is at most `u64::MAX` seconds from the epoch, about `2^74`
/// milliseconds. So the difference fits an `i128` and cannot overflow.
fn duration_cell(start: Option<i128>, end: Option<i128>) -> String {
    match (start, end) {
        (Some(start), Some(end)) => format!("{}ms", end - start),
        _ => dash(),
    }
}

/// The cell of an absent value.
fn dash() -> String {
    "-".to_owned()
}

/// The signed whole milliseconds of `time` from the Unix epoch, truncated
/// toward zero.
///
/// A `Duration` holds at most `u64::MAX` seconds. So the milliseconds fit
/// an `i128` for every `SystemTime`, and the result needs no `Option`. The
/// time cells narrow the value only when they format it, so a time that
/// cannot be printed still gives the duration cell its exact value.
fn unix_millis(time: std::time::SystemTime) -> i128 {
    let millis =
        |d: std::time::Duration| i128::from(d.as_secs()) * 1000 + i128::from(d.subsec_millis());
    match time.duration_since(std::time::UNIX_EPOCH) {
        Ok(after) => millis(after),
        Err(before) => -millis(before.duration()),
    }
}

/// `millis` from the Unix epoch as JS `toISOString` prints it: ISO 8601 UTC
/// with exactly three fractional digits. A value whose seconds do not fit
/// an `i64`, or whose year the formatter does not support, prints `-`.
fn iso_millis(millis: i128) -> String {
    // Euclidean division keeps the remainder in 0..=999, so a time before
    // the epoch borrows from the seconds, as an ISO timestamp does.
    let remainder = millis.rem_euclid(1000);
    i64::try_from(millis.div_euclid(1000))
        .ok()
        .and_then(|seconds| {
            DateTime::from_secs(seconds)
                .fmt(aws_sdk_lambda::primitives::DateTimeFormat::DateTime)
                .ok()
        })
        .map_or_else(dash, |text| {
            // `Format::DateTime` writes RFC 3339 in UTC, which always ends
            // in `Z`. So a successful format always has the suffix.
            #[expect(clippy::expect_used)] // reason: Format::DateTime always ends in Z
            let whole = text.strip_suffix('Z').expect("Format::DateTime ends in Z");
            format!("{whole}.{remainder:03}Z")
        })
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
///     .await?;
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
    fn deliver_to(self, op: &mut StoredOp, now: u64) {
        op.result = None;
        op.error_type = None;
        op.error_message = None;
        op.error_data = None;
        op.stack_trace = Vec::new();
        match self.kind {
            CallbackReplyKind::Success(payload) => {
                op.set_status(OperationStatus::Succeeded, now);
                op.result = Some(payload);
            }
            CallbackReplyKind::Failure(failure) => {
                op.set_status(OperationStatus::Failed, now);
                op.error_type = failure.error_type;
                op.error_message = failure.error_message;
                op.error_data = failure.error_data;
                op.stack_trace = failure.stack_trace;
            }
            CallbackReplyKind::Timeout => {
                op.set_status(OperationStatus::TimedOut, now);
                op.error_message = Some(CALLBACK_TIMEOUT_MESSAGE.to_owned());
            }
            CallbackReplyKind::HeartbeatTimeout => {
                op.set_status(OperationStatus::TimedOut, now);
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
///     .await?;
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
///     .await?;
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
    ///     .await.expect("runner");
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
    ///     .await.expect("runner");
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
                runner.drive_nested(backend, nested, input).await
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
    /// # Errors
    ///
    /// - [`TestRunnerError::EventSerialization`] when `event` does not
    ///   serialize.
    /// - [`TestRunnerError::OutputDeserialization`] when the output of a
    ///   succeeded execution does not decode into `O`.
    /// - [`TestRunnerError::UnexpectedEnvelope`] when a response envelope
    ///   has a status other than `SUCCEEDED`, `FAILED` and `PENDING`.
    ///
    /// An invocation fault is not a runner error: the runner invokes again,
    /// as the service does.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk as durable;
    /// use durable::test_util::LocalRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), durable::BoxError> {
    /// let result = LocalRunner::new()
    ///     .run(
    ///         |n: u32, ctx: durable::DurableContext| async move {
    ///             let v = ctx.step(move |_| async move { Ok(n + 1) }).await?;
    ///             Ok::<_, durable::BoxError>(v)
    ///         },
    ///         41_u32,
    ///     )
    ///     .await?;
    /// assert_eq!(result.output(), Some(&42));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run<E, O, F, Fut>(
        &self,
        handler: F,
        event: E,
    ) -> Result<TestResult<O>, TestRunnerError>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        // Serialize the event once; the envelope re-delivers it per
        // invocation.
        let event_json = serde_json::to_string(&event)
            .map_err(|e| TestRunnerError::EventSerialization(Arc::new(e)))?;
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
    ) -> Result<TestResult<O>, TestRunnerError>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        self.drive(backend, handler, event_json)
            .await?
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
    ) -> Result<RawRun, TestRunnerError>
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
        self.drive_service(backend, service, event_json).await
    }

    /// Drives a nested durable run and returns its outcome.
    async fn drive_nested<E, O, F, Fut>(
        &self,
        backend: Arc<Backend>,
        handler: F,
        event_json: String,
    ) -> NestedOutcome
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        // `drive` runs the handler behind `crate::wrap`. Every envelope
        // `wrap` returns has the status SUCCEEDED, FAILED or PENDING. The
        // only runner error `drive_service` returns is `UnexpectedEnvelope`,
        // for any other status. So `drive` returns no runner error here.
        #[expect(clippy::expect_used)]
        let run = self
            .drive(backend, handler, event_json)
            .await
            .expect("a run behind `wrap` returns no runner error");
        run.nested_outcome()
    }

    /// The invoke → parse → advance loop of [`drive`](Self::drive), with the
    /// service function as a parameter. Internal seam: a unit test passes a
    /// service that returns an envelope `wrap` never produces.
    #[expect(clippy::too_many_lines)] // reason: one loop over the invocations reads better as one flow
    async fn drive_service<S>(
        &self,
        backend: Arc<Backend>,
        service: S,
        event_json: String,
    ) -> Result<RawRun, TestRunnerError>
    where
        S: Fn(
            lambda_runtime::LambdaEvent<crate::InvocationPayload>,
        ) -> crate::BoxedInvocationFuture,
    {
        // A nested durable run uses the settings this run starts with.
        let nested = self.nested_settings();

        let mut invocations = 0_usize;
        // The most recent invocation fault (Lambda runtime error), kept so
        // an execution that never terminates because every invocation
        // faults reports the fault rather than a generic budget message.
        let mut last_invocation_fault: Option<String> = None;

        backend.record_execution_started(&event_json);
        loop {
            invocations += 1;
            if invocations > self.max_invocations {
                return Ok(RawRun {
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
                    events: backend.events(),
                });
            }

            let payload = build_envelope(&backend, &event_json, self.initial_page_size);
            let mut context = lambda_runtime::Context::default();
            context.request_id = format!("local-invocation-{invocations}");
            let started_at = backend.now();
            let lambda_event = lambda_runtime::LambdaEvent::new(
                crate::InvocationPayload(payload),
                context.clone(),
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
                    let fault = e.to_string();
                    backend.record_invocation_completed(
                        started_at,
                        &context.request_id,
                        Some(&crate::WireError::new(None::<String>, Some(fault.as_str()))),
                    );
                    last_invocation_fault = Some(fault);
                    continue;
                }
            };

            let status = response.get("Status").and_then(serde_json::Value::as_str);
            backend.record_invocation_completed(
                started_at,
                &context.request_id,
                (status == Some("FAILED"))
                    .then(|| envelope_wire_error(&response))
                    .as_ref(),
            );
            let end = match status {
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
                other => {
                    return Err(TestRunnerError::UnexpectedEnvelope(
                        other.map(ToOwned::to_owned),
                    ));
                }
            };
            backend.record_execution_end(&end);
            return Ok(RawRun {
                end,
                operations: backend.snapshot_operations(),
                invocations,
                events: backend.events(),
            });
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
    /// The history events the backend recorded.
    events: Vec<Event>,
}

/// The end of a driven execution (see [`RawRun`]).
#[derive(Debug)]
enum RawEnd {
    /// A `SUCCEEDED` envelope, with its raw `Result` string.
    Succeeded(Option<String>),
    /// A `FAILED` envelope, with the wire error of its `Error` object.
    Failed(crate::WireError),
    /// The execution did not terminate.
    Suspended {
        error_type: Option<String>,
        message: String,
    },
}

impl RawRun {
    /// The [`TestResult`] of a top-level run: the output decoded into `O`.
    fn into_test_result<O: DeserializeOwned>(self) -> Result<TestResult<O>, TestRunnerError> {
        let Self {
            end,
            operations,
            invocations,
            events,
        } = self;
        let (status, output, error) = match end {
            RawEnd::Succeeded(raw) => {
                let output = serde_json::from_str::<O>(raw.as_deref().unwrap_or("null"))
                    .map_err(|e| TestRunnerError::OutputDeserialization(Arc::new(e)))?;
                (ExecutionStatus::Succeeded, Some(output), None)
            }
            RawEnd::Failed(wire) => {
                // A missing `ErrorType` reads `Error`, the fallback the
                // handler mapping uses.
                let error = crate::WireError::new(
                    Some(wire.error_type().unwrap_or("Error")),
                    Some(wire.error_message().unwrap_or_default()),
                )
                .with_error_data(wire.error_data())
                .with_stack_trace(wire.stack_trace().to_vec());
                (ExecutionStatus::Failed, None, Some(error))
            }
            RawEnd::Suspended {
                error_type,
                message,
            } => (
                ExecutionStatus::Running,
                None,
                Some(crate::WireError::new(error_type, Some(message))),
            ),
        };
        Ok(TestResult::with_history(
            status,
            output,
            error,
            operations,
            invocations,
            events,
        ))
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
        "Id": EXECUTION_OPERATION_ID,
        "Type": "EXECUTION",
        "Status": "STARTED",
        "StartTimestamp": virtual_epoch_millis(backend.base(), 0),
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

/// Default delay between execution-status polls, as JS.
const DEFAULT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// How [`CloudRunner`] invokes the function.
///
/// `DryRun` is not offered: it returns no execution ARN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InvocationType {
    /// A synchronous invoke. It returns after the execution ended or the
    /// invocation reached its timeout.
    #[default]
    RequestResponse,
    /// An asynchronous invoke. It returns once the execution started.
    Event,
}

impl InvocationType {
    /// The service model value of this invocation type.
    fn to_wire(self) -> WireInvocationType {
        match self {
            Self::RequestResponse => WireInvocationType::RequestResponse,
            Self::Event => WireInvocationType::Event,
        }
    }
}

/// The fields of one `Invoke` request that [`CloudRunner`] sends.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InvokeRequest {
    function_name: String,
    qualifier: String,
    invocation_type: WireInvocationType,
    payload: Vec<u8>,
}

/// The three Lambda calls [`CloudRunner::run`] makes. `run` calls only this
/// trait, so a unit test can supply a fake that returns scripted outputs
/// and errors.
trait CloudApi {
    /// Sends `Invoke`.
    fn send_invoke(
        &self,
        request: InvokeRequest,
    ) -> impl Future<Output = Result<aws_sdk_lambda::operation::invoke::InvokeOutput, BoxError>> + Send;

    /// Sends `GetDurableExecution` with the execution data.
    fn send_get_durable_execution(
        &self,
        execution_arn: &str,
    ) -> impl Future<
        Output = Result<
            aws_sdk_lambda::operation::get_durable_execution::GetDurableExecutionOutput,
            BoxError,
        >,
    > + Send;

    /// Sends `GetDurableExecutionHistory` with the execution data, for the
    /// page that `marker` names.
    fn send_get_durable_execution_history(
        &self,
        execution_arn: &str,
        marker: Option<&str>,
    ) -> impl Future<
        Output = Result<
            aws_sdk_lambda::operation::get_durable_execution_history::GetDurableExecutionHistoryOutput,
            BoxError,
        >,
    > + Send;
}

impl CloudApi for crate::LambdaClient {
    async fn send_invoke(
        &self,
        request: InvokeRequest,
    ) -> Result<aws_sdk_lambda::operation::invoke::InvokeOutput, BoxError> {
        Ok(self
            .invoke()
            .function_name(request.function_name)
            .invocation_type(request.invocation_type)
            .qualifier(request.qualifier)
            .payload(Blob::new(request.payload))
            .send()
            .await?)
    }

    async fn send_get_durable_execution(
        &self,
        execution_arn: &str,
    ) -> Result<aws_sdk_lambda::operation::get_durable_execution::GetDurableExecutionOutput, BoxError>
    {
        Ok(self
            .get_durable_execution()
            .durable_execution_arn(execution_arn)
            .include_execution_data(true)
            .send()
            .await?)
    }

    async fn send_get_durable_execution_history(
        &self,
        execution_arn: &str,
        marker: Option<&str>,
    ) -> Result<
        aws_sdk_lambda::operation::get_durable_execution_history::GetDurableExecutionHistoryOutput,
        BoxError,
    > {
        Ok(self
            .get_durable_execution_history()
            .durable_execution_arn(execution_arn)
            .include_execution_data(true)
            .set_marker(marker.map(ToOwned::to_owned))
            .send()
            .await?)
    }
}

/// The [`TestRunnerError::Aws`] of a failed Lambda call.
fn aws_error(err: BoxError) -> TestRunnerError {
    TestRunnerError::Aws(Arc::from(err))
}

/// Drives a durable function against **real AWS** and returns the same
/// [`TestResult`] / [`TestOperation`] types [`LocalRunner`] returns.
///
/// `CloudRunner` invokes a deployed durable function, with the
/// [`invocation_type`](Self::invocation_type) `RequestResponse` by default,
/// polls [`GetDurableExecution`] until the execution reaches a terminal
/// state, then folds the recorded [`GetDurableExecutionHistory`] event
/// stream into per-operation [`TestOperation`]s. Because both runners return
/// the same types, a test can be re-pointed from in-memory to real AWS by
/// swapping the runner. The runner reads only `DurableExecutionArn` from the
/// invoke response and ignores its `FunctionError` and `Payload`: the
/// history is the source of the result.
///
/// By default the Lambda client is built from the ambient credentials and
/// region (the same way the production entry point builds it). Configure the
/// region with [`region`](Self::region), or supply the client with
/// [`lambda_client`](Self::lambda_client).
///
/// A failed Lambda call or an invoke response without an execution ARN
/// is a [`TestRunnerError`]. A durable execution that fails, times out or
/// stops is an `Ok` [`TestResult`] with that [`ExecutionStatus`].
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
/// # async fn main() -> Result<(), durable::BoxError> {
/// let result = CloudRunner::new("my-durable-function")
///     .region("us-west-2")
///     .poll_interval(Duration::from_secs(3))
///     .run::<_, i32>(21_i32)
///     .await?;
///
/// if result.is_success() {
///     assert_eq!(result.output(), Some(&42));
///     for op in result.operations() {
///         println!("{} {} {}", op.id(), op.kind(), op.status());
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CloudRunner {
    function_name: String,
    region: Option<String>,
    qualifier: String,
    poll_interval: std::time::Duration,
    max_poll_attempts: Option<usize>,
    invocation_type: InvocationType,
    lambda_client: Option<crate::LambdaClient>,
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
            max_poll_attempts: None,
            invocation_type: InvocationType::default(),
            lambda_client: None,
        }
    }

    /// Sets the AWS region used to build the Lambda client.
    ///
    /// When unset, the region is resolved from the ambient environment
    /// (`AWS_REGION`, profile, or instance metadata). Ignored when
    /// [`lambda_client`](Self::lambda_client) is set.
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
    /// Defaults to 1 second.
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

    /// Sets the maximum number of `GetDurableExecution` polls. A run that
    /// reaches it while the execution still runs returns a [`TestResult`]
    /// with [`ExecutionStatus::Running`]. A value of `0` counts as `1`.
    ///
    /// By default there is no limit: the runner polls until the execution
    /// ends or a Lambda call fails.
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
        self.max_poll_attempts = Some(max.max(1));
        self
    }

    /// Sets how the runner invokes the function.
    ///
    /// Default `RequestResponse`, as JS. A test that answers callbacks while the
    /// execution runs sets `InvocationType::Event`, as every JS callback example does.
    /// Under `RequestResponse`, waiting for an operation's data and the live
    /// callback controls return an error and make no Lambda call.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::{CloudRunner, InvocationType};
    ///
    /// let runner = CloudRunner::new("f").invocation_type(InvocationType::Event);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn invocation_type(mut self, invocation_type: InvocationType) -> Self {
        self.invocation_type = invocation_type;
        self
    }

    /// Use this client for every call. `region` is then ignored.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use durable::test_util::CloudRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let config = aws_config::load_from_env().await;
    /// let runner = CloudRunner::new("f").lambda_client(durable::LambdaClient::new(&config));
    /// # drop(runner);
    /// # }
    /// ```
    #[must_use]
    pub fn lambda_client(mut self, client: crate::LambdaClient) -> Self {
        self.lambda_client = Some(client);
        self
    }

    /// Invokes the durable function with `event`, polls to a terminal state,
    /// and returns the folded [`TestResult`].
    ///
    /// The event is serialized to JSON and delivered as the invocation
    /// payload. The handler output (on success) is deserialized into `O`.
    ///
    /// # Errors
    ///
    /// - [`TestRunnerError::EventSerialization`] when `event` does not
    ///   serialize.
    /// - [`TestRunnerError::Aws`] when `Invoke`, `GetDurableExecution` or
    ///   `GetDurableExecutionHistory` fails.
    /// - [`TestRunnerError::NoExecutionArn`] when the invoke response has no
    ///   execution ARN.
    /// - [`TestRunnerError::OutputDeserialization`] when the output of a
    ///   succeeded execution does not decode into `O`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use durable::test_util::CloudRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), durable::BoxError> {
    /// let result = CloudRunner::new("echo-fn")
    ///     .run::<_, String>("hello".to_owned())
    ///     .await?;
    /// assert!(result.is_success() || result.is_failure() || result.is_suspended());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run<E, O>(&self, event: E) -> Result<TestResult<O>, TestRunnerError>
    where
        E: Serialize,
        O: DeserializeOwned,
    {
        let payload = serde_json::to_vec(&event)
            .map_err(|e| TestRunnerError::EventSerialization(Arc::new(e)))?;
        let client = self.resolve_client().await;
        self.run_with(&client, payload).await
    }

    /// The invoke → poll → history → fold flow of [`run`](Self::run), over
    /// any [`CloudApi`].
    async fn run_with<A: CloudApi, O: DeserializeOwned>(
        &self,
        api: &A,
        payload: Vec<u8>,
    ) -> Result<TestResult<O>, TestRunnerError> {
        // 1. Invoke: start the durable execution and capture its ARN. The
        // `FunctionError` and `Payload` of a synchronous invoke are ignored;
        // the history is the source of the result.
        let invoke_output = api
            .send_invoke(self.build_invoke(payload))
            .await
            .map_err(aws_error)?;
        let execution_arn = invoke_output
            .durable_execution_arn()
            .ok_or(TestRunnerError::NoExecutionArn)?
            .to_owned();

        // 2. Poll GetDurableExecution until terminal, or until a configured
        // limit is spent.
        let mut polls = 0_usize;
        let terminal = loop {
            if self.max_poll_attempts.is_some_and(|limit| polls >= limit) {
                let events = fetch_history(api, &execution_arn).await?;
                return Ok(TestResult::with_history(
                    ExecutionStatus::Running,
                    None,
                    Some(crate::WireError::new(
                        None::<String>,
                        Some(format!(
                            "execution {execution_arn} still running after {polls} polls"
                        )),
                    )),
                    fold_history(&events),
                    polls,
                    events,
                ));
            }
            polls += 1;
            tokio::time::sleep(self.poll_interval).await;
            let out = api
                .send_get_durable_execution(&execution_arn)
                .await
                .map_err(aws_error)?;
            if !matches!(out.status(), WireExecutionStatus::Running) {
                break out;
            }
        };

        // 3. Retrieve the recorded history and fold it into operations.
        let events = fetch_history(api, &execution_arn).await?;
        let invocation_events = invocations_from_history(&events);
        let invocations = if invocation_events == 0 {
            polls
        } else {
            invocation_events
        };

        // 4. Map the terminal status into the result. The history's terminal
        // execution event is the source of the output and the error. The
        // `GetDurableExecution` output is used only when the history holds
        // no terminal execution event.
        let status = ExecutionStatus::from_wire(terminal.status());
        let history_end = terminal_execution_event(&events);
        let (output, error) = if status == ExecutionStatus::Succeeded {
            let payload = match history_end.and_then(Event::execution_succeeded_details) {
                Some(details) => event_result_payload(details.result()),
                None => terminal.result().map(ToOwned::to_owned),
            };
            let output = deserialize_output::<O>(payload.as_deref())
                .map_err(|e| TestRunnerError::OutputDeserialization(Arc::new(e)))?;
            (Some(output), None)
        } else {
            let status_name = status.to_string();
            let error_object = history_end
                .and_then(terminal_event_error)
                .or_else(|| terminal.error());
            let error = error_object.map_or_else(
                || {
                    crate::WireError::new(
                        Some(status_name.as_str()),
                        Some(format!("execution ended with status {status_name}")),
                    )
                },
                |err| {
                    crate::WireError::new(
                        Some(err.error_type().unwrap_or(&status_name)),
                        Some(err.error_message().unwrap_or_default()),
                    )
                    .with_error_data(err.error_data())
                    .with_stack_trace(err.stack_trace().to_vec())
                },
            );
            (None, Some(error))
        };
        Ok(TestResult::with_history(
            status,
            output,
            error,
            fold_history(&events),
            invocations,
            events,
        ))
    }

    /// The `Invoke` request for `payload`, with this runner's function,
    /// qualifier and invocation type.
    fn build_invoke(&self, payload: Vec<u8>) -> InvokeRequest {
        InvokeRequest {
            function_name: self.function_name.clone(),
            qualifier: self.qualifier.clone(),
            invocation_type: self.invocation_type.to_wire(),
            payload,
        }
    }

    /// The client `run` uses: the configured
    /// [`lambda_client`](Self::lambda_client) unchanged, or else one built
    /// from the ambient config with the [`region`](Self::region) override.
    async fn resolve_client(&self) -> crate::LambdaClient {
        if let Some(client) = &self.lambda_client {
            return client.clone();
        }
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = &self.region {
            loader = loader.region(aws_sdk_lambda::config::Region::new(region.clone()));
        }
        let config = loader.load().await;
        crate::LambdaClient::new(&config)
    }
}

/// Retrieves the full paginated execution history.
async fn fetch_history<A: CloudApi>(
    api: &A,
    execution_arn: &str,
) -> Result<Vec<Event>, TestRunnerError> {
    let mut events = Vec::new();
    let mut marker: Option<String> = None;
    loop {
        let out = api
            .send_get_durable_execution_history(execution_arn, marker.as_deref())
            .await
            .map_err(aws_error)?;
        events.extend(out.events().iter().cloned());
        match out.next_marker() {
            Some(m) if !m.is_empty() => marker = Some(m.to_owned()),
            _ => break,
        }
    }
    Ok(events)
}

/// The last terminal execution event of the history: `ExecutionSucceeded`,
/// `ExecutionFailed`, `ExecutionTimedOut` or `ExecutionStopped`.
fn terminal_execution_event(events: &[Event]) -> Option<&Event> {
    events.iter().rev().find(|e| {
        matches!(
            e.event_type(),
            Some(
                EventType::ExecutionSucceeded
                    | EventType::ExecutionFailed
                    | EventType::ExecutionTimedOut
                    | EventType::ExecutionStopped
            )
        )
    })
}

/// The error object a failed, timed-out or stopped terminal execution event
/// carries.
fn terminal_event_error(event: &Event) -> Option<&ErrorObject> {
    let error = match event.event_type() {
        Some(EventType::ExecutionFailed) => event
            .execution_failed_details()
            .and_then(aws_sdk_lambda::types::ExecutionFailedDetails::error),
        Some(EventType::ExecutionTimedOut) => event
            .execution_timed_out_details()
            .and_then(aws_sdk_lambda::types::ExecutionTimedOutDetails::error),
        Some(EventType::ExecutionStopped) => event
            .execution_stopped_details()
            .and_then(aws_sdk_lambda::types::ExecutionStoppedDetails::error),
        _ => None,
    };
    error.and_then(EventError::payload)
}

/// Deserializes the handler output from the terminal result
/// string, tolerating a double-encoded (JSON-string-of-JSON) payload.
fn deserialize_output<O: DeserializeOwned>(result: Option<&str>) -> Result<O, serde_json::Error> {
    let raw = result.unwrap_or("null");
    match serde_json::from_str::<O>(raw) {
        Ok(value) => Ok(value),
        Err(first) => match serde_json::from_str::<String>(raw) {
            Ok(inner) => serde_json::from_str::<O>(&inner),
            Err(_) => Err(first),
        },
    }
}

/// Folds a chronological execution-event stream into per-operation
/// [`TestOperation`]s in first-appearance order.
///
/// Only operation-scoped events (`Step*`, `Wait*`, `Context*`, `Callback*`,
/// `ChainedInvoke*`) with an operation id are folded; execution-level and
/// `InvocationCompleted` events are ignored here (the execution outcome comes
/// from [`terminal_execution_event`], the invocation count from
/// [`invocations_from_history`]). Later transitions overwrite earlier ones, so
/// the recorded `status` is the operation's terminal-or-latest state.
fn fold_history(events: &[Event]) -> Vec<TestOperation> {
    use std::collections::HashMap;

    let mut order: Vec<String> = Vec::new();
    let mut acc: HashMap<String, TestOperation> = HashMap::new();

    for event in events {
        let Some(kind) = event_family_op_type(event.event_type()) else {
            continue;
        };
        let Some(id) = event.id() else { continue };

        // The start time is the timestamp of the operation's first event.
        // It is set once, when that event creates the record. So a first
        // event without a representable timestamp leaves it `None`, and a
        // later event does not fill it.
        let op = acc.entry(id.to_owned()).or_insert_with(|| {
            order.push(id.to_owned());
            TestOperation {
                start_time: event_time(event),
                ..TestOperation::started(id.to_owned(), kind)
            }
        });

        if op.sub_type.is_none() {
            op.sub_type = event.sub_type().map(ToOwned::to_owned);
        }
        if op.name.is_none() {
            op.name = event.name().map(ToOwned::to_owned);
        }
        if op.parent_id.is_none() {
            op.parent_id = event.parent_id().map(ToOwned::to_owned);
        }
        apply_event_transition(op, event);
        // The end time is the timestamp of the terminal event. A retried
        // step records `StepFailed` and then `StepStarted`, so a later
        // non-terminal event clears the end time of an earlier failure.
        op.end_time = if is_terminal_status(op.status) {
            event_time(event)
        } else {
            None
        };
    }

    order.iter().filter_map(|id| acc.get(id).cloned()).collect()
}

/// Applies a single event's transition (status, result, error, attempt) to the
/// accumulating operation record.
fn apply_event_transition(op: &mut TestOperation, event: &Event) {
    match event.event_type() {
        Some(EventType::WaitStarted) => {
            op.status = OperationStatus::Started;
            let details = event.wait_started_details();
            op.wait.duration = details
                .and_then(|d| u64::try_from(d.duration()).ok())
                .map(std::time::Duration::from_secs);
            op.wait.scheduled_end_time = details
                .map(aws_sdk_lambda::types::WaitStartedDetails::scheduled_end_timestamp)
                .and_then(|t| std::time::SystemTime::try_from(*t).ok());
        }
        Some(EventType::CallbackStarted) => {
            op.status = OperationStatus::Started;
            op.callback.callback_id = event
                .callback_started_details()
                .map(|d| d.callback_id().to_owned());
        }
        Some(
            EventType::StepStarted | EventType::ContextStarted | EventType::ChainedInvokeStarted,
        ) => op.status = OperationStatus::Started,
        Some(EventType::StepSucceeded) => {
            op.status = OperationStatus::Succeeded;
            if let Some(details) = event.step_succeeded_details() {
                op.result = event_result_payload(details.result());
                set_attempt(op, details.retry_details());
            }
        }
        Some(EventType::StepFailed) => {
            op.status = OperationStatus::Failed;
            if let Some(details) = event.step_failed_details() {
                set_attempt(op, details.retry_details());
                // The latest `StepFailed` that carries a retry delay sets
                // the next attempt time; one without a delay keeps it. A
                // carried delay whose time cannot be represented sets
                // `None`, so the value never describes an older retry.
                if let Some(delay) = details
                    .retry_details()
                    .and_then(RetryDetails::next_attempt_delay_seconds)
                {
                    op.step.next_attempt_time = next_attempt_time(event, delay);
                }
            }
        }
        Some(EventType::WaitSucceeded) => op.status = OperationStatus::Succeeded,
        Some(EventType::WaitCancelled) => op.status = OperationStatus::Cancelled,
        Some(EventType::ContextSucceeded) => {
            op.status = OperationStatus::Succeeded;
            op.result = event
                .context_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::CallbackSucceeded) => {
            op.status = OperationStatus::Succeeded;
            op.result = event
                .callback_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::ChainedInvokeSucceeded) => {
            op.status = OperationStatus::Succeeded;
            op.result = event
                .chained_invoke_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(
            EventType::ContextFailed | EventType::CallbackFailed | EventType::ChainedInvokeFailed,
        ) => op.status = OperationStatus::Failed,
        Some(EventType::CallbackTimedOut | EventType::ChainedInvokeTimedOut) => {
            op.status = OperationStatus::TimedOut;
        }
        Some(EventType::ChainedInvokeStopped) => op.status = OperationStatus::Stopped,
        _ => {}
    }
    set_error(op, operation_event_error(event));
}

/// The event's timestamp, or `None` when it has none or it lies outside the
/// range of `SystemTime`.
fn event_time(event: &Event) -> Option<std::time::SystemTime> {
    event
        .event_timestamp()
        .and_then(|t| std::time::SystemTime::try_from(*t).ok())
}

/// The next attempt time of a `StepFailed` event: its timestamp plus its
/// retry delay `delay_seconds`, as the JS cloud runner derives it. `None`
/// when the event has no timestamp, the delay is negative, or the sum
/// overflows `SystemTime`.
fn next_attempt_time(event: &Event, delay_seconds: i32) -> Option<std::time::SystemTime> {
    let delay = std::time::Duration::from_secs(u64::try_from(delay_seconds).ok()?);
    event_time(event)?.checked_add(delay)
}

/// The `EventError` an operation event carries.
///
/// Eight operation event types carry an error: `StepFailed`, `WaitCancelled`,
/// `ContextFailed`, `CallbackFailed`, `CallbackTimedOut`,
/// `ChainedInvokeFailed`, `ChainedInvokeTimedOut` and `ChainedInvokeStopped`.
/// Every one of them is read here, in one place. So a transition arm cannot
/// set the status of an error-bearing event and drop its error record.
fn operation_event_error(event: &Event) -> Option<&EventError> {
    match event.event_type()? {
        EventType::StepFailed => event.step_failed_details()?.error(),
        EventType::WaitCancelled => event.wait_cancelled_details()?.error(),
        EventType::ContextFailed => event.context_failed_details()?.error(),
        EventType::CallbackFailed => event.callback_failed_details()?.error(),
        EventType::CallbackTimedOut => event.callback_timed_out_details()?.error(),
        EventType::ChainedInvokeFailed => event.chained_invoke_failed_details()?.error(),
        EventType::ChainedInvokeTimedOut => event.chained_invoke_timed_out_details()?.error(),
        EventType::ChainedInvokeStopped => event.chained_invoke_stopped_details()?.error(),
        _ => None,
    }
}

/// Extracts the JSON payload string from an optional `EventResult`.
fn event_result_payload(result: Option<&EventResult>) -> Option<String> {
    result.and_then(EventResult::payload).map(ToOwned::to_owned)
}

/// Records the error of the current event onto the op: its type, message,
/// error data and stack trace, or no error.
///
/// The op holds the error of its latest event. A retried step records
/// `StepFailed` with an error and then `StepStarted` or `StepSucceeded`
/// without one. Keeping the earlier error would report a succeeded step with
/// the error of a failed attempt. So an event without an error clears it, as
/// `LocalRunner` clears a stored error when a retry starts.
fn set_error(op: &mut TestOperation, error: Option<&EventError>) {
    op.error = error
        .and_then(EventError::payload)
        .map(wire_error_from_object);
}

/// The wire error of a service `ErrorObject`, with every field.
fn wire_error_from_object(obj: &ErrorObject) -> crate::WireError {
    crate::WireError::new(obj.error_type(), obj.error_message())
        .with_error_data(obj.error_data())
        .with_stack_trace(obj.stack_trace().to_vec())
}

/// Records the number of completed retries from an optional `RetryDetails`
/// onto the op.
///
/// `RetryDetails.current_attempt` is 1-based (a first-try success reports `1`),
/// while [`StepDetails::attempt`] reports completed retries (a first-try
/// success reports `0`). The two are reconciled here so both runners agree.
/// `saturating_sub` guards the conversion: a non-positive `current_attempt`
/// maps to `0` without underflow.
fn set_attempt(op: &mut TestOperation, retry: Option<&RetryDetails>) {
    if let Some(details) = retry {
        let one_based = u32::try_from(details.current_attempt().max(0)).unwrap_or(0);
        op.step.attempt = one_based.saturating_sub(1);
    }
}

/// Maps an event type to the [`OperationKind`] of its operation, or
/// `None` for execution-level / invocation-level events that are not folded
/// into operations.
fn event_family_op_type(event_type: Option<&EventType>) -> Option<OperationKind> {
    match event_type? {
        EventType::StepStarted | EventType::StepSucceeded | EventType::StepFailed => {
            Some(OperationKind::Step)
        }
        EventType::WaitStarted | EventType::WaitSucceeded | EventType::WaitCancelled => {
            Some(OperationKind::Wait)
        }
        EventType::ContextStarted | EventType::ContextSucceeded | EventType::ContextFailed => {
            Some(OperationKind::Context)
        }
        EventType::CallbackStarted
        | EventType::CallbackSucceeded
        | EventType::CallbackFailed
        | EventType::CallbackTimedOut => Some(OperationKind::Callback),
        EventType::ChainedInvokeStarted
        | EventType::ChainedInvokeSucceeded
        | EventType::ChainedInvokeFailed
        | EventType::ChainedInvokeTimedOut
        | EventType::ChainedInvokeStopped => Some(OperationKind::ChainedInvoke),
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

/// A stored operation record inside the in-memory backend.
#[derive(Debug, Clone)]
struct StoredOp {
    id: String,
    kind: OperationKind,
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
    /// The virtual time, in seconds, at which the record was created.
    start_at: u64,
    /// The virtual time, in seconds, at which the status first became
    /// terminal (see [`StoredOp::set_status`]), or `None` while it never
    /// has.
    end_at: Option<u64>,
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
    kind: OperationKind,
    sub_type: Option<String>,
    parent_id: Option<String>,
    now: u64,
) -> StoredOp {
    StoredOp {
        id,
        kind,
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
        start_at: now,
        end_at: None,
        callback_delivered: false,
        parent_id,
        replay_children: false,
        invoke_function: None,
        invoke_input: None,
        invoke_dispatched: false,
    }
}

impl StoredOp {
    /// Sets the status. Every status write of the backend goes through
    /// this method.
    ///
    /// `end_at` is the time the operation ended. A terminal status ends
    /// the operation, so the first terminal status records `now` as
    /// `end_at`. A later write keeps that first value. Because every path
    /// calls this method, a new status path cannot forget `end_at`.
    fn set_status(&mut self, status: OperationStatus, now: u64) {
        self.status = status;
        if self.end_at.is_none() && is_terminal_status(status) {
            self.end_at = Some(now);
        }
    }
}

/// Whether `status` ends an operation: `Succeeded`, `Failed`, `TimedOut`,
/// `Cancelled` or `Stopped`.
fn is_terminal_status(status: OperationStatus) -> bool {
    matches!(
        status,
        OperationStatus::Succeeded
            | OperationStatus::Failed
            | OperationStatus::TimedOut
            | OperationStatus::Cancelled
            | OperationStatus::Stopped
    )
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
    /// The wall-clock time the run started. A reported time is `base`
    /// plus a virtual time (see [`BackendState::local_time`]).
    base: std::time::SystemTime,
    /// The IDs of the records [`Backend::advance`] changed since the last
    /// envelope, in change order. The next envelope sends them as
    /// `UpdatedOperationIds` and clears them (see
    /// [`Backend::take_updated_ids`]). A checkpoint call never adds to it.
    updated_ids: Vec<String>,
    /// The history events, in the order of their causes (see
    /// [`BackendState::push_event`]).
    events: Vec<Event>,
    /// The event ID of the last event in `events`, or 0 before the first.
    last_event_id: i32,
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
                base: std::time::SystemTime::now(),
                updated_ids: Vec::new(),
                events: Vec::new(),
                last_event_id: 0,
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
        state
            .ops
            .iter()
            .filter_map(|op| build_operation(op, state.base))
            .collect()
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
            .map(|op| stored_op_envelope_json(op, state.base))
            .collect()
    }

    /// Builds the `Operation` records `GetDurableExecutionState` returns:
    /// the current state without the records the service omits (see
    /// [`prune_replay_children`]).
    fn build_pruned_operations(&self) -> Vec<Operation> {
        let state = self.lock();
        prune_replay_children(&state.ops)
            .into_iter()
            .filter_map(|op| build_operation(op, state.base))
            .collect()
    }

    /// The virtual time.
    fn now(&self) -> u64 {
        self.lock().now
    }

    /// The wall-clock time the run started.
    fn base(&self) -> std::time::SystemTime {
        self.lock().base
    }

    /// The recorded history events.
    fn events(&self) -> Vec<Event> {
        self.lock().events.clone()
    }

    /// Records `ExecutionStarted`, with `input` as its input payload.
    fn record_execution_started(&self, input: &str) {
        // `ExecutionStartedDetails` requires `ExecutionTimeout`. The local
        // backend has no execution timeout, so the event records 0.
        #[expect(clippy::expect_used)] // reason: the one required field is set
        let details = aws_sdk_lambda::types::ExecutionStartedDetails::builder()
            .input(EventInput::builder().payload(input).build())
            .execution_timeout(0)
            .build()
            .expect("the execution timeout is set");
        self.lock().push_event(
            execution_event(EventType::ExecutionStarted).execution_started_details(details),
        );
    }

    /// Records `InvocationCompleted` for an invocation that started at the
    /// virtual time `started_at` and ends now, with `error` as its error.
    fn record_invocation_completed(
        &self,
        started_at: u64,
        request_id: &str,
        error: Option<&crate::WireError>,
    ) {
        let mut state = self.lock();
        #[expect(clippy::expect_used)] // reason: the three required fields are set
        let details = InvocationCompletedDetails::builder()
            .start_timestamp(state.event_time(started_at))
            .end_timestamp(state.event_time(state.now))
            .request_id(request_id)
            .set_error(error.map(event_error))
            .build()
            .expect("the timestamps and the request ID are set");
        state.push_event(
            Event::builder()
                .event_type(EventType::InvocationCompleted)
                .invocation_completed_details(details),
        );
    }

    /// Records `ExecutionSucceeded` or `ExecutionFailed` for a run that
    /// ended with `end`. A suspended run records nothing.
    fn record_execution_end(&self, end: &RawEnd) {
        let event = match end {
            RawEnd::Succeeded(result) => execution_event(EventType::ExecutionSucceeded)
                .execution_succeeded_details(
                    aws_sdk_lambda::types::ExecutionSucceededDetails::builder()
                        .set_result(event_result(result.as_deref()))
                        .build(),
                ),
            RawEnd::Failed(wire) => execution_event(EventType::ExecutionFailed)
                .execution_failed_details(
                    aws_sdk_lambda::types::ExecutionFailedDetails::builder()
                        .error(event_error(wire))
                        .build(),
                ),
            RawEnd::Suspended { .. } => return,
        };
        self.lock().push_event(event);
    }

    /// Snapshots the recorded operations as public [`TestOperation`]s.
    fn snapshot_operations(&self) -> Vec<TestOperation> {
        let state = self.lock();
        state
            .ops
            .iter()
            .map(|op| {
                let mut out = TestOperation::started(op.id.clone(), op.kind);
                out.sub_type.clone_from(&op.sub_type);
                out.status = op.status;
                out.result.clone_from(&op.result);
                out.error = stored_wire_error(op);
                out.name.clone_from(&op.name);
                out.parent_id.clone_from(&op.parent_id);
                out.start_time = state.local_time(op.start_at);
                out.end_time = op.end_at.and_then(|t| state.local_time(t));
                // `due_at` is the wait's scheduled end and the step's next
                // attempt time. Only a step `RETRY` sets it on a step.
                let due = op.due_at.and_then(|t| state.local_time(t));
                out.step.attempt = u32::try_from(op.attempt.max(0)).unwrap_or(0);
                out.step.next_attempt_time = due;
                out.wait.scheduled_end_time = due;
                out.wait.duration = op
                    .wait_seconds
                    .and_then(|seconds| u64::try_from(seconds).ok())
                    .map(std::time::Duration::from_secs);
                out.callback.callback_id.clone_from(&op.callback_id);
                out.context.replay_children = op.replay_children;
                out
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
        // The events of the settled records, appended after the pass.
        let mut events = Vec::new();
        let deliveries = take_callback_deliveries(ops, callback_replies);
        if deliveries.iter().any(Option::is_some) {
            // `deliveries` holds one entry per record, in record order, so
            // zipping pairs each reply with the record it matched.
            for (op, reply) in ops.iter_mut().zip(deliveries) {
                if let Some(reply) = reply {
                    reply.deliver_to(op, *now);
                    updated_ids.push(op.id.clone());
                    events.extend(settled_event(op));
                }
            }
        } else {
            let Some(next) = ops.iter().filter_map(timed_due_at).min() else {
                return false;
            };
            *now = next;
            for op in ops.iter_mut() {
                if timed_due_at(op).is_some_and(|due| due <= next) {
                    let fired = if op.kind == OperationKind::Wait {
                        OperationStatus::Succeeded
                    } else {
                        OperationStatus::Ready
                    };
                    op.set_status(fired, next);
                    updated_ids.push(op.id.clone());
                    events.extend(settled_event(op));
                }
            }
        }
        for event in events {
            state.push_event(event);
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
                op.kind == OperationKind::ChainedInvoke
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
            ops,
            updated_ids,
            now,
            ..
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
                op.set_status(OperationStatus::Succeeded, *now);
                op.result = Some(result.unwrap_or_else(|| "null".to_owned()));
            }
            NestedOutcome::Failed(wire) => {
                op.set_status(OperationStatus::Failed, *now);
                op.error_type = wire.error_type().map(ToOwned::to_owned);
                op.error_message = wire.error_message().map(ToOwned::to_owned);
                op.error_data = wire.error_data().map(ToOwned::to_owned);
                op.stack_trace = wire.stack_trace().to_vec();
            }
            NestedOutcome::Suspended => return false,
        }
        updated_ids.push(op.id.clone());
        // A recorded outcome is `Succeeded` or `Failed`, so the invoke has
        // a settled event.
        if let Some(event) = settled_event(op) {
            state.push_event(event);
        }
        true
    }

    /// The function names of the `Started` chained invokes, in stored
    /// order, each name once.
    fn pending_invoke_names(&self) -> Vec<String> {
        let state = self.lock();
        let mut names: Vec<String> = Vec::new();
        for op in &state.ops {
            if op.kind == OperationKind::ChainedInvoke
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
    op.kind == OperationKind::Callback
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
            && parent.kind == OperationKind::Context
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
    match (op.kind, op.status) {
        (OperationKind::Wait, OperationStatus::Started)
        | (OperationKind::Step, OperationStatus::Pending) => op.due_at,
        _ => None,
    }
}

/// Converts a wire duration in seconds to the clock's unit. A missing or
/// negative value counts as 0, as in the JS and Python checkpoint servers.
fn whole_seconds(value: Option<i32>) -> u64 {
    u64::from(value.unwrap_or(0).max(0).unsigned_abs())
}

impl BackendState {
    /// The reported time of the virtual time `t`: the run's start time plus
    /// `t` seconds. A sum past the range of `SystemTime` reports `None`.
    fn local_time(&self, t: u64) -> Option<std::time::SystemTime> {
        self.base.checked_add(std::time::Duration::from_secs(t))
    }

    /// The event timestamp of the virtual time `t`: the run's start time
    /// plus `t` seconds, as [`local_time`](Self::local_time) reports it.
    ///
    /// `DateTime` counts seconds in an `i64`, so the sum cannot leave its
    /// range: `t` is a sum of `i32` wait and retry durations.
    fn event_time(&self, t: u64) -> DateTime {
        virtual_date_time(self.base, t)
    }

    /// Appends the event `builder` describes, with the next event ID
    /// (counting from 1) and the current virtual time as its timestamp.
    fn push_event(&mut self, builder: EventBuilder) {
        self.last_event_id = self.last_event_id.saturating_add(1);
        let event = builder
            .event_id(self.last_event_id)
            .event_timestamp(self.event_time(self.now))
            .build();
        self.events.push(event);
    }

    /// The history event of an accepted `update` of the stored op `op`, as
    /// the table of [`TestResult::history_events`] states, or `None` when
    /// the update has no event of its own (an `EXECUTION` update, or an
    /// action the table does not list).
    #[expect(clippy::too_many_lines)] // reason: one arm per row of the table
    fn update_event(&self, op: &StoredOp, update: &OperationUpdate) -> Option<EventBuilder> {
        let error = || stored_wire_error(op).as_ref().map(event_error);
        let result = || event_result(op.result.as_deref());
        let event = match (op.kind, update.action()) {
            (OperationKind::Step, OperationAction::Start) => op_event(op, EventType::StepStarted)
                .step_started_details(aws_sdk_lambda::types::StepStartedDetails::builder().build()),
            // `attempt` counts completed retries, and `CurrentAttempt` is
            // 1-based.
            (OperationKind::Step, OperationAction::Succeed) => {
                op_event(op, EventType::StepSucceeded).step_succeeded_details(
                    aws_sdk_lambda::types::StepSucceededDetails::builder()
                        .set_result(result())
                        .retry_details(retry_details(op.attempt.saturating_add(1), None))
                        .build(),
                )
            }
            (OperationKind::Step, OperationAction::Fail) => op_event(op, EventType::StepFailed)
                .step_failed_details(
                    aws_sdk_lambda::types::StepFailedDetails::builder()
                        .set_error(error())
                        .retry_details(retry_details(op.attempt.saturating_add(1), None))
                        .build(),
                ),
            // A `RETRY` has already counted the failed attempt in
            // `attempt`, so `attempt` is the 1-based number of that attempt.
            (OperationKind::Step, OperationAction::Retry) => op_event(op, EventType::StepFailed)
                .step_failed_details(
                    aws_sdk_lambda::types::StepFailedDetails::builder()
                        .set_error(error())
                        .retry_details(retry_details(
                            op.attempt,
                            update.step_options().and_then(
                                aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds,
                            ),
                        ))
                        .build(),
                ),
            (OperationKind::Context, OperationAction::Start) => {
                op_event(op, EventType::ContextStarted).context_started_details(
                    aws_sdk_lambda::types::ContextStartedDetails::builder().build(),
                )
            }
            (OperationKind::Context, OperationAction::Succeed) => {
                op_event(op, EventType::ContextSucceeded).context_succeeded_details(
                    aws_sdk_lambda::types::ContextSucceededDetails::builder()
                        .set_result(result())
                        .build(),
                )
            }
            (OperationKind::Context, OperationAction::Fail) => {
                op_event(op, EventType::ContextFailed).context_failed_details(
                    aws_sdk_lambda::types::ContextFailedDetails::builder()
                        .set_error(error())
                        .build(),
                )
            }
            (OperationKind::Wait, OperationAction::Start) => {
                // A missing duration counts as 0, as the stored due time
                // does (see [`whole_seconds`]).
                let duration = op.wait_seconds.unwrap_or(0);
                #[expect(clippy::expect_used)] // reason: both required fields are set
                let details = aws_sdk_lambda::types::WaitStartedDetails::builder()
                    .duration(duration)
                    .scheduled_end_timestamp(
                        self.event_time(self.now.saturating_add(whole_seconds(Some(duration)))),
                    )
                    .build()
                    .expect("the duration and the scheduled end are set");
                op_event(op, EventType::WaitStarted).wait_started_details(details)
            }
            (OperationKind::Wait, OperationAction::Cancel) => {
                op_event(op, EventType::WaitCancelled).wait_cancelled_details(
                    aws_sdk_lambda::types::WaitCancelledDetails::builder()
                        .set_error(error())
                        .build(),
                )
            }
            (OperationKind::Callback, OperationAction::Start) => {
                let options = update.callback_options();
                // A callback `START` assigns the callback ID when the
                // record has none (see `apply_update`).
                #[expect(clippy::expect_used)] // reason: the callback ID is set
                let details = aws_sdk_lambda::types::CallbackStartedDetails::builder()
                    .set_callback_id(op.callback_id.clone())
                    .set_timeout(
                        options.map(aws_sdk_lambda::types::CallbackOptions::timeout_seconds),
                    )
                    .set_heartbeat_timeout(
                        options
                            .map(aws_sdk_lambda::types::CallbackOptions::heartbeat_timeout_seconds),
                    )
                    .build()
                    .expect("a callback START sets the callback ID");
                op_event(op, EventType::CallbackStarted).callback_started_details(details)
            }
            (OperationKind::ChainedInvoke, OperationAction::Start) => {
                // A `START` without `ChainedInvokeOptions` names no
                // function, and records the empty name.
                #[expect(clippy::expect_used)] // reason: the function name is set
                let details = aws_sdk_lambda::types::ChainedInvokeStartedDetails::builder()
                    .function_name(op.invoke_function.clone().unwrap_or_default())
                    .set_input(
                        op.invoke_input
                            .as_deref()
                            .map(|p| EventInput::builder().payload(p).build()),
                    )
                    .build()
                    .expect("the function name is set");
                op_event(op, EventType::ChainedInvokeStarted)
                    .chained_invoke_started_details(details)
            }
            _ => return None,
        };
        Some(event)
    }

    /// Checks a checkpoint request against the history rules (see
    /// [`validate_history_rules`]), then applies its updates in order and
    /// returns the records the response carries.
    ///
    /// The rules read the state stored before the request. So a rejected
    /// request changes no operation, no token and no history event.
    fn apply_request(
        &mut self,
        updates: &[OperationUpdate],
        kinds: Vec<OperationKind>,
    ) -> Result<Vec<Operation>, String> {
        validate_history_rules(&self.ops, updates, &kinds)?;
        self.token_counter += 1;
        let mut out = Vec::with_capacity(updates.len());
        for (update, kind) in updates.iter().zip(kinds) {
            if let Some(stored) = self.apply_update(kind, update)
                && let Some(op) = build_operation(&stored, self.base)
            {
                out.push(op);
            }
        }
        Ok(out)
    }

    /// Finds a stored op by wire ID.
    fn find_mut(&mut self, id: &str) -> Option<&mut StoredOp> {
        self.ops.iter_mut().find(|op| op.id == id)
    }

    /// Applies a single `OperationUpdate` to the store, returning the
    /// resulting stored op (cloned) for the checkpoint response.
    #[expect(clippy::too_many_lines)] // reason: one match over the update actions reads better as one flow
    fn apply_update(&mut self, kind: OperationKind, update: &OperationUpdate) -> Option<StoredOp> {
        let id = update.id().to_owned();
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
                kind,
                sub_type.clone(),
                parent_id,
                now,
            ));
        }

        // Decide callback-id assignment before taking the mutable op borrow
        // (assigning self.callback_counter later would conflict with it).
        let needs_callback_id = matches!(action, OperationAction::Start)
            && kind == OperationKind::Callback
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
                if op.kind == OperationKind::Wait {
                    op.set_status(OperationStatus::Started, now);
                    op.wait_seconds = wait_seconds;
                    op.due_at = Some(now.saturating_add(whole_seconds(wait_seconds)));
                } else if op.kind == OperationKind::Callback {
                    // The service creates a callback in `Started` and keeps
                    // it there until an external outcome. The backend
                    // assigns the callback id here.
                    op.set_status(OperationStatus::Started, now);
                    if let Some(cb) = new_callback_id {
                        op.callback_id = Some(cb);
                    }
                } else {
                    op.set_status(OperationStatus::Started, now);
                    if op.kind == OperationKind::ChainedInvoke {
                        op.invoke_function = invoke_function;
                        op.invoke_input = payload;
                    }
                }
            }
            OperationAction::Succeed => {
                op.set_status(OperationStatus::Succeeded, now);
                op.result = payload;
                if op.kind == OperationKind::Context {
                    op.replay_children = replay_children;
                }
            }
            OperationAction::Fail => {
                op.set_status(OperationStatus::Failed, now);
                if op.kind == OperationKind::Context {
                    op.replay_children = replay_children;
                }
                op.error_type = err_type;
                op.error_message = err_msg;
                op.error_data = err_data;
                op.stack_trace = err_trace;
            }
            OperationAction::Retry => {
                op.set_status(OperationStatus::Pending, now);
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

        let stored = op.clone();
        if let Some(event) = self.update_event(&stored, update) {
            self.push_event(event);
        }
        Some(stored)
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
        // The service rejects an update of a type it does not know, before
        // any state change. So every type converts before the state lock;
        // an unknown type stores nothing and keeps the token.
        let Some(kinds) = update_kinds(updates) else {
            return Box::pin(async {
                Err(ExecutionClientError::new_non_retryable(
                    "Invalid operation type.",
                ))
            });
        };
        let updated_ops = match self.lock().apply_request(updates, kinds) {
            Ok(ops) => ops,
            Err(message) => {
                return Box::pin(
                    async move { Err(ExecutionClientError::new_non_retryable(&message)) },
                );
            }
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

/// The kind of each update, in order, or `None` when an update has a type
/// that [`OperationKind`] does not list.
fn update_kinds(updates: &[OperationUpdate]) -> Option<Vec<OperationKind>> {
    updates
        .iter()
        .map(|update| OperationKind::from_wire(update.r#type()))
        .collect()
}

/// Checks a checkpoint request against the history rules of JS's checkpoint
/// server, and returns JS's message for the first rule it breaks.
///
/// `ops` is the state stored before the request, and `kinds` holds the kind
/// of each update. The checks run in JS's order:
///
/// 1. One pass over the request checks repeated IDs and parents (see
///    [`validate_parents_and_repeats`]).
/// 2. Then each update in turn is checked against the operation stored
///    before the request: first its identity, then its action against its
///    type, then its action against the stored status.
///
/// Every check compares with the stored state, never with an earlier
/// update of the same request.
fn validate_history_rules(
    ops: &[StoredOp],
    updates: &[OperationUpdate],
    kinds: &[OperationKind],
) -> Result<(), String> {
    validate_parents_and_repeats(ops, updates, kinds)?;
    for (update, kind) in updates.iter().zip(kinds) {
        let stored = ops.iter().find(|op| op.id == update.id());
        if let Some(stored) = stored {
            validate_identity(update, *kind, stored)?;
        }
        validate_action_for_type(*kind, update.action())?;
        validate_status_transition(*kind, update.action(), stored)?;
    }
    Ok(())
}

/// Checks the repeated IDs and the parents of a request, in request order.
///
/// 1. An ID may repeat only when the latest earlier update of that ID is a
///    STEP or CONTEXT `START` and the repeat is not a `START`.
/// 2. A supplied `ParentId` must name a CONTEXT that an earlier update of
///    the request starts, or a stored CONTEXT. An earlier `START` of the ID
///    takes precedence over the stored record, as in JS.
///
/// The repeat check of an update runs before its parent check.
fn validate_parents_and_repeats(
    ops: &[StoredOp],
    updates: &[OperationUpdate],
    kinds: &[OperationKind],
) -> Result<(), String> {
    let mut started: BTreeMap<&str, OperationKind> = BTreeMap::new();
    let mut last_seen: BTreeMap<&str, (OperationKind, &OperationAction)> = BTreeMap::new();
    for (update, kind) in updates.iter().zip(kinds) {
        let id = update.id();
        let action = update.action();
        if let Some((last_kind, last_action)) = last_seen.get(id) {
            let allowed = matches!(last_kind, OperationKind::Step | OperationKind::Context)
                && **last_action == OperationAction::Start
                && *action != OperationAction::Start;
            if !allowed {
                return Err("Cannot update the same operation twice in a single request.".into());
            }
        }
        if let Some(parent_id) = update.parent_id() {
            let parent_kind = started
                .get(parent_id)
                .copied()
                .or_else(|| ops.iter().find(|op| op.id == parent_id).map(|op| op.kind));
            if parent_kind != Some(OperationKind::Context) {
                return Err("Invalid parent operation id.".into());
            }
        }
        if *action == OperationAction::Start {
            started.insert(id, *kind);
        }
        last_seen.insert(id, (*kind, action));
    }
    Ok(())
}

/// Checks that an update keeps the `Type`, `SubType`, `Name` and
/// `ParentId` of the stored operation. A field the update omits passes.
fn validate_identity(
    update: &OperationUpdate,
    kind: OperationKind,
    stored: &StoredOp,
) -> Result<(), String> {
    let inconsistent = |field: &str| {
        Err(format!(
            "{field} value in OperationUpdate is inconsistent with existing operation. \
             Check for non-determinism."
        ))
    };
    let differs = |supplied: Option<&str>, stored: Option<&String>| {
        supplied.is_some_and(|value| Some(value) != stored.map(String::as_str))
    };
    if kind != stored.kind {
        return inconsistent("Type");
    }
    if differs(update.sub_type(), stored.sub_type.as_ref()) {
        return inconsistent("SubType");
    }
    if differs(update.name(), stored.name.as_ref()) {
        return inconsistent("Name");
    }
    if differs(update.parent_id(), stored.parent_id.as_ref()) {
        return inconsistent("ParentId");
    }
    Ok(())
}

/// Checks that `action` is one of the actions JS allows for `kind`.
fn validate_action_for_type(kind: OperationKind, action: &OperationAction) -> Result<(), String> {
    use OperationAction as A;
    let allowed = match kind {
        OperationKind::Step => matches!(action, A::Start | A::Fail | A::Retry | A::Succeed),
        OperationKind::Context => matches!(action, A::Start | A::Fail | A::Succeed),
        OperationKind::Wait => matches!(action, A::Start | A::Cancel),
        OperationKind::Callback | OperationKind::ChainedInvoke => matches!(action, A::Start),
        OperationKind::Execution => matches!(action, A::Succeed | A::Fail),
    };
    if allowed {
        Ok(())
    } else {
        Err("Invalid action for the given operation type.".into())
    }
}

/// Checks that `action` is valid for the status of the stored operation,
/// with the message of JS's validator for `kind`. `stored` is `None` when
/// no operation has the update's ID.
///
/// [`validate_action_for_type`] has already accepted the action, so only
/// the actions of `kind` reach this function.
fn validate_status_transition(
    kind: OperationKind,
    action: &OperationAction,
    stored: Option<&StoredOp>,
) -> Result<(), String> {
    use OperationStatus as S;
    let status = stored.map(|op| op.status);
    let message = match (kind, action) {
        (OperationKind::Step, OperationAction::Start) => status
            .is_some_and(|s| s != S::Ready)
            .then_some("Invalid current STEP state to start."),
        (OperationKind::Step, OperationAction::Retry) => status
            .is_some_and(|s| !matches!(s, S::Started | S::Ready))
            .then_some("Invalid current STEP state to retry."),
        (OperationKind::Step, _) => status
            .is_some_and(|s| !matches!(s, S::Started | S::Ready))
            .then_some("Invalid current STEP state to close."),
        (OperationKind::Context, OperationAction::Start) => status
            .is_some()
            .then_some("Cannot start a CONTEXT operation that already exists."),
        (OperationKind::Context, _) => status
            .is_some_and(|s| s != S::Started)
            .then_some("Invalid current CONTEXT state to close."),
        (OperationKind::Wait, OperationAction::Start) => status
            .is_some()
            .then_some("Cannot start a WAIT operation that already exists."),
        (OperationKind::Wait, _) => (status != Some(S::Started)).then_some(
            "Cannot cancel a WAIT operation that does not exist or has already completed.",
        ),
        (OperationKind::Callback, _) => status
            .is_some()
            .then_some("Cannot start a CALLBACK operation that already exists."),
        (OperationKind::ChainedInvoke, _) => status
            .is_some()
            .then_some("Cannot start a CHAINED_INVOKE operation that already exists."),
        (OperationKind::Execution, _) => None,
    };
    message.map_or(Ok(()), |message| Err(message.to_owned()))
}

/// Whether `stored` carries any error field. A callback failure may set
/// only some of the four fields, so any one of them makes an error object.
fn has_error_fields(stored: &StoredOp) -> bool {
    stored.error_type.is_some()
        || stored.error_message.is_some()
        || stored.error_data.is_some()
        || !stored.stack_trace.is_empty()
}

/// The error a stored op carries, or `None` when it has no error field
/// (see [`has_error_fields`]).
fn stored_wire_error(stored: &StoredOp) -> Option<crate::WireError> {
    has_error_fields(stored).then(|| {
        crate::WireError::new(
            stored.error_type.as_deref(),
            stored.error_message.as_deref(),
        )
        .with_error_data(stored.error_data.as_deref())
        .with_stack_trace(stored.stack_trace.clone())
    })
}

/// The service `ErrorObject` of a wire error, with every field it has.
fn error_object(wire: &crate::WireError) -> ErrorObject {
    let mut builder = ErrorObject::builder()
        .set_error_type(wire.error_type().map(ToOwned::to_owned))
        .set_error_message(wire.error_message().map(ToOwned::to_owned))
        .set_error_data(wire.error_data().map(ToOwned::to_owned));
    for frame in wire.stack_trace() {
        builder = builder.stack_trace(frame);
    }
    builder.build()
}

/// The `EventError` of a history event that carries `wire`.
fn event_error(wire: &crate::WireError) -> EventError {
    EventError::builder().payload(error_object(wire)).build()
}

/// The `EventResult` of a history event that carries `payload`.
fn event_result(payload: Option<&str>) -> Option<EventResult> {
    payload.map(|p| EventResult::builder().payload(p).build())
}

/// The ID of the execution operation of every synthesized envelope.
const EXECUTION_OPERATION_ID: &str = "execution";

/// An execution-level history event of `event_type`, with the ID of the
/// execution operation.
fn execution_event(event_type: EventType) -> EventBuilder {
    Event::builder()
        .event_type(event_type)
        .id(EXECUTION_OPERATION_ID)
}

/// A history event of `event_type` for the stored op `op`, with its ID,
/// name, sub-type and parent ID.
fn op_event(op: &StoredOp, event_type: EventType) -> EventBuilder {
    Event::builder()
        .event_type(event_type)
        .id(&op.id)
        .set_name(op.name.clone())
        .set_sub_type(op.sub_type.clone())
        .set_parent_id(op.parent_id.clone())
}

/// The retry details of a step event: the 1-based `current_attempt` and
/// the retry delay.
fn retry_details(current_attempt: i32, next_attempt_delay_seconds: Option<i32>) -> RetryDetails {
    RetryDetails::builder()
        .current_attempt(current_attempt)
        .set_next_attempt_delay_seconds(next_attempt_delay_seconds)
        .build()
}

/// The history event of a record the backend settled between invocations:
/// a fired wait, a callback that received a reply, or a chained invoke whose
/// target returned. A step whose retry delay elapsed (`Ready`) has none.
fn settled_event(op: &StoredOp) -> Option<EventBuilder> {
    let error = || stored_wire_error(op).as_ref().map(event_error);
    let result = || event_result(op.result.as_deref());
    let event = match (op.kind, op.status) {
        (OperationKind::Wait, OperationStatus::Succeeded) => op_event(op, EventType::WaitSucceeded)
            .wait_succeeded_details(
                aws_sdk_lambda::types::WaitSucceededDetails::builder()
                    .set_duration(op.wait_seconds)
                    .build(),
            ),
        (OperationKind::Callback, OperationStatus::Succeeded) => {
            op_event(op, EventType::CallbackSucceeded).callback_succeeded_details(
                aws_sdk_lambda::types::CallbackSucceededDetails::builder()
                    .set_result(result())
                    .build(),
            )
        }
        (OperationKind::Callback, OperationStatus::Failed) => {
            op_event(op, EventType::CallbackFailed).callback_failed_details(
                aws_sdk_lambda::types::CallbackFailedDetails::builder()
                    .set_error(error())
                    .build(),
            )
        }
        (OperationKind::Callback, OperationStatus::TimedOut) => {
            op_event(op, EventType::CallbackTimedOut).callback_timed_out_details(
                aws_sdk_lambda::types::CallbackTimedOutDetails::builder()
                    .set_error(error())
                    .build(),
            )
        }
        (OperationKind::ChainedInvoke, OperationStatus::Succeeded) => {
            op_event(op, EventType::ChainedInvokeSucceeded).chained_invoke_succeeded_details(
                aws_sdk_lambda::types::ChainedInvokeSucceededDetails::builder()
                    .set_result(result())
                    .build(),
            )
        }
        (OperationKind::ChainedInvoke, OperationStatus::Failed) => {
            op_event(op, EventType::ChainedInvokeFailed).chained_invoke_failed_details(
                aws_sdk_lambda::types::ChainedInvokeFailedDetails::builder()
                    .set_error(error())
                    .build(),
            )
        }
        _ => return None,
    };
    Some(event)
}

/// The `DateTime` of the virtual time `t`: the run's start time `base`
/// plus `t` seconds, the time [`TestOperation::start_time`] and
/// [`TestOperation::end_time`] report.
///
/// `DateTime` counts seconds in an `i64`, so the sum cannot leave its
/// range: `t` is a sum of `i32` wait and retry durations. The sum
/// saturates, so no conversion panics.
fn virtual_date_time(base: std::time::SystemTime, t: u64) -> DateTime {
    let base = DateTime::from(base);
    DateTime::from_secs_and_nanos(
        base.secs().saturating_add(t.cast_signed()),
        base.subsec_nanos(),
    )
}

/// The epoch milliseconds of the virtual time `t`: the run's start time
/// `base` in epoch milliseconds plus `t` seconds. The envelope carries
/// operation times in this unit, as the service does. The arithmetic
/// saturates, so no conversion panics.
fn virtual_epoch_millis(base: std::time::SystemTime, t: u64) -> u64 {
    let base_ms = base
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
    base_ms.saturating_add(t.saturating_mul(1000))
}

/// Builds an SDK `Operation` from a stored op, placing result/error in the
/// details bucket the SDK reads for that operation type. The start and end
/// times are the run's start time `base` plus the record's virtual times.
fn build_operation(stored: &StoredOp, base: std::time::SystemTime) -> Option<Operation> {
    let error_obj = stored_wire_error(stored).map(|wire| error_object(&wire));

    let mut builder = Operation::builder()
        .id(&stored.id)
        .r#type(OperationType::from(stored.kind.wire_str()))
        .status(WireOperationStatus::from(stored.status.wire_str()))
        .start_timestamp(virtual_date_time(base, stored.start_at));
    if let Some(end_at) = stored.end_at {
        builder = builder.end_timestamp(virtual_date_time(base, end_at));
    }

    if let Some(ref st) = stored.sub_type {
        builder = builder.sub_type(st);
    }
    if let Some(ref n) = stored.name {
        builder = builder.name(n);
    }
    if let Some(ref parent) = stored.parent_id {
        builder = builder.parent_id(parent);
    }

    match stored.kind {
        OperationKind::Step => {
            let mut sd = WireStepDetails::builder().attempt(stored.attempt);
            if let Some(r) = &stored.result {
                sd = sd.result(r);
            }
            if let Some(e) = error_obj.clone() {
                sd = sd.error(e);
            }
            builder = builder.step_details(sd.build());
        }
        OperationKind::Context => {
            let mut cd = WireContextDetails::builder();
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
        OperationKind::Callback => {
            let mut cb = WireCallbackDetails::builder();
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
        OperationKind::ChainedInvoke => {
            let mut ci = ChainedInvokeDetails::builder();
            if let Some(r) = &stored.result {
                ci = ci.result(r);
            }
            if let Some(e) = error_obj.clone() {
                ci = ci.error(e);
            }
            builder = builder.chained_invoke_details(ci.build());
        }
        OperationKind::Wait | OperationKind::Execution => {}
    }

    builder.build().ok()
}

/// Renders a stored op as an envelope-wire JSON operation object, placing
/// result/error in the details bucket `parse_single_operation` reads for
/// that operation type. Mirrors [`build_operation`] field-for-field so the
/// inline envelope page and the `get_state` fetch describe identical
/// records. The start and end times are epoch milliseconds: the run's start
/// time `base` plus the record's virtual times.
fn stored_op_envelope_json(stored: &StoredOp, base: std::time::SystemTime) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("Id".to_owned(), serde_json::json!(stored.id));
    obj.insert("Type".to_owned(), serde_json::json!(stored.kind.wire_str()));
    obj.insert(
        "Status".to_owned(),
        serde_json::json!(stored.status.wire_str()),
    );
    obj.insert(
        "StartTimestamp".to_owned(),
        serde_json::json!(virtual_epoch_millis(base, stored.start_at)),
    );
    if let Some(end_at) = stored.end_at {
        obj.insert(
            "EndTimestamp".to_owned(),
            serde_json::json!(virtual_epoch_millis(base, end_at)),
        );
    }
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

    match stored.kind {
        OperationKind::Step => {
            details.insert("Attempt".to_owned(), serde_json::json!(stored.attempt));
            obj.insert("StepDetails".to_owned(), serde_json::Value::Object(details));
        }
        OperationKind::Context => {
            if stored.replay_children {
                details.insert("ReplayChildren".to_owned(), serde_json::json!(true));
            }
            obj.insert(
                "ContextDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationKind::Callback => {
            if let Some(id) = &stored.callback_id {
                details.insert("CallbackId".to_owned(), serde_json::json!(id));
            }
            obj.insert(
                "CallbackDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationKind::ChainedInvoke => {
            obj.insert(
                "ChainedInvokeDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationKind::Wait | OperationKind::Execution => {}
    }

    serde_json::Value::Object(obj)
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
            if parent.kind == OperationKind::Context
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
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::builders::map_parallel::{CompletionConfig, CompletionDecision, CompletionOutcome};
    use crate::{RetryDecision, StepError};
    use serde::{Deserialize, Serialize};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// The completed retries of a step operation, from its details, or
    /// `None` for another kind.
    fn step_attempt(op: &TestOperation) -> Option<u32> {
        match op.details() {
            TestOperationDetails::Step(step) => Some(step.attempt()),
            _ => None,
        }
    }

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
            events: Vec::new(),
        }
        .into_test_result()
        .expect("runner");
        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("Error"));
        assert_eq!(result.error_message(), Some("boom"));

        // An `Error` object with neither type nor message reads type `Error`
        // and an empty message.
        let response = serde_json::json!({ "Status": "FAILED", "Error": {} });
        let result: TestResult<()> = RawRun {
            end: RawEnd::Failed(envelope_wire_error(&response)),
            operations: Vec::new(),
            invocations: 1,
            events: Vec::new(),
        }
        .into_test_result()
        .expect("runner");
        assert_eq!(result.error_type(), Some("Error"));
        assert_eq!(result.error_message(), Some(""));
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_failure());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(result.operations().iter().any(TestOperation::failed));
    }

    /// A test error type the `retry_if` `LocalRunner` tests match by type.
    #[derive(Debug)]
    struct Transient(u32);

    impl std::fmt::Display for Transient {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "transient on attempt {}", self.0)
        }
    }

    impl std::error::Error for Transient {}

    /// Describes an awaited step error: `Step(ExecutionFailed)` plus the
    /// source's message, or the error itself for any other shape.
    fn describe_step_failure(err: &crate::OperationError) -> String {
        match err.kind() {
            crate::OperationErrorKind::Step(step) => match step.kind() {
                crate::StepErrorKind::ExecutionFailed => format!(
                    "ExecutionFailed: {}",
                    std::error::Error::source(step).map_or_else(String::new, ToString::to_string)
                ),
                other => format!("other step kind: {other}"),
            },
            _ => format!("not a step error: {err:#}"),
        }
    }

    #[tokio::test]
    async fn step_retry_if_false_fails_after_one_attempt() {
        // A predicate that rejects every error stops at the first failure,
        // although max_attempts(3) would allow two retries.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(3)
                            .retry_if(|_| false)
                            .build();
                        let outcome = ctx
                            .step(move |_| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    Err::<i32, BoxError>("always fails".into())
                                }
                            })
                            .name("filtered")
                            .retry_strategy_config(config)
                            .await;
                        let err = outcome.expect_err("the step fails");
                        Ok::<_, BoxError>(describe_step_failure(&err))
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output().map(String::as_str),
            Some("ExecutionFailed: always fails")
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let op = result.operation("filtered").expect("the step is recorded");
        assert_eq!(op.status(), OperationStatus::Failed);
    }

    #[tokio::test]
    async fn step_retry_if_retries_matching_error_until_cap() {
        // A predicate that accepts `Transient` retries it until the cap,
        // and the awaited error carries the third attempt's `Transient`.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(3)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .retry_if(|err| {
                                std::error::Error::source(err)
                                    .and_then(|e| e.downcast_ref::<Transient>())
                                    .is_some()
                            })
                            .build();
                        let outcome = ctx
                            .step(move |sc| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    Err::<i32, BoxError>(Box::new(Transient(sc.attempt())))
                                }
                            })
                            .name("transient")
                            .retry_strategy_config(config)
                            .await;
                        let err = outcome.expect_err("the step fails");
                        // On the live path the source is the third
                        // attempt's own `Transient`.
                        let third = match err.kind() {
                            crate::OperationErrorKind::Step(step) => {
                                std::error::Error::source(step)
                                    .and_then(|e| e.downcast_ref::<Transient>())
                                    .map(|t| t.0)
                            }
                            _ => None,
                        };
                        Ok::<_, BoxError>((describe_step_failure(&err), third))
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(
            result.output(),
            Some(&(
                "ExecutionFailed: transient on attempt 3".to_owned(),
                Some(3)
            ))
        );
        let op = result.operation("transient").expect("the step is recorded");
        assert_eq!(op.status(), OperationStatus::Failed);
    }

    #[tokio::test]
    async fn wait_for_callback_submitter_retry_if() {
        // A submitter that fails with "boom" and a predicate that rejects
        // it: the submitter runs once and the operation fails.
        let submits = Arc::new(AtomicU32::new(0));
        let submits_seen = Arc::clone(&submits);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let submits_seen = Arc::clone(&submits_seen);
                    async move {
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(3)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .retry_if(|_| false)
                            .build();
                        let outcome = ctx
                            .wait_for_callback::<u32, _, _>(move |_step_ctx, _callback_id| {
                                let submits_seen = Arc::clone(&submits_seen);
                                async move {
                                    if submits_seen.fetch_add(1, Ordering::SeqCst) == 0 {
                                        Err::<(), BoxError>("boom".into())
                                    } else {
                                        Ok(())
                                    }
                                }
                            })
                            .name("approval")
                            .submitter_retry_config(config)
                            .await;
                        Ok::<_, BoxError>(outcome.is_err())
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&true), "the operation fails");
        assert_eq!(submits.load(Ordering::SeqCst), 1);
        let op = result
            .operation("approval")
            .expect("the operation is recorded");
        assert_eq!(op.status(), OperationStatus::Failed);
    }

    #[tokio::test]
    async fn with_retry_retry_if_matches_recorded_message() {
        // Attempt 1 fails with "transient" and attempt 2 succeeds. The
        // predicate reads the attempt's record through `wire()`, so it
        // answers the same live and when replay asks it again about
        // attempt 1 after the backoff suspension.
        let block_runs = Arc::new(AtomicU32::new(0));
        let block_runs_seen = Arc::clone(&block_runs);
        let answers = Arc::new(Mutex::new(Vec::new()));
        let answers_seen = Arc::clone(&answers);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let block_runs_seen = Arc::clone(&block_runs_seen);
                    let answers_seen = Arc::clone(&answers_seen);
                    async move {
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(3)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .retry_if(move |err| {
                                let matched = std::error::Error::source(err)
                                    .and_then(|e| e.downcast_ref::<crate::OperationError>())
                                    .and_then(crate::OperationError::wire)
                                    .and_then(crate::WireError::error_message)
                                    .is_some_and(|message| message.contains("transient"));
                                answers_seen.lock().expect("answers").push(matched);
                                matched
                            })
                            .build();
                        let value = ctx
                            .with_retry(move |_child| {
                                let block_runs_seen = Arc::clone(&block_runs_seen);
                                async move {
                                    if block_runs_seen.fetch_add(1, Ordering::SeqCst) == 0 {
                                        Err::<u32, BoxError>("transient".into())
                                    } else {
                                        Ok(9)
                                    }
                                }
                            })
                            .name("block")
                            .retry_strategy_config(config)
                            .await?;
                        Ok::<_, BoxError>(value)
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&9));
        assert_eq!(block_runs.load(Ordering::SeqCst), 2, "two attempts ran");
        assert!(result.operation("attempt-1").expect("attempt 1").failed());
        assert!(
            result
                .operation("attempt-2")
                .expect("attempt 2")
                .succeeded()
        );
        assert!(result.invocation_count() >= 2, "the backoff suspended");
        let answers = answers.lock().expect("answers").clone();
        assert!(
            answers.len() >= 2,
            "the predicate is asked live and again on replay: {answers:?}"
        );
        assert!(answers.iter().all(|matched| *matched), "{answers:?}");
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
            .await
            .expect("runner");

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
        assert_eq!(
            step_attempt(step),
            Some(2),
            "backend recorded 2 completed retries"
        );
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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("before-after"));
        // The wait forced at least a second invocation.
        assert!(result.invocation_count() >= 2);
        let wait_op = result
            .operations()
            .iter()
            .find(|o| o.kind() == OperationKind::Wait)
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&30));
        // Parent context op + two child steps recorded.
        assert!(
            result
                .operations()
                .iter()
                .any(|o| o.kind() == OperationKind::Context)
        );
        let step_count = result
            .operations()
            .iter()
            .filter(|o| o.kind() == OperationKind::Step)
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&60));
    }

    /// Replay of a predicate-completed batch is stable: the execution
    /// suspends mid-batch (a timer inside an item), resumes on a later
    /// invocation, replays the already-settled item from its checkpoint
    /// record without re-running its body, and the custom predicate then
    /// ends the batch early with the `CUSTOM_COMPLETION_SUCCEEDED` reason.
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
                                    .completion_predicate(|stats| {
                                        if stats.settled() >= 2 {
                                            CompletionDecision::Complete(
                                                CompletionOutcome::Succeeded,
                                            )
                                        } else {
                                            CompletionDecision::Continue
                                        }
                                    })
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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        // The predicate fired after two settles; item 2 never started.
        assert_eq!(
            result.output(),
            Some(&"CUSTOM_COMPLETION_SUCCEEDED:2".to_owned())
        );
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
    /// CONCURRENT batch: on resume, recorded-terminal children settle
    /// first, before any resumed (live) branch joins.
    ///
    /// Three items at `max_concurrency(2)`: items 0 and 2 succeed on the
    /// first invocation, item 1 parks on a timer and fails after resume.
    /// The predicate `failed() > succeeded()` must never fire: on the
    /// resumed invocation the two recorded successes settle before item 1's
    /// live failure joins, so the predicate sees 2 succeeded before it sees
    /// 1 failed, regardless of how the scheduler orders the join events. If
    /// resumed item 1's failure could join before item 0's recorded success,
    /// the predicate would see 1 failed / 0 succeeded and stop the batch,
    /// and item 2, whose success is already in the checkpoint log, would be
    /// dropped from the result.
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
                                // A predicate-only config: only the
                                // predicate could stop early.
                                .completion_predicate(|stats| {
                                    if stats.failed() > stats.succeeded() {
                                        CompletionDecision::Complete(CompletionOutcome::Succeeded)
                                    } else {
                                        CompletionDecision::Continue
                                    }
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
                },
                (),
            )
            .await
            .expect("runner");

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

    /// Reversed live settlement order PLUS a genuine suspension: item 1
    /// succeeds before the delayed item 0 fails (both on the first
    /// invocation), item 2 parks and only succeeds after resume. The
    /// predicate `failed() > succeeded() + 1` stays false on every snapshot,
    /// live and on resume, so the batch runs to completion with every
    /// item's outcome in the result, on both sides of the suspension
    /// boundary.
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
                                            // The gate and the yields are
                                            // not SDK operations: they run
                                            // under a held activity permit,
                                            // so a waiting sibling cannot
                                            // suspend the invocation here.
                                            child
                                                .hold_activity(async {
                                                    gate.notified().await;
                                                    for _ in 0..16 {
                                                        tokio::task::yield_now().await;
                                                    }
                                                })
                                                .await;
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
                                    // Never true here: one failure can never
                                    // outnumber the successes by two.
                                    .completion_predicate(|stats| {
                                        if stats.failed() > stats.succeeded() + 1 {
                                            CompletionDecision::Complete(
                                                CompletionOutcome::Succeeded,
                                            )
                                        } else {
                                            CompletionDecision::Continue
                                        }
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .find(|o| o.kind() == OperationKind::Callback)
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("CallbackError"));
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), OperationStatus::TimedOut);
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
        assert_eq!(cb_op.status(), OperationStatus::Failed);
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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&true),
            "kind is Callback(ExternalFailure)"
        );
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), OperationStatus::Failed);
        assert_eq!(cb_op.error_type(), Some("Declined"));
    }

    /// What a `wait_for_callback` handler saw in one invocation: the
    /// `CallbackErrorKind` name, then the wire type, message and data.
    type WfcbObservation = (String, Option<String>, Option<String>, Option<String>);

    /// How the submitter of [`run_wfcb_classification`] ends.
    #[derive(Clone, Copy)]
    enum Submitter {
        /// It succeeds.
        Succeeds,
        /// It fails with a plain error, `submission failed`.
        FailsUntyped,
        /// It fails with `TypedError::with_type("PaymentError", "declined")`.
        FailsTyped,
    }

    /// The run of a `wait_for_callback` named `approve` whose failure the
    /// handler observes live and on replay.
    struct WfcbClassification {
        result: TestResult<String>,
        /// One entry per invocation that read the failure.
        seen: Vec<WfcbObservation>,
        /// The `{:#}` text of the live error's source, when that source is
        /// a `Step(ExecutionFailed)` operation error.
        live_step_source: Option<String>,
    }

    /// Runs a handler that calls `wait_for_callback` and records the kind
    /// and wire fields of its error outside any step, so a replay records
    /// them again. When the kind is `expected`, the handler waits one
    /// second, so a later invocation replays the failure, and returns
    /// `output`; any other kind returns the error.
    async fn run_wfcb_classification(
        runner: LocalRunner,
        submitter: Submitter,
        expected: &'static str,
        output: &'static str,
    ) -> WfcbClassification {
        let seen: Arc<Mutex<Vec<WfcbObservation>>> = Arc::default();
        let live_source: Arc<Mutex<Option<String>>> = Arc::default();
        let (seen_in, live_in) = (Arc::clone(&seen), Arc::clone(&live_source));
        let result = runner
            .run(
                move |(), ctx: DurableContext| {
                    let (seen, live_source) = (Arc::clone(&seen_in), Arc::clone(&live_in));
                    async move {
                        let outcome = ctx
                            .wait_for_callback::<String, _, _>(move |_step_ctx, _id| async move {
                                match submitter {
                                    Submitter::Succeeds => Ok(()),
                                    Submitter::FailsUntyped => Err("submission failed".into()),
                                    Submitter::FailsTyped => Err(Box::new(
                                        crate::TypedError::with_type("PaymentError", "declined"),
                                    )
                                        as BoxError),
                                }
                            })
                            .name("approve")
                            .submitter_retry(|_, _| RetryDecision::Stop)
                            .await;
                        let Err(err) = outcome else {
                            return Err::<String, BoxError>("wait_for_callback succeeded".into());
                        };
                        let kind = match err.kind() {
                            crate::OperationErrorKind::Callback(e) => format!("{:?}", e.kind()),
                            other => format!("{other:?}"),
                        };
                        let field = |read: fn(&crate::WireError) -> Option<&str>| {
                            err.wire().and_then(read).map(ToOwned::to_owned)
                        };
                        {
                        let mut seen = seen.lock().expect("not poisoned");
                        if seen.is_empty() {
                            // The first invocation that reads the failure
                            // is the live one.
                            let step = std::error::Error::source(&err)
                                .and_then(std::error::Error::source)
                                .and_then(|s| s.downcast_ref::<crate::OperationError>())
                                .filter(|op| {
                                    matches!(
                                        op.kind(),
                                        crate::OperationErrorKind::Step(e)
                                            if matches!(e.kind(), crate::StepErrorKind::ExecutionFailed)
                                    )
                                });
                            *live_source.lock().expect("not poisoned") =
                                step.map(|op| format!("{op:#}"));
                        }
                        seen.push((
                            kind.clone(),
                            field(crate::WireError::error_type),
                            field(crate::WireError::error_message),
                            field(crate::WireError::error_data),
                        ));
                        }
                        if kind != expected {
                            return Err(err.into());
                        }
                        ctx.wait(std::time::Duration::from_secs(1)).await?;
                        Ok(output.to_owned())
                    }
                },
                (),
            )
            .await
            .expect("runner");
        let seen = seen.lock().expect("not poisoned").clone();
        let live_step_source = live_source.lock().expect("not poisoned").clone();
        WfcbClassification {
            result,
            seen,
            live_step_source,
        }
    }

    /// The recorded outer FAIL type of the `wait_for_callback` named
    /// `approve`.
    fn approve_error_type(result: &TestResult<String>) -> Option<&str> {
        result
            .operation("approve")
            .expect("the wait_for_callback is recorded")
            .error_type()
    }

    #[tokio::test]
    async fn wfcb_external_failure_same_kind_live_and_replay() {
        let runner = LocalRunner::new().callback(CallbackReply::failure(
            CallbackFailure::new("PaymentDeclined", "card declined")
                .with_error_data(r#"{"code":7}"#),
        ));
        let run =
            run_wfcb_classification(runner, Submitter::Succeeds, "ExternalFailure", "external")
                .await;
        assert!(run.result.is_success(), "{:?}", run.result.error_message());
        assert_eq!(run.result.output().map(String::as_str), Some("external"));
        let entry = (
            "ExternalFailure".to_owned(),
            Some("Callback.ExternalFailure".to_owned()),
            Some("card declined".to_owned()),
            Some(r#"{"code":7}"#.to_owned()),
        );
        assert_eq!(run.seen, [entry.clone(), entry]);
        assert_eq!(
            callback_record(&run.result).error_type(),
            Some("PaymentDeclined")
        );
        assert_eq!(
            approve_error_type(&run.result),
            Some("Callback.ExternalFailure")
        );
    }

    #[tokio::test]
    async fn wfcb_submitter_failure_same_kind_live_and_replay() {
        for (submitter, text) in [
            (Submitter::FailsUntyped, "submission failed"),
            (Submitter::FailsTyped, "declined"),
        ] {
            let run = run_wfcb_classification(
                LocalRunner::new(),
                submitter,
                "SubmitterFailed",
                "submitter",
            )
            .await;
            assert!(run.result.is_success(), "{:?}", run.result.error_message());
            let kinds: Vec<&str> = run.seen.iter().map(|e| e.0.as_str()).collect();
            assert_eq!(kinds, ["SubmitterFailed", "SubmitterFailed"], "{text}");
            assert_eq!(run.seen[0], run.seen[1], "{text}");
            assert_eq!(
                approve_error_type(&run.result),
                Some("Callback.SubmitterFailed")
            );
            let message = run
                .result
                .operation("approve")
                .and_then(TestOperation::error_message)
                .unwrap_or_default();
            assert!(message.contains(text), "{message}");
            let live = run
                .live_step_source
                .expect("the live source is the step error");
            assert!(live.contains(text), "{live}");
        }
    }

    #[tokio::test]
    async fn wfcb_timeout_same_kind_live_and_replay() {
        let runner = LocalRunner::new().callback(CallbackReply::timeout());
        let run =
            run_wfcb_classification(runner, Submitter::Succeeds, "TimedOut", "timed out").await;
        assert!(run.result.is_success(), "{:?}", run.result.error_message());
        let kinds: Vec<&str> = run.seen.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(kinds, ["TimedOut", "TimedOut"]);
        assert_eq!(run.seen[0], run.seen[1]);
        assert_eq!(approve_error_type(&run.result), Some("Callback.Timeout"));
    }

    #[tokio::test]
    async fn wfcb_heartbeat_timeout_same_kind_live_and_replay() {
        let runner = LocalRunner::new().callback(CallbackReply::heartbeat_timeout());
        let run = run_wfcb_classification(
            runner,
            Submitter::Succeeds,
            "HeartbeatTimedOut",
            "heartbeat",
        )
        .await;
        assert!(run.result.is_success(), "{:?}", run.result.error_message());
        let kinds: Vec<&str> = run.seen.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(kinds, ["HeartbeatTimedOut", "HeartbeatTimedOut"]);
        assert_eq!(run.seen[0], run.seen[1]);
        assert_eq!(approve_error_type(&run.result), Some("Callback.Heartbeat"));
    }

    #[tokio::test]
    async fn wfcb_unmarked_decode_failure_same_kind_live_and_replay() {
        let runner = LocalRunner::new().callback(CallbackReply::raw("not json"));
        let run = run_wfcb_classification(
            runner,
            Submitter::Succeeds,
            "DeserializationFailed",
            "undecodable",
        )
        .await;
        assert!(run.result.is_success(), "{:?}", run.result.error_message());
        let kinds: Vec<&str> = run.seen.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(kinds, ["DeserializationFailed", "DeserializationFailed"]);
        assert_eq!(run.seen[0], run.seen[1]);
        assert_eq!(
            approve_error_type(&run.result),
            Some("Callback.DeserializationFailed")
        );
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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("a=A b=B"));
    }

    /// The recorded subtypes of rows 10 to 14: a step and a callback without
    /// a setter keep the built-in subtype, a custom subtype is recorded on
    /// the operation and on each of its events, and `wait_for_callback`
    /// keeps its three built-in subtypes.
    #[tokio::test]
    async fn step_and_callback_record_custom_sub_type() {
        use crate::{OperationSubType, StepContext};

        let result = LocalRunner::new()
            .callback_for("plain-cb", success(&1_u32))
            .callback_for("approval", success(&2_u32))
            .callback_for("wfcb", success(&3_u32))
            .run(
                |(), ctx: DurableContext| async move {
                    let plain = ctx.step(|_| async { Ok(1_u32) }).name("plain").await?;
                    let paid = ctx
                        .step(|step_ctx: StepContext| async move {
                            if step_ctx.attempt() == 1 {
                                Err::<u32, BoxError>("transient".into())
                            } else {
                                Ok(2)
                            }
                        })
                        .name("pay")
                        .sub_type(OperationSubType::new("Payment")?)
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: std::time::Duration::from_secs(1),
                                }
                            }
                        })
                        .await?;
                    let cb = ctx.create_callback::<u32>().name("plain-cb").await?;
                    let a = cb.result().await?;
                    let cb = ctx
                        .create_callback::<u32>()
                        .name("approval")
                        .sub_type(OperationSubType::new("Approval")?)
                        .await?;
                    let b = cb.result().await?;
                    let c = ctx
                        .wait_for_callback::<u32, _, _>(|_step_ctx, _callback_id| async { Ok(()) })
                        .name("wfcb")
                        .await?;
                    Ok::<_, BoxError>(plain + paid + a + b + c)
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&9));

        let op = |name: &str| result.operation(name).expect("the operation is recorded");
        // Row 10.
        assert_eq!(op("plain").sub_type(), Some("Step"));
        // Row 11: the step retried once, and the operation and every event
        // of it carry the custom subtype.
        let pay = op("pay");
        assert_eq!(pay.sub_type(), Some("Payment"));
        assert_eq!(step_attempt(pay), Some(1));
        assert!(pay.events().len() >= 3, "{:?}", pay.events());
        for event in pay.events() {
            assert_eq!(event.sub_type(), Some("Payment"), "{event:?}");
        }
        // Rows 12 and 13.
        assert_eq!(op("plain-cb").sub_type(), Some("Callback"));
        assert_eq!(op("approval").sub_type(), Some("Approval"));
        // Row 14.
        let wfcb = op("wfcb");
        assert_eq!(wfcb.sub_type(), Some("WaitForCallback"));
        let mut children: Vec<(OperationKind, Option<&str>)> = result
            .operations()
            .iter()
            .filter(|child| child.parent_id() == Some(wfcb.id()))
            .map(|child| (child.kind(), child.sub_type()))
            .collect();
        children.sort_by_key(|(kind, _)| kind.wire_str());
        assert_eq!(
            children,
            [
                (OperationKind::Callback, Some("Callback")),
                (OperationKind::Step, Some("Step"))
            ]
        );
    }

    /// The recorded subtypes of cases A to E: a child context and a
    /// `with_retry` block without a setter keep `RunInChildContext`, a
    /// custom subtype is recorded on a succeeded and on a failed child
    /// context, and on `with_retry` it labels the block only.
    #[tokio::test]
    async fn child_and_with_retry_record_custom_sub_type() {
        use crate::OperationSubType;

        let work_runs = Arc::new(AtomicU32::new(0));
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let work_runs = Arc::clone(&work_runs);
                    async move {
                        let checkout = || OperationSubType::new("Checkout");
                        // Case A.
                        let case_a = ctx
                            .run_in_child_context(|_child| async { Ok(1_u32) })
                            .name("case-a")
                            .await?;
                        // Case B.
                        let case_b = ctx
                            .run_in_child_context(|_child| async { Ok(2_u32) })
                            .name("case-b")
                            .sub_type(checkout()?)
                            .await?;
                        // Case C.
                        let case_c = ctx
                            .run_in_child_context(|_child| async {
                                Err::<u32, BoxError>("declined".into())
                            })
                            .name("case-c")
                            .sub_type(checkout()?)
                            .await;
                        assert!(case_c.is_err(), "case C must fail");
                        // Case D: the step fails on its first run, so attempt
                        // 1 fails and attempt 2 succeeds.
                        let case_d = ctx
                            .with_retry(move |child| {
                                let work_runs = Arc::clone(&work_runs);
                                async move {
                                    let v = child
                                        .step(move |_| async move {
                                            if work_runs.fetch_add(1, Ordering::SeqCst) == 0 {
                                                Err::<u32, BoxError>("transient".into())
                                            } else {
                                                Ok(4)
                                            }
                                        })
                                        .name("work")
                                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                                        .await?;
                                    Ok(v)
                                }
                            })
                            .name("block")
                            .sub_type(checkout()?)
                            .retry_strategy(|_err, attempt| {
                                if attempt >= 2 {
                                    RetryDecision::Stop
                                } else {
                                    RetryDecision::Retry {
                                        delay: std::time::Duration::from_secs(1),
                                    }
                                }
                            })
                            .await?;
                        // Case E.
                        let case_e = ctx
                            .with_retry(|_child| async { Ok(5_u32) })
                            .name("case-e")
                            .await?;
                        Ok::<_, BoxError>(case_a + case_b + case_d + case_e)
                    }
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&12));

        let op = |name: &str| result.operation(name).expect("the operation is recorded");
        assert_eq!(op("case-a").sub_type(), Some("RunInChildContext"));
        assert_eq!(op("case-b").sub_type(), Some("Checkout"));
        assert!(op("case-b").succeeded());
        let case_c_op = op("case-c");
        assert!(case_c_op.failed(), "{case_c_op:?}");
        assert_eq!(case_c_op.sub_type(), Some("Checkout"));
        let block = op("block");
        assert!(block.succeeded(), "{block:?}");
        assert_eq!(block.sub_type(), Some("Checkout"));
        assert!(op("attempt-1").failed());
        assert!(op("attempt-2").succeeded());
        assert_eq!(op("attempt-1").sub_type(), Some("RunInChildContext"));
        assert_eq!(op("attempt-2").sub_type(), Some("RunInChildContext"));
        assert_eq!(op("block-backoff-1").sub_type(), Some("Wait"));
        let works: Vec<Option<&str>> = result
            .operations()
            .iter()
            .filter(|o| o.name() == Some("work"))
            .map(TestOperation::sub_type)
            .collect();
        assert_eq!(works, [Some("Step"), Some("Step")]);
        assert_eq!(op("case-e").sub_type(), Some("RunInChildContext"));
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_failure(), "{:?}", result.error_message());
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), OperationStatus::Failed);
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
                .expect("runner")
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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&true));
        let cb_op = callback_record(&result);
        assert_eq!(cb_op.status(), OperationStatus::TimedOut);
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
            .await
            .expect("runner");

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
                .expect("runner")
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
        assert_eq!(ops[0].kind(), OperationKind::Step);
        assert_eq!(ops[0].status(), OperationStatus::Succeeded);
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
        assert_eq!(ops[0].status(), OperationStatus::Succeeded);
        assert_eq!(
            step_attempt(&ops[0]),
            Some(1),
            "current_attempt 2 normalizes to 1 completed retry"
        );
        assert_eq!(ops[0].result(), Some("99"));
        assert!(
            ops[0].error().is_none(),
            "the failed attempt's error is not kept: {:?}",
            ops[0].error()
        );

        // A retry that has started but not finished is `Started`, with no
        // error: the `StepStarted` arm sets the status, and its event
        // carries no error.
        let ops = fold_history(
            &events[..2]
                .iter()
                .chain(&events[..1])
                .cloned()
                .collect::<Vec<_>>(),
        );
        assert_eq!(ops[0].status(), OperationStatus::Started);
        assert!(ops[0].error().is_none(), "{:?}", ops[0].error());
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
            step_attempt(&fold_history(&first_try)[0]),
            Some(0),
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
            step_attempt(&fold_history(&zero)[0]),
            Some(0),
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
        assert_eq!(ops[0].kind(), OperationKind::Callback);
        assert_eq!(ops[0].status(), OperationStatus::TimedOut);
        assert_eq!(ops[0].error_type(), Some("CallbackError"));
        // The `CallbackStarted` event above has no `CallbackStartedDetails`,
        // so the callback reports no ID.
        assert!(
            matches!(ops[0].details(), TestOperationDetails::Callback(c) if c.callback_id().is_none()),
            "{:?}",
            ops[0].details()
        );
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
        assert_eq!(ops[0].kind(), OperationKind::Step);
        assert_eq!(
            step_attempt(&ops[0]),
            Some(0),
            "a step event without RetryDetails reports attempt 0"
        );
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
        assert_eq!(ops[1].kind(), OperationKind::Wait);
        assert_eq!(ops[1].status(), OperationStatus::Succeeded);
        // The `WaitStarted` event above has no `WaitStartedDetails`, so the
        // wait reports no duration.
        assert!(
            matches!(ops[1].details(), TestOperationDetails::Wait(w) if w.duration().is_none()),
            "{:?}",
            ops[1].details()
        );
    }

    /// An operation whose history holds no started event keeps the
    /// defaults of [`TestOperation::started`]: attempt 0, no wait duration,
    /// no callback ID, and `replay_children` false.
    #[test]
    fn fold_history_defaults_without_started_events() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::WaitSucceeded)
                .id("w")
                .event_id(2)
                .build(),
            Event::builder()
                .event_type(EventType::CallbackSucceeded)
                .id("cb")
                .event_id(3)
                .build(),
            Event::builder()
                .event_type(EventType::ContextSucceeded)
                .id("c")
                .event_id(4)
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 4);
        assert!(
            matches!(ops[0].details(), TestOperationDetails::Step(s) if s.attempt() == 0),
            "{:?}",
            ops[0].details()
        );
        assert!(
            matches!(ops[1].details(), TestOperationDetails::Wait(w) if w.duration().is_none()),
            "{:?}",
            ops[1].details()
        );
        assert!(
            matches!(ops[2].details(), TestOperationDetails::Callback(c) if c.callback_id().is_none()),
            "{:?}",
            ops[2].details()
        );
        assert!(
            matches!(ops[3].details(), TestOperationDetails::Context(c) if !c.replay_children()),
            "{:?}",
            ops[3].details()
        );
        for op in &ops {
            assert_eq!(op.status(), OperationStatus::Succeeded);
            assert!(op.result().is_none(), "{:?}", op.result());
            assert!(op.error().is_none(), "{:?}", op.error());
            assert!(op.parent_id().is_none());
        }
    }

    #[test]
    fn deserialize_output_handles_plain_missing_and_double_encoded() {
        assert_eq!(deserialize_output::<i32>(Some("42")).unwrap(), 42);
        assert_eq!(deserialize_output::<()>(None).unwrap(), ());
        // Double-encoded: a JSON string whose contents are the JSON value.
        assert_eq!(deserialize_output::<i32>(Some("\"7\"")).unwrap(), 7);
        assert!(deserialize_output::<i32>(Some("not json")).is_err());
    }

    /// A payload that is not a JSON string returns the error of decoding it
    /// as `O`. A JSON string whose content does not decode returns the
    /// error of decoding that content.
    #[test]
    fn deserialize_output_returns_the_error_of_the_decoded_value() {
        let plain = deserialize_output::<u32>(Some("{\"a\":1}")).expect_err("not a u32");
        let expected = serde_json::from_str::<u32>("{\"a\":1}").expect_err("not a u32");
        assert_eq!(plain.to_string(), expected.to_string());

        let inner = deserialize_output::<u32>(Some("\"x\"")).expect_err("not a u32");
        let expected = serde_json::from_str::<u32>("x").expect_err("not a u32");
        assert_eq!(inner.to_string(), expected.to_string());
    }

    #[test]
    fn execution_status_and_family_maps() {
        // Each row: the service status, the status it maps to, and the
        // `Display` text of that status. Every variant appears once.
        let rows = [
            (
                WireExecutionStatus::Running,
                ExecutionStatus::Running,
                "Running",
            ),
            (
                WireExecutionStatus::Succeeded,
                ExecutionStatus::Succeeded,
                "Succeeded",
            ),
            (
                WireExecutionStatus::Failed,
                ExecutionStatus::Failed,
                "Failed",
            ),
            (
                WireExecutionStatus::TimedOut,
                ExecutionStatus::TimedOut,
                "TimedOut",
            ),
            (
                WireExecutionStatus::Stopped,
                ExecutionStatus::Stopped,
                "Stopped",
            ),
        ];
        for (wire, status, text) in rows {
            assert_eq!(ExecutionStatus::from_wire(&wire), status, "{wire:?}");
            assert_eq!(status.to_string(), text);
        }
        assert_eq!(
            event_family_op_type(Some(&EventType::ChainedInvokeSucceeded)),
            Some(OperationKind::ChainedInvoke)
        );
        assert_eq!(
            event_family_op_type(Some(&EventType::ExecutionStarted)),
            None
        );
        assert_eq!(event_family_op_type(None), None);
    }

    #[test]
    fn operation_kind_from_wire_round_trips() {
        for kind in [
            OperationKind::Step,
            OperationKind::Wait,
            OperationKind::Context,
            OperationKind::Callback,
            OperationKind::ChainedInvoke,
            OperationKind::Execution,
        ] {
            assert_eq!(
                OperationKind::from_wire(&OperationType::from(kind.wire_str())),
                Some(kind)
            );
        }
        assert_eq!(
            OperationKind::from_wire(&OperationType::from("BOGUS")),
            None
        );
    }

    #[tokio::test]
    async fn step_reports_kind_and_status() {
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_i32) }).name("only").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        let ops = result.operations();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].kind(), OperationKind::Step);
        assert_eq!(ops[0].status(), OperationStatus::Succeeded);
        assert!(ops[0].succeeded());
    }

    #[tokio::test]
    async fn unknown_operation_type_is_rejected() {
        let backend = Backend::new(QueuedReplies::default(), None);
        let token = backend.current_token();
        let update = OperationUpdate::builder()
            .id("bogus")
            .r#type(OperationType::from("BOGUS"))
            .action(OperationAction::Start)
            .build()
            .expect("valid update");
        let e = backend
            .checkpoint("arn", &token, vec![update])
            .await
            .expect_err("an unknown operation type is rejected");
        assert_eq!(e.to_string(), "durable: client: Invalid operation type.");
        assert!(!e.is_retryable());
        assert!(backend.snapshot_operations().is_empty());
        assert_eq!(backend.current_token(), token);
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
                .status(WireOperationStatus::Succeeded)
                .start_timestamp(DateTime::from_secs(0))
                .step_details(
                    WireStepDetails::builder()
                        .result("\"page1-result\"")
                        .build(),
                )
                .build()
                .unwrap(),
            Operation::builder()
                .id("step-2")
                .r#type(OperationType::Step)
                .status(WireOperationStatus::Succeeded)
                .start_timestamp(DateTime::from_secs(0))
                .step_details(
                    WireStepDetails::builder()
                        .result("\"page2-result\"")
                        .build(),
                )
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
            crate::client::BootstrapState {
                log: operations_to_checkpoint_log(&first_page),
                input_payload: Some("{}".to_owned()),
                execution: None,
            },
            Some("marker-1"),
        )
        .await
        .unwrap()
        .log;

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
            crate::client::BootstrapState {
                log: operations_to_checkpoint_log(&first_page),
                input_payload: Some("{}".to_owned()),
                execution: None,
            },
            None,
        )
        .await
        .unwrap()
        .log;

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

    /// The runner awaits the invocation INLINE under the caller's
    /// `block_on` (the production `lambda_runtime` topology), so the context
    /// is created where `tokio::task::try_id()` is `None`, not on a spawned
    /// task. The handler itself then runs as the invocation's root task,
    /// which the SDK blesses, exactly as in production: so it has a task ID,
    /// and its durable operations pass the ownership guard.
    #[tokio::test]
    async fn runner_topology_matches_production_inline_block_on() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    // Same task visibility production gives the handler:
                    // the SDK's root task.
                    let on_root_task = tokio::task::try_id().is_some();
                    let v = ctx.step(|_| async { Ok(1_i32) }).await?;
                    Ok::<_, BoxError>((on_root_task, v))
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&(true, 1)),
            "handler must run on the SDK's root task, matching the deployed topology"
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&42));
        // The step's START and SUCCEED records both landed.
        assert!(
            result
                .operations()
                .iter()
                .any(|op| op.status == OperationStatus::Succeeded
                    && op.name.as_deref() == Some("inc")),
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            orphan_op.status,
            OperationStatus::Failed,
            "a terminal FAIL must be recorded for the operation whose \
             retained outcome rejection was classified at the flush point"
        );
    }

    /// An invocation fault, with a retained non-retryable failure of a
    /// detached batch flush, fails the execution through the flush
    /// failure. The plugins' end hook runs once, with `Failed`.
    ///
    /// The handler records the fault itself, after the rejection was
    /// retained and before any later write, so no contributor republishes
    /// the retained rejection first.
    #[tokio::test]
    async fn end_hook_failed_after_a_fault_with_a_retained_failure() {
        let plugin = crate::plugin::test_support::Hooks::default();
        let runner = LocalRunner::new().options(
            crate::Options::builder()
                .checkpoint_batching(true)
                .plugin(plugin.clone())
                .build()
                .expect("valid options"),
        );
        let backend = Arc::new(Backend::new(
            QueuedReplies::default(),
            runner.checkpoint_page_size,
        ));
        // 1. the orphan's START passes. 2. its SUCCEED is held at the gate,
        // then rejected non-retryably after the contributor was dropped.
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        backend.plan_checkpoint_calls(vec![
            PlannedCheckpoint::Pass,
            PlannedCheckpoint::FailNonRetryable {
                gate: Some(Arc::clone(&gate)),
            },
        ]);
        let backend_h = Arc::clone(&backend);
        let gate_h = Arc::clone(&gate);
        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                move |(), ctx: DurableContext| {
                    let backend = Arc::clone(&backend_h);
                    let gate = Arc::clone(&gate_h);
                    async move {
                        let orphan = ctx
                            .step(|_| async { Ok("orphaned-outcome".to_owned()) })
                            .name("orphan")
                            .spawn();
                        while backend.checkpoint_call_count() < 2 {
                            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        }
                        drop(orphan);
                        gate.add_permits(1);
                        // Let the detached flush receive the rejection and
                        // retain it.
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        ctx.coordinator().record_invocation_fault(
                            crate::error::CHECKPOINT_INVOCATION_FAILED_ERROR_TYPE,
                            "injected invocation fault".to_owned(),
                        );
                        std::future::pending::<()>().await;
                        Ok::<_, BoxError>(())
                    }
                },
                "null".to_owned(),
            )
            .await
            .expect("runner");
        assert!(result.is_failure(), "{:?}", result.error_message());
        assert_eq!(result.invocation_count(), 1);
        assert_eq!(
            result.error_type(),
            Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
        );
        let ends = plugin.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(end.status(), crate::InvocationStatus::Failed);
        assert_eq!(end.execution_result(), None);
        assert_eq!(
            end.execution_error().and_then(crate::WireError::error_type),
            Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("cb"));
        assert_eq!(result.invocation_count(), 2);
        let hour = result
            .operations()
            .iter()
            .find(|op| op.name() == Some("hour"))
            .expect("the hour wait is recorded");
        assert_eq!(hour.status(), OperationStatus::Started);
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
                let kind = OperationKind::from_wire(update.r#type()).expect("a known type");
                state.apply_update(kind, update).expect("update applies");
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
        (op.status, state.now)
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
    fn step_statuses(backend: &Backend) -> Vec<OperationStatus> {
        backend
            .snapshot_operations()
            .iter()
            .filter(|op| op.kind() == OperationKind::Step)
            .map(TestOperation::status)
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

        // The handler runs on its own task, so each step lets that task run
        // before it reads the backend: after it moves the clock, and before
        // the first move, so the handler reaches its first checkpoint.
        let settle = || async {
            for _ in 0..16 {
                tokio::task::yield_now().await;
            }
        };
        let steps = async {
            settle().await;
            tokio::time::advance(std::time::Duration::from_millis(49)).await;
            settle().await;
            assert!(step_statuses(&backend).is_empty(), "START not applied yet");
            tokio::time::advance(std::time::Duration::from_millis(1)).await;
            settle().await;
            assert_eq!(step_statuses(&backend), [OperationStatus::Started]);
            tokio::time::advance(std::time::Duration::from_millis(49)).await;
            settle().await;
            assert_eq!(step_statuses(&backend), [OperationStatus::Started]);
            tokio::time::advance(std::time::Duration::from_millis(1)).await;
            settle().await;
            assert_eq!(step_statuses(&backend), [OperationStatus::Succeeded]);
        };
        let (delayed, ()) = tokio::join!(
            runner.run_on_backend(Arc::clone(&backend), one_step_handler, "null".to_owned()),
            steps
        );
        let delayed = delayed.expect("runner");

        let baseline = unbatched().run(one_step_handler, ()).await.expect("runner");
        assert!(delayed.is_success(), "{:?}", delayed.error_message());
        assert_eq!(delayed.output(), baseline.output());
        // Each run reports times from its own start, so the comparison
        // drops the times.
        assert_eq!(
            format!("{:?}", without_times(delayed.operations())),
            format!("{:?}", without_times(baseline.operations()))
        );
    }

    /// `operations` with every reported time removed, so two runs that
    /// started at different wall-clock times compare equal.
    fn without_times(operations: &[TestOperation]) -> Vec<TestOperation> {
        operations
            .iter()
            .cloned()
            .map(|mut op| {
                op.start_time = None;
                op.end_time = None;
                op.step.next_attempt_time = None;
                op.wait.scheduled_end_time = None;
                for event in &mut op.events {
                    event.event_timestamp = None;
                }
                op
            })
            .collect()
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
            let result = runtime
                .block_on(LocalRunner::new().run(one_step_handler, ()))
                .expect("runner");
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
            .await
            .expect("runner");

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
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&307_200));
    }

    /// The service creates a callback in `Started`, so an open callback
    /// reports `OperationStatus::Started`, as under `CloudRunner`.
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
            .await
            .expect("runner");

        assert!(result.is_suspended(), "{:?}", result.error_message());
        let callback = result
            .operations()
            .iter()
            .find(|op| op.kind() == OperationKind::Callback)
            .expect("the callback is recorded");
        assert_eq!(callback.status(), OperationStatus::Started);
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
                let kind = OperationKind::from_wire(update.r#type()).expect("a known type");
                state.apply_update(kind, update).expect("update applies");
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
                    ctx_op("C", OperationAction::Start, None, None),
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
        assert_eq!(ids, ["C", "S", "C"]);
    }

    /// A paginated checkpoint response returns the first page of the full
    /// state. That page is not pruned either. So with a page size of 2 the
    /// page holds `C` and its child `S`, which were stored first, and not
    /// the top-level step `T`.
    #[tokio::test]
    async fn paginated_checkpoint_response_is_not_pruned() {
        let backend = Backend::new(QueuedReplies::default(), Some(2));
        let output = backend
            .checkpoint(
                "arn:test",
                "token-0",
                vec![
                    ctx_op("C", OperationAction::Start, None, None),
                    step_op("S", OperationAction::Succeed, Some("C")),
                    ctx_op("C", OperationAction::Succeed, None, None),
                    step_op("T", OperationAction::Succeed, None),
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
        assert_eq!(ids, ["C", "S"]);
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
        assert_eq!(context.status(), &WireOperationStatus::Failed);
        assert_eq!(
            context
                .context_details()
                .and_then(WireContextDetails::replay_children),
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
                .and_then(WireContextDetails::replay_children),
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
                .and_then(WireContextDetails::replay_children),
            None
        );

        // A close without the option records `false`, so the child goes.
        backend
            .lock()
            .apply_update(
                OperationKind::Context,
                &ctx_op("C", OperationAction::Succeed, None, None),
            )
            .expect("update applies");
        assert!(!context_has_replay_children(&backend.envelope_history()));
        assert_eq!(envelope_ids(&backend), ["C"]);
        let ops = get_state_ops(&backend).await;
        assert_eq!(ops.iter().map(Operation::id).collect::<Vec<_>>(), ["C"]);
        assert_eq!(
            ops[0]
                .context_details()
                .and_then(WireContextDetails::replay_children),
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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

    // History rules: identity, actions, statuses, parents and repeated IDs.

    /// A builder for an update of `op_type` and `action` with id `id`
    /// (history-rule tests).
    fn rule_update(
        id: &str,
        op_type: OperationType,
        action: OperationAction,
    ) -> aws_sdk_lambda::types::builders::OperationUpdateBuilder {
        OperationUpdate::builder()
            .id(id)
            .r#type(op_type)
            .action(action)
    }

    /// Builds the update of `builder`.
    fn built(builder: aws_sdk_lambda::types::builders::OperationUpdateBuilder) -> OperationUpdate {
        builder.build().expect("valid update")
    }

    /// The text that follows the field name in every identity message.
    const INCONSISTENT: &str = " value in OperationUpdate is inconsistent with existing operation. Check for non-determinism.";

    /// Seeds a backend with `seed`, sets each `(id, status)` of `statuses`
    /// on the stored op, and sends `updates` in one checkpoint call.
    ///
    /// With `expected` set, asserts the call fails with a non-retryable
    /// error carrying that message, and that the stored operations, the
    /// token and the history did not change. Without it, asserts the call
    /// succeeded.
    async fn check_rule_row(
        row: &str,
        seed: &[OperationUpdate],
        statuses: &[(&str, OperationStatus)],
        updates: Vec<OperationUpdate>,
        expected: Option<&str>,
    ) {
        let backend = seeded_backend(Vec::new(), 0, seed);
        {
            let mut state = backend.lock();
            for (id, status) in statuses {
                let op = state.find_mut(id).expect("the op is seeded");
                op.status = *status;
            }
        }
        let token = backend.current_token();
        let ops_before = format!("{:?}", backend.lock().ops);
        let events_before = format!("{:?}", backend.events());
        let result =
            ExecutionClient::checkpoint(&backend, LOCAL_EXECUTION_ARN, &token, updates).await;
        match expected {
            Some(expected) => {
                assert!(result.is_err(), "row {row}: accepted");
                let err = result.expect_err("the call is rejected");
                assert_eq!(
                    err.to_string(),
                    format!("durable: client: {expected}"),
                    "row {row}"
                );
                assert!(!err.is_retryable(), "row {row}");
                assert_eq!(format!("{:?}", backend.lock().ops), ops_before, "row {row}");
                assert_eq!(backend.current_token(), token, "row {row}");
                assert_eq!(
                    format!("{:?}", backend.events()),
                    events_before,
                    "row {row}"
                );
            }
            None => {
                assert_eq!(result.err().map(|e| e.to_string()), None, "row {row}");
            }
        }
    }

    /// The case of #147: a STEP update reaches the ID of a map item's
    /// CONTEXT record. The backend rejects it, and keeps the record.
    #[tokio::test]
    async fn step_update_at_a_context_id_is_rejected() {
        let seed = [
            built(
                rule_update("map", OperationType::Context, OperationAction::Start).sub_type("Map"),
            ),
            built(
                rule_update("item", OperationType::Context, OperationAction::Start)
                    .sub_type("MapIteration")
                    .name("map-item-0")
                    .parent_id("map"),
            ),
        ];
        let update = built(
            rule_update("item", OperationType::Step, OperationAction::Succeed)
                .sub_type("Step")
                .name("second")
                .payload("2"),
        );
        check_rule_row(
            "step at context",
            &seed,
            &[],
            vec![update],
            Some(&format!("Type{INCONSISTENT}")),
        )
        .await;
    }

    /// JS's message for an action outside the type's set.
    const INVALID_ACTION: &str = "Invalid action for the given operation type.";
    /// JS's message for a bad parent.
    const INVALID_PARENT: &str = "Invalid parent operation id.";
    /// JS's message for a forbidden repeated ID.
    const REPEATED_ID: &str = "Cannot update the same operation twice in a single request.";
    /// JS's message for a WAIT `CANCEL` that finds no started WAIT.
    const CANCEL_WAIT: &str =
        "Cannot cancel a WAIT operation that does not exist or has already completed.";

    /// Runs each `(row, updates, expected)` of `rows` on its own backend
    /// seeded with `seed` (see [`check_rule_row`]).
    async fn check_rule_rows(
        seed: &[OperationUpdate],
        rows: Vec<(&str, Vec<OperationUpdate>, Option<String>)>,
    ) {
        for (row, updates, expected) in rows {
            check_rule_row(row, seed, &[], updates, expected.as_deref()).await;
        }
    }

    /// The identity message for `field`.
    fn inconsistent(field: &str) -> String {
        format!("{field}{INCONSISTENT}")
    }

    /// Test 1.1: each identity field equal, changed, and omitted, and the
    /// order in which the fields are checked.
    #[tokio::test]
    async fn identity_rules() {
        use OperationAction as A;
        use OperationType as T;
        let seed = [
            built(rule_update("p", T::Context, A::Start)),
            built(rule_update("q", T::Context, A::Start)),
            built(
                rule_update("c", T::Context, A::Start)
                    .sub_type("Child")
                    .name("n")
                    .parent_id("p"),
            ),
            built(rule_update("bare", T::Context, A::Start)),
        ];
        let close = |id: &str| rule_update(id, T::Context, A::Succeed);
        let rows = vec![
            (
                "all equal",
                vec![built(close("c").sub_type("Child").name("n").parent_id("p"))],
                None,
            ),
            ("all omitted", vec![built(close("c"))], None),
            (
                "type changed",
                vec![built(rule_update("c", T::Step, A::Succeed))],
                Some(inconsistent("Type")),
            ),
            (
                "subtype changed",
                vec![built(close("c").sub_type("Other"))],
                Some(inconsistent("SubType")),
            ),
            (
                "subtype set on a record without one",
                vec![built(close("bare").sub_type("Child"))],
                Some(inconsistent("SubType")),
            ),
            (
                "name changed",
                vec![built(close("c").name("m"))],
                Some(inconsistent("Name")),
            ),
            (
                "name set on a record without one",
                vec![built(close("bare").name("n"))],
                Some(inconsistent("Name")),
            ),
            (
                "parent changed",
                vec![built(close("c").parent_id("q"))],
                Some(inconsistent("ParentId")),
            ),
            (
                "parent set on a record without one",
                vec![built(close("bare").parent_id("p"))],
                Some(inconsistent("ParentId")),
            ),
            (
                "type before name",
                vec![built(rule_update("c", T::Step, A::Succeed).name("m"))],
                Some(inconsistent("Type")),
            ),
            (
                "subtype before name",
                vec![built(close("c").sub_type("Other").name("m"))],
                Some(inconsistent("SubType")),
            ),
            (
                "name before parent",
                vec![built(close("c").name("m").parent_id("q"))],
                Some(inconsistent("Name")),
            ),
            (
                "identity before action",
                vec![built(rule_update("c", T::Context, A::Retry).name("m"))],
                Some(inconsistent("Name")),
            ),
            (
                "an earlier update of the request is not a stored record",
                vec![
                    built(rule_update("new", T::Step, A::Start).name("a")),
                    built(rule_update("new", T::Step, A::Succeed).name("b")),
                ],
                None,
            ),
        ];
        check_rule_rows(&seed, rows).await;
    }

    /// Test 1.2: every action of every type, with no stored operation.
    #[tokio::test]
    async fn action_sets() {
        use OperationAction as A;
        use OperationType as T;
        // For each type: the actions it allows, and the message of an
        // allowed action that needs a stored operation.
        type Row<'a> = (T, &'a [A], Option<(A, &'a str)>);
        let actions = [A::Start, A::Succeed, A::Fail, A::Retry, A::Cancel];
        let table: [Row<'_>; 6] = [
            (T::Step, &[A::Start, A::Succeed, A::Fail, A::Retry], None),
            (T::Context, &[A::Start, A::Succeed, A::Fail], None),
            (
                T::Wait,
                &[A::Start, A::Cancel],
                Some((A::Cancel, CANCEL_WAIT)),
            ),
            (T::Callback, &[A::Start], None),
            (T::ChainedInvoke, &[A::Start], None),
            (T::Execution, &[A::Succeed, A::Fail], None),
        ];
        for (op_type, allowed, needs_record) in table {
            for action in &actions {
                let expected = if allowed.contains(action) {
                    needs_record
                        .as_ref()
                        .filter(|(a, _)| a == action)
                        .map(|(_, message)| (*message).to_owned())
                } else {
                    Some(INVALID_ACTION.to_owned())
                };
                let row = format!("{op_type:?} {action:?}");
                let update = built(rule_update("o", op_type.clone(), action.clone()));
                check_rule_row(&row, &[], &[], vec![update], expected.as_deref()).await;
            }
        }
    }

    /// The action is checked before the status: a forbidden action on a
    /// stored operation gets the action message, not the status message.
    #[tokio::test]
    async fn action_is_checked_before_status() {
        let seed = [built(rule_update(
            "s",
            OperationType::Step,
            OperationAction::Succeed,
        ))];
        let update = built(rule_update(
            "s",
            OperationType::Step,
            OperationAction::Cancel,
        ));
        check_rule_row(
            "cancel a step",
            &seed,
            &[],
            vec![update],
            Some(INVALID_ACTION),
        )
        .await;
    }

    /// Test 1.3: every allowed action of every type against every status
    /// of a stored operation.
    #[tokio::test]
    async fn status_transitions() {
        use OperationAction as A;
        use OperationStatus as S;
        use OperationType as T;
        let statuses = [
            S::Started,
            S::Pending,
            S::Ready,
            S::Succeeded,
            S::Failed,
            S::Cancelled,
            S::TimedOut,
            S::Stopped,
        ];
        // For each type and action: the statuses that allow it, and JS's
        // message for every other status.
        let table: [(T, A, &[S], Option<&str>); 12] = [
            (
                T::Step,
                A::Start,
                &[S::Ready],
                Some("Invalid current STEP state to start."),
            ),
            (
                T::Step,
                A::Succeed,
                &[S::Started, S::Ready],
                Some("Invalid current STEP state to close."),
            ),
            (
                T::Step,
                A::Fail,
                &[S::Started, S::Ready],
                Some("Invalid current STEP state to close."),
            ),
            (
                T::Step,
                A::Retry,
                &[S::Started, S::Ready],
                Some("Invalid current STEP state to retry."),
            ),
            (
                T::Context,
                A::Start,
                &[],
                Some("Cannot start a CONTEXT operation that already exists."),
            ),
            (
                T::Context,
                A::Succeed,
                &[S::Started],
                Some("Invalid current CONTEXT state to close."),
            ),
            (
                T::Context,
                A::Fail,
                &[S::Started],
                Some("Invalid current CONTEXT state to close."),
            ),
            (
                T::Wait,
                A::Start,
                &[],
                Some("Cannot start a WAIT operation that already exists."),
            ),
            (T::Wait, A::Cancel, &[S::Started], Some(CANCEL_WAIT)),
            (
                T::Callback,
                A::Start,
                &[],
                Some("Cannot start a CALLBACK operation that already exists."),
            ),
            (
                T::ChainedInvoke,
                A::Start,
                &[],
                Some("Cannot start a CHAINED_INVOKE operation that already exists."),
            ),
            (T::Execution, A::Succeed, &statuses, None),
        ];
        for (op_type, action, allowed, message) in table {
            // The seed creates the record. Its status is then set directly.
            let seed_action = if op_type == T::Execution {
                A::Succeed
            } else {
                A::Start
            };
            let seed = [built(rule_update("o", op_type.clone(), seed_action))];
            for status in statuses {
                let expected = (!allowed.contains(&status)).then_some(message).flatten();
                let row = format!("{op_type:?} {action:?} at {status:?}");
                let update = built(rule_update("o", op_type.clone(), action.clone()));
                check_rule_row(&row, &seed, &[("o", status)], vec![update], expected).await;
            }
        }
        // EXECUTION FAIL is allowed at any status too.
        let seed = [built(rule_update("o", T::Execution, A::Succeed))];
        let update = built(rule_update("o", T::Execution, A::Fail));
        check_rule_row("Execution Fail", &seed, &[], vec![update], None).await;
    }

    /// Test 1.4: the parent rules.
    #[tokio::test]
    async fn parent_rules() {
        use OperationAction as A;
        use OperationType as T;
        let seed = [
            built(rule_update("ctx", T::Context, A::Start)),
            built(rule_update("step", T::Step, A::Start)),
        ];
        let child = |parent: &str| built(rule_update("a", T::Step, A::Start).parent_id(parent));
        let parent_error = Some(INVALID_PARENT.to_owned());
        let rows = vec![
            (
                "omitted",
                vec![built(rule_update("a", T::Step, A::Start))],
                None,
            ),
            ("stored CONTEXT", vec![child("ctx")], None),
            (
                "CONTEXT started earlier in the request",
                vec![built(rule_update("n", T::Context, A::Start)), child("n")],
                None,
            ),
            ("missing", vec![child("nope")], parent_error.clone()),
            ("stored STEP", vec![child("step")], parent_error.clone()),
            (
                "STEP started earlier in the request",
                vec![built(rule_update("n", T::Step, A::Start)), child("n")],
                parent_error.clone(),
            ),
            (
                "CONTEXT closed, not started, earlier in the request",
                vec![built(rule_update("n", T::Context, A::Succeed)), child("n")],
                parent_error.clone(),
            ),
            (
                "the operation itself",
                vec![built(rule_update("a", T::Context, A::Start).parent_id("a"))],
                parent_error.clone(),
            ),
            (
                "CONTEXT started later in the request",
                vec![child("n"), built(rule_update("n", T::Context, A::Start))],
                parent_error.clone(),
            ),
            (
                "an earlier START in the request wins over the stored record",
                vec![built(rule_update("ctx", T::Step, A::Start)), child("ctx")],
                parent_error,
            ),
        ];
        check_rule_rows(&seed, rows).await;
    }

    /// Test 1.5: every allowed and forbidden form of a repeated ID.
    #[tokio::test]
    async fn repeated_id_rules() {
        use OperationAction as A;
        use OperationType as T;
        let u = |op_type: T, action: A| built(rule_update("r", op_type, action));
        let repeated = Some(REPEATED_ID.to_owned());
        let rows = vec![
            (
                "STEP START then SUCCEED",
                vec![u(T::Step, A::Start), u(T::Step, A::Succeed)],
                None,
            ),
            (
                "STEP START then FAIL",
                vec![u(T::Step, A::Start), u(T::Step, A::Fail)],
                None,
            ),
            (
                "STEP START then RETRY",
                vec![u(T::Step, A::Start), u(T::Step, A::Retry)],
                None,
            ),
            (
                "CONTEXT START then SUCCEED",
                vec![u(T::Context, A::Start), u(T::Context, A::Succeed)],
                None,
            ),
            (
                "CONTEXT START then FAIL",
                vec![u(T::Context, A::Start), u(T::Context, A::Fail)],
                None,
            ),
            (
                "STEP START then START",
                vec![u(T::Step, A::Start), u(T::Step, A::Start)],
                repeated.clone(),
            ),
            (
                "CONTEXT START then START",
                vec![u(T::Context, A::Start), u(T::Context, A::Start)],
                repeated.clone(),
            ),
            (
                "STEP SUCCEED then FAIL",
                vec![u(T::Step, A::Succeed), u(T::Step, A::Fail)],
                repeated.clone(),
            ),
            (
                "STEP START, SUCCEED, then FAIL",
                vec![
                    u(T::Step, A::Start),
                    u(T::Step, A::Succeed),
                    u(T::Step, A::Fail),
                ],
                repeated.clone(),
            ),
            (
                "WAIT START then CANCEL",
                vec![u(T::Wait, A::Start), u(T::Wait, A::Cancel)],
                repeated.clone(),
            ),
            (
                "CALLBACK START twice",
                vec![u(T::Callback, A::Start), u(T::Callback, A::Start)],
                repeated.clone(),
            ),
            (
                "CHAINED_INVOKE START twice",
                vec![u(T::ChainedInvoke, A::Start), u(T::ChainedInvoke, A::Start)],
                repeated.clone(),
            ),
            (
                "EXECUTION SUCCEED then FAIL",
                vec![u(T::Execution, A::Succeed), u(T::Execution, A::Fail)],
                repeated.clone(),
            ),
            (
                "the repeat is checked before the parent",
                vec![
                    u(T::Step, A::Start),
                    built(rule_update("r", T::Step, A::Start).parent_id("nope")),
                ],
                repeated,
            ),
        ];
        check_rule_rows(&[], rows).await;
    }

    /// The repeated-ID and parent pass covers the whole request before any
    /// update is checked on its own. So a repeat later in the request wins
    /// over a status error of an earlier update.
    #[tokio::test]
    async fn request_pass_runs_before_update_checks() {
        use OperationAction as A;
        use OperationType as T;
        let seed = [built(rule_update("s", T::Step, A::Succeed))];
        let updates = vec![
            built(rule_update("s", T::Step, A::Succeed)),
            built(rule_update("w", T::Wait, A::Start)),
            built(rule_update("w", T::Wait, A::Start)),
        ];
        check_rule_row(
            "repeat after a status error",
            &seed,
            &[],
            updates,
            Some(REPEATED_ID),
        )
        .await;
    }

    /// Test 2: a request whose first update is valid and whose second is
    /// rejected stores neither, keeps the token and records no event.
    #[tokio::test]
    async fn rejected_request_applies_no_update() {
        use OperationAction as A;
        use OperationType as T;
        let seed = [built(rule_update("ctx", T::Context, A::Start))];
        let updates = vec![
            built(rule_update("a", T::Step, A::Start)),
            built(rule_update("ctx", T::Context, A::Start)),
        ];
        check_rule_row(
            "second update rejected",
            &seed,
            &[],
            updates,
            Some("Cannot start a CONTEXT operation that already exists."),
        )
        .await;
        // `check_rule_row` compared the whole store, so `a` is not stored.
    }

    /// Test 3: the SDK sends a STEP update to an ID that holds a pruned
    /// WAIT record. The backend rejects it with a non-retryable error, and
    /// the run ends with `CheckpointUnrecoverableExecutionError`.
    ///
    /// A correct handler builds no forbidden update, and the SDK checks the
    /// identity of every record it can see. So the test hides the WAIT
    /// from the SDK: it stores the WAIT as the child of a closed CONTEXT
    /// without `ReplayChildren`, which the invocation input omits.
    #[tokio::test]
    async fn run_with_a_rejected_update_ends_unrecoverable() {
        use OperationAction as A;
        use OperationType as T;
        let handler = |(), ctx: DurableContext| async move {
            let value = ctx.step(|_| async { Ok(1_u32) }).name("s").await?;
            Ok::<_, BoxError>(value)
        };
        // The ID the handler's step uses.
        let probe = LocalRunner::new().run(handler, ()).await.expect("runner");
        let step_id = probe.operations()[0].id().to_owned();

        let backend = Arc::new(seeded_backend(
            Vec::new(),
            0,
            &[
                built(rule_update("hidden", T::Context, A::Start)),
                built(rule_update(&step_id, T::Wait, A::Start).parent_id("hidden")),
                built(rule_update("hidden", T::Context, A::Succeed)),
            ],
        ));
        let result = LocalRunner::new()
            .run_on_backend(Arc::clone(&backend), handler, "null".to_owned())
            .await
            .expect("runner");
        assert!(result.is_failure(), "{result}");
        assert_eq!(
            result.error_type(),
            Some("CheckpointUnrecoverableExecutionError")
        );
        assert_eq!(
            result.error_message(),
            Some(
                format!("[Unrecoverable Execution] Checkpoint failed: Type{INCONSISTENT}").as_str()
            )
        );
        // The WAIT record is unchanged.
        let state = backend.lock();
        let hidden = state
            .ops
            .iter()
            .find(|op| op.id == step_id)
            .expect("the WAIT is stored");
        assert_eq!(hidden.kind, OperationKind::Wait);
        assert_eq!(hidden.status, OperationStatus::Started);
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

    /// Epoch milliseconds count 0 for a start time before the epoch,
    /// saturate a start time past `u64::MAX` milliseconds, and saturate the
    /// sum with the virtual seconds.
    #[test]
    fn virtual_epoch_millis_saturates() {
        let epoch = std::time::SystemTime::UNIX_EPOCH;
        let second = std::time::Duration::from_secs(1);
        assert_eq!(virtual_epoch_millis(epoch + second, 2), 3_000);
        let before_epoch = epoch
            .checked_sub(second)
            .expect("in range on this platform");
        assert_eq!(virtual_epoch_millis(before_epoch, 0), 0);
        let far = epoch
            .checked_add(std::time::Duration::from_secs(u64::MAX / 1000 + 1))
            .expect("in range on this platform");
        assert_eq!(virtual_epoch_millis(far, 0), u64::MAX);
        assert_eq!(virtual_epoch_millis(epoch + second, u64::MAX), u64::MAX);
    }

    /// The envelope and `get_state` render each record's start and end
    /// times as the run's start time plus the record's virtual seconds.
    #[tokio::test]
    async fn envelope_renders_virtual_timestamps() {
        let before = std::time::SystemTime::now();
        let backend = seeded_backend(Vec::new(), 0, &[]);
        checkpoint_ok(&backend, vec![step_op("s", OperationAction::Start, None)]).await;
        checkpoint_ok(&backend, vec![step_op("s", OperationAction::Succeed, None)]).await;
        checkpoint_ok(&backend, vec![wait_start("w", 10)]).await;
        assert!(backend.advance(), "fires the wait");

        let envelope = build_envelope(&backend, "{}", None);
        let ops = envelope["InitialExecutionState"]["Operations"]
            .as_array()
            .expect("Operations is an array");
        let millis = |id: &str, field: &str| {
            ops.iter()
                .find(|op| op["Id"] == id)
                .and_then(|op| op[field].as_u64())
                .expect("the record has an integer timestamp field")
        };
        let e = millis(EXECUTION_OPERATION_ID, "StartTimestamp");
        let before_ms = u64::try_from(
            before
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .expect("after the epoch")
                .as_millis(),
        )
        .expect("fits in u64");
        assert!(e >= before_ms, "execution start {e} before {before_ms}");
        assert_eq!(
            (millis("s", "StartTimestamp"), millis("s", "EndTimestamp")),
            (e, e)
        );
        assert_eq!(
            (millis("w", "StartTimestamp"), millis("w", "EndTimestamp")),
            (e, e + 10_000)
        );

        let state = get_state_ops(&backend).await;
        let times = |id: &str| {
            let op = state
                .iter()
                .find(|op| op.id() == id)
                .expect("get_state holds the record");
            (
                *op.start_timestamp(),
                *op.end_timestamp().expect("an end timestamp"),
            )
        };
        let (step_start, step_end) = times("s");
        assert_eq!(step_start, step_end);
        assert!(step_start >= DateTime::from(before));
        let (wait_start_time, wait_end_time) = times("w");
        assert_eq!(wait_end_time.secs() - wait_start_time.secs(), 10);
        assert_eq!(wait_end_time.subsec_nanos(), wait_start_time.subsec_nanos());
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
            .filter(|op| op.kind() == OperationKind::ChainedInvoke)
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
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&42), "{:?}", result.error_message());
        let op = invoke_record(&result);
        assert_eq!(op.status(), OperationStatus::Succeeded);
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
            .await
            .expect("runner");
        assert_eq!(
            result.output(),
            Some(&Some("PaymentError".to_owned())),
            "{:?}",
            result.error_message()
        );
        let op = invoke_record(&result);
        assert_eq!(op.status(), OperationStatus::Failed);
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        let op = invoke_record(&result);
        assert_eq!(op.status(), OperationStatus::Succeeded);
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&4), "{:?}", result.error_message());
        let op = invoke_record(&result);
        assert_eq!(op.status(), OperationStatus::Succeeded);
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
            .await
            .expect("runner");
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
        assert_eq!(op.status(), OperationStatus::Failed);
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
        assert!(result.is_suspended(), "{:?}", result.error_message());
        assert_eq!(invoke_record(&result).status(), OperationStatus::Started);
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
        assert!(result.is_suspended(), "{:?}", result.error_message());
        assert_eq!(invoke_record(&result).status(), OperationStatus::Started);
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
            .await?;
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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
            .await
            .expect("runner");
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

    // Operation detail, execution status and runner errors

    /// The only operation with `kind` in `result`.
    fn only_of_kind<O>(result: &TestResult<O>, kind: OperationKind) -> &TestOperation {
        let mut ops = result.operations().iter().filter(|op| op.kind() == kind);
        let op = ops.next().expect("an operation of the kind");
        assert!(ops.next().is_none(), "one operation of kind {kind}");
        op
    }

    #[tokio::test]
    async fn wait_reports_duration() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    ctx.wait(std::time::Duration::from_secs(10)).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let op = only_of_kind(&result, OperationKind::Wait);
        assert!(matches!(
            op.details(),
            TestOperationDetails::Wait(w) if w.duration() == Some(std::time::Duration::from_secs(10))
        ));
    }

    #[tokio::test]
    async fn callback_reports_id_seen_by_handler() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<u32>().await?;
                    Ok::<_, BoxError>(cb.id().to_owned())
                },
                (),
            )
            .await
            .expect("runner");
        let output = result.output().expect("the handler returned the ID");
        let op = only_of_kind(&result, OperationKind::Callback);
        assert!(
            matches!(
                op.details(),
                TestOperationDetails::Callback(c) if c.callback_id() == Some(output.as_str())
            ),
            "{:?}",
            op.details()
        );
    }

    #[tokio::test]
    async fn map_children_are_its_items() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let out = ctx
                        .map(vec![1_u32, 2, 3], |child, item, _idx| async move {
                            Ok(child.step(move |_| async move { Ok(item) }).await?)
                        })
                        .name("items")
                        .await?;
                    Ok::<_, BoxError>(out)
                },
                (),
            )
            .await
            .expect("runner");
        let map_op = result.operation("items").expect("the map operation");
        let children: Vec<_> = result.children(map_op).collect();
        assert_eq!(children.len(), 3);
        for child in children {
            assert_eq!(child.sub_type(), Some("MapIteration"));
            assert_eq!(child.parent_id(), Some(map_op.id()));
        }
    }

    #[tokio::test]
    async fn replay_children_is_reported() {
        // The handler of `oversized_map_result_replays_with_replay_children`.
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
            .await
            .expect("runner");
        let replay_children = |op: &TestOperation| match op.details() {
            TestOperationDetails::Context(c) => Some(c.replay_children()),
            _ => None,
        };
        let map_op = result.operation("oversized").expect("the map operation");
        assert_eq!(replay_children(map_op), Some(true));
        let items: Vec<_> = result.children(map_op).collect();
        assert_eq!(items.len(), 2);
        for item in items {
            assert_eq!(replay_children(item), Some(false));
        }
    }

    #[tokio::test]
    async fn details_match_kind() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_u32) }).name("s").await?;
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    let _cb = ctx.create_callback::<u32>().await?;
                    ctx.run_in_child_context(|_child| async { Ok(2_u32) })
                        .await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let kinds: BTreeSet<String> = result
            .operations()
            .iter()
            .map(|op| op.kind().to_string())
            .collect();
        assert_eq!(kinds.len(), 4, "{kinds:?}");
        for op in result.operations() {
            let matches = match op.details() {
                TestOperationDetails::Execution => op.kind() == OperationKind::Execution,
                TestOperationDetails::Context(_) => op.kind() == OperationKind::Context,
                TestOperationDetails::Step(_) => op.kind() == OperationKind::Step,
                TestOperationDetails::Wait(_) => op.kind() == OperationKind::Wait,
                TestOperationDetails::Callback(_) => op.kind() == OperationKind::Callback,
                TestOperationDetails::ChainedInvoke => op.kind() == OperationKind::ChainedInvoke,
            };
            assert!(matches, "{:?} has details {:?}", op.kind(), op.details());
        }
        let step = result.operation("s").expect("the step");
        assert!(!matches!(step.details(), TestOperationDetails::Wait(_)));
    }

    #[test]
    fn details_variant_for_every_kind() {
        // The name of a `details()` variant, so each row below can state
        // the one variant its kind must give.
        fn variant(details: TestOperationDetails<'_>) -> &'static str {
            match details {
                TestOperationDetails::Execution => "Execution",
                TestOperationDetails::Context(_) => "Context",
                TestOperationDetails::Step(_) => "Step",
                TestOperationDetails::Wait(_) => "Wait",
                TestOperationDetails::Callback(_) => "Callback",
                TestOperationDetails::ChainedInvoke => "ChainedInvoke",
            }
        }
        let rows = [
            (OperationKind::Execution, "Execution"),
            (OperationKind::Context, "Context"),
            (OperationKind::Step, "Step"),
            (OperationKind::Wait, "Wait"),
            (OperationKind::Callback, "Callback"),
            (OperationKind::ChainedInvoke, "ChainedInvoke"),
        ];
        for (kind, expected) in rows {
            let op = TestOperation::started("id".to_owned(), kind);
            assert_eq!(variant(op.details()), expected, "{kind:?}");
        }
    }

    /// The error a step body or a handler returns in the error data tests.
    fn bad_error() -> BoxError {
        Box::new(crate::TypedError::with_type("Bad", "no").with_error_data("{\"a\":1}"))
    }

    #[tokio::test]
    async fn failed_step_error_keeps_data() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let _ = ctx
                        .step(|_| async { Err::<u32, BoxError>(bad_error()) })
                        .name("bad")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .await;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let error = result
            .operation("bad")
            .and_then(TestOperation::error)
            .expect("the step error");
        assert_eq!(error.error_type(), Some("Bad"));
        assert!(error.error_message().is_some_and(|m| m.contains("no")));
        assert_eq!(error.error_data(), Some("{\"a\":1}"));
    }

    #[tokio::test]
    async fn test_result_error_keeps_data() {
        let result = LocalRunner::new()
            .run(
                |(), _ctx: DurableContext| async move { Err::<(), BoxError>(bad_error()) },
                (),
            )
            .await
            .expect("runner");
        let error = result.error().expect("the execution error");
        assert_eq!(error.error_type(), Some("Bad"));
        assert_eq!(error.error_data(), Some("{\"a\":1}"));
    }

    #[tokio::test]
    async fn operation_lookup_by_name() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_u32) }).name("process").await?;
                    ctx.step(|_| async { Ok(2_u32) }).name("charge").await?;
                    ctx.step(|_| async { Ok(3_u32) }).name("process").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let ops = result.operations();
        assert_eq!(ops.len(), 3);
        assert_eq!(
            result.operation("charge").map(TestOperation::id),
            Some(ops[1].id())
        );
        assert_eq!(result.operations_named("process").count(), 2);
        assert_eq!(
            result
                .operations_named("process")
                .nth(1)
                .map(TestOperation::id),
            Some(ops[2].id())
        );
        assert!(result.operation("absent").is_none());
    }

    #[test]
    fn fold_history_fills_detail_fields() {
        use aws_sdk_lambda::types::{CallbackStartedDetails, WaitStartedDetails};
        let events = vec![
            Event::builder()
                .event_type(EventType::WaitStarted)
                .id("w")
                .parent_id("p")
                .event_id(1)
                .wait_started_details(
                    WaitStartedDetails::builder()
                        .duration(10)
                        .scheduled_end_timestamp(DateTime::from_secs(10))
                        .build()
                        .expect("wait details"),
                )
                .build(),
            Event::builder()
                .event_type(EventType::CallbackStarted)
                .id("c")
                .event_id(2)
                .callback_started_details(
                    CallbackStartedDetails::builder()
                        .callback_id("cb-1")
                        .build()
                        .expect("callback details"),
                )
                .build(),
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(3)
                .build(),
            Event::builder()
                .event_type(EventType::StepFailed)
                .id("s")
                .event_id(4)
                .step_failed_details(
                    StepFailedDetails::builder()
                        .error(
                            EventError::builder()
                                .payload(
                                    ErrorObject::builder()
                                        .error_type("Bad")
                                        .error_message("no")
                                        .error_data("{\"a\":1}")
                                        .stack_trace("frame")
                                        .build(),
                                )
                                .build(),
                        )
                        .retry_details(RetryDetails::builder().current_attempt(1).build())
                        .build(),
                )
                .build(),
            // A later event of the wait carries no parent. The parent of the
            // first event stays.
            Event::builder()
                .event_type(EventType::WaitSucceeded)
                .id("w")
                .event_id(5)
                .build(),
            Event::builder()
                .event_type(EventType::ContextStarted)
                .id("x")
                .event_id(6)
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].parent_id(), Some("p"));
        // No history event carries `ReplayChildren`, so a folded context
        // reports `false`.
        assert!(
            matches!(
                ops[3].details(),
                TestOperationDetails::Context(c) if !c.replay_children()
            ),
            "{:?}",
            ops[3].details()
        );
        assert!(matches!(
            ops[0].details(),
            TestOperationDetails::Wait(w) if w.duration() == Some(std::time::Duration::from_secs(10))
        ));
        assert!(matches!(
            ops[1].details(),
            TestOperationDetails::Callback(c) if c.callback_id() == Some("cb-1")
        ));
        let error = ops[2].error().expect("the step error");
        assert_eq!(error.error_type(), Some("Bad"));
        assert_eq!(error.error_data(), Some("{\"a\":1}"));
        assert_eq!(error.stack_trace(), ["frame".to_owned()]);
    }

    /// One event per operation event type that carries an `EventError`,
    /// with its operation's kind and the status the event sets. Each error
    /// has every field set.
    fn error_event_rows() -> Vec<(Event, OperationKind, OperationStatus)> {
        use aws_sdk_lambda::types::{
            CallbackFailedDetails, ChainedInvokeFailedDetails, ChainedInvokeStoppedDetails,
            ChainedInvokeTimedOutDetails, ContextFailedDetails, WaitCancelledDetails,
        };
        let error = || {
            EventError::builder()
                .payload(
                    ErrorObject::builder()
                        .error_type("Kind")
                        .error_message("msg")
                        .error_data("{\"d\":1}")
                        .stack_trace("f1")
                        .stack_trace("f2")
                        .build(),
                )
                .build()
        };
        let base =
            |event_type: EventType| Event::builder().event_type(event_type).id("op").event_id(1);
        vec![
            (
                base(EventType::StepFailed)
                    .step_failed_details(StepFailedDetails::builder().error(error()).build())
                    .build(),
                OperationKind::Step,
                OperationStatus::Failed,
            ),
            (
                base(EventType::WaitCancelled)
                    .wait_cancelled_details(WaitCancelledDetails::builder().error(error()).build())
                    .build(),
                OperationKind::Wait,
                OperationStatus::Cancelled,
            ),
            (
                base(EventType::ContextFailed)
                    .context_failed_details(ContextFailedDetails::builder().error(error()).build())
                    .build(),
                OperationKind::Context,
                OperationStatus::Failed,
            ),
            (
                base(EventType::CallbackFailed)
                    .callback_failed_details(
                        CallbackFailedDetails::builder().error(error()).build(),
                    )
                    .build(),
                OperationKind::Callback,
                OperationStatus::Failed,
            ),
            (
                base(EventType::CallbackTimedOut)
                    .callback_timed_out_details(
                        CallbackTimedOutDetails::builder().error(error()).build(),
                    )
                    .build(),
                OperationKind::Callback,
                OperationStatus::TimedOut,
            ),
            (
                base(EventType::ChainedInvokeFailed)
                    .chained_invoke_failed_details(
                        ChainedInvokeFailedDetails::builder().error(error()).build(),
                    )
                    .build(),
                OperationKind::ChainedInvoke,
                OperationStatus::Failed,
            ),
            (
                base(EventType::ChainedInvokeTimedOut)
                    .chained_invoke_timed_out_details(
                        ChainedInvokeTimedOutDetails::builder()
                            .error(error())
                            .build(),
                    )
                    .build(),
                OperationKind::ChainedInvoke,
                OperationStatus::TimedOut,
            ),
            (
                base(EventType::ChainedInvokeStopped)
                    .chained_invoke_stopped_details(
                        ChainedInvokeStoppedDetails::builder()
                            .error(error())
                            .build(),
                    )
                    .build(),
                OperationKind::ChainedInvoke,
                OperationStatus::Stopped,
            ),
        ]
    }

    /// Every operation event type that carries an `EventError` keeps the
    /// whole record: type, message, data and stack trace. One row per type.
    #[test]
    fn fold_history_keeps_every_event_error_field() {
        for (event, kind, status) in error_event_rows() {
            let label = format!("{:?}", event.event_type());
            let ops = fold_history(std::slice::from_ref(&event));
            assert_eq!(ops.len(), 1, "{label}");
            assert_eq!(ops[0].kind(), kind, "{label}");
            assert_eq!(ops[0].status(), status, "{label}");
            assert!(ops[0].error().is_some(), "{label}: no error");
            let wire = ops[0].error().expect("the folded error");
            assert_eq!(wire.error_type(), Some("Kind"), "{label}");
            assert_eq!(wire.error_message(), Some("msg"), "{label}");
            assert_eq!(wire.error_data(), Some("{\"d\":1}"), "{label}");
            assert_eq!(
                wire.stack_trace(),
                ["f1".to_owned(), "f2".to_owned()],
                "{label}"
            );
            assert_eq!(ops[0].error_type(), Some("Kind"), "{label}");
            assert_eq!(ops[0].error_message(), Some("msg"), "{label}");
        }
    }

    /// Asserts the predicates of `result` agree with its status.
    fn assert_predicates_agree<O>(result: &TestResult<O>) {
        let status = result.status();
        assert_eq!(result.is_success(), status == ExecutionStatus::Succeeded);
        assert_eq!(
            result.is_failure(),
            matches!(
                status,
                ExecutionStatus::Failed | ExecutionStatus::TimedOut | ExecutionStatus::Stopped
            )
        );
        assert_eq!(result.is_suspended(), status == ExecutionStatus::Running);
    }

    #[tokio::test]
    async fn local_status_matches_predicates() {
        let ok = LocalRunner::new()
            .run(
                |(), _ctx: DurableContext| async { Ok::<_, BoxError>(1_u32) },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(ok.status(), ExecutionStatus::Succeeded);
        assert_predicates_agree(&ok);

        let failed = LocalRunner::new()
            .run(
                |(), _ctx: DurableContext| async { Err::<u32, BoxError>("boom".into()) },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(failed.status(), ExecutionStatus::Failed);
        assert_predicates_agree(&failed);

        let running = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<u32>().await?;
                    cb.result().await.map_err(BoxError::from)
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(running.status(), ExecutionStatus::Running);
        assert_predicates_agree(&running);
    }

    /// An event whose `Serialize` always fails.
    #[derive(Debug, Deserialize)]
    struct Unserializable;

    impl Serialize for Unserializable {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("refuse"))
        }
    }

    /// Asserts that `source` is a `serde_json::Error` with the text of `e`.
    fn assert_json_source(err: &TestRunnerError, e: &serde_json::Error) {
        let source = std::error::Error::source(err).expect("a source");
        let json = source
            .downcast_ref::<serde_json::Error>()
            .expect("the source is the serde_json error");
        assert_eq!(json.to_string(), e.to_string());
    }

    #[tokio::test]
    async fn event_serialization_is_a_runner_error() {
        let err = LocalRunner::new()
            .run(
                |_e: Unserializable, _ctx: DurableContext| async { Ok::<_, BoxError>(()) },
                Unserializable,
            )
            .await
            .expect_err("a runner error");
        let e = match &err {
            TestRunnerError::EventSerialization(e) => Some(e),
            _ => None,
        }
        .expect("an EventSerialization error");
        assert_eq!(
            err.to_string(),
            format!("durable: test runner: event serialization failed: {e}")
        );
        assert!(e.to_string().contains("refuse"), "{e}");
        assert_json_source(&err, e);
    }

    #[tokio::test]
    async fn output_decode_is_a_runner_error() {
        // `run` ties the handler output to `O`, so the test drives a handler
        // that returns a string and decodes its output as `u32`.
        let runner = LocalRunner::new();
        let err = runner
            .drive(
                Arc::new(runner.new_backend()),
                |(), _ctx: DurableContext| async { Ok::<_, BoxError>("text".to_owned()) },
                "null".to_owned(),
            )
            .await
            .expect("the run ends")
            .into_test_result::<u32>()
            .expect_err("a runner error");
        let e = match &err {
            TestRunnerError::OutputDeserialization(e) => Some(e),
            _ => None,
        }
        .expect("an OutputDeserialization error");
        assert_eq!(
            err.to_string(),
            format!("durable: test runner: output deserialization failed: {e}")
        );
        assert_json_source(&err, e);
    }

    #[tokio::test]
    async fn unexpected_envelope_is_a_runner_error() {
        let runner = LocalRunner::new();
        let service = |_event: lambda_runtime::LambdaEvent<crate::InvocationPayload>| {
            let response: crate::BoxedInvocationFuture = Box::pin(async {
                Ok(crate::InvocationResponse(
                    serde_json::json!({ "Status": "WEIRD" }),
                ))
            });
            response
        };
        let err = runner
            .drive_service(Arc::new(runner.new_backend()), service, "null".to_owned())
            .await
            .expect_err("a runner error");
        assert!(
            matches!(&err, TestRunnerError::UnexpectedEnvelope(Some(s)) if s == "WEIRD"),
            "{err:?}"
        );
        assert_eq!(
            err.to_string(),
            "durable: test runner: unexpected response envelope status: Some(\"WEIRD\")"
        );
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn invocation_type_default_is_request_response() {
        let request = CloudRunner::new("f").build_invoke(Vec::new());
        assert_eq!(request.invocation_type, WireInvocationType::RequestResponse);
    }

    #[test]
    fn invoke_request_uses_configured_type() {
        let request = CloudRunner::new("f")
            .invocation_type(InvocationType::Event)
            .build_invoke(Vec::new());
        assert_eq!(request.invocation_type, WireInvocationType::Event);
    }

    #[tokio::test]
    async fn configured_client_is_used() {
        let config = aws_sdk_lambda::Config::builder()
            .region(aws_sdk_lambda::config::Region::new("eu-west-3"))
            .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
            .build();
        let client = crate::LambdaClient::from_conf(config);
        let resolved = CloudRunner::new("f")
            .lambda_client(client)
            .region("us-east-1")
            .resolve_client()
            .await;
        assert_eq!(
            resolved.config().region().map(ToString::to_string),
            Some("eu-west-3".to_owned())
        );
    }

    /// A Lambda error with a fixed text.
    #[derive(Debug)]
    struct FakeError(&'static str);

    impl std::fmt::Display for FakeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for FakeError {}

    type GetExecutionOutput =
        aws_sdk_lambda::operation::get_durable_execution::GetDurableExecutionOutput;
    type GetHistoryOutput =
        aws_sdk_lambda::operation::get_durable_execution_history::GetDurableExecutionHistoryOutput;
    type InvokeOutput = aws_sdk_lambda::operation::invoke::InvokeOutput;

    /// The execution ARN every scripted invoke returns.
    const ARN: &str = "fake-execution-arn";

    /// One recorded `GetDurableExecution` request, with the tokio time it
    /// was made.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ExecutionCall {
        execution_arn: String,
        at: tokio::time::Instant,
    }

    /// One recorded `GetDurableExecutionHistory` request.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct HistoryCall {
        execution_arn: String,
        marker: Option<String>,
    }

    /// One scripted history page: its events and its next marker.
    type HistoryPage = Result<(Vec<Event>, Option<String>), BoxError>;

    /// A recording [`CloudApi`] that returns scripted outputs and errors,
    /// and records the arguments of every request.
    ///
    /// `Invoke` returns `invoke` once. `GetDurableExecution` pops the front
    /// of `executions`, and returns `RUNNING` once the queue is empty.
    /// `GetDurableExecutionHistory` pops the front of `history`; a request
    /// with no scripted page left panics, so a test notices an extra fetch.
    struct FakeApi {
        invoke: Mutex<Option<Result<InvokeOutput, BoxError>>>,
        executions: Mutex<std::collections::VecDeque<Result<GetExecutionOutput, BoxError>>>,
        history: Mutex<std::collections::VecDeque<HistoryPage>>,
        invokes: Mutex<Vec<InvokeRequest>>,
        execution_calls: Mutex<Vec<ExecutionCall>>,
        history_calls: Mutex<Vec<HistoryCall>>,
    }

    impl FakeApi {
        /// A fake whose invoke returns [`ARN`] and whose history is one
        /// empty page.
        fn new() -> Self {
            Self {
                invoke: Mutex::new(Some(Ok(invoke_output(Some(ARN))))),
                executions: Mutex::new(std::collections::VecDeque::new()),
                history: Mutex::new(std::collections::VecDeque::from([Ok((Vec::new(), None))])),
                invokes: Mutex::new(Vec::new()),
                execution_calls: Mutex::new(Vec::new()),
                history_calls: Mutex::new(Vec::new()),
            }
        }

        fn with_invoke(self, invoke: Result<InvokeOutput, BoxError>) -> Self {
            *self.invoke.lock().expect("lock") = Some(invoke);
            self
        }

        fn with_execution(self, execution: Result<GetExecutionOutput, BoxError>) -> Self {
            self.executions.lock().expect("lock").push_back(execution);
            self
        }

        /// Replaces the scripted history with one page that has no next
        /// marker.
        fn with_history(self, history: Result<Vec<Event>, BoxError>) -> Self {
            self.with_history_pages(vec![history.map(|events| (events, None))])
        }

        /// Replaces the scripted history with `pages`, served in order.
        fn with_history_pages(self, pages: Vec<HistoryPage>) -> Self {
            *self.history.lock().expect("lock") = pages.into();
            self
        }

        fn execution_call_count(&self) -> usize {
            self.execution_calls.lock().expect("lock").len()
        }

        fn invokes(&self) -> Vec<InvokeRequest> {
            self.invokes.lock().expect("lock").clone()
        }

        fn execution_arns(&self) -> Vec<String> {
            self.execution_calls
                .lock()
                .expect("lock")
                .iter()
                .map(|call| call.execution_arn.clone())
                .collect()
        }

        fn history_calls(&self) -> Vec<HistoryCall> {
            self.history_calls.lock().expect("lock").clone()
        }
    }

    impl CloudApi for FakeApi {
        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_invoke(&self, request: InvokeRequest) -> Result<InvokeOutput, BoxError> {
            self.invokes.lock().expect("lock").push(request);
            self.invoke
                .lock()
                .expect("lock")
                .take()
                .expect("one invoke per run")
        }

        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_get_durable_execution(
            &self,
            execution_arn: &str,
        ) -> Result<GetExecutionOutput, BoxError> {
            self.execution_calls
                .lock()
                .expect("lock")
                .push(ExecutionCall {
                    execution_arn: execution_arn.to_owned(),
                    at: tokio::time::Instant::now(),
                });
            self.executions
                .lock()
                .expect("lock")
                .pop_front()
                .unwrap_or_else(|| Ok(execution_output(WireExecutionStatus::Running, None)))
        }

        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_get_durable_execution_history(
            &self,
            execution_arn: &str,
            marker: Option<&str>,
        ) -> Result<GetHistoryOutput, BoxError> {
            self.history_calls.lock().expect("lock").push(HistoryCall {
                execution_arn: execution_arn.to_owned(),
                marker: marker.map(ToOwned::to_owned),
            });
            let (events, next_marker) = self
                .history
                .lock()
                .expect("lock")
                .pop_front()
                .expect("a scripted history page")?;
            Ok(GetHistoryOutput::builder()
                .set_events(Some(events))
                .set_next_marker(next_marker)
                .build()
                .expect("history output"))
        }
    }

    /// The history requests of one fetch of a single page.
    fn one_history_call() -> Vec<HistoryCall> {
        vec![HistoryCall {
            execution_arn: ARN.to_owned(),
            marker: None,
        }]
    }

    fn invoke_output(arn: Option<&str>) -> InvokeOutput {
        InvokeOutput::builder()
            .status_code(200)
            .set_durable_execution_arn(arn.map(ToOwned::to_owned))
            .build()
    }

    fn execution_output(status: WireExecutionStatus, result: Option<&str>) -> GetExecutionOutput {
        GetExecutionOutput::builder()
            .durable_execution_arn(ARN)
            .durable_execution_name("name")
            .function_arn("function")
            .start_timestamp(DateTime::from_secs(0))
            .status(status)
            .set_result(result.map(ToOwned::to_owned))
            .build()
            .expect("execution output")
    }

    fn fake_error(text: &'static str) -> BoxError {
        Box::new(FakeError(text))
    }

    /// Asserts `err` is an `Aws` error whose source is the fake error `text`.
    fn assert_aws(err: &TestRunnerError, text: &str) {
        assert!(matches!(err, TestRunnerError::Aws(_)), "{err:?}");
        let display = err.to_string();
        assert!(
            display.starts_with("durable: test runner: AWS call failed: ")
                && display.ends_with(text),
            "{display}"
        );
        let source = std::error::Error::source(err).expect("a source");
        assert_eq!(
            source.downcast_ref::<FakeError>().map(|e| e.0),
            Some(text),
            "the source is the fake error"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn fold_terminal_timed_out_status() {
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(WireExecutionStatus::TimedOut, None)));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::TimedOut);
        assert!(result.is_failure());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_invoke_error_is_aws() {
        let api = FakeApi::new().with_invoke(Err(fake_error("fake invoke")));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake invoke");
        assert_eq!(api.invokes().len(), 1);
        assert_eq!(api.execution_call_count(), 0);
        assert_eq!(api.history_calls(), Vec::<HistoryCall>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_get_execution_error_is_aws() {
        let api = FakeApi::new().with_execution(Err(fake_error("fake get")));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake get");
        assert_eq!(api.execution_arns(), vec![ARN.to_owned()]);
        assert_eq!(api.history_calls(), Vec::<HistoryCall>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_get_history_error_is_aws() {
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("5"),
            )))
            .with_history(Err(fake_error("fake history")));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake history");
        assert_eq!(api.execution_arns(), vec![ARN.to_owned()]);
        assert_eq!(api.history_calls(), one_history_call());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_missing_execution_arn_is_runner_error() {
        let api = FakeApi::new().with_invoke(Ok(invoke_output(None)));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect_err("a runner error");
        assert!(matches!(err, TestRunnerError::NoExecutionArn), "{err:?}");
        assert_eq!(
            err.to_string(),
            "durable: test runner: the invoke response carried no DurableExecutionArn"
        );
        assert_eq!(api.execution_call_count(), 0);
        assert_eq!(api.history_calls(), Vec::<HistoryCall>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_ignores_invoke_function_error_and_payload_for_both_invocation_types() {
        use aws_sdk_lambda::types::{EventResult, ExecutionSucceededDetails};
        for invocation_type in [InvocationType::RequestResponse, InvocationType::Event] {
            let invoke = InvokeOutput::builder()
                .status_code(200)
                .durable_execution_arn(ARN)
                .function_error("Unhandled")
                .payload(Blob::new("{\"bogus\":true}"))
                .build();
            let history = vec![
                Event::builder()
                    .event_type(EventType::ExecutionSucceeded)
                    .event_id(1)
                    .execution_succeeded_details(
                        ExecutionSucceededDetails::builder()
                            .result(EventResult::builder().payload("5").build())
                            .build(),
                    )
                    .build(),
            ];
            let api = FakeApi::new()
                .with_invoke(Ok(invoke))
                // The GetDurableExecution result differs from the history's,
                // so the assertion below shows the history wins.
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("7"),
                )))
                .with_history(Ok(history));
            let result = CloudRunner::new("f")
                .invocation_type(invocation_type)
                .run_with::<_, u32>(&api, Vec::new())
                .await
                .expect("runner");
            assert_eq!(
                result.status(),
                ExecutionStatus::Succeeded,
                "{invocation_type:?}"
            );
            assert_eq!(result.output(), Some(&5), "{invocation_type:?}");
            assert_eq!(api.invokes().len(), 1);
            assert_eq!(api.invokes()[0].invocation_type, invocation_type.to_wire());
            assert_eq!(api.execution_arns(), vec![ARN.to_owned()]);
            assert_eq!(api.history_calls(), one_history_call());
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_reads_terminal_error_from_history_for_each_terminal_event() {
        use aws_sdk_lambda::types::{
            EventError, ExecutionFailedDetails, ExecutionStoppedDetails, ExecutionTimedOutDetails,
        };
        let history_error = || {
            EventError::builder()
                .payload(
                    ErrorObject::builder()
                        .error_type("HistoryError")
                        .error_message("from history")
                        .error_data("{\"code\":1}")
                        .stack_trace("frame-1")
                        .stack_trace("frame-2")
                        .build(),
                )
                .build()
        };
        let failed = Event::builder()
            .event_type(EventType::ExecutionFailed)
            .execution_failed_details(
                ExecutionFailedDetails::builder()
                    .error(history_error())
                    .build(),
            );
        let timed_out = Event::builder()
            .event_type(EventType::ExecutionTimedOut)
            .execution_timed_out_details(
                ExecutionTimedOutDetails::builder()
                    .error(history_error())
                    .build(),
            );
        let stopped = Event::builder()
            .event_type(EventType::ExecutionStopped)
            .execution_stopped_details(
                ExecutionStoppedDetails::builder()
                    .error(history_error())
                    .build(),
            );
        let rows = [
            (WireExecutionStatus::Failed, ExecutionStatus::Failed, failed),
            (
                WireExecutionStatus::TimedOut,
                ExecutionStatus::TimedOut,
                timed_out,
            ),
            (
                WireExecutionStatus::Stopped,
                ExecutionStatus::Stopped,
                stopped,
            ),
        ];
        for (wire_status, status, event) in rows {
            // GetDurableExecution reports a different error, so the
            // assertions below show the history's terminal event wins.
            let execution = GetExecutionOutput::builder()
                .durable_execution_arn(ARN)
                .durable_execution_name("name")
                .function_arn("function")
                .start_timestamp(DateTime::from_secs(0))
                .status(wire_status)
                .error(
                    ErrorObject::builder()
                        .error_type("ExecutionError")
                        .error_message("from GetDurableExecution")
                        .error_data("{\"code\":2}")
                        .stack_trace("other-frame")
                        .build(),
                )
                .build()
                .expect("execution output");
            let api = FakeApi::new()
                .with_execution(Ok(execution))
                .with_history(Ok(vec![event.event_id(1).build()]));
            let result = CloudRunner::new("f")
                .run_with::<_, u32>(&api, Vec::new())
                .await
                .expect("runner");
            assert_eq!(result.status(), status);
            assert!(result.is_failure(), "{status}");
            let error = result.error().expect("an error");
            assert_eq!(error.error_type(), Some("HistoryError"), "{status}");
            assert_eq!(error.error_message(), Some("from history"), "{status}");
            assert_eq!(error.error_data(), Some("{\"code\":1}"), "{status}");
            assert_eq!(error.stack_trace(), ["frame-1", "frame-2"], "{status}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_poll_interval_defaults_to_one_second() {
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(WireExecutionStatus::Running, None)))
            .with_execution(Ok(execution_output(WireExecutionStatus::Running, None)))
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("1"),
            )));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert!(result.is_success());
        let calls = api.execution_calls.lock().expect("lock").clone();
        assert_eq!(calls.len(), 3);
        assert!(
            calls.iter().all(|call| call.execution_arn == ARN),
            "{calls:?}"
        );
        for pair in calls.windows(2) {
            assert_eq!(pair[1].at - pair[0].at, std::time::Duration::from_secs(1));
        }
        assert_eq!(api.history_calls(), one_history_call());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_default_polling_continues_past_150_fetches() {
        let mut api = FakeApi::new();
        for _ in 0..200 {
            api = api.with_execution(Ok(execution_output(WireExecutionStatus::Running, None)));
        }
        let api = api.with_execution(Ok(execution_output(
            WireExecutionStatus::Succeeded,
            Some("1"),
        )));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        assert_eq!(api.execution_call_count(), 201);
        assert_eq!(api.history_calls(), one_history_call());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_configured_poll_limit_ends_running() {
        let api = FakeApi::new();
        let result = CloudRunner::new("f")
            .max_poll_attempts(3)
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Running);
        assert_eq!(api.execution_arns(), vec![ARN.to_owned(); 3]);
        assert_eq!(api.history_calls(), one_history_call());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_records_request_arguments_and_follows_history_markers() {
        use aws_sdk_lambda::types::{EventResult, ExecutionSucceededDetails, StepStartedDetails};
        let step_started = |id: &str| {
            Event::builder()
                .event_type(EventType::StepStarted)
                .event_id(1)
                .id(id)
                .name(id)
                .step_started_details(StepStartedDetails::builder().build())
                .build()
        };
        let succeeded = Event::builder()
            .event_type(EventType::ExecutionSucceeded)
            .event_id(3)
            .execution_succeeded_details(
                ExecutionSucceededDetails::builder()
                    .result(EventResult::builder().payload("5").build())
                    .build(),
            )
            .build();
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(WireExecutionStatus::Running, None)))
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("5"),
            )))
            .with_history_pages(vec![
                Ok((vec![step_started("first")], Some("page-2".to_owned()))),
                Ok((vec![step_started("second")], Some("page-3".to_owned()))),
                // An empty marker ends the history as an absent one does.
                Ok((vec![succeeded], Some(String::new()))),
            ]);
        let result = CloudRunner::new("my-fn")
            .qualifier("live")
            .invocation_type(InvocationType::Event)
            .run_with::<_, u32>(&api, b"{\"x\":1}".to_vec())
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&5));
        assert_eq!(
            api.invokes(),
            vec![InvokeRequest {
                function_name: "my-fn".to_owned(),
                qualifier: "live".to_owned(),
                invocation_type: WireInvocationType::Event,
                payload: b"{\"x\":1}".to_vec(),
            }]
        );
        assert_eq!(api.execution_arns(), vec![ARN.to_owned(); 2]);
        let marker = |m: Option<&str>| HistoryCall {
            execution_arn: ARN.to_owned(),
            marker: m.map(ToOwned::to_owned),
        };
        assert_eq!(
            api.history_calls(),
            vec![marker(None), marker(Some("page-2")), marker(Some("page-3"))]
        );
        let ids: Vec<&str> = result.operations().iter().map(TestOperation::id).collect();
        assert_eq!(ids, ["first", "second"], "every page is folded");
    }

    // Rule 4 coverage of the runner error mappings and defaults

    #[tokio::test]
    async fn cloud_event_serialization_is_a_runner_error() {
        // `run` serializes the event before it resolves a client, so this
        // test makes no Lambda call.
        let err = CloudRunner::new("f")
            .run::<_, u32>(Unserializable)
            .await
            .expect_err("a runner error");
        let e = match &err {
            TestRunnerError::EventSerialization(e) => Some(e),
            _ => None,
        }
        .expect("an EventSerialization error");
        assert_eq!(
            err.to_string(),
            format!("durable: test runner: event serialization failed: {e}")
        );
        assert!(e.to_string().contains("refuse"), "{e}");
        assert_json_source(&err, e);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_output_decode_is_a_runner_error() {
        use aws_sdk_lambda::types::{EventResult, ExecutionSucceededDetails};
        let succeeded = Event::builder()
            .event_type(EventType::ExecutionSucceeded)
            .event_id(1)
            .execution_succeeded_details(
                ExecutionSucceededDetails::builder()
                    .result(EventResult::builder().payload("{\"a\":1}").build())
                    .build(),
            )
            .build();
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("{\"a\":1}"),
            )))
            .with_history(Ok(vec![succeeded]));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect_err("a runner error");
        let e = match &err {
            TestRunnerError::OutputDeserialization(e) => Some(e),
            _ => None,
        }
        .expect("an OutputDeserialization error");
        assert_eq!(
            err.to_string(),
            format!("durable: test runner: output deserialization failed: {e}")
        );
        assert_json_source(&err, e);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_success_without_terminal_event_reads_execution_result() {
        // The history holds no terminal execution event, so the output
        // comes from the `GetDurableExecution` result.
        let api = FakeApi::new().with_execution(Ok(execution_output(
            WireExecutionStatus::Succeeded,
            Some("9"),
        )));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&9));
        assert!(result.error().is_none());
    }

    /// The history's `ExecutionSucceeded` event wins even when it carries no
    /// result: the output decodes from `null`, not from the
    /// `GetDurableExecution` result.
    #[tokio::test(start_paused = true)]
    async fn cloud_success_terminal_event_without_result_is_null() {
        use aws_sdk_lambda::types::ExecutionSucceededDetails;
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("9"),
            )))
            .with_history(Ok(vec![
                Event::builder()
                    .event_type(EventType::ExecutionSucceeded)
                    .event_id(1)
                    .execution_succeeded_details(ExecutionSucceededDetails::builder().build())
                    .build(),
            ]));
        let result = CloudRunner::new("f")
            .run_with::<_, Option<u32>>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        assert_eq!(result.output(), Some(&None));
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_terminal_error_falls_back_to_execution_error_then_status_name() {
        use aws_sdk_lambda::types::{ExecutionFailedDetails, ExecutionStoppedDetails};
        let execution = |status: WireExecutionStatus, error: Option<ErrorObject>| {
            GetExecutionOutput::builder()
                .durable_execution_arn(ARN)
                .durable_execution_name("name")
                .function_arn("function")
                .start_timestamp(DateTime::from_secs(0))
                .status(status)
                .set_error(error)
                .build()
                .expect("execution output")
        };
        let execution_error = || {
            ErrorObject::builder()
                .error_type("ExecutionError")
                .error_message("from GetDurableExecution")
                .error_data("{\"code\":2}")
                .stack_trace("other-frame")
                .build()
        };
        let from_execution =
            crate::WireError::new(Some("ExecutionError"), Some("from GetDurableExecution"))
                .with_error_data(Some("{\"code\":2}"))
                .with_stack_trace(vec!["other-frame".to_owned()]);
        let rows: Vec<(&str, GetExecutionOutput, Vec<Event>, crate::WireError)> = vec![
            (
                "no terminal event: the GetDurableExecution error",
                execution(WireExecutionStatus::Failed, Some(execution_error())),
                Vec::new(),
                from_execution.clone(),
            ),
            (
                "a terminal event without an error: the GetDurableExecution error",
                execution(WireExecutionStatus::Failed, Some(execution_error())),
                vec![
                    Event::builder()
                        .event_type(EventType::ExecutionFailed)
                        .event_id(1)
                        .execution_failed_details(ExecutionFailedDetails::builder().build())
                        .build(),
                ],
                from_execution,
            ),
            (
                "no error anywhere: the status name",
                execution(WireExecutionStatus::Stopped, None),
                Vec::new(),
                crate::WireError::new(Some("Stopped"), Some("execution ended with status Stopped")),
            ),
            (
                "an error without type or message: the status name and an empty message",
                execution(WireExecutionStatus::Stopped, None),
                vec![
                    Event::builder()
                        .event_type(EventType::ExecutionStopped)
                        .event_id(1)
                        .execution_stopped_details(
                            ExecutionStoppedDetails::builder()
                                .error(
                                    EventError::builder()
                                        .payload(ErrorObject::builder().error_data("d").build())
                                        .build(),
                                )
                                .build(),
                        )
                        .build(),
                ],
                crate::WireError::new(Some("Stopped"), Some("")).with_error_data(Some("d")),
            ),
        ];
        for (label, execution, history, expected) in rows {
            let api = FakeApi::new()
                .with_execution(Ok(execution))
                .with_history(Ok(history));
            let result = CloudRunner::new("f")
                .run_with::<_, u32>(&api, Vec::new())
                .await
                .expect("runner");
            assert_eq!(result.error(), Some(&expected), "{label}");
        }
    }

    #[test]
    fn terminal_execution_event_is_the_last_one() {
        let event = |event_type: EventType, id: i32| {
            Event::builder().event_type(event_type).event_id(id).build()
        };
        let events = [
            event(EventType::ExecutionFailed, 1),
            event(EventType::ExecutionSucceeded, 2),
            event(EventType::InvocationCompleted, 3),
        ];
        assert_eq!(
            terminal_execution_event(&events).map(Event::event_id),
            Some(2)
        );
        assert!(terminal_execution_event(&events[2..]).is_none());
    }

    #[test]
    fn terminal_event_error_of_a_succeeded_event_is_none() {
        let event = Event::builder()
            .event_type(EventType::ExecutionSucceeded)
            .event_id(1)
            .build();
        assert!(terminal_event_error(&event).is_none());
    }

    #[test]
    fn unknown_wire_execution_status_is_failed() {
        assert_eq!(
            ExecutionStatus::from_wire(&WireExecutionStatus::from("NotAStatus")),
            ExecutionStatus::Failed
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_poll_limit_result_reports_polls_and_history() {
        use aws_sdk_lambda::types::StepStartedDetails;
        let api = FakeApi::new().with_history(Ok(vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .event_id(1)
                .id("s")
                .step_started_details(StepStartedDetails::builder().build())
                .build(),
        ]));
        let result = CloudRunner::new("f")
            .max_poll_attempts(2)
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Running);
        assert_eq!(result.invocation_count(), 2);
        let error = result.error().expect("the reason the run ended");
        assert_eq!(error.error_type(), None);
        assert_eq!(
            error.error_message(),
            Some(format!("execution {ARN} still running after 2 polls").as_str())
        );
        let ids: Vec<&str> = result.operations().iter().map(TestOperation::id).collect();
        assert_eq!(ids, ["s"]);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_poll_limit_history_error_is_aws() {
        let api = FakeApi::new().with_history(Err(fake_error("fake history")));
        let err = CloudRunner::new("f")
            .max_poll_attempts(1)
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake history");
        assert_eq!(api.execution_call_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_poll_limit_of_zero_counts_as_one() {
        let api = FakeApi::new();
        let result = CloudRunner::new("f")
            .max_poll_attempts(0)
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Running);
        assert_eq!(api.execution_call_count(), 1);
    }

    #[tokio::test]
    async fn resolve_client_without_a_client_uses_the_region() {
        let resolved = CloudRunner::new("f")
            .region("eu-west-3")
            .resolve_client()
            .await;
        assert_eq!(
            resolved.config().region().map(ToString::to_string),
            Some("eu-west-3".to_owned())
        );
    }

    /// `run` with a configured client sends each request through the
    /// `LambdaClient` implementation of [`CloudApi`], with every argument
    /// the runner sets. Each rule answers one request, so a request with
    /// a wrong argument matches no rule and fails the run.
    #[tokio::test]
    async fn lambda_client_sends_each_request_with_its_arguments() {
        use aws_sdk_lambda::types::{EventResult, ExecutionSucceededDetails};
        use aws_smithy_mocks::{RuleMode, mock, mock_client};
        let invoke = mock!(aws_sdk_lambda::Client::invoke)
            .match_requests(|req| {
                req.function_name() == Some("my-fn")
                    && req.qualifier() == Some("live")
                    && req.invocation_type() == Some(&WireInvocationType::Event)
                    && req.payload().map(Blob::as_ref) == Some(b"{\"x\":1}".as_slice())
            })
            .sequence()
            .output(|| invoke_output(Some(ARN)))
            .build();
        let execution = mock!(aws_sdk_lambda::Client::get_durable_execution)
            .match_requests(|req| {
                req.durable_execution_arn() == Some(ARN)
                    && req.include_execution_data() == Some(true)
            })
            .sequence()
            .output(|| execution_output(WireExecutionStatus::Succeeded, Some("5")))
            .build();
        let first_page = mock!(aws_sdk_lambda::Client::get_durable_execution_history)
            .match_requests(|req| {
                req.durable_execution_arn() == Some(ARN)
                    && req.include_execution_data() == Some(true)
                    && req.marker().is_none()
            })
            .sequence()
            .output(|| {
                GetHistoryOutput::builder()
                    .set_events(Some(Vec::new()))
                    .next_marker("page-2")
                    .build()
                    .expect("history output")
            })
            .build();
        let second_page = mock!(aws_sdk_lambda::Client::get_durable_execution_history)
            .match_requests(|req| {
                req.durable_execution_arn() == Some(ARN)
                    && req.include_execution_data() == Some(true)
                    && req.marker() == Some("page-2")
            })
            .sequence()
            .output(|| {
                GetHistoryOutput::builder()
                    .set_events(Some(vec![
                        Event::builder()
                            .event_type(EventType::ExecutionSucceeded)
                            .event_id(1)
                            .execution_succeeded_details(
                                ExecutionSucceededDetails::builder()
                                    .result(EventResult::builder().payload("6").build())
                                    .build(),
                            )
                            .build(),
                    ]))
                    .build()
                    .expect("history output")
            })
            .build();
        let client = mock_client!(
            aws_sdk_lambda,
            RuleMode::MatchAny,
            [&invoke, &execution, &first_page, &second_page]
        );
        let result = CloudRunner::new("my-fn")
            .qualifier("live")
            .invocation_type(InvocationType::Event)
            .poll_interval(std::time::Duration::ZERO)
            .lambda_client(client)
            .run::<_, u32>(serde_json::json!({ "x": 1 }))
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&6));
        assert_eq!(invoke.num_calls(), 1);
        assert_eq!(execution.num_calls(), 1);
        assert_eq!(first_page.num_calls(), 1);
        assert_eq!(second_page.num_calls(), 1);
    }

    #[tokio::test]
    async fn lambda_client_invoke_error_is_aws() {
        use aws_sdk_lambda::operation::invoke::InvokeError;
        use aws_smithy_mocks::{mock, mock_client};
        let invoke = mock!(aws_sdk_lambda::Client::invoke).then_error(|| {
            InvokeError::ResourceNotFoundException(
                aws_sdk_lambda::types::error::ResourceNotFoundException::builder().build(),
            )
        });
        let err = CloudRunner::new("f")
            .lambda_client(mock_client!(aws_sdk_lambda, [&invoke]))
            .run::<_, u32>(())
            .await
            .expect_err("a runner error");
        assert!(matches!(err, TestRunnerError::Aws(_)), "{err:?}");
        assert!(std::error::Error::source(&err).is_some());
        assert_eq!(invoke.num_calls(), 1);
    }

    /// A `GetDurableExecution` failure of the `LambdaClient` implementation
    /// of [`CloudApi`] is `TestRunnerError::Aws`, and its source is the SDK
    /// error of that call.
    #[tokio::test]
    async fn lambda_client_get_execution_error_is_aws() {
        use aws_sdk_lambda::operation::get_durable_execution::GetDurableExecutionError;
        use aws_smithy_mocks::{RuleMode, mock, mock_client};
        let invoke = mock!(aws_sdk_lambda::Client::invoke)
            .sequence()
            .output(|| invoke_output(Some(ARN)))
            .build();
        let execution = mock!(aws_sdk_lambda::Client::get_durable_execution)
            .sequence()
            .error(|| {
                GetDurableExecutionError::ResourceNotFoundException(
                    aws_sdk_lambda::types::error::ResourceNotFoundException::builder().build(),
                )
            })
            .build();
        let history = mock!(aws_sdk_lambda::Client::get_durable_execution_history)
            .then_output(|| GetHistoryOutput::builder().build().expect("history output"));
        let client = mock_client!(
            aws_sdk_lambda,
            RuleMode::MatchAny,
            [&invoke, &execution, &history]
        );
        let err = CloudRunner::new("f")
            .poll_interval(std::time::Duration::ZERO)
            .lambda_client(client)
            .run::<_, u32>(())
            .await
            .expect_err("a runner error");
        assert!(matches!(err, TestRunnerError::Aws(_)), "{err:?}");
        let source = std::error::Error::source(&err).expect("a source");
        let sdk = source
            .downcast_ref::<aws_sdk_lambda::error::SdkError<
                GetDurableExecutionError,
                aws_sdk_lambda::config::http::HttpResponse,
            >>()
            .expect("the GetDurableExecution SDK error");
        assert!(
            matches!(
                sdk.as_service_error(),
                Some(GetDurableExecutionError::ResourceNotFoundException(_))
            ),
            "{sdk:?}"
        );
        assert_eq!(invoke.num_calls(), 1);
        assert_eq!(execution.num_calls(), 1);
        assert_eq!(history.num_calls(), 0);
    }

    /// A `GetDurableExecutionHistory` failure of the `LambdaClient`
    /// implementation of [`CloudApi`] is `TestRunnerError::Aws`, and its
    /// source is the SDK error of that call.
    #[tokio::test]
    async fn lambda_client_get_history_error_is_aws() {
        use aws_sdk_lambda::operation::get_durable_execution_history::GetDurableExecutionHistoryError;
        use aws_smithy_mocks::{RuleMode, mock, mock_client};
        let invoke = mock!(aws_sdk_lambda::Client::invoke)
            .sequence()
            .output(|| invoke_output(Some(ARN)))
            .build();
        let execution = mock!(aws_sdk_lambda::Client::get_durable_execution)
            .sequence()
            .output(|| execution_output(WireExecutionStatus::Succeeded, Some("5")))
            .build();
        let history = mock!(aws_sdk_lambda::Client::get_durable_execution_history)
            .sequence()
            .error(|| {
                GetDurableExecutionHistoryError::ResourceNotFoundException(
                    aws_sdk_lambda::types::error::ResourceNotFoundException::builder().build(),
                )
            })
            .build();
        let client = mock_client!(
            aws_sdk_lambda,
            RuleMode::MatchAny,
            [&invoke, &execution, &history]
        );
        let err = CloudRunner::new("f")
            .poll_interval(std::time::Duration::ZERO)
            .lambda_client(client)
            .run::<_, u32>(())
            .await
            .expect_err("a runner error");
        assert!(matches!(err, TestRunnerError::Aws(_)), "{err:?}");
        let source = std::error::Error::source(&err).expect("a source");
        let sdk = source
            .downcast_ref::<aws_sdk_lambda::error::SdkError<
                GetDurableExecutionHistoryError,
                aws_sdk_lambda::config::http::HttpResponse,
            >>()
            .expect("the GetDurableExecutionHistory SDK error");
        assert!(
            matches!(
                sdk.as_service_error(),
                Some(GetDurableExecutionHistoryError::ResourceNotFoundException(
                    _
                ))
            ),
            "{sdk:?}"
        );
        assert_eq!(invoke.num_calls(), 1);
        assert_eq!(execution.num_calls(), 1);
        assert_eq!(history.num_calls(), 1);
    }

    #[test]
    fn no_execution_arn_has_no_source() {
        assert!(std::error::Error::source(&TestRunnerError::NoExecutionArn).is_none());
    }

    #[tokio::test]
    async fn unexpected_envelope_without_a_string_status() {
        let runner = LocalRunner::new();
        let service = |_event: lambda_runtime::LambdaEvent<crate::InvocationPayload>| {
            let response: crate::BoxedInvocationFuture = Box::pin(async {
                Ok(crate::InvocationResponse(
                    serde_json::json!({ "Status": 3 }),
                ))
            });
            response
        };
        let err = runner
            .drive_service(Arc::new(runner.new_backend()), service, "null".to_owned())
            .await
            .expect_err("a runner error");
        assert!(
            matches!(&err, TestRunnerError::UnexpectedEnvelope(None)),
            "{err:?}"
        );
        assert_eq!(
            err.to_string(),
            "durable: test runner: unexpected response envelope status: None"
        );
    }

    #[test]
    fn local_failed_envelope_keeps_data_and_stack_trace() {
        let response = serde_json::json!({
            "Status": "FAILED",
            "Error": {
                "ErrorType": "T",
                "ErrorMessage": "m",
                "ErrorData": "d",
                "StackTrace": ["a", "b"],
            },
        });
        let result: TestResult<()> = RawRun {
            end: RawEnd::Failed(envelope_wire_error(&response)),
            operations: Vec::new(),
            invocations: 1,
            events: Vec::new(),
        }
        .into_test_result()
        .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Failed);
        assert_eq!(
            result.error(),
            Some(
                &crate::WireError::new(Some("T"), Some("m"))
                    .with_error_data(Some("d"))
                    .with_stack_trace(vec!["a".to_owned(), "b".to_owned()])
            )
        );
    }

    #[tokio::test]
    async fn local_succeeded_operation_has_no_error() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_u32) }).name("ok").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let step = result.operation("ok").expect("the step");
        assert!(step.error().is_none(), "{:?}", step.error());
        assert!(result.error().is_none());
    }

    #[test]
    fn snapshot_reports_stack_trace_and_drops_a_negative_wait() {
        let step = |action: OperationAction, error: Option<ErrorObject>| {
            OperationUpdate::builder()
                .id("s")
                .r#type(OperationType::Step)
                .action(action)
                .set_error(error)
                .build()
                .expect("valid update")
        };
        let backend = seeded_backend(
            Vec::new(),
            0,
            &[
                step(OperationAction::Start, None),
                step(
                    OperationAction::Fail,
                    Some(
                        ErrorObject::builder()
                            .error_type("T")
                            .stack_trace("frame")
                            .build(),
                    ),
                ),
                wait_start("w", -5),
            ],
        );
        let ops = backend.snapshot_operations();
        let step_op = ops.iter().find(|op| op.id() == "s").expect("the step");
        assert_eq!(
            step_op.error().map(crate::WireError::stack_trace),
            Some(["frame".to_owned()].as_slice())
        );
        let wait_op = ops.iter().find(|op| op.id() == "w").expect("the wait");
        assert!(
            matches!(wait_op.details(), TestOperationDetails::Wait(w) if w.duration().is_none()),
            "{:?}",
            wait_op.details()
        );
    }

    #[test]
    fn fold_history_drops_a_negative_wait_duration() {
        use aws_sdk_lambda::types::WaitStartedDetails;
        let event = Event::builder()
            .event_type(EventType::WaitStarted)
            .id("w")
            .event_id(1)
            .wait_started_details(
                WaitStartedDetails::builder()
                    .duration(-1)
                    .scheduled_end_timestamp(DateTime::from_secs(0))
                    .build()
                    .expect("wait details"),
            )
            .build();
        let ops = fold_history(std::slice::from_ref(&event));
        assert!(
            matches!(ops[0].details(), TestOperationDetails::Wait(w) if w.duration().is_none()),
            "{:?}",
            ops[0].details()
        );
    }

    /// Runs `item` as the only item of a Flat map named `m`, and asserts
    /// that the record named `record` names the map as its parent.
    ///
    /// A Flat item writes no record of its own. So every operation inside it
    /// must record the batch parent as `ParentId`, or its parent is missing
    /// from the history.
    async fn assert_flat_item_records_batch_parent<F, Fut, O>(
        runner: LocalRunner,
        record: &str,
        item: F,
    ) where
        F: Fn(DurableContext) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        use crate::builders::map_parallel::NestingMode;
        let result = runner
            .run(
                move |(), ctx: DurableContext| {
                    let item = item.clone();
                    async move {
                        let out = ctx
                            .map(vec![1_u32], move |child, _item, _idx| item(child))
                            .name("m")
                            .nesting(NestingMode::Flat)
                            .await?;
                        Ok::<_, BoxError>(out)
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        let m = result.operation("m").expect("map record");
        let op = result.operation(record).expect("record");
        assert_eq!(op.parent_id(), Some(m.id()), "{record}");
    }

    #[tokio::test]
    async fn flat_item_child_context_records_batch_parent() {
        assert_flat_item_records_batch_parent(LocalRunner::new(), "nested", |child| async move {
            child
                .run_in_child_context(|nested| async move {
                    Ok(nested
                        .step(|_| async { Ok(1_u32) })
                        .name("inner-step")
                        .await?)
                })
                .name("nested")
                .await
                .map_err(BoxError::from)
        })
        .await;
    }

    #[tokio::test]
    async fn flat_item_combinator_records_batch_parent() {
        assert_flat_item_records_batch_parent(LocalRunner::new(), "gather", |child| async move {
            let a = child.step(|_| async { Ok(1_u32) }).name("a").future();
            let b = child.step(|_| async { Ok(2_u32) }).name("b").future();
            child
                .try_join_all([a, b])
                .name("gather")
                .await
                .map_err(BoxError::from)
        })
        .await;
    }

    #[tokio::test]
    async fn flat_item_nested_map_records_batch_parent() {
        assert_flat_item_records_batch_parent(LocalRunner::new(), "inner", |child| async move {
            child
                .map(vec![1_u32], |grandchild, item, _idx| async move {
                    Ok(grandchild
                        .step(move |_| async move { Ok(item) })
                        .name("inner-step")
                        .await?)
                })
                .name("inner")
                .await
                .map_err(BoxError::from)
        })
        .await;
    }

    #[tokio::test]
    async fn flat_item_callback_records_batch_parent() {
        assert_flat_item_records_batch_parent(LocalRunner::new(), "cb", |child| async move {
            let cb = child.create_callback::<String>().name("cb").await?;
            Ok(cb.id().to_owned())
        })
        .await;
    }

    #[tokio::test]
    async fn flat_item_wait_for_callback_records_batch_parent() {
        let runner = LocalRunner::new()
            .callback_for("approval", CallbackReply::success(&"ok").expect("encodes"));
        assert_flat_item_records_batch_parent(runner, "approval", |child| async move {
            child
                .wait_for_callback::<String, _, _>(|_step_ctx, _callback_id| async { Ok(()) })
                .name("approval")
                .await
                .map_err(BoxError::from)
        })
        .await;
    }

    #[tokio::test]
    async fn flat_item_step_records_batch_parent() {
        assert_flat_item_records_batch_parent(LocalRunner::new(), "s", |child| async move {
            child
                .step(|_| async { Ok(1_u32) })
                .name("s")
                .await
                .map_err(BoxError::from)
        })
        .await;
    }

    /// The detail of a step operation, or `None` for another kind.
    fn step_details(op: &TestOperation) -> Option<&StepDetails> {
        match op.details() {
            TestOperationDetails::Step(step) => Some(step),
            _ => None,
        }
    }

    /// The detail of a wait operation, or `None` for another kind.
    fn wait_details(op: &TestOperation) -> Option<&WaitDetails> {
        match op.details() {
            TestOperationDetails::Wait(wait) => Some(wait),
            _ => None,
        }
    }

    /// A `TestResult` holding `operations`, built without a run (the
    /// crate-private constructor of the table tests).
    fn result_with(operations: Vec<TestOperation>) -> TestResult<()> {
        TestResult {
            status: ExecutionStatus::Succeeded,
            output: Some(()),
            error: None,
            operations,
            invocations: 0,
            history_events: Vec::new(),
            invocation_records: Vec::new(),
        }
    }

    /// Every column except the ones `disabled` names.
    fn columns_without(disabled: &[&str]) -> PrintColumns {
        let on = |name: &str| !disabled.contains(&name);
        PrintColumns {
            parent_id: on("parent_id"),
            name: on("name"),
            kind: on("kind"),
            sub_type: on("sub_type"),
            status: on("status"),
            start_time: on("start_time"),
            end_time: on("end_time"),
            duration: on("duration"),
        }
    }

    /// Only the columns `enabled` names.
    fn columns_only(enabled: &[&str]) -> PrintColumns {
        let on = |name: &str| enabled.contains(&name);
        PrintColumns {
            parent_id: on("parent_id"),
            name: on("name"),
            kind: on("kind"),
            sub_type: on("sub_type"),
            status: on("status"),
            start_time: on("start_time"),
            end_time: on("end_time"),
            duration: on("duration"),
        }
    }

    #[tokio::test]
    async fn wait_times_follow_virtual_clock() {
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    ctx.wait(std::time::Duration::from_secs(10))
                        .name("pause")
                        .await?;
                    ctx.step(|_| async { Ok(2_i32) }).name("b").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success());
        let a = result.operation("a").expect("step a");
        let wait = result.operation("pause").expect("the wait");
        let b = result.operation("b").expect("step b");
        let start = result.operations()[0].start_time().expect("a start time");
        let at = |n: u64| Some(start + std::time::Duration::from_secs(n));
        assert_eq!(wait.start_time(), at(0));
        assert_eq!(wait.end_time(), at(10));
        assert_eq!(
            wait_details(wait)
                .expect("wait details")
                .scheduled_end_time(),
            at(10)
        );
        assert_eq!(b.start_time(), at(10));
        assert_eq!(a.end_time(), a.start_time());
    }

    #[tokio::test]
    async fn retried_step_reports_next_attempt_time() {
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|step| async move {
                        if step.attempt() == 1 {
                            return Err::<i32, BoxError>("first attempt fails".into());
                        }
                        Ok(7)
                    })
                    .name("flaky")
                    .retry_strategy(|_err, _attempt| RetryDecision::Retry {
                        delay: std::time::Duration::from_secs(30),
                    })
                    .await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error());
        let step = result.operation("flaky").expect("the step");
        let start = result.operations()[0].start_time().expect("start time");
        let at30 = Some(start + std::time::Duration::from_secs(30));
        assert_eq!(
            step_details(step)
                .expect("step details")
                .next_attempt_time(),
            at30
        );
        assert_eq!(step.end_time(), at30);
    }

    #[tokio::test]
    async fn step_without_retry_has_no_next_attempt_time() {
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_i32) }).name("once").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let step = result.operation("once").expect("the step");
        assert_eq!(
            step_details(step)
                .expect("step details")
                .next_attempt_time(),
            None
        );
    }

    #[test]
    fn end_time_keeps_the_first_terminal_status() {
        // A record that ends at virtual time 5 keeps that end time when a
        // later write at time 9 sets another terminal status.
        let mut op = new_started_op("x".to_owned(), OperationKind::Step, None, None, 1);
        op.set_status(OperationStatus::Pending, 3);
        assert_eq!(op.end_at, None);
        op.set_status(OperationStatus::Succeeded, 5);
        op.set_status(OperationStatus::Failed, 9);
        assert_eq!((op.start_at, op.end_at), (1, Some(5)));
    }

    #[test]
    fn every_terminal_status_sets_end_at() {
        for (status, terminal) in [
            (OperationStatus::Started, false),
            (OperationStatus::Pending, false),
            (OperationStatus::Ready, false),
            (OperationStatus::Succeeded, true),
            (OperationStatus::Failed, true),
            (OperationStatus::TimedOut, true),
            (OperationStatus::Cancelled, true),
            (OperationStatus::Stopped, true),
        ] {
            let mut op = new_started_op("x".to_owned(), OperationKind::Step, None, None, 0);
            op.set_status(status, 4);
            assert_eq!(op.end_at, terminal.then_some(4), "{status:?}");
        }
    }

    #[tokio::test]
    async fn callback_reply_and_invoke_outcome_set_end_time() {
        // A delivered callback reply and a chained invoke outcome both end
        // their records at the clock time of the advance.
        let result = LocalRunner::new()
            .callback(CallbackReply::success(&true).expect("reply"))
            .register_function("target", |_input: serde_json::Value| async move {
                Ok::<_, BoxError>(serde_json::json!(1))
            })
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.wait(std::time::Duration::from_secs(4))
                        .name("w")
                        .await?;
                    let cb = ctx.create_callback::<bool>().name("cb").await?;
                    cb.result().await?;
                    ctx.invoke::<serde_json::Value, _>("target", serde_json::json!(0))
                        .name("call")
                        .await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error());
        let start = result.operations()[0].start_time().expect("start time");
        let at4 = Some(start + std::time::Duration::from_secs(4));
        assert_eq!(result.operation("cb").expect("callback").end_time(), at4);
        assert_eq!(result.operation("call").expect("invoke").end_time(), at4);
    }

    #[test]
    fn local_time_past_the_system_time_range_is_none() {
        let backend = Backend::new(QueuedReplies::default(), None);
        let state = backend.lock();
        assert_eq!(state.local_time(0), Some(state.base));
        assert_eq!(state.local_time(u64::MAX), None);
    }

    #[test]
    fn fold_history_fills_times() {
        use aws_sdk_lambda::types::WaitStartedDetails;
        let events = vec![
            Event::builder()
                .event_type(EventType::WaitStarted)
                .id("w")
                .event_id(1)
                .event_timestamp(DateTime::from_secs(100))
                .wait_started_details(
                    WaitStartedDetails::builder()
                        .duration(10)
                        .scheduled_end_timestamp(DateTime::from_secs(110))
                        .build()
                        .expect("wait details"),
                )
                .build(),
            Event::builder()
                .event_type(EventType::WaitSucceeded)
                .id("w")
                .event_id(2)
                .event_timestamp(DateTime::from_secs(110))
                .build(),
        ];
        let ops = fold_history(&events);
        let at = |n: u64| Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(n));
        assert_eq!(ops[0].start_time(), at(100));
        assert_eq!(ops[0].end_time(), at(110));
        assert_eq!(
            wait_details(&ops[0])
                .expect("wait details")
                .scheduled_end_time(),
            at(110)
        );
    }

    #[test]
    fn fold_history_derives_next_attempt_time() {
        use aws_sdk_lambda::types::{RetryDetails, StepFailedDetails};
        let failed = |id: i32, at: i64, delay: Option<i32>| {
            let mut retry = RetryDetails::builder().current_attempt(1);
            if let Some(delay) = delay {
                retry = retry.next_attempt_delay_seconds(delay);
            }
            Event::builder()
                .event_type(EventType::StepFailed)
                .id("s")
                .event_id(id)
                .event_timestamp(DateTime::from_secs(at))
                .step_failed_details(
                    StepFailedDetails::builder()
                        .retry_details(retry.build())
                        .build(),
                )
                .build()
        };
        let at = |n: u64| Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(n));
        let ops = fold_history(&[failed(1, 100, Some(30))]);
        assert_eq!(
            step_details(&ops[0]).expect("step").next_attempt_time(),
            at(130)
        );
        // A later failure without a delay keeps the earlier time, and the
        // latest failure with a delay replaces it.
        let ops = fold_history(&[failed(1, 100, Some(30)), failed(2, 200, None)]);
        assert_eq!(
            step_details(&ops[0]).expect("step").next_attempt_time(),
            at(130)
        );
        let ops = fold_history(&[failed(1, 100, Some(30)), failed(2, 200, Some(5))]);
        assert_eq!(
            step_details(&ops[0]).expect("step").next_attempt_time(),
            at(205)
        );
        // No delay, a negative delay or no timestamp gives no time.
        let ops = fold_history(&[failed(1, 100, None)]);
        assert_eq!(
            step_details(&ops[0]).expect("step").next_attempt_time(),
            None
        );
        let ops = fold_history(&[failed(1, 100, Some(-1))]);
        assert_eq!(
            step_details(&ops[0]).expect("step").next_attempt_time(),
            None
        );
        let untimed = Event::builder()
            .event_type(EventType::StepFailed)
            .id("s")
            .event_id(1)
            .step_failed_details(
                StepFailedDetails::builder()
                    .retry_details(
                        RetryDetails::builder()
                            .current_attempt(1)
                            .next_attempt_delay_seconds(30)
                            .build(),
                    )
                    .build(),
            )
            .build();
        let ops = fold_history(&[untimed]);
        assert_eq!(
            step_details(&ops[0]).expect("step").next_attempt_time(),
            None
        );
    }

    #[test]
    fn fold_history_keeps_start_time_of_an_untimed_first_event() {
        // The first event carries no timestamp and a later one does. The
        // start time is the first event's timestamp, so it stays `None`.
        let untimed = Event::builder()
            .event_type(EventType::StepStarted)
            .id("s")
            .event_id(1)
            .build();
        let timed = Event::builder()
            .event_type(EventType::StepSucceeded)
            .id("s")
            .event_id(2)
            .event_timestamp(DateTime::from_secs(200))
            .build();
        let ops = fold_history(&[untimed, timed]);
        assert_eq!(ops[0].start_time(), None);
        assert_eq!(
            ops[0].end_time(),
            Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(200))
        );
    }

    #[test]
    fn fold_history_clears_next_attempt_time_when_the_latest_is_unrepresentable() {
        use aws_sdk_lambda::types::{RetryDetails, StepFailedDetails};
        let failed = |id: i32, at: Option<i64>, delay: i32| {
            let mut event = Event::builder()
                .event_type(EventType::StepFailed)
                .id("s")
                .event_id(id)
                .step_failed_details(
                    StepFailedDetails::builder()
                        .retry_details(
                            RetryDetails::builder()
                                .current_attempt(1)
                                .next_attempt_delay_seconds(delay)
                                .build(),
                        )
                        .build(),
                );
            if let Some(at) = at {
                event = event.event_timestamp(DateTime::from_secs(at));
            }
            event.build()
        };
        let next = |events: &[Event]| {
            step_details(&fold_history(events)[0])
                .expect("step")
                .next_attempt_time()
        };
        let earlier = failed(1, Some(100), 30);
        assert!(next(std::slice::from_ref(&earlier)).is_some());
        // The latest failure carries a delay but no timestamp.
        assert_eq!(next(&[earlier.clone(), failed(2, None, 5)]), None);
        // The latest failure carries a negative delay.
        assert_eq!(next(&[earlier.clone(), failed(2, Some(200), -1)]), None);
        // The latest failure's timestamp plus its delay overflows.
        let last = std::time::UNIX_EPOCH
            .checked_add(std::time::Duration::from_secs(i64::MAX.unsigned_abs()))
            .expect("in the SystemTime range on this platform");
        assert!(
            last.checked_add(std::time::Duration::from_secs(5))
                .is_none()
        );
        assert_eq!(
            std::time::SystemTime::try_from(DateTime::from_secs(i64::MAX)).ok(),
            Some(last)
        );
        assert_eq!(next(&[earlier, failed(2, Some(i64::MAX), 5)]), None);
    }

    #[test]
    fn fold_history_clears_end_time_when_a_retry_starts() {
        let event = |id: i32, kind: EventType, at: i64| {
            Event::builder()
                .event_type(kind)
                .id("s")
                .event_id(id)
                .event_timestamp(DateTime::from_secs(at))
                .build()
        };
        let at = |n: u64| Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(n));
        let failed = [
            event(1, EventType::StepStarted, 100),
            event(2, EventType::StepFailed, 101),
        ];
        assert_eq!(fold_history(&failed)[0].end_time(), at(101));
        let retried = [
            event(1, EventType::StepStarted, 100),
            event(2, EventType::StepFailed, 101),
            event(3, EventType::StepStarted, 131),
        ];
        let ops = fold_history(&retried);
        assert_eq!(ops[0].start_time(), at(100));
        assert_eq!(ops[0].end_time(), None);
    }

    #[tokio::test]
    async fn display_prints_js_table() {
        use std::fmt::Write as _;
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_i32) }).name("fetch").await?;
                    ctx.run_in_child_context(|child| async move {
                        child.step(|_| async { Ok(2_i32) }).name("inner").await?;
                        Ok(())
                    })
                    .name("group")
                    .await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("cooldown")
                        .await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let g = result.operation("group").expect("the context").id();
        let expected = format!(
            "parentId | name | type | subType | status | duration\n\
             - | fetch | STEP | Step | SUCCEEDED | 0ms\n\
             - | group | CONTEXT | RunInChildContext | SUCCEEDED | 0ms\n\
             {g} | inner | STEP | Step | SUCCEEDED | 0ms\n\
             - | cooldown | WAIT | Wait | SUCCEEDED | 1000ms\n"
        );
        assert_eq!(
            result
                .display_with(columns_without(&["start_time", "end_time"]))
                .to_string(),
            expected
        );
        let cell = |t: Option<std::time::SystemTime>| iso_millis(unix_millis(t.expect("a time")));
        let mut full =
            "parentId | name | type | subType | status | startTime | endTime | duration\n"
                .to_owned();
        for (op, duration) in result
            .operations()
            .iter()
            .zip(["0ms", "0ms", "0ms", "1000ms"])
        {
            writeln!(
                full,
                "{} | {} | {} | {} | {} | {} | {} | {duration}",
                op.parent_id().unwrap_or("-"),
                op.name().unwrap_or("-"),
                op.kind().wire_str(),
                op.sub_type().unwrap_or("-"),
                op.status().wire_str(),
                cell(op.start_time()),
                cell(op.end_time()),
            )
            .expect("write to a String");
        }
        assert_eq!(format!("{result}"), full);
    }

    #[tokio::test]
    async fn display_with_selects_columns() {
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_i32) }).name("fetch").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(
            result
                .display_with(columns_only(&["name", "status"]))
                .to_string(),
            "name | status\nfetch | SUCCEEDED\n"
        );
    }

    #[tokio::test]
    async fn display_prints_dash_for_absent_values() {
        let result = LocalRunner::new()
            .run(
                |_e: (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_i32) }).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(
            result
                .display_with(columns_only(&["parent_id", "name"]))
                .to_string(),
            "parentId | name\n- | -\n"
        );
    }

    #[test]
    fn display_prints_dash_for_absent_times_and_sub_type() {
        // A record with no sub-type, no start and no end prints `-` in
        // those cells and in the duration.
        let op = TestOperation::started("x".to_owned(), OperationKind::Step);
        let only_start = TestOperation {
            start_time: Some(std::time::UNIX_EPOCH),
            ..TestOperation::started("y".to_owned(), OperationKind::Step)
        };
        let result = result_with(vec![op, only_start]);
        assert_eq!(
            result
                .display_with(columns_only(&[
                    "sub_type",
                    "start_time",
                    "end_time",
                    "duration"
                ]))
                .to_string(),
            "subType | startTime | endTime | duration\n\
             - | - | - | -\n\
             - | 1970-01-01T00:00:00.000Z | - | -\n"
        );
    }

    #[test]
    fn display_preserves_millisecond_fraction() {
        let op = TestOperation {
            start_time: Some(std::time::UNIX_EPOCH + std::time::Duration::from_millis(100_123)),
            ..TestOperation::started("x".to_owned(), OperationKind::Step)
        };
        let result = result_with(vec![op]);
        let table = result
            .display_with(columns_only(&["start_time"]))
            .to_string();
        assert_eq!(table, "startTime\n1970-01-01T00:01:40.123Z\n");
        assert!(table.ends_with(".123Z\n"));
    }

    #[test]
    fn display_truncates_times_before_the_epoch_toward_zero() {
        // 1.5 ms before the epoch truncates to -1 ms, which prints as the
        // last millisecond of 1969 with the Euclidean remainder 999.
        let before = std::time::UNIX_EPOCH - std::time::Duration::from_micros(1_500);
        assert_eq!(unix_millis(before), -1);
        assert_eq!(iso_millis(-1), "1969-12-31T23:59:59.999Z");
        // A year the formatter cannot print prints `-`.
        assert_eq!(iso_millis(i128::from(i64::MAX)), "-");
        // Seconds that do not fit an `i64` print `-`.
        assert_eq!(iso_millis(i128::MAX), "-");
        assert_eq!(iso_millis(i128::MIN), "-");
    }

    #[test]
    fn unix_millis_keeps_the_i64_endpoints_and_beyond() {
        // The `i64::MIN` endpoint is a valid millisecond value. Narrowing
        // its magnitude to `i64` before negation would reject it.
        let min_span = std::time::Duration::from_millis(i64::MIN.unsigned_abs());
        let at_min = std::time::UNIX_EPOCH
            .checked_sub(min_span)
            .expect("in the SystemTime range on this platform");
        assert_eq!(unix_millis(at_min), i128::from(i64::MIN));
        let max_span = std::time::Duration::from_millis(i64::MAX.unsigned_abs());
        let at_max = std::time::UNIX_EPOCH
            .checked_add(max_span)
            .expect("in the SystemTime range on this platform");
        assert_eq!(unix_millis(at_max), i128::from(i64::MAX));
        // One millisecond past each endpoint is still exact.
        let one = std::time::Duration::from_millis(1);
        assert_eq!(
            unix_millis(at_min.checked_sub(one).expect("in range")),
            i128::from(i64::MIN) - 1
        );
        assert_eq!(
            unix_millis(at_max.checked_add(one).expect("in range")),
            i128::from(i64::MAX) + 1
        );
    }

    #[test]
    fn display_prints_duration_for_times_beyond_the_i64_millisecond_range() {
        // Both times exist, but neither fits `i64` milliseconds and neither
        // time cell can be formatted. The duration still prints exactly.
        let span = std::time::Duration::from_millis(i64::MAX.unsigned_abs())
            + std::time::Duration::from_millis(1);
        let op = TestOperation {
            start_time: std::time::UNIX_EPOCH.checked_sub(span),
            end_time: std::time::UNIX_EPOCH.checked_add(span),
            ..TestOperation::started("x".to_owned(), OperationKind::Step)
        };
        assert!(op.start_time.is_some() && op.end_time.is_some());
        let table = result_with(vec![op])
            .display_with(columns_only(&["start_time", "end_time", "duration"]))
            .to_string();
        assert_eq!(
            table,
            "startTime | endTime | duration\n- | - | 18446744073709551616ms\n"
        );
    }

    #[test]
    fn display_prints_duration_across_the_whole_i64_range() {
        // Both times fit `i64` milliseconds, but `end - start` is about
        // 2^64 ms and does not fit an `i64`. The cell prints it exactly.
        let span = std::time::Duration::from_millis(i64::MAX.unsigned_abs());
        let op = TestOperation {
            start_time: std::time::UNIX_EPOCH.checked_sub(span),
            end_time: std::time::UNIX_EPOCH.checked_add(span),
            ..TestOperation::started("x".to_owned(), OperationKind::Step)
        };
        assert!(op.start_time.is_some() && op.end_time.is_some());
        let table = result_with(vec![op])
            .display_with(columns_only(&["duration"]))
            .to_string();
        assert_eq!(table, "duration\n18446744073709551614ms\n");
    }

    #[test]
    fn display_empty_result() {
        let result = result_with(Vec::new());
        assert_eq!(format!("{result}"), "No operations found.\n");
    }

    #[test]
    fn print_columns_default_enables_every_column() {
        assert_eq!(PrintColumns::default(), columns_without(&[]));
    }

    // History events and invocation records.

    /// The event types of `events`, in order.
    fn event_types(events: &[Event]) -> Vec<EventType> {
        events
            .iter()
            .filter_map(|event| event.event_type().cloned())
            .collect()
    }

    /// The history events and the operations of a result.
    fn history_of<O>(result: &TestResult<O>) -> (Vec<Event>, Vec<TestOperation>) {
        (
            result.history_events().to_vec(),
            result.operations().to_vec(),
        )
    }

    /// Two steps: each step's `events()` holds only its own events.
    #[tokio::test]
    async fn operation_events_are_filtered_by_id() {
        let result = LocalRunner::new()
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
                    ctx.step(|_| async { Ok(2_u32) }).name("b").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        for name in ["a", "b"] {
            let op = result.operation(name).expect("the step");
            assert_eq!(
                event_types(op.events()),
                [EventType::StepStarted, EventType::StepSucceeded],
                "{name}"
            );
            assert!(
                op.events().iter().all(|e| e.id() == Some(op.id())),
                "{name}"
            );
        }
    }

    /// A synthetic event of `event_type` for the operation `id`, at `secs`.
    fn synthetic_event(event_id: i32, event_type: EventType, id: Option<&str>, secs: i64) -> Event {
        Event::builder()
            .event_id(event_id)
            .event_type(event_type)
            .set_id(id.map(ToOwned::to_owned))
            .event_timestamp(DateTime::from_secs(secs))
            .build()
    }

    /// An `ExecutionSucceeded` event whose result is `1`.
    fn execution_succeeded_event(event_id: i32) -> Event {
        Event::builder()
            .event_id(event_id)
            .event_type(EventType::ExecutionSucceeded)
            .execution_succeeded_details(
                aws_sdk_lambda::types::ExecutionSucceededDetails::builder()
                    .result(EventResult::builder().payload("1").build())
                    .build(),
            )
            .build()
    }

    /// `CloudRunner` returns the fetched history unchanged.
    #[tokio::test(start_paused = true)]
    async fn fold_history_keeps_service_events() {
        let events = vec![
            synthetic_event(1, EventType::ExecutionStarted, Some("execution"), 0),
            synthetic_event(2, EventType::StepStarted, Some("s"), 1),
            execution_succeeded_event(3),
        ];
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("1"),
            )))
            .with_history(Ok(events.clone()));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.history_events(), events.as_slice());
        let step = result.operations().first().expect("the step");
        assert_eq!(step.events(), events.get(1..2).expect("the step event"));
    }

    /// `CloudRunner` returns the events of every history page, in page
    /// order.
    #[tokio::test(start_paused = true)]
    async fn cloud_history_keeps_all_pages() {
        let events = vec![
            synthetic_event(1, EventType::ExecutionStarted, Some("execution"), 0),
            synthetic_event(2, EventType::StepStarted, Some("s"), 1),
            synthetic_event(3, EventType::StepSucceeded, Some("s"), 2),
            synthetic_event(4, EventType::InvocationCompleted, None, 3),
            execution_succeeded_event(5),
        ];
        let page = |range: std::ops::Range<usize>| events.get(range).expect("a page").to_vec();
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("1"),
            )))
            .with_history_pages(vec![
                Ok((page(0..2), Some("m1".to_owned()))),
                Ok((page(2..4), Some("m2".to_owned()))),
                Ok((page(4..5), None)),
            ]);
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.history_events(), events.as_slice());
        let markers: Vec<Option<String>> = api
            .history_calls()
            .into_iter()
            .map(|call| call.marker)
            .collect();
        assert_eq!(
            markers,
            [None, Some("m1".to_owned()), Some("m2".to_owned())]
        );
    }

    /// An `InvocationCompleted` event with details.
    fn invocation_completed_event(
        event_id: i32,
        start: i64,
        end: i64,
        request_id: &str,
        error: Option<&crate::WireError>,
    ) -> Event {
        Event::builder()
            .event_id(event_id)
            .event_type(EventType::InvocationCompleted)
            .invocation_completed_details(
                InvocationCompletedDetails::builder()
                    .start_timestamp(DateTime::from_secs(start))
                    .end_timestamp(DateTime::from_secs(end))
                    .request_id(request_id)
                    .set_error(error.map(event_error))
                    .build()
                    .expect("details"),
            )
            .build()
    }

    /// `CloudRunner` builds one record per `InvocationCompleted` event, in
    /// history order, and ignores the other events.
    #[tokio::test(start_paused = true)]
    async fn fold_history_builds_invocations() {
        let failure = crate::WireError::new(Some("Bad"), Some("no"));
        let events = vec![
            synthetic_event(1, EventType::ExecutionStarted, Some("execution"), 0),
            invocation_completed_event(2, 10, 11, "r1", None),
            invocation_completed_event(3, 20, 22, "r2", Some(&failure)),
            execution_succeeded_event(4),
        ];
        let api = FakeApi::new()
            .with_execution(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("1"),
            )))
            .with_history(Ok(events));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(&api, Vec::new())
            .await
            .expect("runner");
        let at = |secs: u64| Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs));
        let records: Vec<_> = result
            .invocations()
            .iter()
            .map(|r| {
                (
                    r.start_time(),
                    r.end_time(),
                    r.request_id().map(ToOwned::to_owned),
                    r.error().cloned(),
                )
            })
            .collect();
        assert_eq!(
            records,
            [
                (at(10), at(11), Some("r1".to_owned()), None),
                (at(20), at(22), Some("r2".to_owned()), Some(failure)),
            ]
        );
        assert_eq!(result.invocation_count(), 2);
    }

    /// An `InvocationCompleted` event without details gives a record with
    /// every field absent.
    #[test]
    fn invocation_without_details_has_no_fields() {
        let record = TestInvocation::from_event(&synthetic_event(
            1,
            EventType::InvocationCompleted,
            None,
            0,
        ));
        assert_eq!(record.start_time(), None);
        assert_eq!(record.end_time(), None);
        assert_eq!(record.request_id(), None);
        assert_eq!(record.error(), None);
    }

    /// One wait: a pending invocation, then a succeeded one, one second
    /// later.
    #[tokio::test]
    async fn invocations_record_pending_then_succeeded() {
        let result = LocalRunner::new()
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        let records = result.invocations();
        let ids: Vec<_> = records.iter().map(TestInvocation::request_id).collect();
        assert_eq!(
            ids,
            [Some("local-invocation-1"), Some("local-invocation-2")]
        );
        let first = records.first().expect("the first record");
        let second = records.get(1).expect("the second record");
        assert_eq!(
            second.start_time(),
            first
                .start_time()
                .map(|t| t + std::time::Duration::from_secs(1))
        );
        assert!(records.iter().all(|r| r.error().is_none()));
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(result.invocation_count(), records.len());
    }

    /// An invocation fault records its message and no error type.
    #[tokio::test]
    async fn invocations_record_retrying_fault() {
        let result = LocalRunner::new()
            .fail_checkpoints_after_retryable(1, 1)
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
                    ctx.step(|_| async { Ok(2_u32) }).name("b").await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error());
        let errors: Vec<_> = result
            .invocations()
            .iter()
            .filter_map(TestInvocation::error)
            .collect();
        assert_eq!(errors.len(), 1, "{errors:?}");
        let fault = errors.first().expect("the fault");
        assert!(
            fault
                .error_message()
                .is_some_and(|m| m.contains("injected retryable checkpoint failure")),
            "{fault:?}"
        );
        assert_eq!(fault.error_type(), None);
        assert_eq!(result.invocation_count(), result.invocations().len());
    }

    /// A `FAILED` invocation records the handler's error.
    #[tokio::test]
    async fn invocations_record_failed() {
        let result = LocalRunner::new()
            .run(
                |(): (), _ctx: DurableContext| async move { Err::<(), BoxError>("boom".into()) },
                (),
            )
            .await
            .expect("runner");
        let records = result.invocations();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records
                .first()
                .and_then(TestInvocation::error)
                .and_then(crate::WireError::error_message),
            Some("boom")
        );
    }

    /// A `FAILED` envelope's stack trace reaches both the
    /// `InvocationCompleted` event and the invocation record.
    #[tokio::test]
    async fn invocations_record_failed_stack_trace() {
        let result = LocalRunner::new()
            .run(
                |(): (), _ctx: DurableContext| async move {
                    Err::<(), BoxError>(error_with_stack_trace())
                },
                (),
            )
            .await
            .expect("runner");
        let frames = ["frame-1".to_owned(), "frame-2".to_owned()];
        let event_frames = result
            .history_events()
            .iter()
            .find(|e| e.event_type() == Some(&EventType::InvocationCompleted))
            .and_then(Event::invocation_completed_details)
            .and_then(InvocationCompletedDetails::error)
            .and_then(crate::lambda_types::EventError::payload)
            .map(|payload| payload.stack_trace().to_vec());
        assert_eq!(event_frames.as_deref(), Some(frames.as_slice()));
        let records = result.invocations();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records
                .first()
                .and_then(TestInvocation::error)
                .map(crate::WireError::stack_trace),
            Some(frames.as_slice())
        );
    }

    /// A run that reaches `max_invocations` with work pending has no
    /// execution end event.
    #[tokio::test]
    async fn max_invocations_run_has_no_execution_end_event() {
        let result = LocalRunner::new()
            .max_invocations(1)
            .run(
                |(): (), ctx: DurableContext| async move {
                    ctx.wait(std::time::Duration::from_secs(1)).await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_suspended());
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.contains("did not terminate within 1 invocations"),
            "{message}"
        );
        let types = event_types(result.history_events());
        assert!(
            !types.contains(&EventType::ExecutionSucceeded)
                && !types.contains(&EventType::ExecutionFailed),
            "{types:?}"
        );
        assert_eq!(types.first(), Some(&EventType::ExecutionStarted));
        assert_eq!(result.invocation_count(), 1);
    }

    /// The handler sees the local request ID of its invocation.
    #[tokio::test]
    async fn handler_sees_local_request_id() {
        let result = LocalRunner::new()
            .run(
                |(): (), ctx: DurableContext| async move {
                    Ok::<_, BoxError>(ctx.lambda_context().request_id.clone())
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(
            result.output().map(String::as_str),
            Some("local-invocation-1")
        );
    }

    /// A run that ends with work still pending has no execution end event.
    #[tokio::test]
    async fn suspended_run_has_no_execution_end_event() {
        let result = LocalRunner::new()
            .run(
                |(): (), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<u32>().await?;
                    cb.result().await?;
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_suspended());
        assert_eq!(
            event_types(result.history_events()),
            [
                EventType::ExecutionStarted,
                EventType::CallbackStarted,
                EventType::InvocationCompleted,
            ]
        );
        // A callback without a timeout or a heartbeat sends no options.
        let details = result
            .history_events()
            .get(1)
            .and_then(Event::callback_started_details)
            .expect("callback started details");
        assert_eq!(details.timeout(), None);
        assert_eq!(details.heartbeat_timeout(), None);
    }

    /// The events a direct checkpoint of `updates` records.
    async fn checkpoint_events(updates: Vec<OperationUpdate>) -> Vec<Event> {
        let backend = Backend::new(QueuedReplies::default(), None);
        backend
            .checkpoint("arn", "token", updates)
            .await
            .expect("checkpoint accepted");
        backend.events()
    }

    /// An `EXECUTION` update produces no event of its own.
    #[tokio::test]
    async fn execution_update_has_no_event() {
        let update = OperationUpdate::builder()
            .id("execution")
            .r#type(OperationType::Execution)
            .action(OperationAction::Succeed)
            .payload("1")
            .build()
            .expect("valid update");
        assert_eq!(checkpoint_events(vec![update]).await, []);
    }

    /// A WAIT `START` without wait options records a duration of 0.
    #[tokio::test]
    async fn wait_start_without_duration_records_zero() {
        let update = OperationUpdate::builder()
            .id("w")
            .r#type(OperationType::Wait)
            .action(OperationAction::Start)
            .build()
            .expect("valid update");
        let events = checkpoint_events(vec![update]).await;
        let event = events.first().expect("the event");
        let details = event.wait_started_details().expect("wait started details");
        assert_eq!(details.duration(), 0);
        assert_eq!(
            Some(details.scheduled_end_timestamp()),
            event.event_timestamp()
        );
    }

    /// A `CHAINED_INVOKE` `START` without options records the empty function
    /// name.
    #[tokio::test]
    async fn invoke_start_without_options_records_no_function() {
        let update = OperationUpdate::builder()
            .id("i")
            .r#type(OperationType::ChainedInvoke)
            .action(OperationAction::Start)
            .build()
            .expect("valid update");
        let events = checkpoint_events(vec![update]).await;
        let details = events
            .first()
            .and_then(Event::chained_invoke_started_details)
            .expect("chained invoke started details");
        assert_eq!(details.function_name(), "");
        assert_eq!(details.input(), None);
    }

    /// The owner of a row's event: the execution, an invocation, or the
    /// operation with a name.
    #[derive(Debug, Clone, Copy)]
    enum Owner {
        Execution,
        Invocation,
        Op(&'static str),
    }

    /// One event a row selects: the first event of `event_type`, its owner,
    /// and the expected detail fields given the owner's operation.
    struct EventCheck {
        event_type: EventType,
        owner: Owner,
        expected: fn(Option<&TestOperation>) -> serde_json::Value,
    }

    type RowFuture = Pin<Box<dyn Future<Output = (Vec<Event>, Vec<TestOperation>)>>>;

    /// One row of the table: its number, its setup, and its checks.
    struct EventRow {
        row: usize,
        setup: fn() -> RowFuture,
        checks: Vec<EventCheck>,
    }

    /// Renders an `EventInput` as JSON.
    fn input_json(input: Option<&EventInput>) -> serde_json::Value {
        input.map_or(
            serde_json::Value::Null,
            |i| serde_json::json!({ "payload": i.payload(), "truncated": i.truncated() }),
        )
    }

    /// Renders an `EventResult` as JSON.
    fn result_json(result: Option<&EventResult>) -> serde_json::Value {
        result.map_or(
            serde_json::Value::Null,
            |r| serde_json::json!({ "payload": r.payload(), "truncated": r.truncated() }),
        )
    }

    /// Renders an `EventError` as JSON.
    fn error_json(error: Option<&EventError>) -> serde_json::Value {
        error.map_or(serde_json::Value::Null, |e| {
            serde_json::json!({
                "payload": e.payload().map(|p| serde_json::json!({
                    "type": p.error_type(),
                    "message": p.error_message(),
                    "data": p.error_data(),
                    "stack": p.stack_trace(),
                })),
                "truncated": e.truncated(),
            })
        })
    }

    /// Renders `RetryDetails` as JSON.
    fn retry_json(retry: Option<&RetryDetails>) -> serde_json::Value {
        retry.map_or(serde_json::Value::Null, |r| {
            serde_json::json!({
                "current_attempt": r.current_attempt(),
                "next_attempt_delay_seconds": r.next_attempt_delay_seconds(),
            })
        })
    }

    /// The seconds from the event's timestamp to `t`.
    fn secs_after(event: &Event, t: &DateTime) -> i64 {
        t.secs() - event.event_timestamp().expect("a timestamp").secs()
    }

    /// Every field of the detail struct of the event's type, as JSON. A
    /// timestamp renders as the seconds after the event's timestamp.
    #[expect(clippy::too_many_lines)] // reason: one arm per event type
    fn details_json(event: &Event) -> serde_json::Value {
        use serde_json::json;
        match event.event_type().expect("an event type") {
            EventType::ExecutionStarted => {
                let d = event.execution_started_details().expect("details");
                {
                    json!({ "input": input_json(d.input()), "execution_timeout": d.execution_timeout() })
                }
            }
            EventType::ExecutionSucceeded => {
                let d = event.execution_succeeded_details().expect("details");
                json!({ "result": result_json(d.result()) })
            }
            EventType::ExecutionFailed => {
                let d = event.execution_failed_details().expect("details");
                json!({ "error": error_json(d.error()) })
            }
            EventType::StepStarted => {
                event.step_started_details().expect("details");
                json!({})
            }
            EventType::StepSucceeded => {
                let d = event.step_succeeded_details().expect("details");
                {
                    json!({ "result": result_json(d.result()), "retry_details": retry_json(d.retry_details()) })
                }
            }
            EventType::StepFailed => {
                let d = event.step_failed_details().expect("details");
                {
                    json!({ "error": error_json(d.error()), "retry_details": retry_json(d.retry_details()) })
                }
            }
            EventType::ContextStarted => {
                event.context_started_details().expect("details");
                json!({})
            }
            EventType::ContextSucceeded => {
                let d = event.context_succeeded_details().expect("details");
                json!({ "result": result_json(d.result()) })
            }
            EventType::ContextFailed => {
                let d = event.context_failed_details().expect("details");
                json!({ "error": error_json(d.error()) })
            }
            EventType::WaitStarted => {
                let d = event.wait_started_details().expect("details");
                {
                    json!({
                        "duration": d.duration(),
                        "scheduled_end_after": secs_after(event, d.scheduled_end_timestamp()),
                    })
                }
            }
            EventType::WaitSucceeded => {
                let d = event.wait_succeeded_details().expect("details");
                json!({ "duration": d.duration() })
            }
            EventType::WaitCancelled => {
                let d = event.wait_cancelled_details().expect("details");
                json!({ "error": error_json(d.error()) })
            }
            EventType::CallbackStarted => {
                let d = event.callback_started_details().expect("details");
                {
                    json!({
                        "callback_id": d.callback_id(),
                        "timeout": d.timeout(),
                        "heartbeat_timeout": d.heartbeat_timeout(),
                    })
                }
            }
            EventType::CallbackSucceeded => {
                let d = event.callback_succeeded_details().expect("details");
                json!({ "result": result_json(d.result()) })
            }
            EventType::CallbackFailed => {
                let d = event.callback_failed_details().expect("details");
                json!({ "error": error_json(d.error()) })
            }
            EventType::CallbackTimedOut => {
                let d = event.callback_timed_out_details().expect("details");
                json!({ "error": error_json(d.error()) })
            }
            EventType::ChainedInvokeStarted => {
                let d = event.chained_invoke_started_details().expect("details");
                {
                    json!({
                        "function_name": d.function_name(),
                        "tenant_id": d.tenant_id(),
                        "input": input_json(d.input()),
                        "executed_version": d.executed_version(),
                        "durable_execution_arn": d.durable_execution_arn(),
                    })
                }
            }
            EventType::ChainedInvokeSucceeded => {
                let d = event.chained_invoke_succeeded_details().expect("details");
                json!({ "result": result_json(d.result()) })
            }
            EventType::ChainedInvokeFailed => {
                let d = event.chained_invoke_failed_details().expect("details");
                json!({ "error": error_json(d.error()) })
            }
            EventType::InvocationCompleted => {
                let d = event.invocation_completed_details().expect("details");
                {
                    json!({
                        "start_after": secs_after(event, d.start_timestamp()),
                        "end_after": secs_after(event, d.end_timestamp()),
                        "request_id": d.request_id(),
                        "error": error_json(d.error()),
                    })
                }
            }
            // The local backend records no other type, so the comparison
            // with the expected fields fails.
            _ => serde_json::Value::Null,
        }
    }

    /// The number of `*_details()` getters of `event` that return a value.
    fn details_count(event: &Event) -> usize {
        [
            event.execution_started_details().is_some(),
            event.execution_succeeded_details().is_some(),
            event.execution_failed_details().is_some(),
            event.execution_timed_out_details().is_some(),
            event.execution_stopped_details().is_some(),
            event.context_started_details().is_some(),
            event.context_succeeded_details().is_some(),
            event.context_failed_details().is_some(),
            event.wait_started_details().is_some(),
            event.wait_succeeded_details().is_some(),
            event.wait_cancelled_details().is_some(),
            event.step_started_details().is_some(),
            event.step_succeeded_details().is_some(),
            event.step_failed_details().is_some(),
            event.chained_invoke_started_details().is_some(),
            event.chained_invoke_succeeded_details().is_some(),
            event.chained_invoke_failed_details().is_some(),
            event.chained_invoke_timed_out_details().is_some(),
            event.chained_invoke_stopped_details().is_some(),
            event.callback_started_details().is_some(),
            event.callback_succeeded_details().is_some(),
            event.callback_failed_details().is_some(),
            event.callback_timed_out_details().is_some(),
            event.invocation_completed_details().is_some(),
        ]
        .into_iter()
        .filter(|set| *set)
        .count()
    }

    /// The JSON of an error payload with these fields and no stack trace.
    fn error_payload(
        error_type: Option<&str>,
        message: Option<&str>,
        data: Option<&str>,
    ) -> serde_json::Value {
        serde_json::json!({
            "payload": { "type": error_type, "message": message, "data": data, "stack": [] },
            "truncated": null,
        })
    }

    /// The JSON of an `EventResult` or `EventInput` with `payload`.
    fn payload(payload: &str) -> serde_json::Value {
        serde_json::json!({ "payload": payload, "truncated": null })
    }

    /// The JSON of retry details.
    fn retry(current_attempt: i32, delay: Option<i32>) -> serde_json::Value {
        serde_json::json!({ "current_attempt": current_attempt, "next_attempt_delay_seconds": delay })
    }

    /// The message the SDK records for [`bad_no`]: the error chain, the
    /// type name and then the message.
    const BAD_NO_MESSAGE: &str = "Bad: no";

    /// The error `Bad` / `no`.
    fn bad_no() -> BoxError {
        crate::TypedError::with_type("Bad", "no").into()
    }

    /// A callback `cb` with a 60 s timeout and a 10 s heartbeat, run with
    /// `reply` queued.
    fn callback_row(reply: CallbackReply) -> RowFuture {
        Box::pin(async move {
            let result = LocalRunner::new()
                .callback(reply)
                .run(
                    |(): (), ctx: DurableContext| async move {
                        let cb = ctx
                            .create_callback::<u32>()
                            .name("cb")
                            .timeout(std::time::Duration::from_secs(60))
                            .heartbeat(std::time::Duration::from_secs(10))
                            .await?;
                        Ok::<_, BoxError>(cb.result().await?)
                    },
                    (),
                )
                .await
                .expect("runner");
            history_of(&result)
        })
    }

    /// A handler run once on a fresh runner, with event `()`.
    fn handler_row<O, F, Fut>(handler: F) -> RowFuture
    where
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    {
        Box::pin(async move {
            let result = LocalRunner::new().run(handler, ()).await.expect("runner");
            history_of(&result)
        })
    }

    /// The table of rule 5: one row per cause, with the detail fields of
    /// rule 4.
    #[expect(clippy::too_many_lines)] // reason: one entry per row of the table
    fn event_rows() -> Vec<EventRow> {
        use serde_json::json;
        vec![
            EventRow {
                row: 1,
                setup: || {
                    Box::pin(async {
                        let result = LocalRunner::new()
                            .run(
                                |_e: serde_json::Value, _ctx: DurableContext| async move {
                                    Ok::<_, BoxError>(1_u32)
                                },
                                json!({ "k": 1 }),
                            )
                            .await
                            .expect("runner");
                        history_of(&result)
                    })
                },
                checks: vec![
                    EventCheck {
                        event_type: EventType::ExecutionStarted,
                        owner: Owner::Execution,
                        expected: |_| json!({ "input": payload("{\"k\":1}"), "execution_timeout": 0 }),
                    },
                    EventCheck {
                        event_type: EventType::ExecutionSucceeded,
                        owner: Owner::Execution,
                        expected: |_| json!({ "result": payload("1") }),
                    },
                ],
            },
            EventRow {
                row: 2,
                setup: || {
                    handler_row(|(), ctx: DurableContext| async move {
                        Ok(ctx.step(|_| async { Ok(7_u32) }).name("s").await?)
                    })
                },
                checks: vec![
                    EventCheck {
                        event_type: EventType::StepStarted,
                        owner: Owner::Op("s"),
                        expected: |_| json!({}),
                    },
                    EventCheck {
                        event_type: EventType::StepSucceeded,
                        owner: Owner::Op("s"),
                        expected: |_| json!({ "result": payload("7"), "retry_details": retry(1, None) }),
                    },
                ],
            },
            EventRow {
                row: 3,
                setup: || {
                    handler_row(|(), ctx: DurableContext| async move {
                        ctx.step(|_| async { Err::<u32, BoxError>(bad_no()) })
                            .name("s")
                            .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                            .await?;
                        Ok(())
                    })
                },
                checks: vec![EventCheck {
                    event_type: EventType::StepFailed,
                    owner: Owner::Op("s"),
                    expected: |_| {
                        json!({
                            "error": error_payload(Some("Bad"), Some(BAD_NO_MESSAGE), None),
                            "retry_details": retry(1, None),
                        })
                    },
                }],
            },
            EventRow {
                row: 4,
                setup: || {
                    handler_row(|(), ctx: DurableContext| async move {
                        Ok(ctx
                            .step(|step| async move {
                                if step.attempt() == 1 {
                                    return Err::<u32, BoxError>(bad_no());
                                }
                                Ok(1)
                            })
                            .name("s")
                            .retry_strategy(|_err, _attempt| RetryDecision::Retry {
                                delay: std::time::Duration::from_secs(5),
                            })
                            .await?)
                    })
                },
                checks: vec![EventCheck {
                    event_type: EventType::StepFailed,
                    owner: Owner::Op("s"),
                    expected: |_| {
                        json!({
                            "error": error_payload(Some("Bad"), Some(BAD_NO_MESSAGE), None),
                            "retry_details": retry(1, Some(5)),
                        })
                    },
                }],
            },
            EventRow {
                row: 5,
                setup: || {
                    handler_row(|(), ctx: DurableContext| async move {
                        Ok(ctx
                            .run_in_child_context(|_child| async move { Ok(2_u32) })
                            .name("c")
                            .await?)
                    })
                },
                checks: vec![
                    EventCheck {
                        event_type: EventType::ContextStarted,
                        owner: Owner::Op("c"),
                        expected: |_| json!({}),
                    },
                    EventCheck {
                        event_type: EventType::ContextSucceeded,
                        owner: Owner::Op("c"),
                        expected: |_| json!({ "result": payload("2") }),
                    },
                ],
            },
            EventRow {
                row: 6,
                setup: || {
                    handler_row(|(), ctx: DurableContext| async move {
                        Ok(ctx
                            .run_in_child_context(
                                |_child| async move { Err::<u32, BoxError>(bad_no()) },
                            )
                            .name("c")
                            .await?)
                    })
                },
                checks: vec![EventCheck {
                    event_type: EventType::ContextFailed,
                    owner: Owner::Op("c"),
                    expected: |_| json!({ "error": error_payload(Some("Bad"), Some(BAD_NO_MESSAGE), None) }),
                }],
            },
            EventRow {
                row: 7,
                setup: || {
                    handler_row(|(), ctx: DurableContext| async move {
                        Ok(ctx
                            .wait(std::time::Duration::from_secs(3))
                            .name("w")
                            .await?)
                    })
                },
                checks: vec![
                    EventCheck {
                        event_type: EventType::WaitStarted,
                        owner: Owner::Op("w"),
                        expected: |_| json!({ "duration": 3, "scheduled_end_after": 3 }),
                    },
                    EventCheck {
                        event_type: EventType::WaitSucceeded,
                        owner: Owner::Op("w"),
                        expected: |_| json!({ "duration": 3 }),
                    },
                ],
            },
            EventRow {
                row: 8,
                setup: || {
                    Box::pin(async {
                        let backend = Backend::new(QueuedReplies::default(), None);
                        let start = OperationUpdate::builder()
                            .id("w")
                            .name("w")
                            .r#type(OperationType::Wait)
                            .action(OperationAction::Start)
                            .wait_options(
                                aws_sdk_lambda::types::WaitOptions::builder()
                                    .wait_seconds(3)
                                    .build(),
                            )
                            .build()
                            .expect("valid update");
                        let cancel = OperationUpdate::builder()
                            .id("w")
                            .name("w")
                            .r#type(OperationType::Wait)
                            .action(OperationAction::Cancel)
                            .build()
                            .expect("valid update");
                        for update in [start, cancel] {
                            backend
                                .checkpoint("arn", "token", vec![update])
                                .await
                                .expect("checkpoint accepted");
                        }
                        (backend.events(), backend.snapshot_operations())
                    })
                },
                checks: vec![EventCheck {
                    event_type: EventType::WaitCancelled,
                    owner: Owner::Op("w"),
                    expected: |_| json!({ "error": null }),
                }],
            },
            EventRow {
                row: 9,
                setup: || callback_row(CallbackReply::success(&4_u32).expect("encodes")),
                checks: vec![
                    EventCheck {
                        event_type: EventType::CallbackStarted,
                        owner: Owner::Op("cb"),
                        expected: |op| {
                            let callback_id = match op.expect("the callback").details() {
                                TestOperationDetails::Callback(c) => {
                                    c.callback_id().map(ToOwned::to_owned)
                                }
                                _ => None,
                            };
                            json!({ "callback_id": callback_id, "timeout": 60, "heartbeat_timeout": 10 })
                        },
                    },
                    EventCheck {
                        event_type: EventType::CallbackSucceeded,
                        owner: Owner::Op("cb"),
                        expected: |_| json!({ "result": payload("4") }),
                    },
                ],
            },
            EventRow {
                row: 10,
                setup: || callback_row(CallbackReply::raw("\"r\"")),
                checks: vec![EventCheck {
                    event_type: EventType::CallbackSucceeded,
                    owner: Owner::Op("cb"),
                    expected: |_| json!({ "result": payload("\"r\"") }),
                }],
            },
            EventRow {
                row: 11,
                setup: || {
                    callback_row(CallbackReply::failure(
                        CallbackFailure::new("Declined", "no").with_error_data("d"),
                    ))
                },
                checks: vec![EventCheck {
                    event_type: EventType::CallbackFailed,
                    owner: Owner::Op("cb"),
                    expected: |_| json!({ "error": error_payload(Some("Declined"), Some("no"), Some("d")) }),
                }],
            },
            EventRow {
                row: 12,
                setup: || callback_row(CallbackReply::timeout()),
                checks: vec![EventCheck {
                    event_type: EventType::CallbackTimedOut,
                    owner: Owner::Op("cb"),
                    expected: |_| json!({ "error": error_payload(None, Some("Callback timed out"), None) }),
                }],
            },
            EventRow {
                row: 13,
                setup: || callback_row(CallbackReply::heartbeat_timeout()),
                checks: vec![EventCheck {
                    event_type: EventType::CallbackTimedOut,
                    owner: Owner::Op("cb"),
                    expected: |_| {
                        json!({
                            "error": error_payload(
                                Some("Callback.Heartbeat"),
                                Some("Callback timed out on heartbeat"),
                                None,
                            ),
                        })
                    },
                }],
            },
            EventRow {
                row: 14,
                setup: || {
                    Box::pin(async {
                        let result = LocalRunner::new()
                            .register_function("target", |_x: i32| async move {
                                Ok::<_, BoxError>(6_i32)
                            })
                            .run(
                                |(): (), ctx: DurableContext| async move {
                                    Ok(ctx.invoke::<i32, _>("target", 5).name("call").await?)
                                },
                                (),
                            )
                            .await
                            .expect("runner");
                        history_of(&result)
                    })
                },
                checks: vec![
                    EventCheck {
                        event_type: EventType::ChainedInvokeStarted,
                        owner: Owner::Op("call"),
                        expected: |_| {
                            json!({
                                "function_name": "target",
                                "tenant_id": null,
                                "input": payload("5"),
                                "executed_version": null,
                                "durable_execution_arn": null,
                            })
                        },
                    },
                    EventCheck {
                        event_type: EventType::ChainedInvokeSucceeded,
                        owner: Owner::Op("call"),
                        expected: |_| json!({ "result": payload("6") }),
                    },
                ],
            },
            EventRow {
                row: 15,
                setup: || {
                    Box::pin(async {
                        let result = LocalRunner::new()
                            .register_function("target", |_x: i32| async move {
                                Err::<i32, BoxError>(bad_no())
                            })
                            .run(
                                |(): (), ctx: DurableContext| async move {
                                    Ok(ctx.invoke::<i32, _>("target", 5).name("call").await?)
                                },
                                (),
                            )
                            .await
                            .expect("runner");
                        history_of(&result)
                    })
                },
                checks: vec![EventCheck {
                    event_type: EventType::ChainedInvokeFailed,
                    owner: Owner::Op("call"),
                    expected: |_| json!({ "error": error_payload(Some("Bad"), Some(BAD_NO_MESSAGE), None) }),
                }],
            },
            EventRow {
                row: 16,
                setup: || {
                    handler_row(|(), _ctx: DurableContext| async move {
                        Err::<(), BoxError>(
                            crate::TypedError::with_type("Bad", "no")
                                .with_error_data("d")
                                .into(),
                        )
                    })
                },
                checks: vec![
                    EventCheck {
                        event_type: EventType::InvocationCompleted,
                        owner: Owner::Invocation,
                        expected: |_| {
                            json!({
                                "start_after": 0,
                                "end_after": 0,
                                "request_id": "local-invocation-1",
                                "error": error_payload(Some("Bad"), Some(BAD_NO_MESSAGE), Some("d")),
                            })
                        },
                    },
                    EventCheck {
                        event_type: EventType::ExecutionFailed,
                        owner: Owner::Execution,
                        expected: |_| json!({ "error": error_payload(Some("Bad"), Some(BAD_NO_MESSAGE), Some("d")) }),
                    },
                ],
            },
        ]
    }

    /// Each cause of rule 5 records its event, with the operation's
    /// identity and exactly the detail fields of rule 4.
    #[tokio::test]
    async fn local_events_cover_every_cause_and_detail() {
        let rows = event_rows();
        assert_eq!(rows.len(), 16);
        for EventRow { row, setup, checks } in rows {
            let (events, operations) = setup().await;
            for check in checks {
                let context = format!("row {row}, {:?}", check.event_type);
                let event = events
                    .iter()
                    .find(|e| e.event_type() == Some(&check.event_type))
                    .expect("the row records the event");
                let op = match check.owner {
                    Owner::Op(name) => Some(
                        operations
                            .iter()
                            .find(|op| op.name() == Some(name))
                            .expect("the row records the operation"),
                    ),
                    Owner::Execution | Owner::Invocation => None,
                };
                let identity = (
                    event.id(),
                    event.name(),
                    event.sub_type(),
                    event.parent_id(),
                );
                let expected_identity = match (check.owner, op) {
                    (Owner::Op(_), Some(op)) => {
                        (Some(op.id()), op.name(), op.sub_type(), op.parent_id())
                    }
                    (Owner::Execution, _) => (Some(EXECUTION_OPERATION_ID), None, None, None),
                    _ => (None, None, None, None),
                };
                assert_eq!(identity, expected_identity, "{context}");
                assert_eq!(details_json(event), (check.expected)(op), "{context}");
                assert_eq!(details_count(event), 1, "{context}");
            }
        }
    }
}
