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
//! - Callback replies are queued before the run, or sent while it runs
//!   through [`OperationHandle::send_callback`]. The runner runs no
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
//! # Reacting to a run in progress
//!
//! [`LocalRunner::run`] and [`CloudRunner::run`] return a [`RunHandle`].
//! Before awaiting it, a test can take an [`OperationHandle`] with
//! [`RunHandle::operation`], wait until the operation reaches a
//! [`WaitingStatus`] with [`OperationHandle::wait_for_data`], read it, and
//! answer its callback with [`OperationHandle::send_callback`]. Under
//! [`LocalRunner`] it can also [`pause`](RunHandle::pause) and
//! [`resume`](RunHandle::resume) the run.
//!
//! The run moves on only while the test awaits a runner future: the
//! `RunHandle`, `wait_for_data`, or `pause`. So after `wait_for_data`
//! returns, the test can read the operation and reply before the run moves.
//! Await the operation controls first and the `RunHandle` last, as JS tests
//! do. While the `RunHandle` itself is awaited, the run does not wait for
//! other test code.
//!
//! Under [`LocalRunner`], a checkpoint that the handler sends while no
//! runner future waits is answered only once one waits, and no invocation
//! starts and the clock does not move meanwhile. [`CloudRunner`] cannot hold
//! the deployed function. It supports `wait_for_data` and the callback
//! controls only with [`InvocationType::Event`], and reads the whole history
//! every [`poll_interval`](CloudRunner::poll_interval). It makes up to three
//! attempts per `GetDurableExecution` call and per history page request, as
//! JS does.
//!
//! This test pauses a run while a step runs and while it waits for a
//! callback:
//!
//! ```
//! use aws_durable_execution_sdk as durable;
//! use durable::test_util::{CallbackReply, LocalRunner, WaitingStatus};
//! use std::time::Duration;
//!
//! async fn handler(
//!     event: serde_json::Value,
//!     ctx: durable::DurableContext,
//! ) -> Result<serde_json::Value, durable::BoxError> {
//!     let order_id = event
//!         .get("orderId")
//!         .and_then(serde_json::Value::as_str)
//!         .unwrap_or("order-1")
//!         .to_owned();
//!     let id = order_id.clone();
//!     // Slow enough that a test can pause while it runs. The pause takes
//!     // effect when this step checkpoints its result: that checkpoint is
//!     // kept, but the invocation stops there.
//!     let reservation = ctx
//!         .step(move |_| async move {
//!             tokio::time::sleep(Duration::from_millis(200)).await;
//!             Ok(format!("reserved-{id}"))
//!         })
//!         .name("reserve-stock")
//!         .await?;
//!     let approval = ctx
//!         .wait_for_callback::<String>(|_step_ctx, _callback_id| async {
//!             // In a real workflow, the callback ID would be sent to an
//!             // approver here.
//!             Ok(())
//!         })
//!         .name("manager-approval")
//!         .await?;
//!     let shipment = ctx
//!         .step(move |_| async move { Ok(format!("shipped-{order_id}")) })
//!         .name("ship-order")
//!         .await?;
//!     Ok(serde_json::json!({
//!         "reservation": reservation,
//!         "approval": approval,
//!         "shipment": shipment,
//!     }))
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), durable::BoxError> {
//! // Pause while "reserve-stock" runs. `pause` returns once the invocation
//! // stopped, which is when the step checkpoints its result.
//! let mut run = LocalRunner::new().run(handler, serde_json::json!({"orderId": "order-7"}));
//! let reserve_stock = run.operation("reserve-stock");
//! reserve_stock.wait_for_data(WaitingStatus::Started).await?;
//! run.pause().await?;
//!
//! // The step's result was kept, but the invocation stopped there: the
//! // approval was never requested, and the run does not finish while
//! // paused.
//! assert!(reserve_stock.snapshot().is_some_and(|op| op.succeeded()));
//! assert!(run.operation("manager-approval").snapshot().is_none());
//! assert!(tokio::time::timeout(Duration::from_millis(50), &mut run).await.is_err());
//!
//! // Resuming starts a new invocation, which replays "reserve-stock" and
//! // carries on to the approval.
//! run.resume().await?;
//! let approval = run.operation("manager-approval");
//! approval.wait_for_data(WaitingStatus::Submitted).await?;
//! approval.send_callback(CallbackReply::success(&"approved")?).await?;
//!
//! let result = run.await?;
//! assert_eq!(
//!     result.output(),
//!     Some(&serde_json::json!({
//!         "reservation": "reserved-order-7",
//!         "approval": "approved",
//!         "shipment": "shipped-order-7",
//!     }))
//! );
//!
//! // A pause while the run waits for its approval returns at once. The
//! // reply can still be sent, but the invocation it would start waits for
//! // `resume`.
//! let mut run = LocalRunner::new().run(handler, serde_json::json!({"orderId": "order-8"}));
//! let approval = run.operation("manager-approval");
//! approval.wait_for_data(WaitingStatus::Submitted).await?;
//! run.pause().await?;
//! approval.send_callback(CallbackReply::success(&"approved")?).await?;
//! assert!(tokio::time::timeout(Duration::from_millis(50), &mut run).await.is_err());
//! assert!(run.operation("ship-order").snapshot().is_none());
//!
//! run.resume().await?;
//! let result = run.await?;
//! assert_eq!(
//!     result.output(),
//!     Some(&serde_json::json!({
//!         "reservation": "reserved-order-8",
//!         "approval": "approved",
//!         "shipment": "shipped-order-8",
//!     }))
//! );
//! # Ok(())
//! # }
//! ```
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

mod live;

use self::live::{
    CloudControl, CloudShared, Engine, LiveState, LocalPhase, RunCore, RunState, Selector,
    callback_target, wake,
};
pub use self::live::{OperationHandle, RunHandle, WaitingStatus};
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
    kind: RunnerKind,
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
    fn started(id: String, kind: RunnerKind) -> Self {
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
        self.kind.operation_kind()
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

    /// The [`result`](Self::result) payload decoded as JSON, the format
    /// [`JsonSerdes`](crate::JsonSerdes) writes.
    ///
    /// Returns `Ok(None)` when the operation stored no result, which
    /// includes a [`CloudRunner`] result that recorded history withheld
    /// because it was oversized, and the decoding error when the payload is
    /// not JSON of `T`.
    ///
    /// # Errors
    ///
    /// Returns the [`serde_json::Error`] of a payload that does not decode
    /// into `T`.
    pub fn result_json<T: DeserializeOwned>(&self) -> Result<Option<T>, serde_json::Error> {
        self.result.as_deref().map(serde_json::from_str).transpose()
    }

    /// Returns `true` for the child context of a `wait_for_callback`: an
    /// [`OperationKind::Context`] whose [`sub_type`](Self::sub_type) is
    /// `WaitForCallback`.
    pub fn is_wait_for_callback(&self) -> bool {
        self.kind == RunnerKind::Context
            && self.sub_type.as_deref() == Some(crate::callback::WFCB_SUB_TYPE)
    }

    /// The recorded duration of a wait, in whole seconds.
    ///
    /// `None` for an operation of any other kind, and for a wait that
    /// recorded no duration.
    pub fn wait_duration(&self) -> Option<std::time::Duration> {
        match self.details() {
            TestOperationDetails::Wait(wait) => wait.duration(),
            _ => None,
        }
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
            RunnerKind::Context => TestOperationDetails::Context(&self.context),
            RunnerKind::Step => TestOperationDetails::Step(&self.step),
            RunnerKind::Wait => TestOperationDetails::Wait(&self.wait),
            RunnerKind::Callback => TestOperationDetails::Callback(&self.callback),
            RunnerKind::ChainedInvoke => TestOperationDetails::ChainedInvoke,
        }
    }
}

/// The kind of an operation that a runner records: every
/// [`OperationKind`] except [`OperationKind::Execution`], which no runner
/// stores as an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunnerKind {
    Context,
    Step,
    Wait,
    Callback,
    ChainedInvoke,
}

impl RunnerKind {
    /// The runner kind of `kind`, or `None` for
    /// [`OperationKind::Execution`].
    fn from_operation_kind(kind: OperationKind) -> Option<Self> {
        match kind {
            OperationKind::Execution => None,
            OperationKind::Context => Some(Self::Context),
            OperationKind::Step => Some(Self::Step),
            OperationKind::Wait => Some(Self::Wait),
            OperationKind::Callback => Some(Self::Callback),
            OperationKind::ChainedInvoke => Some(Self::ChainedInvoke),
        }
    }

    /// The public kind of this runner kind.
    pub(crate) fn operation_kind(self) -> OperationKind {
        match self {
            Self::Context => OperationKind::Context,
            Self::Step => OperationKind::Step,
            Self::Wait => OperationKind::Wait,
            Self::Callback => OperationKind::Callback,
            Self::ChainedInvoke => OperationKind::ChainedInvoke,
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
///
/// No runner returns the execution itself as a [`TestOperation`], so the
/// enum has no variant for it:
///
/// ```rust,compile_fail
/// use aws_durable_execution_sdk::test_util::TestOperationDetails;
///
/// fn is_execution(details: TestOperationDetails<'_>) -> bool {
///     matches!(details, TestOperationDetails::Execution)
/// }
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum TestOperationDetails<'a> {
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
    /// A step retry and a `wait_for_condition` check that continued both
    /// set it. Under [`LocalRunner`] it is the virtual time the latest retry
    /// delay ended. Under [`CloudRunner`] it is the timestamp of the latest
    /// step event that carries a retry delay, plus that delay: a
    /// `StepFailed` for a step retry, or a `StepSucceeded` for a check that
    /// continued. The two runners report the same value.
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
    /// The runner does not support the named control, for example
    /// [`RunHandle::pause`] under [`CloudRunner`].
    Unsupported(&'static str),
    /// No operation of the run matches the [`OperationHandle`].
    OperationNotFound,
    /// The execution ended before the operation reached the status
    /// [`OperationHandle::wait_for_data`] waited for.
    OperationNotReached(WaitingStatus),
    /// The operation is neither a callback nor a `wait_for_callback`
    /// context.
    NotACallback,
    /// The callback is not waiting for a reply.
    CallbackNotPending,
    /// The callback has no heartbeat timeout.
    NoHeartbeatTimeout,
    /// The named control needs a run in progress, and the run already
    /// finished.
    RunFinished(&'static str),
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
            Self::Unsupported(what) => write!(
                f,
                "durable: test runner: {what} is not supported by this runner"
            ),
            Self::OperationNotFound => {
                f.write_str("durable: test runner: no operation matches this handle")
            }
            Self::OperationNotReached(status) => write!(
                f,
                "durable: test runner: the operation did not reach {status:?} before the \
                 execution ended"
            ),
            Self::NotACallback => f.write_str(
                "durable: test runner: the operation is not a callback or a wait_for_callback \
                 context",
            ),
            Self::CallbackNotPending => {
                f.write_str("durable: test runner: the callback is not waiting for a reply")
            }
            Self::NoHeartbeatTimeout => {
                f.write_str("durable: test runner: the callback has no heartbeat timeout")
            }
            Self::RunFinished(what) => write!(
                f,
                "durable: test runner: {what} needs a run in progress; call it before awaiting \
                 the RunHandle"
            ),
        }
    }
}

impl std::error::Error for TestRunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EventSerialization(e) | Self::OutputDeserialization(e) => Some(e.as_ref()),
            Self::Aws(e) => Some(e.as_ref()),
            Self::UnexpectedEnvelope(_)
            | Self::NoExecutionArn
            | Self::Unsupported(_)
            | Self::OperationNotFound
            | Self::OperationNotReached(_)
            | Self::NotACallback
            | Self::CallbackNotPending
            | Self::NoHeartbeatTimeout
            | Self::RunFinished(_) => None,
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

    /// The service `ErrorObject` of this failure, with each field it sets.
    fn error_object(&self) -> ErrorObject {
        let mut builder = ErrorObject::builder()
            .set_error_type(self.error_type.clone())
            .set_error_message(self.error_message.clone())
            .set_error_data(self.error_data.clone());
        for frame in &self.stack_trace {
            builder = builder.stack_trace(frame);
        }
        builder.build()
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
///                 .wait_for_callback::<u32>(|_step_ctx, _callback_id| async { Ok(()) })
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
    /// fault; the runner re-invokes) rather than non-retryable (execution
    /// failure, with no further checkpoint).
    checkpoint_failures_retryable: bool,
    /// How long the backend waits before it applies and answers each
    /// checkpoint request (see
    /// [`checkpoint_latency`](Self::checkpoint_latency)). Zero by default.
    checkpoint_latency: std::time::Duration,
    /// The number of checkpoint requests answered with a token before the
    /// one answered without (see
    /// [`withdraw_checkpoints_after`](Self::withdraw_checkpoints_after)).
    withdraw_after: Option<usize>,
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
            withdraw_after: None,
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
    /// A non-retryable checkpoint failure never reaches the handler: the
    /// SDK sends no further checkpoint and fails the execution. Use this
    /// to test that model, and to assert no record-transition lifecycle
    /// event is emitted for the rejected write (see
    /// [`crate::observability`]).
    ///
    /// It is [`fail_checkpoints_after(0, count)`](Self::fail_checkpoints_after).
    ///
    /// One checkpoint fault applies at a time: this call replaces the fault
    /// that an earlier `fail_next_checkpoints`, `fail_checkpoints_after` or
    /// `fail_checkpoints_after_retryable` call set.
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
    pub fn fail_next_checkpoints(self, count: usize) -> Self {
        self.fail_checkpoints_after(0, count)
    }

    /// Like [`fail_next_checkpoints`](Self::fail_next_checkpoints), but
    /// lets the first `skip` checkpoint calls through before rejecting
    /// the following `count` non-retryably.
    ///
    /// Use it to reject a specific write: a step's live path writes START
    /// first and its outcome second, so `fail_checkpoints_after(1, 1)`
    /// rejects exactly the step's outcome write, while the `START` before
    /// it persists.
    ///
    /// One checkpoint fault applies at a time: this call replaces the fault
    /// that an earlier `fail_next_checkpoints`, `fail_checkpoints_after` or
    /// `fail_checkpoints_after_retryable` call set.
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
    /// A retryable checkpoint failure fails the invocation, and the SDK
    /// sends no further checkpoint in it; the runner then re-invokes,
    /// exactly as the durable service does, and replay resumes from the
    /// recorded state.
    ///
    /// One checkpoint fault applies at a time: this call replaces the fault
    /// that an earlier `fail_next_checkpoints`, `fail_checkpoints_after` or
    /// `fail_checkpoints_after_retryable` call set.
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

    /// Answers checkpoint request number `n + 1` of the execution without a
    /// token, once.
    ///
    /// The count is 1-based over the checkpoint requests the execution
    /// receives, across invocations. The backend applies that request and
    /// returns no `CheckpointToken`, as the service does when it accepts no
    /// further checkpoints from an invocation. The SDK then sends nothing
    /// more and answers `PENDING`. The runner invokes the function again at
    /// once, as after an invocation fault, and the next invocation replays
    /// the accepted updates.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().withdraw_checkpoints_after(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn withdraw_checkpoints_after(mut self, n: usize) -> Self {
        self.withdraw_after = Some(n);
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
    ///             let doubled = ctx.invoke::<i32>("double", n).await?;
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
    ///             let out = ctx.invoke::<u32>("workflow", n).await?;
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
        nested.withdraw_after = defaults.withdraw_after;
        nested
    }

    /// Starts a run of `handler` with `event`, and returns its
    /// [`RunHandle`].
    ///
    /// The run starts nothing until the test polls one of its futures.
    /// Awaiting the handle drives `handler` to a terminal outcome. Before
    /// that, the test can wait for an operation and answer its callback
    /// through [`RunHandle::operation`], and pause the run with
    /// [`RunHandle::pause`] (see "Reacting to a run in progress" in the
    /// module docs).
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
    /// Awaiting the handle returns these runner errors, the run failures:
    ///
    /// - [`TestRunnerError::EventSerialization`] when `event` does not
    ///   serialize.
    /// - [`TestRunnerError::OutputDeserialization`] when the output of a
    ///   succeeded execution does not decode into `O`.
    /// - [`TestRunnerError::UnexpectedEnvelope`] when a response envelope
    ///   has a status other than `SUCCEEDED`, `FAILED` and `PENDING`.
    ///
    /// A [`wait_for_data`](OperationHandle::wait_for_data) pending when the
    /// run fails returns the run failure. One first polled after the run
    /// failed returns the stored operation when it already reached the
    /// requested status, and the run failure otherwise.
    /// [`send_callback`](OperationHandle::send_callback),
    /// [`send_callback_heartbeat`](OperationHandle::send_callback_heartbeat),
    /// [`pause`](RunHandle::pause) and [`resume`](RunHandle::resume) return
    /// the run failure.
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
    pub fn run<E, O, F, Fut>(&self, handler: F, event: E) -> RunHandle<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        self.run_configured(handler, event, |_backend| {})
    }

    /// [`run`](Self::run), with `configure` applied to the run's backend
    /// before the first invocation. Internal seam: a unit test plans the
    /// backend's checkpoint answers for a run it controls through the
    /// returned handle.
    #[expect(clippy::needless_pass_by_value)] // reason: the run takes the event, as the service delivers it
    fn run_configured<E, O, F, Fut>(
        &self,
        handler: F,
        event: E,
        configure: impl FnOnce(&Backend),
    ) -> RunHandle<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let live = Arc::new(LiveState::new(BackendState::new(
            self.callback_replies.clone(),
        )));
        let backend = Arc::new(self.backend_on(StateCell::Live(Arc::clone(&live))));
        configure(&backend);
        let engine = Engine::Local;
        // Serialize the event once; the envelope re-delivers it per
        // invocation.
        let event_json = match serde_json::to_string(&event) {
            Ok(json) => json,
            Err(e) => {
                return RunHandle::from_shared(RunCore::failed(
                    live,
                    engine,
                    TestRunnerError::EventSerialization(Arc::new(e)),
                ));
            }
        };
        let runner = self.clone();
        let driver = async move {
            let result: TestResult<O> = runner
                .drive(backend, handler, event_json)
                .await?
                .into_test_result()?;
            Ok(Box::new(result) as Box<dyn std::any::Any + Send>)
        };
        RunHandle::from_shared(RunCore::new(live, engine, Box::pin(driver)))
    }

    /// A fresh backend configured from this runner's settings: queued
    /// callback replies, checkpoint page size, registered function names,
    /// injected checkpoint failures and checkpoint latency.
    fn new_backend(&self) -> Backend {
        self.backend_on(StateCell::own(BackendState::new(
            self.callback_replies.clone(),
        )))
    }

    /// [`new_backend`](Self::new_backend) with its records in `cell`.
    fn backend_on(&self, cell: StateCell) -> Backend {
        let mut backend = Backend::on(cell, self.checkpoint_page_size);
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
        backend.withdraw_at = self.withdraw_after.map(|n| n.saturating_add(1));
        backend
    }

    /// The invoke → parse → advance loop over an externally supplied
    /// backend, with the event as JSON. Internal seam: unit tests inject a
    /// backend they retain a handle to, so they can assert on
    /// transport-level facts (e.g. how many `get_state` fetches pagination
    /// forced).
    #[cfg(test)]
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
                backend.close();
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

            // The hold rule: an invocation starts only while a runner
            // future waits and the run is not paused. The check and the
            // start of the invocation are one transition on the run state, so
            // a `pause` either sees the invocation in progress or keeps it
            // from starting.
            backend.admit_invocation().await;
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
            let response = service(lambda_event).await;
            // An accepted `EXECUTION` update ends the execution, as on the
            // service, even when the invocation then failed.
            let checkpointed = backend.execution_result();
            // A terminal envelope ends the run, and so does an accepted
            // `EXECUTION` update when the invocation then failed: a reply
            // after this point is refused, in the same transition that ends
            // the invocation.
            let closing = match &response {
                Ok(envelope) => {
                    envelope.0.get("Status").and_then(serde_json::Value::as_str) != Some("PENDING")
                }
                Err(_) => checkpointed.is_some(),
            };
            let response = match response {
                Ok(envelope) => envelope.0,
                Err(e) => {
                    // The invocation itself failed (a Lambda runtime
                    // error), e.g. a retryable checkpoint failure that
                    // exhausted transport retries (issue #43), or a
                    // bootstrap failure. The durable service re-invokes
                    // on an invocation fault, so the runner does too;
                    // `max_invocations` bounds a persistent fault.
                    let fault = e.to_string();
                    backend.finish_invocation(
                        closing,
                        started_at,
                        &context.request_id,
                        Some(&crate::WireError::new(None::<String>, Some(fault.as_str()))),
                    );
                    if checkpointed.is_some() {
                        let end = RawEnd::Succeeded(checkpointed);
                        backend.record_execution_end(&end);
                        return Ok(RawRun {
                            end,
                            operations: backend.snapshot_operations(),
                            invocations,
                            events: backend.events(),
                        });
                    }
                    last_invocation_fault = Some(fault);
                    continue;
                }
            };

            let status = response.get("Status").and_then(serde_json::Value::as_str);
            backend.finish_invocation(
                closing,
                started_at,
                &context.request_id,
                (status == Some("FAILED"))
                    .then(|| envelope_wire_error(&response))
                    .as_ref(),
            );
            let end = match status {
                Some("SUCCEEDED") => RawEnd::Succeeded(
                    match response.get("Result").and_then(serde_json::Value::as_str) {
                        // The result is too large for the envelope: the SDK
                        // checkpointed it as the `EXECUTION` update.
                        Some("") => Some(checkpointed.unwrap_or_default()),
                        result => result.map(ToOwned::to_owned),
                    },
                ),
                Some("FAILED") => RawEnd::Failed(envelope_wire_error(&response)),
                Some("PENDING") => {
                    // A withdrawn checkpoint token ends the invocation as an
                    // invocation fault does: the next one starts at once.
                    let withdrawn = backend.take_withdrawal();
                    if self.between_invocations(&backend, &nested, withdrawn).await {
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

    /// Decides, after a `PENDING` invocation, whether the next invocation
    /// starts. Returns `false` when the execution is stuck: nothing can
    /// advance it.
    ///
    /// `start` is set when the invocation's checkpoint token was withdrawn.
    /// A callback reply sent while the run was in progress also starts the
    /// next invocation. Otherwise pending invokes run before the clock moves,
    /// and then the backend advances (timers, retries, queued callback
    /// replies).
    ///
    /// While the run is paused, the backend keeps advancing, and the
    /// invocation those completions would start waits for `resume`. Each
    /// decision is one transition of [`Backend::between_step`] on the run
    /// state, so the hold rule, a live reply and the clock advance are
    /// decided together.
    async fn between_invocations(&self, backend: &Backend, nested: &Self, mut start: bool) -> bool {
        let mut invokes_tried = false;
        loop {
            match backend.between_step(&mut start, invokes_tried).await {
                BetweenStep::Start => return true,
                BetweenStep::Stuck => return false,
                BetweenStep::RunInvokes => {
                    invokes_tried = true;
                    // Pending invokes run before the clock moves. When one
                    // of them changed its record, the next invocation starts
                    // without an advance.
                    if self.run_pending_invokes(backend, nested).await {
                        return true;
                    }
                }
            }
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

/// One decision of [`LocalRunner::between_invocations`] (see
/// [`between_decision`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BetweenStep {
    /// Start the next invocation.
    Start,
    /// Nothing can advance the execution: the run ends suspended.
    Stuck,
    /// Run the pending invokes, then decide again.
    RunInvokes,
}

/// What one [`between_decision`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decided {
    /// The step [`LocalRunner::between_invocations`] takes.
    Step(BetweenStep),
    /// A paused run advanced: decide again at once.
    Again,
    /// Wait for the run state to change: no runner future waits, or the
    /// run is paused and nothing can advance.
    Wait,
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

/// The Lambda calls [`CloudRunner::run`] and its live controls make. They
/// call only this trait, so a unit test can supply a fake that returns
/// scripted outputs and errors.
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

    /// Sends `SendDurableExecutionCallbackSuccess` with `result` as the
    /// payload.
    fn send_callback_success(
        &self,
        callback_id: &str,
        result: Vec<u8>,
    ) -> impl Future<Output = Result<(), BoxError>> + Send;

    /// Sends `SendDurableExecutionCallbackFailure` with `error`.
    fn send_callback_failure(
        &self,
        callback_id: &str,
        error: ErrorObject,
    ) -> impl Future<Output = Result<(), BoxError>> + Send;

    /// Sends `SendDurableExecutionCallbackHeartbeat`.
    fn send_callback_heartbeat(
        &self,
        callback_id: &str,
    ) -> impl Future<Output = Result<(), BoxError>> + Send;
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

    async fn send_callback_success(
        &self,
        callback_id: &str,
        result: Vec<u8>,
    ) -> Result<(), BoxError> {
        self.send_durable_execution_callback_success()
            .callback_id(callback_id)
            .result(Blob::new(result))
            .send()
            .await?;
        Ok(())
    }

    async fn send_callback_failure(
        &self,
        callback_id: &str,
        error: ErrorObject,
    ) -> Result<(), BoxError> {
        self.send_durable_execution_callback_failure()
            .callback_id(callback_id)
            .error(error)
            .send()
            .await?;
        Ok(())
    }

    async fn send_callback_heartbeat(&self, callback_id: &str) -> Result<(), BoxError> {
        self.send_durable_execution_callback_heartbeat()
            .callback_id(callback_id)
            .send()
            .await?;
        Ok(())
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
/// The runner paces and retries its reads as the JS history poller does. It
/// waits one [`poll_interval`](Self::poll_interval) before each
/// `GetDurableExecution` poll, and one interval before each history page
/// that follows a page with a next marker. Each `GetDurableExecution` call
/// and each history page request makes up to three attempts on any error:
/// the second starts 1000 ms after the first fails, the third 1450 ms after
/// the second fails, and the third attempt's error is the call's error.
/// `Invoke` and the callback calls make one attempt.
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
///     .run::<i32>(21_i32)
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
    /// ends or a `GetDurableExecution` poll fails.
    ///
    /// A poll counts once however many attempts it makes (see
    /// [`CloudRunner`]), and a history read of a live control counts once
    /// however many pages and attempts it makes. The history read that
    /// builds the final [`TestResult`] does not count.
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

    /// Starts a run with `event`, and returns its [`RunHandle`].
    ///
    /// The run starts nothing until the test polls one of its futures.
    /// Awaiting the handle invokes the durable function, polls it to a
    /// terminal state, and returns the folded [`TestResult`]. Under
    /// [`InvocationType::Event`] the test can wait for an operation and
    /// answer its callback before that, through [`RunHandle::operation`].
    ///
    /// The event is serialized to JSON and delivered as the invocation
    /// payload. The handler output (on success) is deserialized into `O`,
    /// the only type parameter: name it with `run::<O>(event)`.
    ///
    /// # Errors
    ///
    /// Awaiting the handle returns these runner errors, the run failures.
    /// Only the run's own calls cause one:
    ///
    /// - [`TestRunnerError::EventSerialization`] when `event` does not
    ///   serialize.
    /// - [`TestRunnerError::Aws`] when `Invoke` fails, when the third
    ///   attempt of a `GetDurableExecution` poll fails, or when the history
    ///   read that builds the [`TestResult`] fails.
    /// - [`TestRunnerError::NoExecutionArn`] when the invoke response has no
    ///   execution ARN.
    /// - [`TestRunnerError::OutputDeserialization`] when the output of a
    ///   succeeded execution does not decode into `O`.
    ///
    /// A [`wait_for_data`](OperationHandle::wait_for_data) pending when the
    /// run fails returns the run failure. One first polled after the run
    /// failed returns the operation in the newest history read when it
    /// already reached the requested status, and the run failure otherwise.
    ///
    /// A [`send_callback`](OperationHandle::send_callback) or
    /// [`send_callback_heartbeat`](OperationHandle::send_callback_heartbeat)
    /// returns the run failure when the run fails before the invoke
    /// returned the execution ARN. Once the ARN is known, it goes on after
    /// a run failure and returns its own outcome.
    ///
    /// [`pause`](RunHandle::pause) and [`resume`](RunHandle::resume), every
    /// control under [`InvocationType::RequestResponse`], and
    /// [`CallbackReply::timeout`] and [`CallbackReply::heartbeat_timeout`]
    /// return their [`TestRunnerError::Unsupported`] error, also after a run
    /// failure.
    ///
    /// A failed history read or callback call of a control returns
    /// [`TestRunnerError::Aws`] from that control only. It is not a run
    /// failure: the run goes on, and the other controls are not affected.
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
    ///     .run::<String>("hello".to_owned())
    ///     .await?;
    /// assert!(result.is_success() || result.is_failure() || result.is_suspended());
    /// # Ok(())
    /// # }
    /// ```
    pub fn run<O>(&self, event: impl Serialize) -> RunHandle<O>
    where
        O: DeserializeOwned + Send + 'static,
    {
        let runner = self.clone();
        self.start(
            async move { Arc::new(runner.resolve_client().await) },
            serde_json::to_vec(&event),
        )
    }

    /// The run of [`run`](Self::run) over the [`CloudApi`] `api`. Internal
    /// seam: unit tests pass a fake.
    #[cfg(test)]
    fn run_with<A, O>(&self, api: Arc<A>, payload: Vec<u8>) -> RunHandle<O>
    where
        A: CloudApi + Send + Sync + 'static,
        O: DeserializeOwned + Send + 'static,
    {
        self.start(async move { api }, Ok(payload))
    }

    /// Starts a run whose client `resolve` returns, with the serialized
    /// event `payload`.
    fn start<A, O, R>(
        &self,
        resolve: R,
        payload: Result<Vec<u8>, serde_json::Error>,
    ) -> RunHandle<O>
    where
        A: CloudApi + Send + Sync + 'static,
        O: DeserializeOwned + Send + 'static,
        R: Future<Output = Arc<A>> + Send + 'static,
    {
        // A `CloudRunner` run keeps no backend records.
        let live = Arc::new(LiveState::new(BackendState::new(QueuedReplies::default())));
        let cloud = Arc::new(CloudShared::<A>::new(self, Arc::clone(&live)));
        let engine = Engine::Cloud(Arc::clone(&cloud) as Arc<dyn CloudControl>);
        let payload = match payload {
            Ok(payload) => payload,
            Err(e) => {
                return RunHandle::from_shared(RunCore::failed(
                    live,
                    engine,
                    TestRunnerError::EventSerialization(Arc::new(e)),
                ));
            }
        };
        let runner = self.clone();
        let driver = async move {
            let api = resolve.await;
            let result: TestResult<O> = runner.drive(&cloud, api, payload).await?;
            Ok(Box::new(result) as Box<dyn std::any::Any + Send>)
        };
        RunHandle::from_shared(RunCore::new(live, engine, Box::pin(driver)))
    }

    /// The invoke → poll → history → fold flow of [`run`](Self::run), over
    /// any [`CloudApi`]. The driver's `GetDurableExecution` calls spend the
    /// run's poll budget, which the live controls share (see
    /// [`max_poll_attempts`](Self::max_poll_attempts)).
    async fn drive<A: CloudApi + Send + Sync + 'static, O: DeserializeOwned>(
        &self,
        cloud: &CloudShared<A>,
        api: Arc<A>,
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
        cloud.set_target(Arc::clone(&api), execution_arn.clone());
        let api = &*api;

        // 2. Poll GetDurableExecution until terminal, or until a configured
        // budget is spent.
        let mut polls = 0_usize;
        let terminal = loop {
            if !cloud.budget_spent() {
                tokio::time::sleep(self.poll_interval).await;
                if cloud.take_poll() {
                    polls += 1;
                    let out = with_read_retries(|| api.send_get_durable_execution(&execution_arn))
                        .await
                        .map_err(aws_error)?;
                    if !matches!(out.status(), WireExecutionStatus::Running) {
                        break out;
                    }
                    continue;
                }
            }
            let (events, operations) = cloud.driver_history(api, &execution_arn).await?;
            return Ok(TestResult::with_history(
                ExecutionStatus::Running,
                None,
                Some(crate::WireError::new(
                    None::<String>,
                    Some(format!(
                        "execution {execution_arn} still running after {} polls",
                        cloud.used()
                    )),
                )),
                operations,
                polls,
                events,
            ));
        };

        // 3. Retrieve the recorded history and fold it into operations.
        let (events, operations) = cloud.driver_history(api, &execution_arn).await?;
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
            operations,
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

/// The attempts [`with_read_retries`] makes, as JS
/// `HistoryPoller.getMaxRetryAttempts`.
const READ_ATTEMPTS: u32 = 3;

/// The wait after `failed` failed attempts of a read, as JS
/// `callWithRetries`: `(failed² - 1) × 150 + 1000` ms, so 1000 ms after the
/// first failure and 1450 ms after the second.
fn read_retry_delay(failed: u32) -> std::time::Duration {
    std::time::Duration::from_millis(u64::from(failed * failed - 1) * 150 + 1000)
}

/// Calls `call` until it succeeds, up to [`READ_ATTEMPTS`] times, waiting
/// [`read_retry_delay`] after each failed attempt but the last. Any error
/// fails an attempt, and the last attempt's error is the result.
async fn with_read_retries<T, F, Fut>(mut call: F) -> Result<T, BoxError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, BoxError>>,
{
    let mut failed = 0;
    loop {
        match call().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                failed += 1;
                if failed >= READ_ATTEMPTS {
                    return Err(err);
                }
                tokio::time::sleep(read_retry_delay(failed)).await;
            }
        }
    }
}

/// Retrieves the full paginated execution history. Each page request makes
/// the attempts of [`with_read_retries`], and a page that follows a page
/// with a next marker is requested `poll_interval` after that page arrived.
async fn fetch_history<A: CloudApi>(
    api: &A,
    execution_arn: &str,
    poll_interval: std::time::Duration,
) -> Result<Vec<Event>, TestRunnerError> {
    let mut events = Vec::new();
    let mut marker: Option<String> = None;
    loop {
        let page = marker.as_deref();
        let out = with_read_retries(|| api.send_get_durable_execution_history(execution_arn, page))
            .await
            .map_err(aws_error)?;
        events.extend(out.events().iter().cloned());
        match out.next_marker() {
            Some(m) if !m.is_empty() => marker = Some(m.to_owned()),
            _ => break,
        }
        tokio::time::sleep(poll_interval).await;
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
        // step records `StepFailed`, and a `wait_for_condition` check that
        // continued records `StepSucceeded`, and then `StepStarted`, so a
        // later non-terminal event clears the end time of an earlier one.
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
                // A `wait_for_condition` check that continued records
                // `StepSucceeded` with a retry delay.
                set_next_attempt_time(op, event, details.retry_details());
            }
        }
        Some(EventType::StepFailed) => {
            op.status = OperationStatus::Failed;
            if let Some(details) = event.step_failed_details() {
                set_attempt(op, details.retry_details());
                set_next_attempt_time(op, event, details.retry_details());
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

/// Sets the next attempt time of a step from the retry details of its
/// `StepFailed` or `StepSucceeded` event `event`.
///
/// The latest step event that carries a retry delay sets the next attempt
/// time; one without a delay keeps it. A carried delay whose time cannot
/// be represented sets `None`, so the value never describes an older retry.
fn set_next_attempt_time(op: &mut TestOperation, event: &Event, retry: Option<&RetryDetails>) {
    if let Some(delay) = retry.and_then(RetryDetails::next_attempt_delay_seconds) {
        op.step.next_attempt_time = next_attempt_time(event, delay);
    }
}

/// The next attempt time of a step event that carries a retry delay (a
/// `StepFailed` of a step retry, or a `StepSucceeded` of a
/// `wait_for_condition` check that continued): its timestamp plus the
/// delay `delay_seconds`. `None` when the event has no timestamp, the
/// delay is negative, or the sum overflows `SystemTime`.
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
/// without one, and a `wait_for_condition` check that continued records
/// `StepSucceeded` without one. Keeping the earlier error would report a succeeded step with
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

/// Maps an event type to the [`RunnerKind`] of its operation, or
/// `None` for execution-level / invocation-level events that are not folded
/// into operations.
fn event_family_op_type(event_type: Option<&EventType>) -> Option<RunnerKind> {
    match event_type? {
        EventType::StepStarted | EventType::StepSucceeded | EventType::StepFailed => {
            Some(RunnerKind::Step)
        }
        EventType::WaitStarted | EventType::WaitSucceeded | EventType::WaitCancelled => {
            Some(RunnerKind::Wait)
        }
        EventType::ContextStarted | EventType::ContextSucceeded | EventType::ContextFailed => {
            Some(RunnerKind::Context)
        }
        EventType::CallbackStarted
        | EventType::CallbackSucceeded
        | EventType::CallbackFailed
        | EventType::CallbackTimedOut => Some(RunnerKind::Callback),
        EventType::ChainedInvokeStarted
        | EventType::ChainedInvokeSucceeded
        | EventType::ChainedInvokeFailed
        | EventType::ChainedInvokeTimedOut
        | EventType::ChainedInvokeStopped => Some(RunnerKind::ChainedInvoke),
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
    kind: RunnerKind,
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
    /// The `HeartbeatTimeoutSeconds` of a callback's `START`.
    heartbeat_seconds: Option<i32>,
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
    kind: RunnerKind,
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
        heartbeat_seconds: None,
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
    /// The payload of the accepted `EXECUTION` update: a result the SDK
    /// checkpointed because it is over the response limit. No operation
    /// stores it.
    execution_result: Option<String>,
}

impl BackendState {
    /// The records of a new run, with the queued callback replies
    /// `callback_replies`.
    fn new(callback_replies: QueuedReplies) -> Self {
        Self {
            ops: Vec::new(),
            token_counter: 0,
            callback_counter: 0,
            callback_replies,
            now: 0,
            base: std::time::SystemTime::now(),
            updated_ids: Vec::new(),
            events: Vec::new(),
            last_event_id: 0,
            execution_result: None,
        }
    }
}

/// Where a [`Backend`] keeps its records.
#[derive(Debug)]
enum StateCell {
    /// In its own mutex: a nested durable target and a unit test's
    /// injected backend, which no run control observes.
    Own(Mutex<BackendState>),
    /// In the run's state (see [`RunState::local`]), so every decision of
    /// the run's controls, its driver and its backend is one transition on
    /// one mutex.
    Live(Arc<LiveState>),
}

impl StateCell {
    /// Records in their own mutex.
    fn own(records: BackendState) -> Self {
        Self::Own(Mutex::new(records))
    }
}

/// The locked records of a [`Backend`].
enum StateGuard<'a> {
    Own(std::sync::MutexGuard<'a, BackendState>),
    Live(std::sync::MutexGuard<'a, RunState>),
}

impl StateGuard<'_> {
    /// [`RunState::recheck`] for the records of a top-level run. A backend
    /// with its own records has no controls.
    fn recheck(&mut self) -> Vec<std::task::Waker> {
        match self {
            Self::Own(_) => Vec::new(),
            Self::Live(state) => state.recheck(),
        }
    }
}

impl std::ops::Deref for StateGuard<'_> {
    type Target = BackendState;

    fn deref(&self) -> &BackendState {
        match self {
            Self::Own(guard) => guard,
            Self::Live(guard) => &guard.local.records,
        }
    }
}

impl std::ops::DerefMut for StateGuard<'_> {
    fn deref_mut(&mut self) -> &mut BackendState {
        match self {
            Self::Own(guard) => guard,
            Self::Live(guard) => &mut guard.local.records,
        }
    }
}

/// The in-memory execution client. Shared (via `Arc`) between the runner and
/// the durable context. Debug-printable and `Send + Sync` as the
/// `ExecutionClient` trait requires.
#[derive(Debug)]
struct Backend {
    state: StateCell,
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
    /// The 1-based number of the checkpoint request answered without a
    /// token (see [`LocalRunner::withdraw_checkpoints_after`]).
    withdraw_at: Option<usize>,
    /// Set when a request was answered without a token, and taken by the
    /// runner after the invocation answered `PENDING`.
    withdrew: std::sync::atomic::AtomicBool,
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
    /// The call applies the updates as [`Self::Pass`] does, then holds its
    /// answer until the test adds a permit to `gate`. So the SDK merges
    /// the response only after the release.
    PassGated { gate: Arc<tokio::sync::Semaphore> },
    /// The call applies the updates as [`Self::Pass`] does, then answers
    /// with a retryable error carrying the message, as when the service
    /// recorded a write whose response was lost.
    PassThenRetryableError(String),
}

impl Backend {
    /// A backend with its own records, as a unit test injects it.
    #[cfg(test)]
    fn new(callback_replies: QueuedReplies, checkpoint_page_size: Option<usize>) -> Self {
        Self::on(
            StateCell::own(BackendState::new(callback_replies)),
            checkpoint_page_size,
        )
    }

    /// A backend whose records are in `state`.
    fn on(state: StateCell, checkpoint_page_size: Option<usize>) -> Self {
        Self {
            state,
            checkpoint_page_size,
            get_state_calls: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_calls: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_remaining: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_skip: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_retryable: std::sync::atomic::AtomicBool::new(false),
            checkpoint_latency: std::time::Duration::ZERO,
            registered_functions: BTreeSet::new(),
            withdraw_at: None,
            withdrew: std::sync::atomic::AtomicBool::new(false),
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

    /// Queues `response` as the answer to an upcoming checkpoint call, after
    /// the calls queued before it. Only
    /// [`TestResponse::AcceptThenRetryableError`] has a meaning for the
    /// backend: it applies the call's updates, then answers with a
    /// retryable error.
    #[cfg(test)]
    fn enqueue_response(&self, response: crate::client::TestResponse) {
        let crate::client::TestResponse::AcceptThenRetryableError(message) = response else {
            unreachable!("the LocalRunner backend answers only AcceptThenRetryableError");
        };
        self.plan_checkpoint_calls(vec![PlannedCheckpoint::PassThenRetryableError(message)]);
    }

    /// The payload of the `EXECUTION` update the backend accepted, if any.
    fn execution_result(&self) -> Option<String> {
        self.lock().execution_result.clone()
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

    fn lock(&self) -> StateGuard<'_> {
        match &self.state {
            StateCell::Own(state) => StateGuard::Own(
                state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ),
            StateCell::Live(live) => StateGuard::Live(live.lock()),
        }
    }

    /// The run's state cell of a top-level run (see [`RunHandle`]). A
    /// nested durable target and a unit test's injected backend have none,
    /// so the hold rule does not apply to them.
    fn live(&self) -> Option<&Arc<LiveState>> {
        match &self.state {
            StateCell::Own(_) => None,
            StateCell::Live(live) => Some(live),
        }
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
        state: &mut BackendState,
        started_at: u64,
        request_id: &str,
        error: Option<&crate::WireError>,
    ) {
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
        snapshot_of(&self.lock())
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
        advance_records(&mut self.lock())
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
                op.kind == RunnerKind::ChainedInvoke
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
        // The waiting `wait_for_data` calls see the outcome in the same
        // transition.
        let wakers = state.recheck();
        drop(state);
        wake(wakers);
        true
    }

    /// The function names of the `Started` chained invokes, in stored
    /// order, each name once.
    fn pending_invoke_names(&self) -> Vec<String> {
        let state = self.lock();
        let mut names: Vec<String> = Vec::new();
        for op in &state.ops {
            if op.kind == RunnerKind::ChainedInvoke
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

/// Snapshots the recorded operations as public [`TestOperation`]s.
fn snapshot_of(state: &BackendState) -> Vec<TestOperation> {
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

/// [`Backend::advance`] on the records `state`.
fn advance_records(state: &mut BackendState) -> bool {
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
                let fired = if op.kind == RunnerKind::Wait {
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

/// The progress decision between invocations when nothing holds the run:
/// start the next invocation for `start`, ask for the pending invokes once,
/// then advance the backend with `advance`, else the run is stuck.
fn progress_decision(
    start: bool,
    invokes_tried: bool,
    advance: impl FnOnce() -> bool,
) -> BetweenStep {
    if start {
        return BetweenStep::Start;
    }
    if !invokes_tried {
        return BetweenStep::RunInvokes;
    }
    // Advance the simulated backend (timers, retries, callbacks). If
    // nothing can advance, the execution is genuinely stuck.
    if advance() {
        BetweenStep::Start
    } else {
        BetweenStep::Stuck
    }
}

/// The decision of one [`Backend::between_step`] of a top-level run, as one
/// transition on the run state `state`, so no control starts or stops
/// waiting, no reply is published and no pause or resume happens while it
/// is made.
///
/// While no control lets the run move, it decides to wait. It takes a
/// published live reply into `start`. While paused it advances the
/// records, or decides to wait for `resume` when nothing can advance.
/// Otherwise it makes the [`progress_decision`], and closes the run when it
/// is stuck. The wakers of the `wait_for_data` calls an advance made ready
/// are returned, to wake after the lock is released.
fn between_decision(
    state: &mut RunState,
    start: &mut bool,
    invokes_tried: bool,
) -> (Decided, Vec<std::task::Waker>) {
    if !state.hold_open() {
        return (Decided::Wait, Vec::new());
    }
    *start |= std::mem::take(&mut state.local.invocation_due);
    if state.local.paused {
        if advance_records(&mut state.local.records) {
            *start = true;
            return (Decided::Again, state.recheck());
        }
        return (Decided::Wait, Vec::new());
    }
    let mut advanced = false;
    let step = progress_decision(*start, invokes_tried, || {
        advanced = advance_records(&mut state.local.records);
        advanced
    });
    if step == BetweenStep::Stuck {
        state.local.phase = LocalPhase::Closed;
    }
    let wakers = if advanced {
        state.recheck()
    } else {
        Vec::new()
    };
    (Decided::Step(step), wakers)
}

/// Delivers `reply` to the pending callback that `selector` targets (see
/// [`callback_target`]) while the run is in progress, as a queued reply is
/// delivered: the record, the next envelope's `UpdatedOperationIds` and the
/// history event. Returns the wakers of the `wait_for_data` calls the reply
/// made ready, to wake after the lock is released.
///
/// It is one transition on the run state `state`: the target is chosen, the
/// outcome committed (see [`RunState::commit`]) and the reply published.
/// So the driver's next decision (see [`between_decision`]) either sees the
/// reply and starts the next invocation without moving the clock, or closed
/// the run before it, and the reply is refused with
/// [`TestRunnerError::CallbackNotPending`].
fn reply_live(
    state: &mut RunState,
    selector: &Selector,
    reply: CallbackReply,
) -> Result<Vec<std::task::Waker>, TestRunnerError> {
    let closed = state.local.phase == LocalPhase::Closed;
    let target =
        callback_target(selector, &snapshot_of(&state.local.records)).map(|op| op.id.clone());
    let records = &mut state.local.records;
    let now = records.now;
    let candidate = match target {
        Ok(id) => records
            .ops
            .iter()
            .position(|op| op.id == id && !closed && is_pending_callback(op))
            .ok_or(TestRunnerError::CallbackNotPending),
        Err(err) => Err(err),
    };
    let index = state.commit(candidate)?;
    let records = &mut state.local.records;
    // The commit above found the record at `index` in this transition, and
    // a record is never removed.
    #[expect(clippy::expect_used)]
    let op = records
        .ops
        .get_mut(index)
        .expect("the committed callback record is stored");
    reply.deliver_to(op, now);
    let event = settled_event(op);
    let id = op.id.clone();
    records.updated_ids.push(id);
    if let Some(event) = event {
        records.push_event(event);
    }
    state.local.invocation_due = true;
    Ok(state.recheck())
}

/// Checks a heartbeat for the callback that `selector` targets, as one
/// transition on the run state `state`: it must be pending, as for
/// [`reply_live`], and its `START` must have set a positive heartbeat
/// timeout. The outcome is committed (see [`RunState::commit`]).
fn check_heartbeat(state: &mut RunState, selector: &Selector) -> Result<(), TestRunnerError> {
    let closed = state.local.phase == LocalPhase::Closed;
    let records = &state.local.records;
    let candidate = callback_target(selector, &snapshot_of(records)).and_then(|target| {
        let op = records
            .ops
            .iter()
            .find(|op| op.id == target.id)
            .filter(|op| !closed && is_pending_callback(op))
            .ok_or(TestRunnerError::CallbackNotPending)?;
        if op.heartbeat_seconds.unwrap_or(0) > 0 {
            Ok(())
        } else {
            Err(TestRunnerError::NoHeartbeatTimeout)
        }
    });
    state.commit(candidate)
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
    op.kind == RunnerKind::Callback
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
            && parent.kind == RunnerKind::Context
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
        (RunnerKind::Wait, OperationStatus::Started)
        | (RunnerKind::Step, OperationStatus::Pending) => op.due_at,
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
            (RunnerKind::Step, OperationAction::Start) => op_event(op, EventType::StepStarted)
                .step_started_details(aws_sdk_lambda::types::StepStartedDetails::builder().build()),
            // `attempt` counts completed retries, and `CurrentAttempt` is
            // 1-based.
            (RunnerKind::Step, OperationAction::Succeed) => op_event(op, EventType::StepSucceeded)
                .step_succeeded_details(
                    aws_sdk_lambda::types::StepSucceededDetails::builder()
                        .set_result(result())
                        .retry_details(retry_details(op.attempt.saturating_add(1), None))
                        .build(),
                ),
            (RunnerKind::Step, OperationAction::Fail) => op_event(op, EventType::StepFailed)
                .step_failed_details(
                    aws_sdk_lambda::types::StepFailedDetails::builder()
                        .set_error(error())
                        .retry_details(retry_details(op.attempt.saturating_add(1), None))
                        .build(),
                ),
            // A `RETRY` has already counted the attempt in `attempt`, so
            // `attempt` is the 1-based number of that attempt. A `RETRY`
            // with an error is a failed step attempt, which the service
            // records as `StepFailed`. A `RETRY` without an error is a
            // `wait_for_condition` check that asked to continue, which the
            // service records as `StepSucceeded` with the update's payload,
            // the new state.
            (RunnerKind::Step, OperationAction::Retry) => {
                let retry = retry_details(
                    op.attempt,
                    update
                        .step_options()
                        .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds),
                );
                match error() {
                    Some(error) => op_event(op, EventType::StepFailed).step_failed_details(
                        aws_sdk_lambda::types::StepFailedDetails::builder()
                            .error(error)
                            .retry_details(retry)
                            .build(),
                    ),
                    None => op_event(op, EventType::StepSucceeded).step_succeeded_details(
                        aws_sdk_lambda::types::StepSucceededDetails::builder()
                            .set_result(event_result(update.payload()))
                            .retry_details(retry)
                            .build(),
                    ),
                }
            }
            (RunnerKind::Context, OperationAction::Start) => {
                op_event(op, EventType::ContextStarted).context_started_details(
                    aws_sdk_lambda::types::ContextStartedDetails::builder().build(),
                )
            }
            (RunnerKind::Context, OperationAction::Succeed) => {
                op_event(op, EventType::ContextSucceeded).context_succeeded_details(
                    aws_sdk_lambda::types::ContextSucceededDetails::builder()
                        .set_result(result())
                        .build(),
                )
            }
            (RunnerKind::Context, OperationAction::Fail) => op_event(op, EventType::ContextFailed)
                .context_failed_details(
                    aws_sdk_lambda::types::ContextFailedDetails::builder()
                        .set_error(error())
                        .build(),
                ),
            (RunnerKind::Wait, OperationAction::Start) => {
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
            (RunnerKind::Wait, OperationAction::Cancel) => op_event(op, EventType::WaitCancelled)
                .wait_cancelled_details(
                    aws_sdk_lambda::types::WaitCancelledDetails::builder()
                        .set_error(error())
                        .build(),
                ),
            (RunnerKind::Callback, OperationAction::Start) => {
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
            (RunnerKind::ChainedInvoke, OperationAction::Start) => {
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
        // An `EXECUTION` update carries the execution's result. It is kept
        // apart, so it never appears as an operation.
        let Some(kind) = RunnerKind::from_operation_kind(kind) else {
            self.execution_result = update.payload().map(ToOwned::to_owned);
            return None;
        };
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
        let heartbeat_seconds = update
            .callback_options()
            .map(aws_sdk_lambda::types::CallbackOptions::heartbeat_timeout_seconds);
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
            && kind == RunnerKind::Callback
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
                if op.kind == RunnerKind::Wait {
                    op.set_status(OperationStatus::Started, now);
                    op.wait_seconds = wait_seconds;
                    op.due_at = Some(now.saturating_add(whole_seconds(wait_seconds)));
                } else if op.kind == RunnerKind::Callback {
                    // The service creates a callback in `Started` and keeps
                    // it there until an external outcome. The backend
                    // assigns the callback id here.
                    op.set_status(OperationStatus::Started, now);
                    if let Some(cb) = new_callback_id {
                        op.callback_id = Some(cb);
                    }
                    op.heartbeat_seconds = heartbeat_seconds;
                } else {
                    op.set_status(OperationStatus::Started, now);
                    if op.kind == RunnerKind::ChainedInvoke {
                        op.invoke_function = invoke_function;
                        op.invoke_input = payload;
                    }
                }
            }
            OperationAction::Succeed => {
                op.set_status(OperationStatus::Succeeded, now);
                op.result = payload;
                if op.kind == RunnerKind::Context {
                    op.replay_children = replay_children;
                }
            }
            OperationAction::Fail => {
                op.set_status(OperationStatus::Failed, now);
                if op.kind == RunnerKind::Context {
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
    /// Applies and answers one checkpoint request with no latency. With
    /// `records`, the request is applied to those locked records, in the
    /// caller's transition; else the backend locks its records.
    fn checkpoint_now<'a>(
        &'a self,
        updates: &[OperationUpdate],
        records: Option<&mut BackendState>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + 'a>>
    {
        let number = self
            .checkpoint_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .saturating_add(1);
        let answer = self.checkpoint_counted(updates, records);
        if self.withdraw_at != Some(number) {
            return answer;
        }
        // `LocalRunner::withdraw_checkpoints_after`: the request is applied,
        // and its answer carries no token.
        Box::pin(async move {
            let output = answer.await?;
            self.withdrew
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(CheckpointOutput {
                checkpoint_token: None,
                ..output
            })
        })
    }

    /// Returns once an invocation may start under the hold rule, and starts
    /// it in the same transition (see [`RunState::admit_invocation`]).
    /// Without a run state it returns at once.
    async fn admit_invocation(&self) {
        if let Some(live) = self.live() {
            live.wait_until(RunState::admit_invocation).await;
        }
    }

    /// Records that the invocation that started at the virtual time
    /// `started_at` ended, as one transition: its `InvocationCompleted`
    /// event with `error`, the run's invocation phase and the pauses that
    /// waited for it (see [`RunState::end_invocation`]), and the recheck of
    /// the waiting `wait_for_data` calls. Then it wakes the controls.
    /// `closing` is set for a terminal envelope: the run ends, so a reply
    /// or heartbeat after this step is refused.
    fn finish_invocation(
        &self,
        closing: bool,
        started_at: u64,
        request_id: &str,
        error: Option<&crate::WireError>,
    ) {
        let Some(live) = self.live() else {
            Self::record_invocation_completed(&mut self.lock(), started_at, request_id, error);
            return;
        };
        let wakers = {
            let mut state = live.lock();
            Self::record_invocation_completed(
                &mut state.local.records,
                started_at,
                request_id,
                error,
            );
            let mut wakers = state.end_invocation(closing);
            wakers.extend(state.recheck());
            wakers
        };
        wake(wakers);
        live.wake_all();
    }

    /// Records that the run ends, so a later reply or heartbeat is refused.
    fn close(&self) {
        if let Some(live) = self.live() {
            live.lock().local.phase = LocalPhase::Closed;
        }
    }

    /// One decision of [`LocalRunner::between_invocations`]. A top-level run
    /// makes each [`between_decision`] as one transition on the run state,
    /// and waits for the state to change while the decision is to wait. A
    /// nested run has no run state and no hold rule, so it never waits.
    async fn between_step(&self, start: &mut bool, invokes_tried: bool) -> BetweenStep {
        let Some(live) = self.live() else {
            return progress_decision(*start, invokes_tried, || self.advance());
        };
        loop {
            let notified = live.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            let (step, wakers) = between_decision(&mut live.lock(), start, invokes_tried);
            wake(wakers);
            match step {
                Decided::Step(step) => return step,
                Decided::Again => {}
                Decided::Wait => notified.await,
            }
        }
    }

    /// Applies and answers one checkpoint request of a top-level run, under
    /// the hold rule: the request waits until a control lets the run move.
    /// The hold check, the request, a pause's tokenless answer and the
    /// recheck of the waiting `wait_for_data` calls are one transition on
    /// the run state.
    async fn checkpoint_held(
        &self,
        live: &LiveState,
        updates: &[OperationUpdate],
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        let (answer, tokenless, wakers) = loop {
            let notified = live.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            {
                let mut state = live.lock();
                if state.hold_open() {
                    let answer = self.checkpoint_now(updates, Some(&mut state.local.records));
                    let tokenless = state.take_tokenless();
                    let wakers = state.recheck();
                    break (answer, tokenless, wakers);
                }
            }
            notified.await;
        };
        wake(wakers);
        let output = answer.await?;
        if tokenless {
            self.withdrew
                .store(true, std::sync::atomic::Ordering::SeqCst);
            return Ok(CheckpointOutput {
                checkpoint_token: None,
                ..output
            });
        }
        Ok(output)
    }

    /// Whether a request was answered without a token since the last call.
    fn take_withdrawal(&self) -> bool {
        self.withdrew
            .swap(false, std::sync::atomic::Ordering::SeqCst)
    }

    /// [`Self::checkpoint_now`] after the request was counted.
    fn checkpoint_counted<'a>(
        &'a self,
        updates: &[OperationUpdate],
        records: Option<&mut BackendState>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + 'a>>
    {
        // In-crate test seam: a per-call plan (see
        // [`Backend::plan_checkpoint_calls`]) takes precedence over the
        // counter-based injection below while entries remain.
        #[cfg(test)]
        if let Some(planned) = self.next_planned_checkpoint() {
            match planned {
                PlannedCheckpoint::Pass => {}
                PlannedCheckpoint::PassGated { gate } => {
                    let answer = self.checkpoint_unplanned(updates, records);
                    return Box::pin(async move {
                        drop(gate.acquire().await);
                        answer.await
                    });
                }
                PlannedCheckpoint::PassThenRetryableError(message) => {
                    let answer = self.checkpoint_unplanned(updates, records);
                    return Box::pin(async move {
                        // The answer is replaced whatever it was: the test
                        // sees only the lost response.
                        drop(answer.await);
                        Err(ExecutionClientError::new(
                            crate::ExecutionClientErrorScope::Invocation,
                            message,
                        ))
                    });
                }
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
        self.checkpoint_unplanned(updates, records)
    }

    /// [`Self::checkpoint_now`] after the per-call plan: the injected
    /// faults, then the request against the simulated store.
    fn checkpoint_unplanned<'a>(
        &'a self,
        updates: &[OperationUpdate],
        records: Option<&mut BackendState>,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + 'a>>
    {
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
        // The request, the token and the response page are read in one
        // locked section of the records.
        let mut own = None;
        let records = match records {
            Some(records) => records,
            None => &mut **own.insert(self.lock()),
        };
        let updated_ops = match records.apply_request(updates, kinds) {
            Ok(ops) => ops,
            Err(message) => {
                return Box::pin(
                    async move { Err(ExecutionClientError::new_non_retryable(&message)) },
                );
            }
        };
        let token = format!("local-token-{}", records.token_counter);

        // If a checkpoint page size is configured and the total stored
        // operations exceed it, simulate a genuinely paginated response:
        // return only the FIRST PAGE of the full execution state (mirroring
        // the service's NewExecutionState) and set next_marker. Operations
        // beyond the page, including the ones this call just updated when
        // they fall outside it, are only observable through the follow-up
        // get_state fetch, so a caller that ignores the marker genuinely
        // misses state.
        let (response_ops, next_marker) = if let Some(page_size) = self.checkpoint_page_size {
            let all_ops: Vec<Operation> = records
                .ops
                .iter()
                .filter_map(|op| build_operation(op, records.base))
                .collect();
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
        if let Some(live) = self.live() {
            return Box::pin(async move {
                if !latency.is_zero() {
                    tokio::time::sleep(latency).await;
                }
                self.checkpoint_held(live, &updates).await
            });
        }
        if latency.is_zero() {
            return self.checkpoint_now(&updates, None);
        }
        Box::pin(async move {
            tokio::time::sleep(latency).await;
            self.checkpoint_now(&updates, None).await
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
            let parent_kind = started.get(parent_id).copied().or_else(|| {
                ops.iter()
                    .find(|op| op.id == parent_id)
                    .map(|op| op.kind.operation_kind())
            });
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
    if kind != stored.kind.operation_kind() {
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
        (RunnerKind::Wait, OperationStatus::Succeeded) => op_event(op, EventType::WaitSucceeded)
            .wait_succeeded_details(
                aws_sdk_lambda::types::WaitSucceededDetails::builder()
                    .set_duration(op.wait_seconds)
                    .build(),
            ),
        (RunnerKind::Callback, OperationStatus::Succeeded) => {
            op_event(op, EventType::CallbackSucceeded).callback_succeeded_details(
                aws_sdk_lambda::types::CallbackSucceededDetails::builder()
                    .set_result(result())
                    .build(),
            )
        }
        (RunnerKind::Callback, OperationStatus::Failed) => op_event(op, EventType::CallbackFailed)
            .callback_failed_details(
                aws_sdk_lambda::types::CallbackFailedDetails::builder()
                    .set_error(error())
                    .build(),
            ),
        (RunnerKind::Callback, OperationStatus::TimedOut) => {
            op_event(op, EventType::CallbackTimedOut).callback_timed_out_details(
                aws_sdk_lambda::types::CallbackTimedOutDetails::builder()
                    .set_error(error())
                    .build(),
            )
        }
        (RunnerKind::ChainedInvoke, OperationStatus::Succeeded) => {
            op_event(op, EventType::ChainedInvokeSucceeded).chained_invoke_succeeded_details(
                aws_sdk_lambda::types::ChainedInvokeSucceededDetails::builder()
                    .set_result(result())
                    .build(),
            )
        }
        (RunnerKind::ChainedInvoke, OperationStatus::Failed) => {
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
        .r#type(OperationType::from(stored.kind.operation_kind().wire_str()))
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
        RunnerKind::Step => {
            let mut sd = WireStepDetails::builder().attempt(stored.attempt);
            if let Some(r) = &stored.result {
                sd = sd.result(r);
            }
            if let Some(e) = error_obj.clone() {
                sd = sd.error(e);
            }
            builder = builder.step_details(sd.build());
        }
        RunnerKind::Context => {
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
        RunnerKind::Callback => {
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
        RunnerKind::ChainedInvoke => {
            let mut ci = ChainedInvokeDetails::builder();
            if let Some(r) = &stored.result {
                ci = ci.result(r);
            }
            if let Some(e) = error_obj.clone() {
                ci = ci.error(e);
            }
            builder = builder.chained_invoke_details(ci.build());
        }
        RunnerKind::Wait => {}
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
    obj.insert(
        "Type".to_owned(),
        serde_json::json!(stored.kind.operation_kind().wire_str()),
    );
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
        RunnerKind::Step => {
            details.insert("Attempt".to_owned(), serde_json::json!(stored.attempt));
            obj.insert("StepDetails".to_owned(), serde_json::Value::Object(details));
        }
        RunnerKind::Context => {
            if stored.replay_children {
                details.insert("ReplayChildren".to_owned(), serde_json::json!(true));
            }
            obj.insert(
                "ContextDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        RunnerKind::Callback => {
            if let Some(id) = &stored.callback_id {
                details.insert("CallbackId".to_owned(), serde_json::json!(id));
            }
            obj.insert(
                "CallbackDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        RunnerKind::ChainedInvoke => {
            obj.insert(
                "ChainedInvokeDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        RunnerKind::Wait => {}
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
            if parent.kind == RunnerKind::Context
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
    use super::live::{
        CLOUD_PAUSE_UNSUPPORTED, CLOUD_TIMEOUT_REPLY_UNSUPPORTED, ControlKind, ControlPhase,
        ControlRole, DriverFuture, HistoryStart, Registration, RunEnd,
        SEND_CALLBACK_HEARTBEAT_REQUEST_RESPONSE_UNSUPPORTED,
        SEND_CALLBACK_REQUEST_RESPONSE_UNSUPPORTED, WAIT_FOR_DATA_REQUEST_RESPONSE_UNSUPPORTED,
        await_in_flight, status_reached,
    };
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
                            .wait_for_callback::<u32>(move |_step_ctx, _callback_id| {
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
                            .with_retry(move |_child, _attempt| {
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
                                RetryDecision::retry(std::time::Duration::from_secs(1))
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

    /// A `Serdes<String>` whose `serialize` upper-cases the string and whose
    /// `deserialize` is the identity.
    struct UpperSerdes;

    impl crate::Serdes<String> for UpperSerdes {
        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: String,
            _context: crate::serdes::SerdesContext,
        ) -> Result<String, BoxError> {
            Ok(value.to_uppercase())
        }

        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            wire: String,
            _context: crate::serdes::SerdesContext,
        ) -> Result<String, BoxError> {
            Ok(wire)
        }
    }

    #[tokio::test]
    async fn wfc_first_check_sees_round_tripped_initial_state() {
        // The first check receives the initial state after a serdes round
        // trip, as JS does, so `UpperSerdes` shows it "A", not "a".
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let seen = Arc::new(Mutex::new(None::<String>));
                    let seen_check = Arc::clone(&seen);
                    ctx.wait_for_condition(
                        move |_sc, state: String| {
                            let seen = Arc::clone(&seen_check);
                            async move {
                                *seen.lock().expect("lock") = Some(state.clone());
                                Ok(state)
                            }
                        },
                        "a".to_owned(),
                    )
                    .wait_strategy_fn(|_state, _attempt| {
                        crate::builders::wait_for_condition::WaitDecision::complete()
                    })
                    .name("poll")
                    .serdes(UpperSerdes)
                    .await?;
                    let received = seen.lock().expect("lock").clone();
                    Ok::<_, BoxError>(received)
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&Some("A".to_owned())));
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
        // The execution records the condition's recorded message.
        assert_eq!(
            result.error_message(),
            Some("max attempts exceeded: 2 checks completed (max_attempts = 2)"),
            "failure must name exhaustion with the attempt count"
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
                                            CompletionDecision::complete(
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
                                        CompletionDecision::complete(CompletionOutcome::Succeeded)
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
                                            CompletionDecision::complete(
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
                        .wait_for_callback::<String>(|_step_ctx, _callback_id| async { Ok(()) })
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
                            .wait_for_callback::<String>(move |_step_ctx, _id| async move {
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

    /// An external failure with no message, or an empty one, gives the
    /// outer `Callback.ExternalFailure` record the message
    /// `Callback failed`, live and on replay, and leaves the callback's
    /// own record unchanged.
    #[tokio::test]
    async fn wfcb_external_failure_without_message_same_kind_live_and_replay() {
        let cases = [
            (CallbackFailure::default(), None, None, None),
            (
                CallbackFailure::new("Declined", "").with_error_data("d"),
                Some("Declined"),
                Some(""),
                Some("d"),
            ),
        ];
        for (failure, stored_type, stored_message, data) in cases {
            let runner = LocalRunner::new().callback(CallbackReply::failure(failure));
            let run =
                run_wfcb_classification(runner, Submitter::Succeeds, "ExternalFailure", "external")
                    .await;
            assert!(run.result.is_success(), "{:?}", run.result.error_message());
            let entry = (
                "ExternalFailure".to_owned(),
                Some("Callback.ExternalFailure".to_owned()),
                Some("Callback failed".to_owned()),
                data.map(ToOwned::to_owned),
            );
            assert_eq!(run.seen, [entry.clone(), entry]);
            let stored = callback_record(&run.result);
            assert_eq!(stored.error_type(), stored_type);
            assert_eq!(stored.error_message(), stored_message);
            assert_eq!(stored.error().and_then(crate::WireError::error_data), data);
            let outer = run
                .result
                .operation("approve")
                .expect("the wait_for_callback is recorded");
            assert_eq!(outer.error_type(), Some("Callback.ExternalFailure"));
            assert_eq!(outer.error_message(), Some("Callback failed"));
        }
    }

    #[tokio::test]
    async fn wfcb_submitter_failure_same_kind_live_and_replay() {
        // The recorded message omits a `TypedError`'s name; the live
        // source's alternate `Display` keeps it.
        for (submitter, text, live_text) in [
            (
                Submitter::FailsUntyped,
                "submission failed",
                "operation error: step: execution failed: submission failed",
            ),
            (
                Submitter::FailsTyped,
                "declined",
                "operation error: step: execution failed: PaymentError: declined",
            ),
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
            assert_eq!(message, text);
            let live = run
                .live_step_source
                .expect("the live source is the step error");
            assert_eq!(live, live_text);
        }
    }

    #[tokio::test]
    async fn wfcb_timeout_same_kind_live_and_replay() {
        let runner = LocalRunner::new().callback(CallbackReply::timeout());
        let run =
            run_wfcb_classification(runner, Submitter::Succeeds, "TimedOut", "timed out").await;
        assert!(run.result.is_success(), "{:?}", run.result.error_message());
        let entry = (
            "TimedOut".to_owned(),
            Some("Callback.Timeout".to_owned()),
            Some("Callback timed out".to_owned()),
            None,
        );
        assert_eq!(run.seen, [entry.clone(), entry]);
        assert_eq!(approve_error_type(&run.result), Some("Callback.Timeout"));
        assert_eq!(
            run.result
                .operation("approve")
                .and_then(TestOperation::error_message),
            Some("Callback timed out")
        );
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
        let entry = (
            "HeartbeatTimedOut".to_owned(),
            Some("Callback.Heartbeat".to_owned()),
            Some("Callback timed out on heartbeat".to_owned()),
            None,
        );
        assert_eq!(run.seen, [entry.clone(), entry]);
        assert_eq!(approve_error_type(&run.result), Some("Callback.Heartbeat"));
        assert_eq!(
            run.result
                .operation("approve")
                .and_then(TestOperation::error_message),
            Some("Callback timed out on heartbeat")
        );
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
        // The outer context records the raw payload, byte for byte, and the
        // decode after it fails.
        let approve = run
            .result
            .operation("approve")
            .expect("the wait_for_callback is recorded");
        assert_eq!(approve.status(), OperationStatus::Succeeded);
        assert_eq!(approve.result(), Some("not json"));
        assert_eq!(approve.error_type(), None);
    }

    /// `PassthroughSerdes` on `wait_for_callback` returns the raw reply the
    /// external caller sent, which is not JSON.
    #[tokio::test]
    async fn wfcb_passthrough_reads_raw_reply() {
        let result = LocalRunner::new()
            .callback(CallbackReply::raw("approved"))
            .run(
                |(), ctx: DurableContext| async move {
                    let approval = ctx
                        .wait_for_callback::<String>(|_step_ctx, _id| async { Ok(()) })
                        .name("approve")
                        .serdes(crate::PassthroughSerdes)
                        .await?;
                    Ok::<_, BoxError>(approval)
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("approved"));
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
                                RetryDecision::retry(std::time::Duration::from_secs(1))
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
                        .wait_for_callback::<u32>(|_step_ctx, _callback_id| async { Ok(()) })
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
                            .with_retry(move |child, _attempt| {
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
                                    RetryDecision::retry(std::time::Duration::from_secs(1))
                                }
                            })
                            .await?;
                        // Case E.
                        let case_e = ctx
                            .with_retry(|_child, _attempt| async { Ok(5_u32) })
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
                                    .wait_for_callback::<u32>(|_step_ctx, _callback_id| async {
                                        Ok(())
                                    })
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
        assert_eq!(result.error_message(), Some("Callback failed"));
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
                        .heartbeat_timeout(std::time::Duration::from_secs(60))
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
            Some(RunnerKind::ChainedInvoke)
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
            msg.starts_with("context usage error") && msg.contains(".spawn()"),
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

    /// A non-retryable failure of a detached batch flush whose only
    /// contributor was dropped before the rejection arrived fails the
    /// execution in this invocation. The failure is recorded where the
    /// response arrives, so no reader is needed. The SDK sends nothing
    /// after it: the later live step's START never reaches the backend,
    /// and no terminal FAIL is written, so the orphan's record stays at its
    /// accepted START.
    #[tokio::test]
    async fn nonretryable_failure_of_dropped_contributor_fails_execution() {
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
        //      so the rejection has no reader.
        // Nothing is sent after the rejection.
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
                        // gate: the rejection has no reader.
                        while backend.checkpoint_call_count() < 2 {
                            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        }
                        drop(orphan);
                        gate.add_permits(1);

                        // A later LIVE operation: its START write is never
                        // sent, because the failure ends the drive and
                        // stops every later write.
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
            "the orphaned body runs exactly once"
        );
        assert_eq!(
            result.invocation_count(),
            1,
            "the non-retryable rejection is deterministic: the execution \
             must die in this invocation, not re-invoke into the same \
             rejection"
        );
        assert!(
            result.is_failure(),
            "the execution must fail; got {:?} / {:?}",
            result.error_type(),
            result.error_message()
        );
        assert_eq!(
            result.error_type(),
            Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
        );

        // No terminal FAIL: the orphan's record stays at its accepted
        // START, and nothing is sent after the rejection.
        let ops = result.operations();
        let orphan_op = ops
            .iter()
            .find(|op| op.name.as_deref() == Some("orphan"))
            .expect("the orphaned step's operation record exists");
        assert_eq!(orphan_op.status, OperationStatus::Started);
        assert!(
            ops.iter().all(|op| op.name.as_deref() != Some("live")),
            "the live step's START is never sent"
        );
        assert_eq!(backend.checkpoint_call_count(), 2);
    }

    /// The change of a write whose caller dropped it, and that no
    /// operation task owns, is reported before `on_invocation_end`.
    ///
    /// Call 2, the `SUCCEED`, is applied but its answer is held until the
    /// handler dropped the write. So the SDK merges that response only
    /// after the drop, and only the drain before the end hook delivers it.
    #[tokio::test]
    async fn dropped_contributor_change_reported_before_invocation_end() {
        let recorder = crate::plugin::test_support::Changes::default();
        let runner = LocalRunner::new().options(
            crate::Options::builder()
                .plugin(recorder.clone())
                .build()
                .expect("valid options"),
        );
        let backend = Arc::new(Backend::new(
            QueuedReplies::default(),
            runner.checkpoint_page_size,
        ));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        backend.plan_checkpoint_calls(vec![
            PlannedCheckpoint::Pass,
            PlannedCheckpoint::PassGated {
                gate: Arc::clone(&gate),
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
                        let update = |action: OperationAction| {
                            OperationUpdate::builder()
                                .id("orphan-id")
                                .name("orphan")
                                .r#type(OperationType::Step)
                                .action(action)
                                .build()
                                .expect("valid update")
                        };
                        ctx.checkpoint_updates(vec![update(OperationAction::Start)])
                            .await?;
                        let write =
                            ctx.submit_checkpoint_updates(vec![update(OperationAction::Succeed)]);
                        while backend.checkpoint_call_count() < 2 {
                            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        }
                        drop(write);
                        gate.add_permits(1);
                        Ok::<_, BoxError>("done".to_owned())
                    }
                },
                "null".to_owned(),
            )
            .await
            .expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("done"));
        let timeline = recorder.timeline();
        assert!(
            timeline.ends_with(&[
                "change:orphan:SUCCEEDED".to_owned(),
                "invocation_end:Succeeded".to_owned(),
            ]),
            "{timeline:?}"
        );
    }

    /// A non-retryable failure of a detached batch flush whose contributor
    /// was dropped fails the execution, and the plugins' end hook runs
    /// once, with `Failed`. The failure ends the drive before the handler
    /// records an invocation fault, and no terminal FAIL is written, so
    /// the orphan's record stays at its accepted START.
    #[tokio::test]
    async fn end_hook_failed_after_a_dropped_contributor_failure() {
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
                        // The rejection ends the drive while this sleeps.
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
        let orphan = result
            .operations()
            .iter()
            .find(|op| op.name.as_deref() == Some("orphan"))
            .map(|op| op.status);
        assert_eq!(orphan, Some(OperationStatus::Started));
    }

    // Harness fidelity: production wire-error mapping

    /// The runner reports the SAME wire error message production reports.
    /// The execution propagates the child's failure, so it records the
    /// child's recorded message, which is the body's raw message.
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
        assert_eq!(
            result.error_message(),
            Some("boom-child"),
            "the execution records the child's recorded message"
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
                                .retry_strategy(|_err, _attempt| {
                                    RetryDecision::retry(std::time::Duration::from_secs(300))
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

    // Issue #137: summaries of oversized children and batches.

    /// An oversized child records its summary, and the replay after the
    /// wait runs the body again and ignores the summary.
    #[tokio::test]
    async fn child_summary_ignored_on_replay() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let big = ctx
                            .run_in_child_context(|_child| async { Ok("x".repeat(300 * 1024)) })
                            .name("big")
                            .summary(move |s: &String| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                format!("len={}", s.len())
                            })
                            .await?;
                        ctx.wait(std::time::Duration::from_secs(1))
                            .name("pause")
                            .await?;
                        Ok::<_, BoxError>(big.len())
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&307_200));
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the replay calls no summary"
        );
        let big = result.operation("big").expect("the child's record");
        assert_eq!(big.result(), Some("len=307200"));
        assert!(
            matches!(big.details(), TestOperationDetails::Context(c) if c.replay_children()),
            "{:?}",
            big.details()
        );
    }

    /// The batch record `m` of an oversized batch, parsed as JSON, with
    /// the assertions every oversized batch summary shares.
    fn oversized_batch_record(result: &TestResult<Vec<usize>>, type_name: &str) {
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&vec![153_600, 153_600]));
        let text = result
            .operation("m")
            .and_then(TestOperation::result)
            .expect("the batch record has a payload");
        let record: serde_json::Value = serde_json::from_str(text).expect("the record is JSON");
        assert_eq!(record["type"], type_name, "{text}");
        assert_eq!(record["summary"], "ok=2", "{text}");
        assert_eq!(record["totalCount"], 2, "{text}");
        assert_eq!(record["itemStatuses"], "SS", "{text}");
    }

    /// An oversized map stores the caller's summary under the key
    /// `summary` of its `MapResult` record, and replays from the record.
    #[tokio::test]
    async fn map_summary_stored_under_summary_key() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let items = ctx
                            .map(vec![0_u32, 1], |child, _item, _idx| async move {
                                Ok(child
                                    .step(|_| async { Ok("x".repeat(150 * 1024)) })
                                    .name("big")
                                    .await?)
                            })
                            .name("m")
                            .summary(move |r| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                format!("ok={}", r.success_count())
                            })
                            .await?;
                        ctx.wait(std::time::Duration::from_secs(1))
                            .name("pause")
                            .await?;
                        Ok::<_, BoxError>(items.iter().map(String::len).collect::<Vec<_>>())
                    }
                },
                (),
            )
            .await
            .expect("runner");
        oversized_batch_record(&result, "MapResult");
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the write calls the summary once, and the replay calls none"
        );
    }

    /// An oversized parallel stores the caller's summary under the key
    /// `summary` of its `ParallelResult` record.
    #[tokio::test]
    async fn parallel_summary_stored_under_summary_key() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let branch = |name: &'static str| {
                            crate::Branch::new(name, |child| async move {
                                Ok(child
                                    .step(|_| async { Ok("x".repeat(150 * 1024)) })
                                    .name("big")
                                    .await?)
                            })
                        };
                        let items = ctx
                            .parallel(vec![branch("a"), branch("b")])
                            .name("m")
                            .summary(move |r| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                format!("ok={}", r.success_count())
                            })
                            .await?;
                        ctx.wait(std::time::Duration::from_secs(1))
                            .name("pause")
                            .await?;
                        Ok::<_, BoxError>(items.iter().map(String::len).collect::<Vec<_>>())
                    }
                },
                (),
            )
            .await
            .expect("runner");
        oversized_batch_record(&result, "ParallelResult");
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the write calls the summary once, and the replay calls none"
        );
    }

    /// A batch whose record fits never calls the summary, and records the
    /// normal batch payload.
    #[tokio::test]
    async fn map_summary_not_called_under_limit() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let items = ctx
                            .map(vec![0_u32, 1], |_child, item, _idx| async move { Ok(item) })
                            .name("m")
                            .summary(move |_r| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                "unused".to_owned()
                            })
                            .await?;
                        Ok::<_, BoxError>(items)
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&vec![0, 1]));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let text = result
            .operation("m")
            .and_then(TestOperation::result)
            .expect("the batch record has a payload");
        let record: serde_json::Value = serde_json::from_str(text).expect("the record is JSON");
        assert_eq!(record["completionReason"], "ALL_COMPLETED", "{text}");
        assert_eq!(record["all"].as_array().map(Vec::len), Some(2), "{text}");
        assert!(record.get("summary").is_none(), "{text}");
    }

    /// A `with_retry` block's summary becomes the payload of the block's
    /// own oversized record.
    #[tokio::test]
    async fn with_retry_summary_forwarded_to_block() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let big = ctx
                            .with_retry(|_attempt_ctx, _attempt| async {
                                Ok("x".repeat(300 * 1024))
                            })
                            .name("block")
                            .summary(move |s: &String| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                format!("len={}", s.len())
                            })
                            .await?;
                        Ok::<_, BoxError>(big.len())
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.operation("block").and_then(TestOperation::result),
            Some("len=307200")
        );
        // The attempt context records its oversized result without the
        // summary, so the summary ran once, for the block's own record.
        assert_eq!(
            result
                .operation("attempt-1")
                .and_then(TestOperation::result),
            Some("")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// A `with_retry` block whose result fits never calls its summary.
    #[tokio::test]
    async fn with_retry_summary_not_called_under_limit() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let value = ctx
                            .with_retry(|_attempt_ctx, _attempt| async { Ok("small".to_owned()) })
                            .name("block")
                            .summary(move |_s: &String| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                "unused".to_owned()
                            })
                            .await?;
                        Ok::<_, BoxError>(value)
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("small"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    /// A `with_retry` block without a child context writes no record of
    /// its own, so its summary is never called, even for an oversized
    /// result.
    #[tokio::test]
    async fn with_retry_summary_never_called_without_child_context() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let big = ctx
                            .with_retry(|_attempt_ctx, _attempt| async {
                                Ok("x".repeat(300 * 1024))
                            })
                            .name("block")
                            .without_child_context()
                            .summary(move |_s: &String| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                "unused".to_owned()
                            })
                            .await?;
                        Ok::<_, BoxError>(big.len())
                    }
                },
                (),
            )
            .await
            .expect("runner");

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&307_200));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            result
                .operation("block-attempt-1")
                .and_then(TestOperation::result),
            Some("")
        );
    }

    /// An oversized `Flat` map calls its summary once, when it writes the
    /// record, and the replay after the wait runs the items again to
    /// rebuild the result without calling the summary.
    #[tokio::test]
    async fn flat_map_summary_called_once_and_items_rebuilt() {
        let calls = Arc::new(AtomicU32::new(0));
        let bodies = Arc::new(AtomicU32::new(0));
        let (handler_calls, handler_bodies) = (Arc::clone(&calls), Arc::clone(&bodies));
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    let bodies = Arc::clone(&handler_bodies);
                    async move {
                        let items = ctx
                            .map(vec![0_u32, 1], move |child, _item, _idx| {
                                bodies.fetch_add(1, Ordering::SeqCst);
                                async move {
                                    Ok(child.step(|_| async { Ok("x".repeat(150 * 1024)) }).await?)
                                }
                            })
                            .name("m")
                            .nesting(crate::builders::map_parallel::NestingMode::Flat)
                            .summary(move |r| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                format!("ok={}", r.success_count())
                            })
                            .await?;
                        ctx.wait(std::time::Duration::from_secs(1))
                            .name("pause")
                            .await?;
                        Ok::<_, BoxError>(items.iter().map(String::len).collect::<Vec<_>>())
                    }
                },
                (),
            )
            .await
            .expect("runner");

        oversized_batch_record(&result, "MapResult");
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            bodies.load(Ordering::SeqCst),
            4,
            "each Flat item runs once live and once to rebuild the result"
        );
    }

    /// The summary set on a map or parallel reaches the batch record
    /// through `.await_batch()` and `.spawn_batch()`, which build the
    /// batch apart from `.await`.
    #[tokio::test]
    async fn batch_summary_kept_by_await_batch_and_spawn_batch() {
        for (kind, spawn) in [
            ("map", false),
            ("map", true),
            ("parallel", false),
            ("parallel", true),
        ] {
            let result = LocalRunner::new()
                .run(
                    move |(), ctx: DurableContext| async move {
                        let big = || async { Ok("x".repeat(150 * 1024)) };
                        let batch = if kind == "map" {
                            let builder = ctx
                                .map(vec![0_u32, 1], move |child, _item, _idx| async move {
                                    Ok(child.step(move |_| big()).name("big").await?)
                                })
                                .name("m")
                                .summary(|r| format!("ok={}", r.success_count()));
                            if spawn {
                                builder.spawn_batch().await?
                            } else {
                                builder.await_batch().await?
                            }
                        } else {
                            let branch = |name: &'static str| {
                                crate::Branch::new(name, move |child| async move {
                                    Ok(child.step(move |_| big()).name("big").await?)
                                })
                            };
                            let builder = ctx
                                .parallel(vec![branch("a"), branch("b")])
                                .name("m")
                                .summary(|r| format!("ok={}", r.success_count()));
                            if spawn {
                                builder.spawn_batch().await?
                            } else {
                                builder.await_batch().await?
                            }
                        };
                        let items = batch.into_result()?;
                        Ok::<_, BoxError>(items.results().map(String::len).collect::<Vec<_>>())
                    },
                    (),
                )
                .await
                .expect("runner");
            let type_name = if kind == "map" {
                "MapResult"
            } else {
                "ParallelResult"
            };
            oversized_batch_record(&result, type_name);
        }
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
            // The backend stores no `EXECUTION` record (it keeps the
            // payload apart), so an `EXECUTION` update never meets a
            // stored status.
            if op_type == T::Execution {
                let update = built(rule_update("o", op_type.clone(), action.clone()));
                check_rule_row("Execution Succeed", &[], &[], vec![update], message).await;
                continue;
            }
            // The seed creates the record. Its status is then set directly.
            let seed = [built(rule_update("o", op_type.clone(), A::Start))];
            for status in statuses {
                let expected = (!allowed.contains(&status)).then_some(message).flatten();
                let row = format!("{op_type:?} {action:?} at {status:?}");
                let update = built(rule_update("o", op_type.clone(), action.clone()));
                check_rule_row(&row, &seed, &[("o", status)], vec![update], expected).await;
            }
        }
        // EXECUTION FAIL is allowed too.
        let update = built(rule_update("o", T::Execution, A::Fail));
        check_rule_row("Execution Fail", &[], &[], vec![update], None).await;
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
        assert_eq!(hidden.kind, RunnerKind::Wait);
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
                    let doubled = ctx.invoke::<i32>("double", 21).await?;
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
                        .invoke::<()>("pay", ())
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
            Some(&(Some("PaymentError".to_owned()), Some("declined".to_owned()))),
            "{:?}",
            result.error_message()
        );
        let op = invoke_record(&result);
        assert_eq!(op.status(), OperationStatus::Failed);
        assert_eq!(op.error_type(), Some("PaymentError"));
        assert_eq!(op.error_message(), Some("declined"));
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
                        .invoke::<()>("pay", ())
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
                    ctx.invoke::<()>("notify", 1).await?;
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
                    ctx.invoke::<()>("child-fn", ()).await?;
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
                        .invoke::<usize>("len", 5_i32)
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
                        .invoke::<serde_json::Value>("map", ())
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
                    let count = ctx.invoke::<u32>("count", ()).await?;
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
                            Ok(child.invoke::<String>("fast", ()).await?)
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
                    let out = ctx.invoke::<u32>("workflow", 1).await?;
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
                        .invoke::<()>("pay", ())
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
        assert_eq!(op.error_message(), Some("declined"));
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
        ctx.invoke::<()>("slow", ()).await?;
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
                    ctx.invoke::<()>("count", ()).await?;
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
                    let approved = ctx.invoke::<bool>("approval", ()).await?;
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
                    let approval = ctx.invoke::<bool>("approval", ()).spawn();
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
                let leaf = ctx.invoke::<u32>("leaf", ()).await?;
                Ok::<_, BoxError>(leaf)
            })
            .run(
                |(): (), ctx: DurableContext| async move {
                    let err = ctx
                        .invoke::<u32>("mid", ())
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
                    let millis = ctx.invoke::<u64>("timed", ()).await?;
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
                    let out = ctx.invoke::<u32>("t", ()).await?;
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
            .invoke::<()>("fail", ())
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
                TestOperationDetails::Context(_) => "Context",
                TestOperationDetails::Step(_) => "Step",
                TestOperationDetails::Wait(_) => "Wait",
                TestOperationDetails::Callback(_) => "Callback",
                TestOperationDetails::ChainedInvoke => "ChainedInvoke",
            }
        }
        let rows = [
            (RunnerKind::Context, OperationKind::Context, "Context"),
            (RunnerKind::Step, OperationKind::Step, "Step"),
            (RunnerKind::Wait, OperationKind::Wait, "Wait"),
            (RunnerKind::Callback, OperationKind::Callback, "Callback"),
            (
                RunnerKind::ChainedInvoke,
                OperationKind::ChainedInvoke,
                "ChainedInvoke",
            ),
        ];
        for (kind, public, expected) in rows {
            let op = TestOperation::started("id".to_owned(), kind);
            assert_eq!(variant(op.details()), expected, "{kind:?}");
            assert_eq!(op.kind(), public, "{kind:?}");
            assert_eq!(RunnerKind::from_operation_kind(public), Some(kind));
        }
        assert_eq!(
            RunnerKind::from_operation_kind(OperationKind::Execution),
            None
        );
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
        assert_eq!(error.error_message(), Some("no"));
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
        assert_eq!(error.error_message(), Some("no"));
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
    fn checkpoint_fault_setters_replace_all_fault_fields() {
        let fault = |r: &LocalRunner| {
            (
                r.checkpoint_failure_skip,
                r.checkpoint_failures,
                r.checkpoint_failures_retryable,
                r.withdraw_after,
            )
        };
        let next = LocalRunner::new()
            .withdraw_checkpoints_after(4)
            .fail_checkpoints_after_retryable(2, 3)
            .fail_next_checkpoints(1);
        assert_eq!(fault(&next), (0, 1, false, Some(4)));

        let after = LocalRunner::new()
            .withdraw_checkpoints_after(4)
            .fail_checkpoints_after_retryable(2, 3)
            .fail_checkpoints_after(5, 6);
        assert_eq!(fault(&after), (5, 6, false, Some(4)));

        let retryable = LocalRunner::new()
            .withdraw_checkpoints_after(4)
            .fail_checkpoints_after(2, 3)
            .fail_next_checkpoints(7)
            .fail_checkpoints_after_retryable(5, 6);
        assert_eq!(fault(&retryable), (5, 6, true, Some(4)));
    }

    #[test]
    fn typed_accessors_read_folded_cloud_history() {
        use aws_sdk_lambda::types::WaitStartedDetails;
        #[derive(Debug, PartialEq, serde::Deserialize)]
        struct Order {
            id: u32,
        }
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("step")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("step")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload(r#"{"id":7}"#).build())
                        .build(),
                )
                .build(),
            Event::builder()
                .event_type(EventType::ContextStarted)
                .id("wfcb")
                .sub_type("WaitForCallback")
                .event_id(3)
                .build(),
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("custom")
                .sub_type("WaitForCallback")
                .event_id(4)
                .build(),
            Event::builder()
                .event_type(EventType::WaitStarted)
                .id("wait")
                .event_id(5)
                .wait_started_details(
                    WaitStartedDetails::builder()
                        .duration(10)
                        .scheduled_end_timestamp(DateTime::from_secs(10))
                        .build()
                        .expect("wait details"),
                )
                .build(),
            Event::builder()
                .event_type(EventType::WaitSucceeded)
                .id("bare-wait")
                .event_id(6)
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 5, "{ops:?}");
        let (step, wfcb, custom, wait, bare_wait) = (&ops[0], &ops[1], &ops[2], &ops[3], &ops[4]);

        assert_eq!(
            step.result_json::<Order>().expect("decodes"),
            Some(Order { id: 7 })
        );
        assert!(step.result_json::<String>().is_err());
        assert_eq!(wfcb.result_json::<Order>().expect("no result"), None);

        assert!(wfcb.is_wait_for_callback());
        assert!(!custom.is_wait_for_callback());
        assert!(!step.is_wait_for_callback());

        assert_eq!(
            wait.wait_duration(),
            Some(std::time::Duration::from_secs(10))
        );
        assert_eq!(bare_wait.wait_duration(), None);
        assert_eq!(step.wait_duration(), None);

        // A non-wait reports no duration even when its wait detail holds
        // one: the accessor reads the detail of the operation's kind.
        let mut not_a_wait = TestOperation::started("s".to_owned(), RunnerKind::Step);
        not_a_wait.wait.duration = Some(std::time::Duration::from_secs(3));
        assert_eq!(not_a_wait.wait_duration(), None);
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
    ///
    /// With [`repeat_history`](Self::repeat_history), a history request
    /// with no scripted page left gets the repeated page instead. `calls`
    /// records every request in order.
    struct FakeApi {
        invoke: Mutex<Option<Result<InvokeOutput, BoxError>>>,
        executions: Mutex<std::collections::VecDeque<Result<GetExecutionOutput, BoxError>>>,
        history: Mutex<std::collections::VecDeque<HistoryPage>>,
        repeated_history: Mutex<Option<Result<Vec<Event>, &'static str>>>,
        invokes: Mutex<Vec<InvokeRequest>>,
        execution_calls: Mutex<Vec<ExecutionCall>>,
        history_calls: Mutex<Vec<HistoryCall>>,
        history_times: Mutex<Vec<tokio::time::Instant>>,
        calls: Mutex<Vec<FakeCall>>,
    }

    /// One request a [`FakeApi`] received.
    #[derive(Debug, Clone, PartialEq)]
    enum FakeCall {
        Invoke,
        GetExecution,
        History,
        CallbackSuccess(String, Vec<u8>),
        CallbackFailure(String, ErrorObject),
        Heartbeat(String),
    }

    impl FakeApi {
        /// A fake whose invoke returns [`ARN`] and whose history is one
        /// empty page.
        fn new() -> Self {
            Self {
                invoke: Mutex::new(Some(Ok(invoke_output(Some(ARN))))),
                executions: Mutex::new(std::collections::VecDeque::new()),
                history: Mutex::new(std::collections::VecDeque::from([Ok((Vec::new(), None))])),
                repeated_history: Mutex::new(None),
                invokes: Mutex::new(Vec::new()),
                execution_calls: Mutex::new(Vec::new()),
                history_calls: Mutex::new(Vec::new()),
                history_times: Mutex::new(Vec::new()),
                calls: Mutex::new(Vec::new()),
            }
        }

        /// Serves `page` (events, or a fake error with this text) to every
        /// history request after the scripted pages ran out.
        fn repeat_history(self, page: Result<Vec<Event>, &'static str>) -> Self {
            *self.repeated_history.lock().expect("lock") = Some(page);
            self
        }

        fn calls(&self) -> Vec<FakeCall> {
            self.calls.lock().expect("lock").clone()
        }

        fn record(&self, call: FakeCall) {
            self.calls.lock().expect("lock").push(call);
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

        /// The milliseconds after `start` of every `GetDurableExecution`
        /// request.
        fn execution_ms(&self, start: tokio::time::Instant) -> Vec<u128> {
            self.execution_calls
                .lock()
                .expect("lock")
                .iter()
                .map(|call| (call.at - start).as_millis())
                .collect()
        }

        /// The milliseconds after `start` and the marker of every
        /// `GetDurableExecutionHistory` request.
        fn history_ms(&self, start: tokio::time::Instant) -> Vec<(u128, Option<String>)> {
            let times = self.history_times.lock().expect("lock").clone();
            times
                .iter()
                .zip(self.history_calls())
                .map(|(at, call)| ((*at - start).as_millis(), call.marker))
                .collect()
        }
    }

    /// `(ms, None)` for each of `ms`: history requests without a marker.
    fn unmarked(ms: &[u128]) -> Vec<(u128, Option<String>)> {
        ms.iter().map(|ms| (*ms, None)).collect()
    }

    impl CloudApi for FakeApi {
        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_invoke(&self, request: InvokeRequest) -> Result<InvokeOutput, BoxError> {
            self.record(FakeCall::Invoke);
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
            self.record(FakeCall::GetExecution);
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
            self.record(FakeCall::History);
            self.history_calls.lock().expect("lock").push(HistoryCall {
                execution_arn: execution_arn.to_owned(),
                marker: marker.map(ToOwned::to_owned),
            });
            self.history_times
                .lock()
                .expect("lock")
                .push(tokio::time::Instant::now());
            let scripted = self.history.lock().expect("lock").pop_front();
            let (events, next_marker) = match scripted {
                Some(page) => page?,
                None => match self
                    .repeated_history
                    .lock()
                    .expect("lock")
                    .clone()
                    .expect("a scripted history page")
                {
                    Ok(events) => (events, None),
                    Err(text) => return Err(fake_error(text)),
                },
            };
            Ok(GetHistoryOutput::builder()
                .set_events(Some(events))
                .set_next_marker(next_marker)
                .build()
                .expect("history output"))
        }

        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_callback_success(
            &self,
            callback_id: &str,
            result: Vec<u8>,
        ) -> Result<(), BoxError> {
            self.record(FakeCall::CallbackSuccess(callback_id.to_owned(), result));
            Ok(())
        }

        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_callback_failure(
            &self,
            callback_id: &str,
            error: ErrorObject,
        ) -> Result<(), BoxError> {
            self.record(FakeCall::CallbackFailure(callback_id.to_owned(), error));
            Ok(())
        }

        #[expect(clippy::unused_async_trait_impl)] // reason: the fake answers at once
        async fn send_callback_heartbeat(&self, callback_id: &str) -> Result<(), BoxError> {
            self.record(FakeCall::Heartbeat(callback_id.to_owned()));
            Ok(())
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
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(WireExecutionStatus::TimedOut, None))),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::TimedOut);
        assert!(result.is_failure());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_invoke_error_is_aws() {
        let api = Arc::new(FakeApi::new().with_invoke(Err(fake_error("fake invoke"))));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake invoke");
        assert_eq!(api.invokes().len(), 1);
        assert_eq!(api.execution_call_count(), 0);
        assert_eq!(api.history_calls(), Vec::<HistoryCall>::new());
    }

    /// Every `GetDurableExecution` attempt fails: the poll makes three
    /// attempts, 1000 ms and then 1450 ms apart, and the run returns the
    /// third error.
    #[tokio::test(start_paused = true)]
    async fn cloud_get_execution_error_is_aws() {
        let start = tokio::time::Instant::now();
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Err(fake_error("fake get 1")))
                .with_execution(Err(fake_error("fake get 2")))
                .with_execution(Err(fake_error("fake get 3"))),
        );
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake get 3");
        assert_eq!(api.execution_arns(), vec![ARN.to_owned(); 3]);
        assert_eq!(api.execution_ms(start), [1000, 2000, 3450]);
        assert_eq!(api.history_calls(), Vec::<HistoryCall>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_get_execution_succeeds_on_the_third_attempt() {
        let start = tokio::time::Instant::now();
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Err(fake_error("fake get 1")))
                .with_execution(Err(fake_error("fake get 2")))
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("5"),
                ))),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect("runner");
        assert!(result.is_success());
        assert_eq!(result.output(), Some(&5));
        assert_eq!(api.execution_ms(start), [1000, 2000, 3450]);
        assert_eq!(api.history_ms(start), unmarked(&[3450]));
    }

    /// Every history attempt fails: the page request makes three attempts,
    /// and the run returns the third error.
    #[tokio::test(start_paused = true)]
    async fn cloud_get_history_error_is_aws() {
        let start = tokio::time::Instant::now();
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("5"),
                )))
                .with_history_pages(vec![
                    Err(fake_error("fake history 1")),
                    Err(fake_error("fake history 2")),
                    Err(fake_error("fake history 3")),
                ]),
        );
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake history 3");
        assert_eq!(api.execution_ms(start), [1000]);
        assert_eq!(api.history_ms(start), unmarked(&[1000, 2000, 3450]));
    }

    /// The page after a marker starts one poll interval after the marked
    /// page, and its retries repeat the marker after their retry delays.
    #[tokio::test(start_paused = true)]
    async fn cloud_history_retry_repeats_the_marker() {
        let start = tokio::time::Instant::now();
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("5"),
                )))
                .with_history_pages(vec![
                    Ok((Vec::new(), Some("m1".to_owned()))),
                    Err(fake_error("fake m1 1")),
                    Err(fake_error("fake m1 2")),
                    Ok((Vec::new(), None)),
                ]),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect("runner");
        assert!(result.is_success());
        assert_eq!(api.execution_ms(start), [1000]);
        let m1 = Some("m1".to_owned());
        assert_eq!(
            api.history_ms(start),
            [
                (1000, None),
                (2000, m1.clone()),
                (3000, m1.clone()),
                (4450, m1)
            ]
        );
    }

    /// A status poll that retries counts once against the poll budget.
    #[tokio::test(start_paused = true)]
    async fn cloud_retried_poll_counts_once() {
        let start = tokio::time::Instant::now();
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Err(fake_error("fake get 1")))
                .with_execution(Err(fake_error("fake get 2")))
                .with_execution(Ok(execution_output(WireExecutionStatus::Running, None)))
                .with_execution(Ok(execution_output(WireExecutionStatus::Running, None))),
        );
        let result = CloudRunner::new("f")
            .max_poll_attempts(2)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Running);
        assert_eq!(api.execution_ms(start), [1000, 2000, 3450, 4450]);
        assert_eq!(api.history_ms(start), unmarked(&[4450]));
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(
            result.error().and_then(crate::WireError::error_message),
            Some("execution fake-execution-arn still running after 2 polls")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_missing_execution_arn_is_runner_error() {
        let api = Arc::new(FakeApi::new().with_invoke(Ok(invoke_output(None))));
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
            let api = Arc::new(
                FakeApi::new()
                    .with_invoke(Ok(invoke))
                    // The GetDurableExecution result differs from the history's,
                    // so the assertion below shows the history wins.
                    .with_execution(Ok(execution_output(
                        WireExecutionStatus::Succeeded,
                        Some("7"),
                    )))
                    .with_history(Ok(history)),
            );
            let result = CloudRunner::new("f")
                .invocation_type(invocation_type)
                .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
            let api = Arc::new(
                FakeApi::new()
                    .with_execution(Ok(execution))
                    .with_history(Ok(vec![event.event_id(1).build()])),
            );
            let result = CloudRunner::new("f")
                .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(WireExecutionStatus::Running, None)))
                .with_execution(Ok(execution_output(WireExecutionStatus::Running, None)))
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("1"),
                ))),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(api.with_execution(Ok(execution_output(
            WireExecutionStatus::Succeeded,
            Some("1"),
        ))));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        assert_eq!(api.execution_call_count(), 201);
        assert_eq!(api.history_calls(), one_history_call());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_configured_poll_limit_ends_running() {
        let api = Arc::new(FakeApi::new());
        let result = CloudRunner::new("f")
            .max_poll_attempts(3)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(
            FakeApi::new()
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
                ]),
        );
        let result = CloudRunner::new("my-fn")
            .qualifier("live")
            .invocation_type(InvocationType::Event)
            .run_with::<_, u32>(Arc::clone(&api), b"{\"x\":1}".to_vec())
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
            .run::<u32>(Unserializable)
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
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("{\"a\":1}"),
                )))
                .with_history(Ok(vec![succeeded])),
        );
        let err = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(FakeApi::new().with_execution(Ok(execution_output(
            WireExecutionStatus::Succeeded,
            Some("9"),
        ))));
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(
            FakeApi::new()
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
                ])),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, Option<u32>>(Arc::clone(&api), Vec::new())
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
            let api = Arc::new(
                FakeApi::new()
                    .with_execution(Ok(execution))
                    .with_history(Ok(history)),
            );
            let result = CloudRunner::new("f")
                .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(FakeApi::new().with_history(Ok(vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .event_id(1)
                .id("s")
                .step_started_details(StepStartedDetails::builder().build())
                .build(),
        ])));
        let result = CloudRunner::new("f")
            .max_poll_attempts(2)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(
            FakeApi::new()
                .with_history_pages(Vec::new())
                .repeat_history(Err("fake history")),
        );
        let err = CloudRunner::new("f")
            .max_poll_attempts(1)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
            .await
            .expect_err("a runner error");
        assert_aws(&err, "fake history");
        assert_eq!(api.execution_call_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_poll_limit_of_zero_counts_as_one() {
        let api = Arc::new(FakeApi::new());
        let result = CloudRunner::new("f")
            .max_poll_attempts(0)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
            .run::<u32>(serde_json::json!({ "x": 1 }))
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
            .run::<u32>(())
            .await
            .expect_err("a runner error");
        assert!(matches!(err, TestRunnerError::Aws(_)), "{err:?}");
        assert!(std::error::Error::source(&err).is_some());
        assert_eq!(invoke.num_calls(), 1);
    }

    /// A `GetDurableExecution` failure of the `LambdaClient` implementation
    /// of [`CloudApi`] is `TestRunnerError::Aws` after three attempts, and
    /// its source is the SDK error of that call.
    #[tokio::test(start_paused = true)]
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
            .times(3)
            .build();
        let history = mock!(aws_sdk_lambda::Client::get_durable_execution_history)
            .then_output(|| GetHistoryOutput::builder().build().expect("history output"));
        let client = mock_client!(
            aws_sdk_lambda,
            RuleMode::MatchAny,
            [&invoke, &execution, &history]
        );
        let err = CloudRunner::new("f")
            .lambda_client(client)
            .run::<u32>(())
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
        assert_eq!(execution.num_calls(), 3);
        assert_eq!(history.num_calls(), 0);
    }

    /// A `GetDurableExecutionHistory` failure of the `LambdaClient`
    /// implementation of [`CloudApi`] is `TestRunnerError::Aws` after three
    /// attempts, and its source is the SDK error of that call.
    #[tokio::test(start_paused = true)]
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
            .times(3)
            .build();
        let client = mock_client!(
            aws_sdk_lambda,
            RuleMode::MatchAny,
            [&invoke, &execution, &history]
        );
        let err = CloudRunner::new("f")
            .lambda_client(client)
            .run::<u32>(())
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
        assert_eq!(history.num_calls(), 3);
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
                .wait_for_callback::<String>(|_step_ctx, _callback_id| async { Ok(()) })
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
                    .retry_strategy(|_err, _attempt| {
                        RetryDecision::retry(std::time::Duration::from_secs(30))
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
        let mut op = new_started_op("x".to_owned(), RunnerKind::Step, None, None, 1);
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
            let mut op = new_started_op("x".to_owned(), RunnerKind::Step, None, None, 0);
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
                    ctx.invoke::<serde_json::Value>("target", serde_json::json!(0))
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
        let op = TestOperation::started("x".to_owned(), RunnerKind::Step);
        let only_start = TestOperation {
            start_time: Some(std::time::UNIX_EPOCH),
            ..TestOperation::started("y".to_owned(), RunnerKind::Step)
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
            ..TestOperation::started("x".to_owned(), RunnerKind::Step)
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
            ..TestOperation::started("x".to_owned(), RunnerKind::Step)
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
            ..TestOperation::started("x".to_owned(), RunnerKind::Step)
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
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("1"),
                )))
                .with_history(Ok(events.clone())),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("1"),
                )))
                .with_history_pages(vec![
                    Ok((page(0..2), Some("m1".to_owned()))),
                    Ok((page(2..4), Some("m2".to_owned()))),
                    Ok((page(4..5), None)),
                ]),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("1"),
                )))
                .with_history(Ok(events)),
        );
        let result = CloudRunner::new("f")
            .run_with::<_, u32>(Arc::clone(&api), Vec::new())
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

    /// The message the SDK records for [`bad_no`]: the wrapped error's
    /// message, without the type name, which the record carries as its
    /// type.
    const BAD_NO_MESSAGE: &str = "no";

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
                            .heartbeat_timeout(std::time::Duration::from_secs(10))
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
                            .retry_strategy(|_err, _attempt| {
                                RetryDecision::retry(std::time::Duration::from_secs(5))
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
                                    Ok(ctx.invoke::<i32>("target", 5).name("call").await?)
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
                                    Ok(ctx.invoke::<i32>("target", 5).name("call").await?)
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
            EventRow {
                row: 17,
                setup: || handler_row(poll_handler),
                checks: vec![EventCheck {
                    event_type: EventType::StepSucceeded,
                    owner: Owner::Op("poll"),
                    expected: |_| json!({ "result": payload("1"), "retry_details": retry(1, Some(1)) }),
                }],
            },
        ]
    }

    /// A `wait_for_condition` named `poll` whose check adds 1 to a state
    /// that starts at 0, and whose strategy continues with a 1 s delay
    /// while the state is below 2.
    async fn poll_handler((): (), ctx: DurableContext) -> Result<i32, BoxError> {
        use crate::builders::wait_for_condition::WaitDecision;
        Ok(ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
            .wait_strategy_fn(|state: i32, _attempt| {
                if state < 2 {
                    WaitDecision::continue_with(std::time::Duration::from_secs(1))
                } else {
                    WaitDecision::complete()
                }
            })
            .name("poll")
            .await?)
    }

    /// A `StepSucceeded` that carries a retry delay, as the service records
    /// a `wait_for_condition` check that continued, sets the next attempt
    /// time, and a later `StepSucceeded` without a delay keeps it.
    #[tokio::test]
    async fn fold_history_reads_the_delay_of_a_step_succeeded() {
        use aws_sdk_lambda::types::{
            EventResult, RetryDetails, StepStartedDetails, StepSucceededDetails,
        };
        let started = |id: i32, at: i64| {
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(id)
                .event_timestamp(DateTime::from_secs(at))
                .step_started_details(StepStartedDetails::builder().build())
                .build()
        };
        let succeeded = |id: i32, at: i64, result: &str, attempt: i32, delay: Option<i32>| {
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(id)
                .event_timestamp(DateTime::from_secs(at))
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload(result).build())
                        .retry_details(
                            RetryDetails::builder()
                                .current_attempt(attempt)
                                .set_next_attempt_delay_seconds(delay)
                                .build(),
                        )
                        .build(),
                )
                .build()
        };
        let ops = fold_history(&[
            started(1, 100),
            succeeded(2, 100, "1", 1, Some(1)),
            started(3, 101),
            succeeded(4, 101, "2", 2, None),
        ]);
        let op = ops.first().expect("one operation");
        let step = step_details(op).expect("step");
        assert_eq!(
            (
                op.status(),
                op.result(),
                step.attempt(),
                step.next_attempt_time()
            ),
            (
                OperationStatus::Succeeded,
                Some("2"),
                1,
                Some(std::time::UNIX_EPOCH + std::time::Duration::from_secs(101)),
            )
        );

        // `CloudRunner`'s fold of `LocalRunner`'s history reports the next
        // attempt time `LocalRunner` reports.
        let result = LocalRunner::new()
            .run(poll_handler, ())
            .await
            .expect("runner");
        let local = result.operation("poll").expect("the condition");
        let folded = fold_history(result.history_events());
        let cloud = folded
            .iter()
            .find(|op| op.name() == Some("poll"))
            .expect("the folded condition");
        let local_time = step_details(local).expect("step").next_attempt_time();
        assert!(local_time.is_some());
        assert_eq!(
            step_details(cloud).expect("step").next_attempt_time(),
            local_time
        );
    }

    /// Each cause of rule 5 records its event, with the operation's
    /// identity and exactly the detail fields of rule 4.
    #[tokio::test]
    async fn local_events_cover_every_cause_and_detail() {
        let rows = event_rows();
        assert_eq!(rows.len(), 17);
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

    /// The kind of an operation error, as text, for a handler to record.
    fn child_kind(err: &crate::OperationError) -> String {
        match err.kind() {
            crate::OperationErrorKind::ChildContext(child) => {
                format!("ChildContext({:?})", child.kind())
            }
            other => format!("{other:?}"),
        }
    }

    /// A virtual child `v` inside child `outer` writes no record: its step
    /// names `outer` as its parent, and replays after the wait without
    /// running again.
    #[tokio::test]
    async fn virtual_child_writes_no_context_record() {
        let runs = Arc::new(AtomicU32::new(0));
        let handler_runs = Arc::clone(&runs);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let runs = Arc::clone(&handler_runs);
                    async move {
                        let value = ctx
                            .run_in_child_context(move |outer| async move {
                                let value = outer
                                    .run_in_child_context(move |v| async move {
                                        Ok(v.step(move |_| async move {
                                            runs.fetch_add(1, Ordering::SeqCst);
                                            Ok(7_u32)
                                        })
                                        .name("inner")
                                        .await?)
                                    })
                                    .name("v")
                                    .virtual_context()
                                    .await?;
                                outer
                                    .wait(std::time::Duration::from_secs(1))
                                    .name("pause")
                                    .await?;
                                Ok(value)
                            })
                            .name("outer")
                            .await?;
                        Ok::<_, BoxError>(value)
                    }
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&7));
        assert!(result.operation("v").is_none(), "{result}");
        let outer = result.operation("outer").expect("outer");
        let inner = result.operation("inner").expect("inner");
        assert_eq!(inner.parent_id(), Some(outer.id()));
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    /// A virtual child whose body fails returns `ChildContext(ChildFailed)`
    /// and writes no CONTEXT record.
    #[tokio::test]
    async fn virtual_child_failure_is_child_failed() {
        let kind: Arc<Mutex<Option<String>>> = Arc::default();
        let handler_kind = Arc::clone(&kind);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let kind = Arc::clone(&handler_kind);
                    async move {
                        let err = ctx
                            .run_in_child_context(|_v| async {
                                Err::<u32, BoxError>("nope".into())
                            })
                            .name("v")
                            .virtual_context()
                            .await
                            .expect_err("the body fails");
                        *kind.lock().expect("lock") = Some(child_kind(&err));
                        Ok::<_, BoxError>(())
                    }
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            kind.lock().expect("lock").as_deref(),
            Some("ChildContext(ChildFailed)")
        );
        assert!(
            result
                .operations()
                .iter()
                .all(|op| op.kind() != OperationKind::Context),
            "{result}"
        );
    }

    /// A virtual child's step and wait replay: the step body runs once
    /// over the two invocations the wait causes.
    #[tokio::test]
    async fn virtual_child_replays_its_operations() {
        let runs = Arc::new(AtomicU32::new(0));
        let handler_runs = Arc::clone(&runs);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let runs = Arc::clone(&handler_runs);
                    async move {
                        let value = ctx
                            .run_in_child_context(move |child| async move {
                                let value = child
                                    .step(move |_| async move {
                                        runs.fetch_add(1, Ordering::SeqCst);
                                        Ok(3_u32)
                                    })
                                    .name("s")
                                    .await?;
                                child
                                    .wait(std::time::Duration::from_secs(1))
                                    .name("w")
                                    .await?;
                                Ok(value)
                            })
                            .name("v")
                            .virtual_context()
                            .await?;
                        Ok::<_, BoxError>(value)
                    }
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&3));
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    /// A summary set on a virtual child is never called, even for an
    /// oversized result.
    #[tokio::test]
    async fn virtual_child_never_calls_summary() {
        let calls = Arc::new(AtomicU32::new(0));
        let handler_calls = Arc::clone(&calls);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let calls = Arc::clone(&handler_calls);
                    async move {
                        let big = ctx
                            .run_in_child_context(|_v| async { Ok("x".repeat(300 * 1024)) })
                            .name("v")
                            .virtual_context()
                            .summary(move |_s: &String| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                "unused".to_owned()
                            })
                            .await?;
                        Ok::<_, BoxError>(big.len())
                    }
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&307_200));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(result.operations().is_empty(), "{result}");
    }

    /// Runs, inside child `outer`, a virtual `with_retry` block named
    /// `name` whose attempt 1 fails and whose later attempts succeed, with
    /// a strategy that stops at attempt `stop_at`. Returns the run and the
    /// kind of the block's error, when it failed, with whether its source
    /// is the error of attempt 2.
    async fn virtual_with_retry_run(
        name: Option<&'static str>,
        fail_until: u32,
    ) -> (TestResult<u32>, Option<(String, bool)>) {
        let failure: Arc<Mutex<Option<(String, bool)>>> = Arc::default();
        let handler_failure = Arc::clone(&failure);
        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let failure = Arc::clone(&handler_failure);
                    async move {
                        let value = ctx
                            .run_in_child_context(move |outer| async move {
                                let block = outer
                                    .with_retry(move |_attempt_ctx, attempt| async move {
                                        if attempt <= fail_until {
                                            return Err::<u32, BoxError>(
                                                format!("fail on attempt {attempt}").into(),
                                            );
                                        }
                                        Ok(attempt)
                                    })
                                    .virtual_context()
                                    .retry_strategy(|_err, attempt| {
                                        if attempt >= 2 {
                                            RetryDecision::Stop
                                        } else {
                                            RetryDecision::retry(std::time::Duration::from_secs(1))
                                        }
                                    });
                                let block = match name {
                                    Some(name) => block.name(name),
                                    None => block,
                                };
                                match block.await {
                                    Ok(value) => Ok(value),
                                    Err(err) => {
                                        let last_is_attempt_2 = std::error::Error::source(&err)
                                            .and_then(std::error::Error::source)
                                            .and_then(|s| s.downcast_ref::<crate::OperationError>())
                                            .is_some_and(|last| {
                                                format!("{last:#}").contains("fail on attempt 2")
                                            });
                                        *failure.lock().expect("lock") =
                                            Some((child_kind(&err), last_is_attempt_2));
                                        Ok(0)
                                    }
                                }
                            })
                            .name("outer")
                            .await?;
                        Ok::<_, BoxError>(value)
                    }
                },
                (),
            )
            .await
            .expect("runner");
        let failure = failure.lock().expect("lock").clone();
        (result, failure)
    }

    /// An unnamed virtual `with_retry` block writes no record at its slot:
    /// its attempts name `outer` as their parent, and its wait is unnamed.
    #[tokio::test]
    async fn virtual_with_retry_writes_no_block_record() {
        let (result, failure) = virtual_with_retry_run(None, 1).await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(failure, None);
        assert_eq!(result.output(), Some(&2));
        let outer = result.operation("outer").expect("outer");
        let attempt = result.operation("attempt-1").expect("attempt-1");
        assert_eq!(attempt.parent_id(), Some(outer.id()));
        assert!(result.operation("attempt-2").is_some(), "{result}");
        let contexts: Vec<_> = result
            .operations()
            .iter()
            .filter(|op| op.kind() == OperationKind::Context)
            .filter_map(TestOperation::name)
            .collect();
        assert_eq!(contexts, ["outer", "attempt-1", "attempt-2"], "{result}");
        let waits: Vec<_> = result
            .operations()
            .iter()
            .filter(|op| op.kind() == OperationKind::Wait)
            .collect();
        assert_eq!(waits.len(), 1, "{result}");
        assert!(waits.iter().all(|w| w.name().is_none()), "{result}");
        assert!(
            waits.iter().all(|w| w.parent_id() == Some(outer.id())),
            "{result}"
        );
    }

    /// An exhausted virtual block fails with `ChildContext(ChildFailed)`
    /// whose source is the error of the last attempt, attempt 2.
    #[tokio::test]
    async fn virtual_with_retry_exhaustion_kind() {
        let (result, failure) = virtual_with_retry_run(None, 2).await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            failure,
            Some(("ChildContext(ChildFailed)".to_owned(), true)),
            "{result}"
        );
    }

    /// A virtual block named `b` names its attempts `b-attempt-N` and its
    /// waits `b-backoff-N`, under `outer`.
    #[tokio::test]
    async fn named_virtual_with_retry_prefixes_children() {
        let (result, failure) = virtual_with_retry_run(Some("b"), 1).await;
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(failure, None);
        let outer = result.operation("outer").expect("outer");
        let attempt = result.operation("b-attempt-1").expect("b-attempt-1");
        assert_eq!(attempt.parent_id(), Some(outer.id()));
        assert!(result.operation("b-attempt-2").is_some(), "{result}");
        assert!(result.operation("b-backoff-1").is_some(), "{result}");
        assert!(result.operation("b").is_none(), "{result}");
    }

    /// `.spawn()` of a virtual child and of a virtual `with_retry` block
    /// runs them, and writes no record for either.
    #[tokio::test]
    async fn spawned_virtual_child_and_block_write_no_record() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let child = ctx
                        .run_in_child_context(|v| async move {
                            Ok(v.step(|_| async { Ok(2_u32) }).name("s").await?)
                        })
                        .name("v")
                        .virtual_context()
                        .spawn();
                    let block = ctx
                        .with_retry(|_attempt_ctx, _attempt| async { Ok(3_u32) })
                        .name("b")
                        .virtual_context()
                        .spawn();
                    Ok::<_, BoxError>(child.await? + block.await?)
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&5));
        assert!(result.operation("v").is_none(), "{result}");
        assert!(result.operation("b").is_none(), "{result}");
        assert_eq!(
            result.operation("s").and_then(TestOperation::parent_id),
            None
        );
        assert_eq!(
            result
                .operation("b-attempt-1")
                .and_then(TestOperation::parent_id),
            None
        );
    }

    // ---- Live control: RunHandle and OperationHandle (LocalRunner) ----

    /// The pause-resume handler: step `reserve-stock`, the
    /// `wait_for_callback` `manager-approval`, and step `ship-order`. The
    /// first step takes 200 ms, so a test can pause while it runs.
    async fn pause_resume_handler(
        event: serde_json::Value,
        ctx: DurableContext,
    ) -> Result<serde_json::Value, BoxError> {
        let order_id = event
            .get("orderId")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("order-1")
            .to_owned();
        let id = order_id.clone();
        let reservation = ctx
            .step(move |_| async move {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                Ok(format!("reserved-{id}"))
            })
            .name("reserve-stock")
            .await?;
        let approval = ctx
            .wait_for_callback::<String>(|_step_ctx, _callback_id| async { Ok(()) })
            .name("manager-approval")
            .await?;
        let shipment = ctx
            .step(move |_| async move { Ok(format!("shipped-{order_id}")) })
            .name("ship-order")
            .await?;
        Ok(serde_json::json!({
            "reservation": reservation,
            "approval": approval,
            "shipment": shipment,
        }))
    }

    /// The output of the pause-resume handler for `order`.
    fn pause_resume_output(order: &str) -> serde_json::Value {
        serde_json::json!({
            "reservation": format!("reserved-{order}"),
            "approval": "approved",
            "shipment": format!("shipped-{order}"),
        })
    }

    /// The state cell of a run with no backend records.
    fn new_live() -> Arc<LiveState> {
        Arc::new(LiveState::new(BackendState::new(QueuedReplies::default())))
    }

    /// Asserts that `run` does not finish within 50 ms.
    async fn assert_run_waits<O: DeserializeOwned + Send + 'static>(run: &mut RunHandle<O>) {
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), run)
                .await
                .is_err(),
            "the run waits"
        );
    }

    /// The handler of the race tests: a callback named `approval` races a
    /// one-hour wait, and the handler returns which won.
    async fn callback_races_one_hour((): (), ctx: DurableContext) -> Result<String, BoxError> {
        let callback = ctx
            .run_in_child_context(|child| async move {
                let cb = child.create_callback::<String>().name("approval").await?;
                cb.result().await?;
                Ok("callback".to_owned())
            })
            .name("callback-branch")
            .future();
        let wait = ctx
            .run_in_child_context(|child| async move {
                child
                    .wait(std::time::Duration::from_secs(3600))
                    .name("one-hour")
                    .await?;
                Ok("wait".to_owned())
            })
            .name("wait-branch")
            .future();
        Ok(ctx.race([callback, wait]).await?)
    }

    /// A `wait_for_callback` named `approve` with a submitter that does
    /// nothing.
    async fn wait_for_approve((): (), ctx: DurableContext) -> Result<u32, BoxError> {
        Ok(ctx
            .wait_for_callback::<u32>(|_step_ctx, _callback_id| async { Ok(()) })
            .name("approve")
            .await?)
    }

    /// Steps named by `names`, in order, each returning its index.
    fn steps_named(
        names: &'static [&'static str],
    ) -> impl Fn((), DurableContext) -> Pin<Box<dyn Future<Output = Result<(), BoxError>> + Send>>
    + Send
    + Sync
    + 'static {
        move |(), ctx| {
            Box::pin(async move {
                for (index, name) in names.iter().enumerate() {
                    ctx.step(move |_| async move { Ok(index) })
                        .name(*name)
                        .await?;
                }
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn reply_chosen_from_callback_id() {
        let run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<String>().name("approval").await?;
                Ok::<_, BoxError>(cb.result().await?)
            },
            (),
        );
        let approval = run.operation("approval");
        let op = approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        let id = match op.details() {
            TestOperationDetails::Callback(details) => details.callback_id(),
            _ => None,
        }
        .expect("a callback ID")
        .to_owned();
        approval
            .send_callback(CallbackReply::raw(format!("\"{id}\"")))
            .await
            .expect("sent");
        let result = run.await.expect("runner");
        assert_eq!(result.output(), Some(&id));
    }

    #[tokio::test]
    async fn wait_for_callback_submitted_then_reply() {
        let run = LocalRunner::new().run(wait_for_approve, ());
        let approve = run.operation("approve");
        let op = approve
            .wait_for_data(WaitingStatus::Submitted)
            .await
            .expect("submitted");
        assert_eq!(op.kind(), OperationKind::Context);
        assert_eq!(op.status(), OperationStatus::Started);
        approve.send_callback(success(&5_u32)).await.expect("sent");
        let result = run.await.expect("runner");
        assert_eq!(result.output(), Some(&5));
    }

    #[tokio::test]
    async fn submitted_differs_from_completed() {
        let run = LocalRunner::new().run(wait_for_approve, ());
        let approve = run.operation("approve");
        let submitted = approve
            .wait_for_data(WaitingStatus::Submitted)
            .await
            .expect("submitted");
        approve.send_callback(success(&5_u32)).await.expect("sent");
        let completed = approve
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect("completed");
        assert_eq!(submitted.status(), OperationStatus::Started);
        assert_eq!(completed.status(), OperationStatus::Succeeded);
        let result = run.await.expect("runner");
        assert_eq!(
            result.operation("approve").map(TestOperation::status),
            Some(OperationStatus::Succeeded)
        );
    }

    #[tokio::test]
    async fn reply_after_wait_for_data_beats_a_one_hour_wait() {
        let run = LocalRunner::new().run(callback_races_one_hour, ());
        let approval = run.operation("approval");
        approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        approval
            .send_callback(success(&"yes".to_owned()))
            .await
            .expect("sent");
        let result = run.await.expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("callback"));
    }

    #[tokio::test]
    async fn wait_wins_without_a_reply() {
        let result = LocalRunner::new()
            .run(callback_races_one_hour, ())
            .await
            .expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("wait"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn checkpoint_held_while_no_runner_future_waits() {
        let run = LocalRunner::new().run(steps_named(&["a", "b"]), ());
        let a = run.operation("a");
        let b = run.operation("b");
        a.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            a.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started)
        );
        assert!(b.snapshot().is_none());
        let result = run.await.expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
    }

    #[tokio::test]
    async fn pause_mid_step_keeps_the_checkpoint() {
        let mut run = LocalRunner::new().run(
            pause_resume_handler,
            serde_json::json!({"orderId": "order-7"}),
        );
        let reserve = run.operation("reserve-stock");
        let approval = run.operation("manager-approval");
        reserve
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        run.pause().await.expect("paused");
        assert_eq!(
            reserve.snapshot().map(|op| op.status()),
            Some(OperationStatus::Succeeded)
        );
        assert!(approval.snapshot().is_none());
        assert_run_waits(&mut run).await;
        run.resume().await.expect("resumed");
        approval
            .wait_for_data(WaitingStatus::Submitted)
            .await
            .expect("submitted");
        approval
            .send_callback(success(&"approved".to_owned()))
            .await
            .expect("sent");
        let result = run.await.expect("runner");
        assert_eq!(result.output(), Some(&pause_resume_output("order-7")));
    }

    #[tokio::test]
    async fn pause_while_idle_holds_the_next_invocation() {
        let mut run = LocalRunner::new().run(
            pause_resume_handler,
            serde_json::json!({"orderId": "order-8"}),
        );
        let approval = run.operation("manager-approval");
        approval
            .wait_for_data(WaitingStatus::Submitted)
            .await
            .expect("submitted");
        run.pause().await.expect("paused");
        approval
            .send_callback(success(&"approved".to_owned()))
            .await
            .expect("sent");
        assert_run_waits(&mut run).await;
        assert!(run.operation("ship-order").snapshot().is_none());
        run.resume().await.expect("resumed");
        let result = run.await.expect("runner");
        assert_eq!(result.output(), Some(&pause_resume_output("order-8")));
    }

    /// `handler` wrapped so that each invocation increments `counter` on
    /// entry.
    fn counting<E: 'static, O, Fut>(
        counter: &Arc<AtomicU32>,
        handler: fn(E, DurableContext) -> Fut,
    ) -> impl Fn(E, DurableContext) -> Fut + Send + Sync + 'static
    where
        Fut: Future<Output = Result<O, BoxError>> + 'static,
    {
        let counter = Arc::clone(counter);
        move |event, ctx| {
            counter.fetch_add(1, Ordering::SeqCst);
            handler(event, ctx)
        }
    }

    async fn cool_down((): (), ctx: DurableContext) -> Result<String, BoxError> {
        ctx.wait(std::time::Duration::from_secs(3600))
            .name("cool-down")
            .await?;
        Ok("done".to_owned())
    }

    #[tokio::test]
    async fn pause_holds_invocation_after_wait_completes() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut run = LocalRunner::new().run(counting(&counter, cool_down), ());
        let wait = run.operation("cool-down");
        wait.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        run.pause().await.expect("paused");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        wait.wait_for_data(WaitingStatus::Completed)
            .await
            .expect("completed while paused");
        assert_run_waits(&mut run).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        let op = wait.snapshot().expect("the wait");
        assert_eq!(op.status(), OperationStatus::Succeeded);
        assert_eq!(
            op.end_time(),
            op.start_time()
                .map(|t| t + std::time::Duration::from_secs(3600))
        );
        run.resume().await.expect("resumed");
        let result = run.await.expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("done"));
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(result.invocations().len(), 2);
    }

    async fn two_waits((): (), ctx: DurableContext) -> Result<String, BoxError> {
        let branches: Vec<crate::Branch<()>> = vec![
            crate::Branch::new("one-hour-branch", |child| async move {
                child
                    .wait(std::time::Duration::from_secs(3600))
                    .name("one hour")
                    .await?;
                Ok(())
            }),
            crate::Branch::new("two-hours-branch", |child| async move {
                child
                    .wait(std::time::Duration::from_secs(7200))
                    .name("two hours")
                    .await?;
                Ok(())
            }),
        ];
        ctx.parallel(branches).await?;
        Ok("done".to_owned())
    }

    #[tokio::test]
    async fn pause_keeps_completing_waits_in_due_order() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut run = LocalRunner::new().run(counting(&counter, two_waits), ());
        let one = run.operation("one hour");
        let two = run.operation("two hours");
        two.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        run.pause().await.expect("paused");
        let (one_done, two_done) = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            tokio::join!(
                one.wait_for_data(WaitingStatus::Completed),
                two.wait_for_data(WaitingStatus::Completed)
            )
        })
        .await
        .expect("both waits complete while paused");
        let one_done = one_done.expect("one hour");
        let two_done = two_done.expect("two hours");
        let after = |op: &TestOperation, seconds| {
            op.start_time()
                .map(|t| t + std::time::Duration::from_secs(seconds))
        };
        assert_eq!(one_done.status(), OperationStatus::Succeeded);
        assert_eq!(one_done.end_time(), after(&one_done, 3600));
        assert_eq!(two_done.status(), OperationStatus::Succeeded);
        assert_eq!(two_done.end_time(), after(&two_done, 7200));
        assert!(one_done.end_time() < two_done.end_time());
        assert_run_waits(&mut run).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        run.resume().await.expect("resumed");
        let result = run.await.expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("done"));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    /// A parallel of a step that takes 200 ms and a one-hour wait.
    async fn slow_step_beside_a_long_wait((): (), ctx: DurableContext) -> Result<String, BoxError> {
        let branches: Vec<crate::Branch<()>> = vec![
            crate::Branch::new("slow-branch", |child| async move {
                child
                    .step(|_| async {
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                        Ok(1_u32)
                    })
                    .name("slow")
                    .await?;
                Ok(())
            }),
            crate::Branch::new("wait-branch", |child| async move {
                child
                    .wait(std::time::Duration::from_secs(3600))
                    .name("long")
                    .await?;
                Ok(())
            }),
        ];
        ctx.parallel(branches).await?;
        Ok("done".to_owned())
    }

    /// A pause that waits for the invocation to end stops counting as a
    /// waiting runner future when the invocation ends, so the paused run
    /// advances nothing until another runner future waits.
    #[tokio::test]
    async fn pause_mid_step_does_not_advance_a_started_wait() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut run = LocalRunner::new().run(counting(&counter, slow_step_beside_a_long_wait), ());
        let slow = run.operation("slow");
        let long = run.operation("long");
        let (slow_started, long_started) = tokio::join!(
            slow.wait_for_data(WaitingStatus::Started),
            long.wait_for_data(WaitingStatus::Started)
        );
        slow_started.expect("the step started");
        long_started.expect("the wait started");
        run.pause().await.expect("paused");
        assert_eq!(run.shared.live.lock().local.phase, LocalPhase::Idle);
        assert_eq!(
            long.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started),
            "the wait did not advance before the pause returned"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            long.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started),
            "nothing advances while no runner future waits"
        );
        let done = long
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect("a waiting runner future lets the wait complete");
        assert_eq!(done.status(), OperationStatus::Succeeded);
        assert_run_waits(&mut run).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        run.resume().await.expect("resumed");
        let result = run.await.expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("done"));
    }

    /// A driver that never finishes and counts its polls.
    fn counting_driver(polls: Arc<AtomicU32>) -> DriverFuture {
        Box::pin(std::future::poll_fn(move |_| {
            polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::Pending
        }))
    }

    /// A `LocalRunner` run on the state cell `live` whose driver never
    /// finishes and counts its polls.
    fn counting_run(live: &Arc<LiveState>) -> (Arc<RunCore>, Arc<AtomicU32>) {
        let polls = Arc::new(AtomicU32::new(0));
        let shared = RunCore::new(
            Arc::clone(live),
            Engine::Local,
            counting_driver(Arc::clone(&polls)),
        );
        (shared, polls)
    }

    /// Adds a control of `kind` to `live` and marks it pending, as a polled
    /// runner future that returned `Pending`.
    fn pending_control(live: &LiveState, kind: ControlKind) -> u64 {
        let mut state = live.lock();
        let id = state.add_control(kind);
        assert!(state.register(id, std::task::Waker::noop()).is_some());
        id
    }

    /// A `wait_for_data` control for the operation `id` and `status`.
    fn data_control(id: &str, status: WaitingStatus) -> ControlKind {
        ControlKind::Data {
            selector: Selector::Id(id.to_owned()),
            status,
        }
    }

    /// A backend of a top-level run, with its records in the run state.
    fn live_backend() -> (Arc<Backend>, Arc<LiveState>) {
        let live = new_live();
        let backend = Backend::on(StateCell::Live(Arc::clone(&live)), None);
        (Arc::new(backend), live)
    }

    /// Stores a started wait `id` that ends at `due` seconds.
    fn store_wait(backend: &Backend, id: &str, due: u64) {
        backend.lock().ops.push(StoredOp {
            due_at: Some(due),
            ..new_started_op(id.to_owned(), RunnerKind::Wait, None, None, 0)
        });
    }

    /// Stores a pending callback `id`, with a heartbeat timeout.
    fn store_callback(backend: &Backend, id: &str) {
        backend.lock().ops.push(StoredOp {
            callback_id: Some(format!("cb-{id}")),
            heartbeat_seconds: Some(60),
            ..new_started_op(id.to_owned(), RunnerKind::Callback, None, None, 0)
        });
    }

    fn stored_status(backend: &Backend, id: &str) -> OperationStatus {
        backend
            .lock()
            .ops
            .iter()
            .find(|op| op.id == id)
            .map(|op| op.status)
            .expect("stored")
    }

    /// Pause and admission are transitions on one state, in either order.
    /// A pause first keeps the invocation from starting. An admission
    /// first makes the pause wait for that invocation's end, and the
    /// invocation's end makes the pause ready.
    #[test]
    fn pause_and_admission_are_ordered_transitions() {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());

        // Pause first.
        let live = new_live();
        let (shared, _polls) = counting_run(&live);
        let run = RunHandle::<u32>::from_shared(Arc::clone(&shared));
        pending_control(&live, ControlKind::Run);
        let mut pause = std::pin::pin!(run.pause());
        assert!(matches!(
            pause.as_mut().poll(&mut cx),
            std::task::Poll::Ready(Ok(()))
        ));
        assert!(!live.lock().admit_invocation(), "a paused run admits none");
        assert_eq!(live.lock().local.phase, LocalPhase::Idle);

        // Admission first.
        let live = new_live();
        let (shared, _polls) = counting_run(&live);
        let run = RunHandle::<u32>::from_shared(Arc::clone(&shared));
        pending_control(&live, ControlKind::Run);
        assert!(live.lock().admit_invocation());
        assert!(!live.lock().admit_invocation(), "one invocation at a time");
        let mut pause = std::pin::pin!(run.pause());
        assert!(pause.as_mut().poll(&mut cx).is_pending());
        assert_eq!(
            live.lock().local.phase,
            LocalPhase::Invoking {
                tokenless_requested: true
            },
            "the pause asks for a tokenless answer in its own transition"
        );
        assert!(live.lock().take_tokenless());
        assert!(!live.lock().take_tokenless(), "taken once");
        wake(live.lock().end_invocation(false));
        assert!(matches!(
            pause.as_mut().poll(&mut cx),
            std::task::Poll::Ready(Ok(()))
        ));
        assert!(!live.lock().admit_invocation(), "still paused");
    }

    /// An invocation's end makes every pending pause ready in the same
    /// transition, so none of them lets the run move when the driver decides
    /// what to advance next.
    #[test]
    fn invocation_end_makes_every_pause_ready_before_the_decision() {
        let (backend, live) = live_backend();
        store_wait(&backend, "w", 3600);
        let first = pending_control(&live, ControlKind::Pause);
        let second = pending_control(&live, ControlKind::Pause);
        assert!(live.lock().admit_invocation());
        backend.finish_invocation(false, 0, "r", None);
        {
            let state = live.lock();
            assert!(state.pause_done(first) && state.pause_done(second));
            assert_eq!(
                state
                    .local
                    .records
                    .events
                    .last()
                    .and_then(Event::event_type),
                Some(&EventType::InvocationCompleted),
                "the end event is recorded in the same transition"
            );
        }
        assert!(!live.lock().hold_open());
        let mut start = false;
        assert_eq!(
            between_decision(&mut live.lock(), &mut start, true).0,
            Decided::Wait
        );
        assert_eq!(backend.now(), 0, "no control waits: the clock holds");
    }

    /// A control that is ready, or no longer exists, is not registered and
    /// does not poll the driver. A control that is not ready is registered
    /// and polls it. After a run failure no control registers, so none
    /// polls the driver.
    #[test]
    fn a_ready_control_neither_registers_nor_polls() {
        let (backend, live) = live_backend();
        let (shared, polls) = counting_run(&live);
        backend.lock().ops.push(new_started_op(
            "x".to_owned(),
            RunnerKind::Step,
            None,
            None,
            0,
        ));
        let ready = live
            .lock()
            .add_control(data_control("x", WaitingStatus::Started));
        let waiting = live
            .lock()
            .add_control(data_control("ghost", WaitingStatus::Started));
        assert!(live.lock().recheck().is_empty(), "nothing was pending yet");
        shared.drive(ready, std::task::Waker::noop());
        assert!(!live.lock().hold_open());
        assert_eq!(polls.load(Ordering::SeqCst), 0);

        let pause = live.lock().add_control(ControlKind::Pause);
        drop(live.lock().end_invocation(false));
        shared.drive(pause, std::task::Waker::noop());
        assert!(!live.lock().hold_open(), "a ready pause does not wait");
        assert_eq!(polls.load(Ordering::SeqCst), 0);

        shared.leave(waiting);
        shared.drive(waiting, std::task::Waker::noop());
        assert_eq!(polls.load(Ordering::SeqCst), 0, "a dropped control");

        let other = live
            .lock()
            .add_control(data_control("ghost", WaitingStatus::Started));
        shared.drive(other, std::task::Waker::noop());
        assert!(live.lock().hold_open());
        assert_eq!(polls.load(Ordering::SeqCst), 1);

        live.lock().failure = Some(TestRunnerError::NoExecutionArn);
        let late = live.lock().add_control(ControlKind::Run);
        shared.drive(late, std::task::Waker::noop());
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }

    /// Dropping a pending control of each `LocalRunner` kind closes the hold
    /// it opened, so it no longer lets the run move. A pending `CloudRunner`
    /// control never opens the hold.
    #[test]
    fn a_dropped_control_no_longer_lets_the_run_move() {
        for kind in [
            ControlKind::Run,
            data_control("x", WaitingStatus::Started),
            ControlKind::Pause,
        ] {
            let live = new_live();
            let (shared, _polls) = counting_run(&live);
            let id = pending_control(&live, kind);
            assert!(live.lock().hold_open());
            shared.leave(id);
            assert!(!live.lock().hold_open());
            assert!(!live.lock().admit_invocation());
        }
        let live = new_live();
        pending_control(&live, ControlKind::Cloud);
        assert!(!live.lock().hold_open());
    }

    /// The hold check and the clock advance are one transition: without a
    /// pending control the clock does not move, with one it moves in the
    /// same step.
    #[test]
    fn the_clock_advances_only_in_the_step_that_sees_a_waiter() {
        let (backend, live) = live_backend();
        store_wait(&backend, "w", 3600);
        let mut start = false;
        let decide = |start: &mut bool| between_decision(&mut live.lock(), start, true).0;
        assert_eq!(decide(&mut start), Decided::Wait);
        assert_eq!(backend.now(), 0, "no waiter: the clock holds");
        assert_eq!(stored_status(&backend, "w"), OperationStatus::Started);
        let run = pending_control(&live, ControlKind::Run);
        assert_eq!(decide(&mut start), Decided::Step(BetweenStep::Start));
        assert_eq!(backend.now(), 3600);
        assert_eq!(stored_status(&backend, "w"), OperationStatus::Succeeded);
        // Paused: an advance decides again at once, and with nothing to
        // advance the driver waits for `resume`. Without a waiter, nothing
        // advances.
        store_wait(&backend, "w2", 7200);
        live.lock().local.paused = true;
        live.lock().controls.remove(&run);
        assert_eq!(decide(&mut start), Decided::Wait);
        assert_eq!(backend.now(), 3600);
        pending_control(&live, ControlKind::Run);
        assert_eq!(decide(&mut start), Decided::Again);
        assert!(start, "the advance starts the held invocation");
        assert_eq!(backend.now(), 7200);
        assert_eq!(decide(&mut start), Decided::Wait, "wait for resume");
        live.lock().local.paused = false;
        assert_eq!(decide(&mut start), Decided::Step(BetweenStep::Start));
        // Nothing can advance: the first decision asks for the pending
        // invokes, the next one ends the run and closes it.
        let mut start = false;
        assert_eq!(
            between_decision(&mut live.lock(), &mut start, false).0,
            Decided::Step(BetweenStep::RunInvokes)
        );
        assert_eq!(live.lock().local.phase, LocalPhase::Idle);
        assert_eq!(decide(&mut start), Decided::Step(BetweenStep::Stuck));
        assert_eq!(live.lock().local.phase, LocalPhase::Closed);
    }

    /// A paused advance that makes the only pending `wait_for_data` ready
    /// closes the hold in the same transition: the second due wait and the
    /// clock stay until another control waits.
    #[test]
    fn a_waiter_made_ready_by_an_advance_stops_the_next_advance() {
        let (backend, live) = live_backend();
        store_wait(&backend, "one", 3600);
        store_wait(&backend, "two", 7200);
        live.lock().local.paused = true;
        pending_control(&live, data_control("one", WaitingStatus::Completed));
        let mut start = false;
        let (decided, wakers) = between_decision(&mut live.lock(), &mut start, true);
        assert_eq!(decided, Decided::Again);
        assert_eq!(wakers.len(), 1, "the waiter is woken with its snapshot");
        assert!(!live.lock().hold_open());
        assert_eq!(
            between_decision(&mut live.lock(), &mut start, true).0,
            Decided::Wait
        );
        assert_eq!(backend.now(), 3600);
        assert_eq!(stored_status(&backend, "two"), OperationStatus::Started);
        pending_control(&live, ControlKind::Run);
        assert_eq!(
            between_decision(&mut live.lock(), &mut start, true).0,
            Decided::Again
        );
        assert_eq!(backend.now(), 7200);
    }

    /// A live reply is one transition with its record change. A reply
    /// before the driver's decision starts the next invocation without
    /// moving the clock; a reply after the driver closed the run is refused
    /// and changes nothing.
    #[test]
    fn a_live_reply_starts_the_next_invocation_or_is_refused() {
        let reply_c = |live: &LiveState| {
            reply_live(
                &mut live.lock(),
                &Selector::Id("c".to_owned()),
                CallbackReply::raw("1"),
            )
        };
        let (backend, live) = live_backend();
        store_callback(&backend, "c");
        store_wait(&backend, "w", 3600);
        pending_control(&live, ControlKind::Run);
        reply_c(&live).expect("pending");
        let mut start = false;
        assert_eq!(
            between_decision(&mut live.lock(), &mut start, true).0,
            Decided::Step(BetweenStep::Start)
        );
        assert_eq!(backend.now(), 0, "the reply starts it, not the clock");
        assert!(
            !live.lock().local.invocation_due,
            "the decision took the reply"
        );

        let (backend, live) = live_backend();
        store_callback(&backend, "c");
        pending_control(&live, ControlKind::Run);
        let mut start = false;
        assert_eq!(
            between_decision(&mut live.lock(), &mut start, true).0,
            Decided::Step(BetweenStep::Stuck)
        );
        let err = reply_c(&live).expect_err("the run ends");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        assert_eq!(stored_status(&backend, "c"), OperationStatus::Started);
        assert_eq!(backend.lock().updated_ids, Vec::<String>::new());
        assert!(!live.lock().local.invocation_due);
    }

    /// A terminal envelope and the invocation budget close the run too, and
    /// a closed run refuses a heartbeat to a callback that is still
    /// `Started`, as it refuses a reply.
    #[test]
    fn the_run_closes_at_its_end() {
        let heartbeat_c =
            |live: &LiveState| check_heartbeat(&mut live.lock(), &Selector::Id("c".to_owned()));
        let (backend, live) = live_backend();
        store_callback(&backend, "c");
        heartbeat_c(&live).expect("an open run");
        backend.finish_invocation(false, 0, "r", None);
        assert_eq!(
            live.lock().local.phase,
            LocalPhase::Idle,
            "a PENDING invocation keeps it open"
        );
        backend.finish_invocation(true, 0, "r", None);
        assert_eq!(live.lock().local.phase, LocalPhase::Closed);
        assert!(matches!(
            heartbeat_c(&live),
            Err(TestRunnerError::CallbackNotPending)
        ));
        backend.finish_invocation(false, 0, "r", None);
        assert_eq!(live.lock().local.phase, LocalPhase::Closed, "closed stays");

        let (backend, live) = live_backend();
        store_callback(&backend, "c");
        backend.close();
        assert_eq!(live.lock().local.phase, LocalPhase::Closed);
        assert!(matches!(
            heartbeat_c(&live),
            Err(TestRunnerError::CallbackNotPending)
        ));
        assert_eq!(stored_status(&backend, "c"), OperationStatus::Started);
    }

    /// A commit decides against the failure recorded before it, and only
    /// that one: an outcome committed first stays, a later commit returns
    /// the first failure, and a second failure does not replace it.
    #[test]
    fn a_commit_orders_an_outcome_against_the_first_failure() {
        let live = new_live();
        assert_eq!(live.lock().commit(Ok(1_u32)).expect("no failure"), 1);
        let (winner, _) = live.lock().record_failure(TestRunnerError::NoExecutionArn);
        assert!(matches!(winner, TestRunnerError::NoExecutionArn));
        let (winner, _) = live
            .lock()
            .record_failure(TestRunnerError::OperationNotFound);
        assert!(matches!(winner, TestRunnerError::NoExecutionArn));
        assert!(matches!(
            live.lock().commit(Ok(2_u32)),
            Err(TestRunnerError::NoExecutionArn)
        ));
    }

    /// A handler that creates the callback `c`, then waits an hour, and never
    /// awaits the callback's result.
    async fn callback_then_wait((): (), ctx: DurableContext) -> Result<u32, BoxError> {
        ctx.create_callback::<u32>().name("c").await?;
        ctx.wait(std::time::Duration::from_secs(3600))
            .name("w")
            .await?;
        Ok(1)
    }

    /// A reply to a callback that is still `Started` after the run decided to
    /// end is refused: after a terminal envelope, and after the invocation
    /// budget ran out.
    #[tokio::test]
    async fn a_reply_after_the_run_ended_is_refused() {
        for runner in [LocalRunner::new(), LocalRunner::new().max_invocations(1)] {
            let mut run = runner.run(callback_then_wait, ());
            let c = run.operation("c");
            let result = (&mut run).await.expect("runner");
            assert_eq!(
                c.snapshot().map(|op| op.status()),
                Some(OperationStatus::Started)
            );
            let err = c
                .send_callback(CallbackReply::success(&1_u32).expect("encodes"))
                .await
                .expect_err("the run ended");
            assert!(
                matches!(err, TestRunnerError::CallbackNotPending),
                "{:?}: {err:?}",
                result.status()
            );
        }
    }

    /// A handler that waits for the callback `c` and returns its result.
    async fn await_callback_c((): (), ctx: DurableContext) -> Result<u32, BoxError> {
        let cb = ctx.create_callback::<u32>().name("c").await?;
        Ok(cb.result().await?)
    }

    #[tokio::test]
    async fn pause_and_resume_are_idempotent_but_reject_after_run() {
        let mut run = LocalRunner::new().run(steps_named(&["s"]), ());
        run.pause().await.expect("pause");
        run.pause().await.expect("pause again");
        run.resume().await.expect("resume");
        run.resume().await.expect("resume again");
        let result = (&mut run).await.expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        let pause = run.pause().await.expect_err("after the run");
        let resume = run.resume().await.expect_err("after the run");
        assert!(
            matches!(pause, TestRunnerError::RunFinished("pause")),
            "{pause:?}"
        );
        assert!(
            matches!(resume, TestRunnerError::RunFinished("resume")),
            "{resume:?}"
        );
        assert_eq!(
            pause.to_string(),
            "durable: test runner: pause needs a run in progress; call it before awaiting the \
             RunHandle"
        );
        assert_eq!(
            resume.to_string(),
            "durable: test runner: resume needs a run in progress; call it before awaiting the \
             RunHandle"
        );
    }

    #[tokio::test]
    async fn send_callback_errors() {
        let run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) }).name("s").await?;
                let cb = ctx.create_callback::<u32>().name("c").await?;
                Ok::<_, BoxError>(cb.result().await?)
            },
            (),
        );
        let c = run.operation("c");
        c.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        let step = run
            .operation("s")
            .send_callback(success(&1_u32))
            .await
            .expect_err("a step");
        assert!(matches!(step, TestRunnerError::NotACallback), "{step:?}");
        c.send_callback(success(&1_u32)).await.expect("sent");
        let again = c
            .send_callback(success(&1_u32))
            .await
            .expect_err("answered");
        assert!(
            matches!(again, TestRunnerError::CallbackNotPending),
            "{again:?}"
        );
        let ghost = run
            .operation("ghost")
            .send_callback(success(&1_u32))
            .await
            .expect_err("no operation");
        assert!(
            matches!(ghost, TestRunnerError::OperationNotFound),
            "{ghost:?}"
        );
        run.await.expect("runner");
    }

    #[tokio::test]
    async fn heartbeat_rules() {
        let run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                let h = ctx
                    .create_callback::<u32>()
                    .name("h")
                    .heartbeat_timeout(std::time::Duration::from_secs(60))
                    .await?;
                let n = ctx.create_callback::<u32>().name("n").await?;
                Ok::<_, BoxError>(h.result().await? + n.result().await?)
            },
            (),
        );
        let h = run.operation("h");
        let n = run.operation("n");
        n.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        h.send_callback_heartbeat()
            .await
            .expect("h has a heartbeat");
        assert_eq!(
            h.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started)
        );
        let err = n.send_callback_heartbeat().await.expect_err("no heartbeat");
        assert!(
            matches!(err, TestRunnerError::NoHeartbeatTimeout),
            "{err:?}"
        );
        h.send_callback(success(&1_u32)).await.expect("sent");
        n.send_callback(success(&2_u32)).await.expect("sent");
        let result = run.await.expect("runner");
        assert_eq!(result.output(), Some(&3));
    }

    #[tokio::test]
    async fn heartbeat_to_an_answered_callback_is_not_pending() {
        let run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                let h = ctx
                    .create_callback::<u32>()
                    .name("h")
                    .heartbeat_timeout(std::time::Duration::from_secs(60))
                    .await?;
                Ok::<_, BoxError>(h.result().await?)
            },
            (),
        );
        let h = run.operation("h");
        h.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        h.send_callback(success(&1_u32)).await.expect("sent");
        let err = h.send_callback_heartbeat().await.expect_err("answered");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        run.await.expect("runner");
    }

    #[tokio::test]
    async fn wait_for_data_not_reached() {
        let run = LocalRunner::new().run(steps_named(&["s"]), ());
        let ghost = run.operation("ghost");
        let (result, waited) = tokio::join!(run, ghost.wait_for_data(WaitingStatus::Started));
        result.expect("runner");
        let err = waited.expect_err("never reached");
        assert!(
            matches!(
                err,
                TestRunnerError::OperationNotReached(WaitingStatus::Started)
            ),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn late_unmatched_waiter_returns_not_reached() {
        let mut run = LocalRunner::new().run(steps_named(&["s"]), ());
        (&mut run).await.expect("runner");
        let err = run
            .operation("ghost")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect_err("never reached");
        assert!(
            matches!(
                err,
                TestRunnerError::OperationNotReached(WaitingStatus::Started)
            ),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn wait_for_callback_started_by_callback_child() {
        let run = LocalRunner::new().run(wait_for_approve, ());
        let context = run
            .operation("approve")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        assert_eq!(context.kind(), OperationKind::Context);
        assert_eq!(context.status(), OperationStatus::Started);
        let callback = run
            .operation_by_index(1)
            .snapshot()
            .expect("the callback child");
        assert_eq!(callback.kind(), OperationKind::Callback);
        assert_eq!(callback.parent_id(), Some(context.id()));
        run.operation("approve")
            .send_callback(success(&1_u32))
            .await
            .expect("sent");
        run.await.expect("runner");
    }

    #[test]
    fn wait_for_callback_without_callback_child_is_not_pending() {
        let context = TestOperation {
            sub_type: Some(crate::callback::WFCB_SUB_TYPE.to_owned()),
            name: Some("approve".to_owned()),
            ..TestOperation::started("ctx".to_owned(), RunnerKind::Context)
        };
        let plain = TestOperation {
            name: Some("plain".to_owned()),
            ..TestOperation::started("p".to_owned(), RunnerKind::Context)
        };
        let ops = [context, plain];
        let err = callback_target(&Selector::Name("approve".to_owned(), 0), &ops)
            .expect_err("no callback child");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        let err = callback_target(&Selector::Name("plain".to_owned(), 0), &ops)
            .expect_err("a plain context");
        assert!(matches!(err, TestRunnerError::NotACallback), "{err:?}");
    }

    #[tokio::test]
    async fn send_callback_records_updated_id_and_history_event() {
        let run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<u32>().name("c").await?;
                Ok::<_, BoxError>(cb.result().await?)
            },
            (),
        );
        let c = run.operation("c");
        let op = c
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        c.send_callback(success(&1_u32)).await.expect("sent");
        let updated = run.shared.live.lock().local.records.updated_ids.clone();
        assert!(updated.contains(&op.id().to_owned()), "{updated:?}");
        let result = run.await.expect("runner");
        assert_eq!(result.output(), Some(&1));
        let events = result.history_events();
        let position = |event_type: EventType| {
            events
                .iter()
                .position(|e| e.event_type() == Some(&event_type) && e.id() == Some(op.id()))
                .expect("the event")
        };
        let started = position(EventType::CallbackStarted);
        let succeeded = position(EventType::CallbackSucceeded);
        assert!(started < succeeded);
        assert!(
            events
                .iter()
                .skip(succeeded)
                .any(|e| e.event_type() == Some(&EventType::InvocationCompleted)),
            "an invocation completes after the reply"
        );
    }

    /// Sets a flag when dropped.
    struct DropFlag(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn dropping_last_handle_drops_driver() {
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = DropFlag(Arc::clone(&dropped));
        let driver: DriverFuture = Box::pin(async move {
            let _flag = flag;
            std::future::pending::<()>().await;
            Err(TestRunnerError::NoExecutionArn)
        });
        let run = RunHandle::<u32>::from_shared(RunCore::new(new_live(), Engine::Local, driver));
        let op = run.operation("x");
        drop(run);
        assert!(!dropped.load(Ordering::SeqCst));
        drop(op);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn wait_for_data_after_run_returns_final_state() {
        let mut run = LocalRunner::new().run(steps_named(&["s"]), ());
        let s = run.operation("s");
        (&mut run).await.expect("runner");
        let op = s
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect("completed");
        assert_eq!(op.status(), OperationStatus::Succeeded);
    }

    #[tokio::test]
    async fn operation_handle_outlives_run_handle() {
        let run = LocalRunner::new().run(steps_named(&["a", "b"]), ());
        let b = run.operation("b");
        drop(run);
        let op = b
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect("completed");
        assert_eq!(op.status(), OperationStatus::Succeeded);
    }

    #[test]
    fn handle_auto_traits() {
        fn assert_send_unpin<T: Send + Unpin>() {}
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_unpin::<RunHandle<u32>>();
        assert_send_sync::<OperationHandle>();
    }

    #[test]
    fn runner_error_texts() {
        let cases = [
            (
                TestRunnerError::Unsupported("pause"),
                "pause is not supported by this runner",
            ),
            (
                TestRunnerError::OperationNotFound,
                "no operation matches this handle",
            ),
            (
                TestRunnerError::OperationNotReached(WaitingStatus::Submitted),
                "the operation did not reach Submitted before the execution ended",
            ),
            (
                TestRunnerError::NotACallback,
                "the operation is not a callback or a wait_for_callback context",
            ),
            (
                TestRunnerError::CallbackNotPending,
                "the callback is not waiting for a reply",
            ),
            (
                TestRunnerError::NoHeartbeatTimeout,
                "the callback has no heartbeat timeout",
            ),
            (
                TestRunnerError::RunFinished("pause"),
                "pause needs a run in progress; call it before awaiting the RunHandle",
            ),
        ];
        for (err, text) in cases {
            assert_eq!(err.to_string(), format!("durable: test runner: {text}"));
            assert!(std::error::Error::source(&err).is_none(), "{err:?}");
        }
    }

    #[tokio::test]
    async fn operation_selectors() {
        let mut run = LocalRunner::new().run(steps_named(&["x", "y", "x"]), ());
        let first = run.operation("x");
        let second = run.operation_by_index(1);
        let third = run.operation_by_name_and_index("x", 1);
        let result = (&mut run).await.expect("runner");
        let ids: Vec<&str> = result.operations().iter().map(TestOperation::id).collect();
        assert_eq!(ids.len(), 3, "{ids:?}");
        let (first_id, second_id, third_id) = (
            ids.first().expect("first"),
            ids.get(1).expect("second"),
            ids.get(2).expect("third"),
        );
        let id_of = |handle: &OperationHandle| handle.snapshot().map(|op| op.id().to_owned());
        assert_eq!(id_of(&first).as_deref(), Some(*first_id));
        assert_eq!(id_of(&second).as_deref(), Some(*second_id));
        assert_eq!(
            second
                .snapshot()
                .and_then(|op| op.name().map(ToOwned::to_owned)),
            Some("y".to_owned())
        );
        assert_eq!(id_of(&third).as_deref(), Some(*third_id));
        assert_eq!(
            id_of(&run.operation_by_id(third_id)).as_deref(),
            Some(*third_id)
        );
        assert!(run.operation_by_id("ghost").snapshot().is_none());
        assert!(run.operation_by_index(3).snapshot().is_none());
    }

    #[tokio::test]
    async fn run_error_reaches_every_runner_future() {
        let run = LocalRunner::new().run(
            |_event: Unserializable, _ctx: DurableContext| async move { Ok::<_, BoxError>(()) },
            Unserializable,
        );
        let a = run.operation("a");
        let b = run.operation("b");
        let (paused, a_done, b_done) = tokio::join!(
            run.pause(),
            a.wait_for_data(WaitingStatus::Started),
            b.wait_for_data(WaitingStatus::Completed)
        );
        let resumed = run.resume().await;
        let sent = a.send_callback(success(&1_u32)).await;
        let beat = a.send_callback_heartbeat().await;
        let result = run.await;
        let texts = [
            result.expect_err("run").to_string(),
            paused.expect_err("pause").to_string(),
            a_done.expect_err("a").to_string(),
            b_done.expect_err("b").to_string(),
            resumed.expect_err("resume").to_string(),
            sent.expect_err("send").to_string(),
            beat.expect_err("heartbeat").to_string(),
        ];
        for text in &texts {
            assert!(
                text.starts_with("durable: test runner: event serialization failed"),
                "{text}"
            );
            assert_eq!(text, &texts[0]);
        }
    }

    #[tokio::test]
    async fn plain_run_await_returns_test_result() {
        let result = LocalRunner::new()
            .run(
                |_event: serde_json::Value, ctx: DurableContext| async move {
                    let name = ctx
                        .step(|_| async { Ok("world".to_owned()) })
                        .name("fetch-name")
                        .await?;
                    ctx.wait(std::time::Duration::from_secs(2))
                        .name("cooldown")
                        .await?;
                    let greeting = ctx
                        .step(move |_| async move { Ok(format!("hello, {name}")) })
                        .name("format")
                        .await?;
                    Ok::<_, BoxError>(greeting)
                },
                serde_json::json!({}),
            )
            .await
            .expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        assert_eq!(result.output().map(String::as_str), Some("hello, world"));
        let ops: Vec<(Option<&str>, OperationKind, OperationStatus)> = result
            .operations()
            .iter()
            .map(|op| (op.name(), op.kind(), op.status()))
            .collect();
        assert_eq!(
            ops,
            [
                (
                    Some("fetch-name"),
                    OperationKind::Step,
                    OperationStatus::Succeeded
                ),
                (
                    Some("cooldown"),
                    OperationKind::Wait,
                    OperationStatus::Succeeded
                ),
                (
                    Some("format"),
                    OperationKind::Step,
                    OperationStatus::Succeeded
                ),
            ]
        );
    }

    #[tokio::test]
    async fn run_handle_polled_after_completion_stays_pending() {
        let mut run = LocalRunner::new().run(steps_named(&["s"]), ());
        (&mut run).await.expect("runner");
        let poll = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(Pin::new(&mut run).poll(cx).is_pending())
        })
        .await;
        assert!(poll, "the result was already taken");
    }

    /// A `LocalRunner` run that completes the step `s` and then returns an
    /// output that does not decode. A wait pending when the run fails
    /// returns that error. A later wait returns `s` for every status `s`
    /// reached, and the error for an operation that never existed. The
    /// callback, heartbeat, `pause` and `resume` controls keep returning
    /// the error.
    #[tokio::test]
    async fn local_waits_after_run_failure_use_snapshot_first() {
        let mut run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) }).name("s").await?;
                Ok::<_, BoxError>(SerializedAs::Text)
            },
            (),
        );
        let s = run.operation("s");
        let ghost = run.operation("ghost");
        let (pending, end) = tokio::join!(ghost.wait_for_data(WaitingStatus::Started), &mut run);
        let err = end.expect_err("the output does not decode");
        assert!(
            matches!(err, TestRunnerError::OutputDeserialization(_)),
            "{err:?}"
        );
        assert_same_error(&pending.expect_err("the pending wait"), &err);
        for status in [
            WaitingStatus::Started,
            WaitingStatus::Submitted,
            WaitingStatus::Completed,
        ] {
            let op = s.wait_for_data(status).await.expect("stored data");
            assert_eq!(op.status(), OperationStatus::Succeeded);
            assert_eq!(op.name(), Some("s"));
        }
        assert_same_error(
            &ghost
                .wait_for_data(WaitingStatus::Started)
                .await
                .expect_err("never reached"),
            &err,
        );
        assert_same_error(
            &s.send_callback(CallbackReply::raw("1"))
                .await
                .expect_err("send_callback"),
            &err,
        );
        assert_same_error(
            &s.send_callback_heartbeat().await.expect_err("heartbeat"),
            &err,
        );
        assert_same_error(&run.pause().await.expect_err("pause"), &err);
        assert_same_error(&run.resume().await.expect_err("resume"), &err);
        assert_eq!(
            s.snapshot().map(|op| op.status()),
            Some(OperationStatus::Succeeded),
            "the failure keeps the records"
        );
    }

    /// An output that serializes as a string and deserializes only from a
    /// number.
    #[derive(Debug)]
    enum SerializedAs {
        Text,
    }

    impl Serialize for SerializedAs {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_str("text")
        }
    }

    impl<'de> Deserialize<'de> for SerializedAs {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            u32::deserialize(deserializer).map(|_| Self::Text)
        }
    }

    #[test]
    fn waiting_status_default_is_started() {
        assert_eq!(WaitingStatus::default(), WaitingStatus::Started);
    }

    #[test]
    fn status_matching_follows_js() {
        let op = |id: &str, kind, status| TestOperation {
            status,
            ..TestOperation::started(id.to_owned(), kind)
        };
        let step = op("s", RunnerKind::Step, OperationStatus::Pending);
        assert!(status_reached(&step, WaitingStatus::Started, &[]));
        assert!(!status_reached(&step, WaitingStatus::Submitted, &[]));
        for status in [
            OperationStatus::Succeeded,
            OperationStatus::Failed,
            OperationStatus::Cancelled,
            OperationStatus::TimedOut,
            OperationStatus::Stopped,
        ] {
            let done = op("s", RunnerKind::Step, status);
            assert!(status_reached(&done, WaitingStatus::Submitted, &[]));
            assert!(status_reached(&done, WaitingStatus::Completed, &[]));
        }
        let context = TestOperation {
            sub_type: Some(crate::callback::WFCB_SUB_TYPE.to_owned()),
            ..op("ctx", RunnerKind::Context, OperationStatus::Started)
        };
        let child = |id: &str, kind, status| TestOperation {
            parent_id: Some("ctx".to_owned()),
            ..op(id, kind, status)
        };
        let all = |ops: &[TestOperation]| {
            [
                WaitingStatus::Started,
                WaitingStatus::Submitted,
                WaitingStatus::Completed,
            ]
            .map(|status| status_reached(&context, status, ops))
        };
        // A wait child does not count.
        assert_eq!(
            all(&[child("w", RunnerKind::Wait, OperationStatus::Succeeded)]),
            [false, false, false]
        );
        assert_eq!(
            all(&[child("cb", RunnerKind::Callback, OperationStatus::Started)]),
            [true, false, false]
        );
        assert_eq!(
            all(&[
                child("cb", RunnerKind::Callback, OperationStatus::Started),
                child("sub", RunnerKind::Step, OperationStatus::Succeeded),
            ]),
            [true, true, false]
        );
        // A finished callback whose context has not finished yet is not
        // `Completed`: the context's own end comes after it.
        assert_eq!(
            all(&[child(
                "cb",
                RunnerKind::Callback,
                OperationStatus::Succeeded
            )]),
            [true, true, false]
        );
        let finished = |status| TestOperation {
            status,
            ..context.clone()
        };
        for status in [OperationStatus::Succeeded, OperationStatus::Failed] {
            let done = finished(status);
            let terminal_callback = [child(
                "cb",
                RunnerKind::Callback,
                OperationStatus::Succeeded,
            )];
            assert!(status_reached(
                &done,
                WaitingStatus::Completed,
                &terminal_callback
            ));
            // A finished context whose callback never ended, such as one
            // whose submitter failed, is not `Completed`, as in JS.
            let open_callback = [
                child("cb", RunnerKind::Callback, OperationStatus::Started),
                child("sub", RunnerKind::Step, OperationStatus::Failed),
            ];
            assert!(status_reached(
                &done,
                WaitingStatus::Submitted,
                &open_callback
            ));
            assert!(!status_reached(
                &done,
                WaitingStatus::Completed,
                &open_callback
            ));
        }
        // A plain child context's status is its own.
        let plain = op("p", RunnerKind::Context, OperationStatus::Started);
        assert!(status_reached(&plain, WaitingStatus::Started, &[]));
        assert!(!status_reached(&plain, WaitingStatus::Submitted, &[]));
    }

    // ---- Live control under CloudRunner ----

    /// A history in which the callback `c`, named `approval`, started with
    /// the callback ID `cb-1`.
    fn callback_started_history() -> Vec<Event> {
        vec![
            Event::builder()
                .event_type(EventType::CallbackStarted)
                .id("c")
                .name("approval")
                .event_id(1)
                .callback_started_details(
                    aws_sdk_lambda::types::CallbackStartedDetails::builder()
                        .callback_id("cb-1")
                        .build()
                        .expect("callback details"),
                )
                .build(),
        ]
    }

    /// A fake whose every history read returns `history`.
    fn fake_with_history(history: Vec<Event>) -> Arc<FakeApi> {
        Arc::new(
            FakeApi::new()
                .with_history_pages(Vec::new())
                .repeat_history(Ok(history)),
        )
    }

    /// A cloud runner with `InvocationType::Event`.
    fn event_runner() -> CloudRunner {
        CloudRunner::new("f").invocation_type(InvocationType::Event)
    }

    /// The number of full-history fetches in `calls` (every history is one
    /// page).
    fn history_fetches(calls: &[FakeCall]) -> usize {
        calls
            .iter()
            .filter(|call| **call == FakeCall::History)
            .count()
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_reply_after_wait_for_data() {
        let api = fake_with_history(callback_started_history());
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        let before = api.calls();
        assert_eq!(history_fetches(&before), 1, "{before:?}");
        approval
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect("sent");
        let calls = api.calls();
        let after = calls.get(before.len()..).expect("later calls");
        assert_eq!(
            after,
            [
                FakeCall::History,
                FakeCall::CallbackSuccess("cb-1".to_owned(), b"1".to_vec())
            ],
            "{calls:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_failure_sends_error_object() {
        let api = fake_with_history(callback_started_history());
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        run.operation("approval")
            .send_callback(CallbackReply::failure(
                CallbackFailure::new("T", "m")
                    .with_error_data("d")
                    .with_stack_trace(vec!["f".into()]),
            ))
            .await
            .expect("sent");
        let expected = ErrorObject::builder()
            .error_type("T")
            .error_message("m")
            .error_data("d")
            .stack_trace("f")
            .build();
        assert_eq!(
            api.calls().last(),
            Some(&FakeCall::CallbackFailure("cb-1".to_owned(), expected))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_timeout_reply_and_pause_are_unsupported() {
        let api = fake_with_history(callback_started_history());
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        for reply in [CallbackReply::timeout(), CallbackReply::heartbeat_timeout()] {
            let err = approval.send_callback(reply).await.expect_err("timeout");
            assert!(
                matches!(
                    err,
                    TestRunnerError::Unsupported(
                        "CallbackReply::timeout and heartbeat_timeout under CloudRunner"
                    )
                ),
                "{err:?}"
            );
        }
        for err in [
            run.pause().await.expect_err("pause"),
            run.resume().await.expect_err("resume"),
        ] {
            assert!(
                matches!(
                    err,
                    TestRunnerError::Unsupported("pause and resume under CloudRunner")
                ),
                "{err:?}"
            );
        }
        assert_eq!(api.calls(), Vec::<FakeCall>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_heartbeat_calls_service() {
        // Row 1: no poll limit.
        let api = fake_with_history(callback_started_history());
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        let before = api.calls().len();
        approval.send_callback_heartbeat().await.expect("heartbeat");
        let calls = api.calls();
        assert_eq!(
            calls.get(before..).expect("later calls"),
            [FakeCall::History, FakeCall::Heartbeat("cb-1".to_owned())],
            "{calls:?}"
        );
        drop(run);

        // Row 2: a budget of two polls.
        let api = fake_with_history(callback_started_history());
        let run = event_runner()
            .max_poll_attempts(2)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        approval.send_callback_heartbeat().await.expect("heartbeat");
        approval
            .send_callback_heartbeat()
            .await
            .expect("heartbeat from the last fold");
        let result = run.await.expect("runner");
        assert_eq!(result.status(), ExecutionStatus::Running);
        let heartbeat = FakeCall::Heartbeat("cb-1".to_owned());
        assert_eq!(
            api.calls(),
            [
                FakeCall::Invoke,
                FakeCall::History,
                FakeCall::History,
                heartbeat.clone(),
                heartbeat,
                FakeCall::History,
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_requires_event_invocation() {
        let api = fake_with_history(callback_started_history());
        let run = CloudRunner::new("f").run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let wait = approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect_err("wait_for_data");
        let send = approval
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect_err("send_callback");
        let heartbeat = approval
            .send_callback_heartbeat()
            .await
            .expect_err("heartbeat");
        let unsupported = |err: &TestRunnerError| match err {
            TestRunnerError::Unsupported(text) => Some(*text),
            _ => None,
        };
        assert_eq!(
            unsupported(&wait),
            Some("wait_for_data with InvocationType::RequestResponse (use InvocationType::Event)"),
            "{wait:?}"
        );
        assert_eq!(
            unsupported(&send),
            Some("send_callback with InvocationType::RequestResponse (use InvocationType::Event)"),
            "{send:?}"
        );
        assert_eq!(
            unsupported(&heartbeat),
            Some(
                "send_callback_heartbeat with InvocationType::RequestResponse \
                 (use InvocationType::Event)"
            ),
            "{heartbeat:?}"
        );
        assert_eq!(
            send.to_string(),
            "durable: test runner: send_callback with InvocationType::RequestResponse \
             (use InvocationType::Event) is not supported by this runner"
        );
        assert_eq!(api.calls(), Vec::<FakeCall>::new());
        drop(run);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_configured_poll_budget_is_shared() {
        let api = fake_with_history(callback_started_history());
        let run = event_runner()
            .max_poll_attempts(4)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let ghost = run.operation("ghost");
        let (result, waited) = tokio::join!(run, ghost.wait_for_data(WaitingStatus::Started));
        assert_eq!(result.expect("runner").status(), ExecutionStatus::Running);
        let err = waited.expect_err("never reached");
        assert!(
            matches!(
                err,
                TestRunnerError::OperationNotReached(WaitingStatus::Started)
            ),
            "{err:?}"
        );
        let calls = api.calls();
        let polls = calls
            .iter()
            .filter(|call| matches!(call, FakeCall::GetExecution | FakeCall::History))
            .count();
        // Four budgeted polls, plus the history read of the final result.
        assert_eq!(polls, 5, "{calls:?}");
        assert_eq!(calls.last(), Some(&FakeCall::History));
    }

    /// A waiter whose history read fails returns its own error, and the
    /// run goes on. A waiter pending when a status poll fails on its third
    /// attempt, and the `RunHandle`, return that run failure.
    #[tokio::test(start_paused = true)]
    async fn cloud_run_failure_is_not_replaced_by_a_control_failure() {
        let running = || Ok(execution_output(WireExecutionStatus::Running, None));
        let api = Arc::new(
            FakeApi::new()
                .with_execution(running())
                .with_execution(running())
                .with_execution(running())
                .with_execution(Err(fake_error("fake execution 1")))
                .with_execution(Err(fake_error("fake execution 2")))
                .with_execution(Err(fake_error("fake execution 3")))
                .with_history_pages(vec![
                    Err(fake_error("fake history 1")),
                    Err(fake_error("fake history 2")),
                    Err(fake_error("fake history 3")),
                ])
                .repeat_history(Ok(Vec::new())),
        );
        let start = tokio::time::Instant::now();
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let ghost = run.operation("ghost");
        let (end, (own, pending)) = tokio::join!(run, async {
            let own = approval.wait_for_data(WaitingStatus::Started).await;
            let pending = ghost.wait_for_data(WaitingStatus::Completed).await;
            (own, pending)
        });
        assert_aws(&own.expect_err("the waiter's read fails"), "fake history 3");
        let end = end.expect_err("the status poll fails");
        assert_aws(&end, "fake execution 3");
        let pending = pending.expect_err("the pending waiter gets the run failure");
        assert_aws(&pending, "fake execution 3");
        assert_eq!(pending.to_string(), end.to_string());
        assert_eq!(
            api.execution_ms(start),
            [1000, 2000, 3000, 4000, 5000, 6450]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_snapshot_is_last_fold() {
        let step = |event_type| {
            Event::builder()
                .event_type(event_type)
                .id("s")
                .name("s")
                .event_id(1)
                .build()
        };
        let api = Arc::new(
            FakeApi::new()
                .with_history_pages(vec![Ok((vec![step(EventType::StepStarted)], None))])
                .repeat_history(Ok(vec![step(EventType::StepSucceeded)])),
        );
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let s = run.operation("s");
        assert!(s.snapshot().is_none());
        let op = s
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        assert_eq!(op.status(), OperationStatus::Started);
        assert_eq!(
            s.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started)
        );
        drop(run);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_not_reached_on_terminal() {
        let mut history = callback_started_history();
        history.push(
            Event::builder()
                .event_type(EventType::ExecutionSucceeded)
                .event_id(2)
                .build(),
        );
        let api = fake_with_history(history);
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let err = run
            .operation("ghost")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect_err("never reached");
        assert!(
            matches!(
                err,
                TestRunnerError::OperationNotReached(WaitingStatus::Started)
            ),
            "{err:?}"
        );
        drop(run);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_polls_until_reached() {
        let api = Arc::new(
            FakeApi::new()
                .with_history_pages(vec![Ok((Vec::new(), None))])
                .repeat_history(Ok(callback_started_history())),
        );
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let op = run
            .operation("approval")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started on the second read");
        assert_eq!(op.id(), "c");
        assert_eq!(history_fetches(&api.calls()), 2);
        drop(run);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_after_the_run_reads_the_last_fold() {
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(WireExecutionStatus::Succeeded, None)))
                .with_history_pages(vec![Ok((callback_started_history(), None))]),
        );
        let mut run = event_runner().run_with::<_, ()>(Arc::clone(&api), Vec::new());
        (&mut run).await.expect("runner");
        let calls = api.calls().len();
        let op = run
            .operation("approval")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("in the final history");
        assert_eq!(op.id(), "c");
        assert_eq!(api.calls().len(), calls, "no further call");
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_send_callback_needs_a_started_callback() {
        let mut history = callback_started_history();
        history.push(
            Event::builder()
                .event_type(EventType::CallbackSucceeded)
                .id("c")
                .event_id(2)
                .build(),
        );
        let api = fake_with_history(history);
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let err = run
            .operation("approval")
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect_err("answered");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        drop(run);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_heartbeat_needs_a_callback_id() {
        let api = fake_with_history(vec![
            Event::builder()
                .event_type(EventType::CallbackStarted)
                .id("c")
                .name("approval")
                .event_id(1)
                .build(),
        ]);
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let err = run
            .operation("approval")
            .send_callback_heartbeat()
            .await
            .expect_err("no callback ID");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        drop(run);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_spent_budget_without_a_fold_is_not_found() {
        /// A fake whose `GetDurableExecution` never answers.
        struct HangingExecution(FakeApi);
        impl CloudApi for HangingExecution {
            async fn send_invoke(&self, request: InvokeRequest) -> Result<InvokeOutput, BoxError> {
                self.0.send_invoke(request).await
            }
            async fn send_get_durable_execution(
                &self,
                _execution_arn: &str,
            ) -> Result<GetExecutionOutput, BoxError> {
                std::future::pending().await
            }
            async fn send_get_durable_execution_history(
                &self,
                execution_arn: &str,
                marker: Option<&str>,
            ) -> Result<GetHistoryOutput, BoxError> {
                self.0
                    .send_get_durable_execution_history(execution_arn, marker)
                    .await
            }
            async fn send_callback_success(
                &self,
                id: &str,
                result: Vec<u8>,
            ) -> Result<(), BoxError> {
                self.0.send_callback_success(id, result).await
            }
            async fn send_callback_failure(
                &self,
                id: &str,
                error: ErrorObject,
            ) -> Result<(), BoxError> {
                self.0.send_callback_failure(id, error).await
            }
            async fn send_callback_heartbeat(&self, id: &str) -> Result<(), BoxError> {
                self.0.send_callback_heartbeat(id).await
            }
        }
        let api = Arc::new(HangingExecution(FakeApi::new()));
        let mut run = event_runner()
            .max_poll_attempts(1)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new());
        // The driver spends the budget of one poll on a `GetDurableExecution`
        // that never answers.
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(5), &mut run)
                .await
                .is_err()
        );
        let err = run
            .operation("approval")
            .send_callback_heartbeat()
            .await
            .expect_err("no history read yet");
        assert!(matches!(err, TestRunnerError::OperationNotFound), "{err:?}");
        assert_eq!(api.0.history_calls(), Vec::<HistoryCall>::new());
    }

    /// The SDK error a `GetDurableExecutionHistory` call returns for an
    /// execution the service does not show.
    fn history_not_found() -> BoxError {
        use aws_sdk_lambda::operation::get_durable_execution_history::GetDurableExecutionHistoryError;
        Box::new(aws_sdk_lambda::error::SdkError::service_error(
            GetDurableExecutionHistoryError::ResourceNotFoundException(
                aws_sdk_lambda::types::error::ResourceNotFoundException::builder()
                    .message("Durable Execution does not exist")
                    .build(),
            ),
            aws_sdk_lambda::config::http::HttpResponse::new(
                404.try_into().expect("a status code"),
                aws_smithy_types::body::SdkBody::empty(),
            ),
        ))
    }

    /// A fake whose history reads answer `ResourceNotFoundException` while
    /// `not_found` is above zero, then as `api` does. `reads` records the
    /// time of every history read.
    struct LateExecution {
        api: FakeApi,
        not_found: Mutex<u32>,
        reads: Mutex<Vec<tokio::time::Instant>>,
    }

    impl LateExecution {
        /// A fake whose first `not_found` history reads fail, and whose
        /// later reads return a started callback `approval`.
        fn new(not_found: u32) -> Arc<Self> {
            Arc::new(Self {
                api: FakeApi::new()
                    .with_history_pages(Vec::new())
                    .repeat_history(Ok(callback_started_history())),
                not_found: Mutex::new(not_found),
                reads: Mutex::new(Vec::new()),
            })
        }

        /// The milliseconds after `start` of every history read.
        fn read_ms(&self, start: tokio::time::Instant) -> Vec<u128> {
            self.reads
                .lock()
                .expect("lock")
                .iter()
                .map(|at| (*at - start).as_millis())
                .collect()
        }
    }

    /// Asserts `err` is the `Aws` error of [`history_not_found`].
    fn assert_history_not_found(err: &TestRunnerError) {
        use aws_sdk_lambda::operation::get_durable_execution_history::GetDurableExecutionHistoryError;
        assert!(matches!(err, TestRunnerError::Aws(_)), "{err:?}");
        let source = std::error::Error::source(err).expect("a source");
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
    }
    impl CloudApi for LateExecution {
        async fn send_invoke(&self, request: InvokeRequest) -> Result<InvokeOutput, BoxError> {
            self.api.send_invoke(request).await
        }
        async fn send_get_durable_execution(
            &self,
            execution_arn: &str,
        ) -> Result<GetExecutionOutput, BoxError> {
            self.api.send_get_durable_execution(execution_arn).await
        }
        async fn send_get_durable_execution_history(
            &self,
            execution_arn: &str,
            marker: Option<&str>,
        ) -> Result<GetHistoryOutput, BoxError> {
            self.reads
                .lock()
                .expect("lock")
                .push(tokio::time::Instant::now());
            {
                let mut not_found = self.not_found.lock().expect("lock");
                if *not_found > 0 {
                    *not_found -= 1;
                    return Err(history_not_found());
                }
            }
            self.api
                .send_get_durable_execution_history(execution_arn, marker)
                .await
        }
        async fn send_callback_success(&self, id: &str, payload: Vec<u8>) -> Result<(), BoxError> {
            self.api.send_callback_success(id, payload).await
        }
        async fn send_callback_failure(
            &self,
            id: &str,
            error: ErrorObject,
        ) -> Result<(), BoxError> {
            self.api.send_callback_failure(id, error).await
        }
        async fn send_callback_heartbeat(&self, id: &str) -> Result<(), BoxError> {
            self.api.send_callback_heartbeat(id).await
        }
    }

    /// A `ResourceNotFoundException` fails a history attempt as any error
    /// does, before and after the first successful read: a waiter retries it
    /// twice, and the third one is that waiter's error only. A later waiter
    /// still reads the history.
    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_retries_a_not_found() {
        let start = tokio::time::Instant::now();
        let api = LateExecution::new(2);
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let op = approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("the callback started");
        assert_eq!(op.kind(), OperationKind::Callback);
        assert_eq!(api.read_ms(start), [1000, 2000, 3450]);
        *api.not_found.lock().expect("lock") = 3;
        let err = approval
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect_err("the third not-found is an AWS error");
        assert_history_not_found(&err);
        assert_eq!(
            api.read_ms(start),
            [1000, 2000, 3450, 4450, 5450, 6900],
            "six requests in all"
        );
        let op = approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("the run goes on");
        assert_eq!(op.id(), "c");
        assert_eq!(
            api.read_ms(start),
            [1000, 2000, 3450, 4450, 5450, 6900, 7900]
        );
        assert!(run.shared.terminal_error().is_none());
        drop(run);
    }

    /// Three `ResourceNotFoundException` answers before the first read are
    /// the waiter's error, and the waiter reads no fourth answer.
    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_not_found_three_times_is_aws() {
        let start = tokio::time::Instant::now();
        let api = LateExecution::new(3);
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let err = run
            .operation("approval")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect_err("the third not-found is an AWS error");
        assert_history_not_found(&err);
        assert_eq!(api.read_ms(start), [1000, 2000, 3450]);
        drop(run);
    }

    /// A waiter whose every history attempt fails returns the third error.
    /// It is the waiter's only: the driver keeps its one-second status
    /// polls, and the run succeeds.
    #[tokio::test(start_paused = true)]
    async fn cloud_wait_for_data_history_error_is_aws() {
        let start = tokio::time::Instant::now();
        let running = || Ok(execution_output(WireExecutionStatus::Running, None));
        let api = Arc::new(
            FakeApi::new()
                .with_execution(running())
                .with_execution(running())
                .with_execution(running())
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("7"),
                )))
                .with_history_pages(vec![
                    Err(fake_error("fake history 1")),
                    Err(fake_error("fake history 2")),
                    Err(fake_error("fake history 3")),
                    Ok((Vec::new(), None)),
                ]),
        );
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let (result, waited) = tokio::join!(run, approval.wait_for_data(WaitingStatus::Started));
        let waited = waited.expect_err("the waiter fails");
        assert_aws(&waited, "fake history 3");
        let result = result.expect("the run succeeds");
        assert_eq!(result.output(), Some(&7));
        assert_eq!(api.execution_ms(start), [1000, 2000, 3000, 4000]);
        assert_eq!(api.history_ms(start), unmarked(&[1000, 2000, 3450, 4000]));
    }

    /// A fake whose callback calls fail with `fake success`, `fake failure`
    /// and `fake heartbeat`. With a barrier, each callback call waits at it
    /// before it fails, so two calls fail together. The counter counts the
    /// callback calls, and the inner fake records each one in its `calls`.
    struct FailingCallbacks(FakeApi, Option<Arc<tokio::sync::Barrier>>, AtomicU32);

    impl FailingCallbacks {
        async fn meet(&self, call: FakeCall) {
            self.0.record(call);
            self.2.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = &self.1 {
                barrier.wait().await;
            }
        }
    }

    /// A [`FailingCallbacks`] whose every history read returns a started
    /// callback `approval`.
    fn failing_callbacks(barrier: Option<Arc<tokio::sync::Barrier>>) -> Arc<FailingCallbacks> {
        Arc::new(FailingCallbacks(
            FakeApi::new()
                .with_history_pages(Vec::new())
                .repeat_history(Ok(callback_started_history())),
            barrier,
            AtomicU32::new(0),
        ))
    }

    /// A [`failing_callbacks`] fake whose first status poll answers
    /// `SUCCEEDED` with the output `7`.
    fn failing_callbacks_then_success(
        barrier: Option<Arc<tokio::sync::Barrier>>,
    ) -> Arc<FailingCallbacks> {
        let api = failing_callbacks(barrier);
        api.0
            .executions
            .lock()
            .expect("lock")
            .push_back(Ok(execution_output(
                WireExecutionStatus::Succeeded,
                Some("7"),
            )));
        api
    }

    /// A [`FakeApi`] whose first three status poll attempts fail with
    /// `fake execution 1`, `2` and `3`, so the poll at 1000 ms fails the
    /// run at 3450 ms, and whose every history read returns a started
    /// callback `approval`.
    fn failing_polls() -> FakeApi {
        FakeApi::new()
            .with_execution(Err(fake_error("fake execution 1")))
            .with_execution(Err(fake_error("fake execution 2")))
            .with_execution(Err(fake_error("fake execution 3")))
            .with_history_pages(Vec::new())
            .repeat_history(Ok(callback_started_history()))
    }

    /// Asserts that `err` is `Unsupported` with `text`.
    #[track_caller]
    fn assert_unsupported(err: &TestRunnerError, text: &str) {
        assert!(
            matches!(err, TestRunnerError::Unsupported(t) if *t == text),
            "{err:?}"
        );
    }

    /// Asserts that `err` is `expected`: the same variant and text.
    #[track_caller]
    fn assert_same_error(err: &TestRunnerError, expected: &TestRunnerError) {
        assert_eq!(
            std::mem::discriminant(err),
            std::mem::discriminant(expected),
            "{err:?}"
        );
        assert_eq!(err.to_string(), expected.to_string());
    }

    impl CloudApi for FailingCallbacks {
        async fn send_invoke(&self, request: InvokeRequest) -> Result<InvokeOutput, BoxError> {
            self.0.send_invoke(request).await
        }
        async fn send_get_durable_execution(
            &self,
            execution_arn: &str,
        ) -> Result<GetExecutionOutput, BoxError> {
            self.0.send_get_durable_execution(execution_arn).await
        }
        async fn send_get_durable_execution_history(
            &self,
            execution_arn: &str,
            marker: Option<&str>,
        ) -> Result<GetHistoryOutput, BoxError> {
            self.0
                .send_get_durable_execution_history(execution_arn, marker)
                .await
        }
        async fn send_callback_success(&self, id: &str, result: Vec<u8>) -> Result<(), BoxError> {
            self.meet(FakeCall::CallbackSuccess(id.to_owned(), result))
                .await;
            Err(fake_error("fake success"))
        }
        async fn send_callback_failure(
            &self,
            id: &str,
            error: ErrorObject,
        ) -> Result<(), BoxError> {
            self.meet(FakeCall::CallbackFailure(id.to_owned(), error))
                .await;
            Err(fake_error("fake failure"))
        }
        async fn send_callback_heartbeat(&self, id: &str) -> Result<(), BoxError> {
            self.meet(FakeCall::Heartbeat(id.to_owned())).await;
            Err(fake_error("fake heartbeat"))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_callback_error_is_aws() {
        // Each callback API error belongs to its call: the call returns it,
        // a later control still works, and the run succeeds.
        for (case, text) in [
            (0_u8, "fake success"),
            (1, "fake failure"),
            (2, "fake heartbeat"),
        ] {
            let start = tokio::time::Instant::now();
            let api = failing_callbacks_then_success(None);
            let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
            let approval = run.operation("approval");
            let err = match case {
                0 => approval.send_callback(CallbackReply::raw("1")).await,
                1 => {
                    approval
                        .send_callback(CallbackReply::failure(CallbackFailure::default()))
                        .await
                }
                _ => approval.send_callback_heartbeat().await,
            }
            .expect_err("the callback call fails");
            assert_aws(&err, text);
            // The control reads the history at once, then makes its one
            // callback call.
            assert_eq!(api.0.history_ms(start), unmarked(&[0]));
            let calls = api.0.calls();
            assert!(
                matches!(
                    (case, calls.as_slice()),
                    (
                        0,
                        [
                            FakeCall::Invoke,
                            FakeCall::History,
                            FakeCall::CallbackSuccess(..)
                        ]
                    ) | (
                        1,
                        [
                            FakeCall::Invoke,
                            FakeCall::History,
                            FakeCall::CallbackFailure(..)
                        ]
                    ) | (
                        2,
                        [FakeCall::Invoke, FakeCall::History, FakeCall::Heartbeat(_)]
                    )
                ),
                "{calls:?}"
            );
            assert_eq!(api.2.load(Ordering::SeqCst), 1, "one callback call");
            let later = approval
                .wait_for_data(WaitingStatus::Started)
                .await
                .expect("a later wait reaches the callback");
            assert_eq!(later.id(), "c");
            let result = run.await.expect("the run succeeds");
            assert!(result.is_success());
            assert_eq!(result.output(), Some(&7));
            assert_eq!(api.2.load(Ordering::SeqCst), 1, "no other callback call");
        }
    }

    /// The run failures before the invoke published the execution ARN:
    /// an event that does not serialize, a failed `Invoke` and an invoke
    /// response without an ARN. Under `InvocationType::Event` a wait and a
    /// callback control return the run failure, because no target exists.
    /// Under `RequestResponse` they return their `Unsupported` errors, and
    /// so do timeout replies, `pause` and `resume` under both. No history or
    /// callback call is made.
    #[tokio::test(start_paused = true)]
    async fn cloud_pre_target_run_failures_keep_their_cause() {
        for row in 0_u8..3 {
            for runner in [event_runner(), CloudRunner::new("f")] {
                let api = failing_callbacks(None);
                let mut run = match row {
                    0 => runner.start::<_, u32, _>(
                        {
                            let api = Arc::clone(&api);
                            async move { api }
                        },
                        serde_json::to_vec(&Unserializable),
                    ),
                    1 => {
                        *api.0.invoke.lock().expect("lock") = Some(Err(fake_error("fake invoke")));
                        runner.run_with::<_, u32>(Arc::clone(&api), Vec::new())
                    }
                    _ => {
                        *api.0.invoke.lock().expect("lock") = Some(Ok(invoke_output(None)));
                        runner.run_with::<_, u32>(Arc::clone(&api), Vec::new())
                    }
                };
                let approval = run.operation("approval");
                let end = (&mut run).await.expect_err("the run fails");
                match row {
                    0 => assert!(
                        matches!(end, TestRunnerError::EventSerialization(_)),
                        "{end:?}"
                    ),
                    1 => assert_aws(&end, "fake invoke"),
                    _ => assert!(matches!(end, TestRunnerError::NoExecutionArn), "{end:?}"),
                }
                let wait = approval
                    .wait_for_data(WaitingStatus::Started)
                    .await
                    .expect_err("wait_for_data");
                let send = approval
                    .send_callback(CallbackReply::raw("1"))
                    .await
                    .expect_err("send_callback");
                let heartbeat = approval
                    .send_callback_heartbeat()
                    .await
                    .expect_err("heartbeat");
                let event = runner.invocation_type == InvocationType::Event;
                if event {
                    for err in [&wait, &send, &heartbeat] {
                        assert_same_error(err, &end);
                    }
                } else {
                    assert_unsupported(&wait, WAIT_FOR_DATA_REQUEST_RESPONSE_UNSUPPORTED);
                    assert_unsupported(&send, SEND_CALLBACK_REQUEST_RESPONSE_UNSUPPORTED);
                    assert_unsupported(
                        &heartbeat,
                        SEND_CALLBACK_HEARTBEAT_REQUEST_RESPONSE_UNSUPPORTED,
                    );
                }
                for reply in [CallbackReply::timeout(), CallbackReply::heartbeat_timeout()] {
                    let err = approval.send_callback(reply).await.expect_err("timeout");
                    let text = if event {
                        CLOUD_TIMEOUT_REPLY_UNSUPPORTED
                    } else {
                        SEND_CALLBACK_REQUEST_RESPONSE_UNSUPPORTED
                    };
                    assert_unsupported(&err, text);
                }
                for err in [
                    run.pause().await.expect_err("pause"),
                    run.resume().await.expect_err("resume"),
                ] {
                    assert_unsupported(&err, CLOUD_PAUSE_UNSUPPORTED);
                }
                assert_same_error(&(&mut run).await.expect_err("the run failure"), &end);
                let calls = api.0.calls();
                assert!(
                    calls.iter().all(|call| matches!(call, FakeCall::Invoke)),
                    "{calls:?}"
                );
                assert_eq!(api.2.load(Ordering::SeqCst), 0);
            }
        }
    }

    /// A callback API failure after the driver returned a `Running` result
    /// (its poll budget spent) fails only that call. A heartbeat fails with
    /// its own error, a reached wait reads the last fold, the `Unsupported`
    /// controls keep their errors, and the consumed `RunHandle` stays
    /// pending.
    #[tokio::test(start_paused = true)]
    async fn cloud_callback_error_after_the_result_fails_only_that_call() {
        let api = failing_callbacks(None);
        let mut run = event_runner()
            .max_poll_attempts(2)
            .run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        let result = (&mut run).await.expect("the result");
        assert_eq!(result.status(), ExecutionStatus::Running);
        let history = history_fetches(&api.0.calls());
        let err = approval
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect_err("the callback call fails");
        assert_aws(&err, "fake success");
        let err = approval
            .send_callback_heartbeat()
            .await
            .expect_err("the heartbeat fails");
        assert_aws(&err, "fake heartbeat");
        assert_eq!(api.2.load(Ordering::SeqCst), 2, "both calls were made");
        let op = approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("the last fold");
        assert_eq!(op.id(), "c");
        assert_eq!(
            history_fetches(&api.0.calls()),
            history,
            "the spent budget starts no read"
        );
        let err = approval
            .send_callback(CallbackReply::timeout())
            .await
            .expect_err("timeout reply");
        assert_unsupported(&err, CLOUD_TIMEOUT_REPLY_UNSUPPORTED);
        for err in [
            run.pause().await.expect_err("pause"),
            run.resume().await.expect_err("resume"),
        ] {
            assert_unsupported(&err, CLOUD_PAUSE_UNSUPPORTED);
        }
        assert!(run.shared.terminal_error().is_none());
        let pending = std::future::poll_fn(|cx| {
            std::task::Poll::Ready(Pin::new(&mut run).poll(cx).is_pending())
        })
        .await;
        assert!(pending, "the result was already taken");
    }

    /// Two callback calls that fail together return their own errors: both
    /// calls are made, and the run succeeds.
    #[tokio::test(start_paused = true)]
    async fn cloud_concurrent_callback_errors_are_separate() {
        let api = failing_callbacks_then_success(Some(Arc::new(tokio::sync::Barrier::new(2))));
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let (success, failure) = tokio::join!(
            approval.send_callback(CallbackReply::raw("1")),
            approval.send_callback(CallbackReply::failure(CallbackFailure::default())),
        );
        assert_aws(&success.expect_err("success fails"), "fake success");
        assert_aws(&failure.expect_err("failure fails"), "fake failure");
        assert_eq!(api.2.load(Ordering::SeqCst), 2, "both calls were sent");
        let calls = api.0.calls();
        assert!(
            calls
                .iter()
                .any(|call| matches!(call, FakeCall::CallbackSuccess(..)))
                && calls
                    .iter()
                    .any(|call| matches!(call, FakeCall::CallbackFailure(..))),
            "{calls:?}"
        );
        let result = run.await.expect("the run succeeds");
        assert_eq!(result.output(), Some(&7));
    }

    /// A callback whose history read fails on its third attempt returns
    /// that error and makes no callback call. A second callback reads the
    /// history at once and sends its reply, and the run succeeds.
    #[tokio::test(start_paused = true)]
    async fn cloud_callback_history_error_fails_only_that_call() {
        let start = tokio::time::Instant::now();
        let running = || Ok(execution_output(WireExecutionStatus::Running, None));
        let api = Arc::new(
            FakeApi::new()
                .with_execution(running())
                .with_execution(running())
                .with_execution(Ok(execution_output(
                    WireExecutionStatus::Succeeded,
                    Some("7"),
                )))
                .with_history_pages(vec![
                    Err(fake_error("fake history 1")),
                    Err(fake_error("fake history 2")),
                    Err(fake_error("fake history 3")),
                ])
                .repeat_history(Ok(callback_started_history())),
        );
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let err = approval
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect_err("the history read fails");
        assert_aws(&err, "fake history 3");
        assert_eq!(api.history_ms(start), unmarked(&[0, 1000, 2450]));
        assert!(
            !api.calls()
                .iter()
                .any(|call| matches!(call, FakeCall::CallbackSuccess(..))),
            "no callback call"
        );
        approval
            .send_callback(CallbackReply::raw("2"))
            .await
            .expect("the second reply is sent");
        assert_eq!(api.history_ms(start), unmarked(&[0, 1000, 2450, 2450]));
        assert_eq!(
            api.calls().last(),
            Some(&FakeCall::CallbackSuccess("cb-1".to_owned(), b"2".to_vec()))
        );
        let result = run.await.expect("the run succeeds");
        assert_eq!(result.output(), Some(&7));
    }

    /// After a run failure past the invoke, a late wait decides from the
    /// last fold: for each of a status poll that fails on its third
    /// attempt, a failed final history read and an output that does not
    /// decode, a wait for a reached status returns the stored operation and
    /// one for an unreached status returns the run failure. Neither makes a
    /// Lambda call or spends a poll.
    #[tokio::test(start_paused = true)]
    async fn cloud_late_wait_after_post_target_failure_uses_the_last_fold() {
        for row in 0_u8..3 {
            let api = Arc::new(match row {
                0 => failing_polls(),
                1 => FakeApi::new()
                    .with_execution(Ok(execution_output(
                        WireExecutionStatus::Succeeded,
                        Some("7"),
                    )))
                    .with_history_pages(vec![Ok((callback_started_history(), None))])
                    .repeat_history(Err("fake final")),
                _ => FakeApi::new()
                    .with_execution(Ok(execution_output(
                        WireExecutionStatus::Succeeded,
                        Some("\"text\""),
                    )))
                    .with_history_pages(Vec::new())
                    .repeat_history(Ok(callback_started_history())),
            });
            let mut run = event_runner()
                .max_poll_attempts(10)
                .run_with::<_, u32>(Arc::clone(&api), Vec::new());
            let approval = run.operation("approval");
            approval
                .send_callback_heartbeat()
                .await
                .expect("the heartbeat stores a fold");
            let end = (&mut run).await.expect_err("the run fails");
            match row {
                0 => assert_aws(&end, "fake execution 3"),
                1 => assert_aws(&end, "fake final"),
                _ => assert!(
                    matches!(end, TestRunnerError::OutputDeserialization(_)),
                    "{end:?}"
                ),
            }
            let calls = api.calls();
            let used = run.shared.live.lock().cloud.used;
            let op = approval
                .wait_for_data(WaitingStatus::Started)
                .await
                .expect("the fold reaches Started");
            assert_eq!(op.id(), "c");
            let err = approval
                .wait_for_data(WaitingStatus::Completed)
                .await
                .expect_err("the fold does not reach Completed");
            assert_same_error(&err, &end);
            assert_eq!(api.calls(), calls, "row {row}: no Lambda call");
            assert_eq!(run.shared.live.lock().cloud.used, used, "no poll spent");
        }
    }

    /// The fake behind a [`post_failure_callbacks`] row.
    trait Recorded {
        fn fake(&self) -> &FakeApi;
    }

    impl Recorded for FakeApi {
        fn fake(&self) -> &FakeApi {
            self
        }
    }

    impl Recorded for FailingCallbacks {
        fn fake(&self) -> &FakeApi {
            &self.0
        }
    }

    /// How much poll budget a [`post_failure_callbacks`] row leaves.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Budget {
        /// No limit: a callback control reads the history.
        Remaining,
        /// Spent, after a wait stored a fold.
        SpentWithFold,
        /// Spent by the driver's one poll, with no fold.
        SpentWithoutFold,
    }

    /// Runs `api`, whose status poll at 1000 ms fails the run, with
    /// `budget`, then sends a success reply, a failure reply and a
    /// heartbeat, and checks each control's calls. With budget remaining,
    /// it also checks that a read made after the run failure became the
    /// fold that `snapshot()` returns. Returns the controls' outcomes.
    async fn post_failure_callbacks<A>(
        api: Arc<A>,
        budget: Budget,
    ) -> Vec<Result<(), TestRunnerError>>
    where
        A: CloudApi + Recorded + Send + Sync + 'static,
    {
        let runner = match budget {
            Budget::Remaining => event_runner(),
            Budget::SpentWithFold => event_runner().max_poll_attempts(2),
            Budget::SpentWithoutFold => event_runner().max_poll_attempts(1),
        };
        let mut run = runner.run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        if budget == Budget::SpentWithFold {
            approval
                .wait_for_data(WaitingStatus::Started)
                .await
                .expect("the wait stores a fold");
        }
        let end = (&mut run).await.expect_err("the run fails");
        assert_aws(&end, "fake execution 3");
        if budget == Budget::Remaining {
            assert!(
                approval.snapshot().is_none(),
                "no fold before the callbacks"
            );
        }
        let mut outcomes = Vec::new();
        for case in 0_u8..3 {
            let before = api.fake().calls().len();
            outcomes.push(match case {
                0 => approval.send_callback(CallbackReply::raw("1")).await,
                1 => {
                    approval
                        .send_callback(CallbackReply::failure(CallbackFailure::default()))
                        .await
                }
                _ => approval.send_callback_heartbeat().await,
            });
            let calls = api.fake().calls();
            let new = calls.get(before..).expect("later calls");
            let read = budget == Budget::Remaining;
            let made = match (case, new) {
                (_, []) => budget == Budget::SpentWithoutFold,
                (0, [FakeCall::History, FakeCall::CallbackSuccess(id, payload)]) => {
                    read && id == "cb-1" && payload == b"1"
                }
                (0, [FakeCall::CallbackSuccess(id, payload)]) => {
                    !read && id == "cb-1" && payload == b"1"
                }
                (1, [FakeCall::History, FakeCall::CallbackFailure(id, _)])
                | (2, [FakeCall::History, FakeCall::Heartbeat(id)]) => read && id == "cb-1",
                (1, [FakeCall::CallbackFailure(id, _)]) | (2, [FakeCall::Heartbeat(id)]) => {
                    !read && id == "cb-1"
                }
                _ => false,
            };
            assert!(made, "{budget:?} case {case}: {new:?}");
        }
        if budget == Budget::Remaining {
            // A successful read after the run failure became the fold.
            assert_eq!(
                approval.snapshot().map(|op| op.id().to_owned()).as_deref(),
                Some("c"),
                "the post-failure read is the fold"
            );
        }
        assert_same_error(&(&mut run).await.expect_err("the run failure stays"), &end);
        outcomes
    }

    /// After a run failure past the invoke, a callback control follows its
    /// normal path: it reads the history while budget remains, else uses
    /// the last fold, makes its callback call and returns its own outcome.
    /// With the budget spent and no fold it returns `OperationNotFound`
    /// without a call. The `RunHandle` keeps the run failure.
    #[tokio::test(start_paused = true)]
    async fn cloud_callback_started_after_run_failure_returns_its_own_outcome() {
        for budget in [Budget::Remaining, Budget::SpentWithFold] {
            for outcome in post_failure_callbacks(Arc::new(failing_polls()), budget).await {
                outcome.expect("the callback call succeeds");
            }
            let api = Arc::new(FailingCallbacks(failing_polls(), None, AtomicU32::new(0)));
            let outcomes = post_failure_callbacks(Arc::clone(&api), budget).await;
            for (outcome, text) in
                outcomes
                    .iter()
                    .zip(["fake success", "fake failure", "fake heartbeat"])
            {
                assert_aws(outcome.as_ref().expect_err("the call fails"), text);
            }
            assert_eq!(api.2.load(Ordering::SeqCst), 3);
        }
        for outcome in
            post_failure_callbacks(Arc::new(failing_polls()), Budget::SpentWithoutFold).await
        {
            let err = outcome.expect_err("no fold");
            assert!(matches!(err, TestRunnerError::OperationNotFound), "{err:?}");
        }
    }

    /// A fake whose status polls and history are those of
    /// [`failing_polls`], and whose heartbeat fails with `fake heartbeat`.
    /// With `fail_success`, a callback success call fails with
    /// `fake success` after the inner fake recorded it. One call is held at
    /// a gate after it signals `entered`: the history read with the 0-based
    /// index `gate_history`, or, without one, every callback success call.
    struct GatedApi {
        inner: FakeApi,
        gate_history: Option<usize>,
        fail_success: bool,
        history_reads: std::sync::atomic::AtomicUsize,
        gate: tokio::sync::Semaphore,
        entered: tokio::sync::Notify,
    }

    impl GatedApi {
        fn new(gate_history: Option<usize>, fail_success: bool) -> Self {
            Self {
                inner: failing_polls(),
                gate_history,
                fail_success,
                history_reads: std::sync::atomic::AtomicUsize::new(0),
                gate: tokio::sync::Semaphore::new(0),
                entered: tokio::sync::Notify::new(),
            }
        }

        async fn hold(&self) {
            self.entered.notify_one();
            drop(self.gate.acquire().await.expect("the gate stays open"));
        }
    }

    impl CloudApi for GatedApi {
        async fn send_invoke(&self, request: InvokeRequest) -> Result<InvokeOutput, BoxError> {
            self.inner.send_invoke(request).await
        }
        async fn send_get_durable_execution(
            &self,
            execution_arn: &str,
        ) -> Result<GetExecutionOutput, BoxError> {
            self.inner.send_get_durable_execution(execution_arn).await
        }
        async fn send_get_durable_execution_history(
            &self,
            execution_arn: &str,
            marker: Option<&str>,
        ) -> Result<GetHistoryOutput, BoxError> {
            let index = self.history_reads.fetch_add(1, Ordering::SeqCst);
            if self.gate_history == Some(index) {
                self.hold().await;
            }
            self.inner
                .send_get_durable_execution_history(execution_arn, marker)
                .await
        }
        async fn send_callback_success(&self, id: &str, result: Vec<u8>) -> Result<(), BoxError> {
            if self.gate_history.is_none() {
                self.hold().await;
            }
            self.inner.send_callback_success(id, result).await?;
            if self.fail_success {
                return Err(fake_error("fake success"));
            }
            Ok(())
        }
        async fn send_callback_failure(
            &self,
            id: &str,
            error: ErrorObject,
        ) -> Result<(), BoxError> {
            self.inner.send_callback_failure(id, error).await
        }
        #[expect(clippy::unused_async_trait_impl)] // reason: the fake fails at once
        async fn send_callback_heartbeat(&self, _: &str) -> Result<(), BoxError> {
            Err(fake_error("fake heartbeat"))
        }
    }

    /// A wait whose history read is outstanding when a status poll fails
    /// the run returns the run failure and drops the read. A callback whose
    /// history read or callback call is outstanding then stays pending while
    /// the call is held, and once it is released returns its own outcome.
    #[tokio::test(start_paused = true)]
    async fn cloud_outstanding_controls_follow_the_run_failure_rules() {
        let api = Arc::new(GatedApi::new(Some(0), false));
        let mut run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        let (end, waited) = tokio::join!(&mut run, approval.wait_for_data(WaitingStatus::Started));
        let end = end.expect_err("the status poll fails");
        assert_aws(&end, "fake execution 3");
        assert_same_error(&waited.expect_err("the waiter gets the run failure"), &end);
        api.gate.add_permits(1);
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        assert_eq!(
            api.inner.history_calls(),
            Vec::<HistoryCall>::new(),
            "the held read was dropped"
        );

        for (gate_history, fail_success) in [
            (Some(0), false),
            (Some(0), true),
            (None, false),
            (None, true),
        ] {
            let api = Arc::new(GatedApi::new(gate_history, fail_success));
            let mut run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
            let approval = run.operation("approval");
            let send = approval.send_callback(CallbackReply::raw("1"));
            let mut send = std::pin::pin!(send);
            tokio::select! {
                biased;
                outcome = &mut send => panic!("the call is held: {outcome:?}"),
                () = api.entered.notified() => {}
            }
            let end = (&mut run).await.expect_err("the status poll fails");
            assert_aws(&end, "fake execution 3");
            let case = format!("{gate_history:?}/{fail_success}");
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(3600), &mut send)
                    .await
                    .is_err(),
                "{case}: the callback stays pending while its call is held"
            );
            api.gate.add_permits(1);
            let outcome = send.await;
            if fail_success {
                assert_aws(&outcome.expect_err("the call fails"), "fake success");
            } else {
                outcome.expect("the call succeeds");
            }
            assert_eq!(
                api.inner.calls().last(),
                Some(&FakeCall::CallbackSuccess("cb-1".to_owned(), b"1".to_vec())),
                "{case}"
            );
            assert_same_error(&(&mut run).await.expect_err("the run failure stays"), &end);
        }
    }

    // ---- Live control: the hold rule and wakeups ----

    /// A step `a` that takes 200 ms, then a step `b`.
    async fn slow_then_fast((): (), ctx: DurableContext) -> Result<(), BoxError> {
        ctx.step(|_| async {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(1_u32)
        })
        .name("a")
        .await?;
        ctx.step(|_| async { Ok(2_u32) }).name("b").await?;
        Ok(())
    }

    #[tokio::test]
    async fn dropped_wait_for_data_stops_counting() {
        let run = LocalRunner::new().run(slow_then_fast, ());
        let a = run.operation("a");
        a.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        // The waiter never completes. When the timeout drops it, nothing
        // waits any more, while the body of `a` still runs.
        let ghost = run.operation("ghost");
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(20),
                ghost.wait_for_data(WaitingStatus::Started)
            )
            .await
            .is_err()
        );
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert_eq!(
            a.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started),
            "the SUCCEED of `a` is held"
        );
        run.await.expect("runner");
    }

    #[tokio::test]
    async fn dropped_run_handle_stops_counting() {
        let mut run = LocalRunner::new().run(slow_then_fast, ());
        let a = run.operation("a");
        a.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut run)
                .await
                .is_err()
        );
        drop(run);
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert_eq!(
            a.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started),
            "the SUCCEED of `a` is held"
        );
        let op = a
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect("the waiter drives the run on");
        assert_eq!(op.status(), OperationStatus::Succeeded);
    }

    #[tokio::test]
    async fn wait_for_data_returns_the_snapshot_at_the_recheck() {
        // The run itself keeps waiting, so the step's SUCCEED follows its
        // START at once. The waiter still reports the step as the START
        // left it.
        let run = LocalRunner::new().run(steps_named(&["a"]), ());
        let a = run.operation("a");
        let (result, op) = tokio::join!(run, a.wait_for_data(WaitingStatus::Started));
        result.expect("runner");
        assert_eq!(
            op.expect("started").status(),
            OperationStatus::Started,
            "the snapshot taken when the START was applied"
        );
    }

    #[tokio::test]
    async fn waiter_on_another_task_sees_the_run_end() {
        let run = LocalRunner::new().run(steps_named(&["s"]), ());
        let ghost = run.operation("ghost");
        let waiter = tokio::spawn(async move { ghost.wait_for_data(WaitingStatus::Started).await });
        tokio::task::yield_now().await;
        run.await.expect("runner");
        let err = tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("the run end wakes the waiter")
            .expect("joined")
            .expect_err("never reached");
        assert!(
            matches!(
                err,
                TestRunnerError::OperationNotReached(WaitingStatus::Started)
            ),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn reply_wakes_a_waiting_wait_for_data() {
        let run = LocalRunner::new().run(
            |(), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<u32>().name("c").await?;
                Ok::<_, BoxError>(cb.result().await?)
            },
            (),
        );
        let c = run.operation("c");
        c.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        // Paused, the run moves no further, so only the reply can make the
        // waiter ready.
        run.pause().await.expect("paused");
        let completed = run.operation("c");
        let waiter =
            tokio::spawn(async move { completed.wait_for_data(WaitingStatus::Completed).await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        c.send_callback(success(&4_u32)).await.expect("sent");
        let op = tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .expect("the reply wakes the waiter")
            .expect("joined")
            .expect("completed");
        assert_eq!(op.status(), OperationStatus::Succeeded);
        run.resume().await.expect("resumed");
        assert_eq!(run.await.expect("runner").output(), Some(&4));
    }

    async fn wait_then_step((): (), ctx: DurableContext) -> Result<u32, BoxError> {
        ctx.wait(std::time::Duration::from_secs(60))
            .name("w")
            .await?;
        Ok(ctx.step(|_| async { Ok(1_u32) }).name("after").await?)
    }

    #[tokio::test]
    async fn pause_after_the_last_checkpoint_keeps_the_next_invocation_token() {
        let mut run = LocalRunner::new().run(wait_then_step, ());
        run.operation("w")
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        // The invocation still runs, and sends no further checkpoint.
        run.pause().await.expect("paused");
        run.resume().await.expect("resumed");
        let result = (&mut run).await.expect("runner");
        assert_eq!(result.output(), Some(&1));
        assert_eq!(
            result.invocations().len(),
            2,
            "the step after the wait keeps its token"
        );
    }

    #[test]
    fn handle_debug_names_the_types() {
        let run = RunHandle::<u32>::from_shared(RunCore::failed(
            new_live(),
            Engine::Local,
            TestRunnerError::NoExecutionArn,
        ));
        let op = run.operation("x");
        let text = format!("{run:?} {op:?}");
        for name in [
            "RunHandle",
            "RunCore",
            "OperationHandle",
            "failure: Some(NoExecutionArn)",
        ] {
            assert!(text.contains(name), "{text}");
        }
        assert_eq!(format!("{:?}", RunEnd::Running), "Running");
        assert_eq!(format!("{:?}", RunEnd::Finished(None)), "Finished");
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_invoke_error_reaches_wait_for_data() {
        let api = Arc::new(FakeApi::new().with_invoke(Err(fake_error("fake invoke"))));
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let a = run.operation("a");
        let err = a
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect_err("the invoke failed");
        assert_aws(&err, "fake invoke");
        let err = run.await.expect_err("the run failed");
        assert_aws(&err, "fake invoke");
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_run_handle_reads_the_last_fold() {
        let api = Arc::new(
            FakeApi::new()
                .with_execution(Ok(execution_output(WireExecutionStatus::Succeeded, None)))
                .with_history_pages(vec![Ok((callback_started_history(), None))]),
        );
        let mut run = event_runner().run_with::<_, ()>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        assert!(approval.snapshot().is_none());
        (&mut run).await.expect("runner");
        assert_eq!(
            approval.snapshot().map(|op| op.id().to_owned()),
            Some("c".to_owned()),
            "the final history read is the last fold"
        );
    }

    /// A waker that sets a flag.
    struct FlagWaker(Arc<std::sync::atomic::AtomicBool>);

    impl std::task::Wake for FlagWaker {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn driver_end_wakes_a_future_that_found_it_busy() {
        let woken = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let other = std::task::Waker::from(Arc::new(FlagWaker(Arc::clone(&woken))));
        let cell: Arc<std::sync::OnceLock<std::sync::Weak<RunCore>>> =
            Arc::new(std::sync::OnceLock::new());
        let in_driver = Arc::clone(&cell);
        let driver: DriverFuture = Box::pin(async move {
            // Another runner future polls while this poll runs: it finds the
            // driver busy and returns without polling it.
            let shared = in_driver
                .get()
                .and_then(std::sync::Weak::upgrade)
                .expect("set");
            let id = shared.live.lock().add_control(ControlKind::Run);
            shared.drive(id, &other);
            Err(TestRunnerError::NoExecutionArn)
        });
        let shared = RunCore::new(new_live(), Engine::Local, driver);
        cell.set(Arc::downgrade(&shared)).expect("unset");
        let id = shared.live.lock().add_control(ControlKind::Run);
        shared.drive(id, std::task::Waker::noop());
        assert!(shared.ended());
        assert!(
            woken.load(Ordering::SeqCst),
            "the driver's end wakes the future that found it busy"
        );
    }

    #[tokio::test]
    async fn pause_before_the_first_invocation_holds_it() {
        let counter = Arc::new(AtomicU32::new(0));
        let mut run = LocalRunner::new().run(counting(&counter, cool_down), ());
        run.pause().await.expect("paused");
        assert_run_waits(&mut run).await;
        assert_eq!(counter.load(Ordering::SeqCst), 0, "no invocation started");
        run.resume().await.expect("resumed");
        let result = run.await.expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("done"));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    /// Arms the commit seam of `run`: the next runner future that commits an
    /// outcome finds this run failure recorded just before its commit, as a
    /// concurrent failure of the run would record it.
    fn fail_at_next_commit<O>(run: &RunHandle<O>) {
        run.shared.live.lock().fail_at_commit = Some(TestRunnerError::NoExecutionArn);
    }

    /// Asserts that `outcome` is the error [`fail_at_next_commit`] records.
    #[track_caller]
    fn assert_failure_won<T: std::fmt::Debug>(outcome: Result<T, TestRunnerError>) {
        let err = outcome.expect_err("the run failure wins");
        assert!(matches!(err, TestRunnerError::NoExecutionArn), "{err:?}");
    }

    /// A `LocalRunner` run of [`await_callback_c`], paused while `c` waits
    /// for its reply, so nothing moves until the test acts.
    async fn paused_callback_run() -> (RunHandle<u32>, OperationHandle) {
        let run = LocalRunner::new().run(await_callback_c, ());
        let c = run.operation("c");
        c.wait_for_data(WaitingStatus::Started)
            .await
            .expect("started");
        run.pause().await.expect("paused");
        (run, c)
    }

    /// A run failure recorded after a `wait_for_data` that started while the
    /// run was active became ready, and before it returned, is what it
    /// returns, under `LocalRunner` and under `CloudRunner`. A wait first
    /// polled after that failure returns the reached data.
    #[tokio::test(start_paused = true)]
    async fn a_run_failure_before_wait_commit_beats_ready_data() {
        let (run, c) = paused_callback_run().await;
        let mut waiter = std::pin::pin!(c.wait_for_data(WaitingStatus::Completed));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(waiter.as_mut().poll(&mut cx).is_pending());
        c.send_callback(CallbackReply::success(&1_u32).expect("encodes"))
            .await
            .expect("pending");
        assert!(
            run.shared
                .live
                .lock()
                .controls
                .values()
                .any(|control| matches!(control.phase, ControlPhase::Ready(_))),
            "the reply made the waiter ready"
        );
        fail_at_next_commit(&run);
        let poll = waiter.as_mut().poll(&mut cx);
        assert!(
            matches!(
                poll,
                std::task::Poll::Ready(Err(TestRunnerError::NoExecutionArn))
            ),
            "a ready waiter returns the run failure: {poll:?}"
        );
        let op = c
            .wait_for_data(WaitingStatus::Completed)
            .await
            .expect("a later wait returns the reached data");
        assert_eq!(op.status(), OperationStatus::Succeeded);

        let api = fake_with_history(callback_started_history());
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        let approval = run.operation("approval");
        fail_at_next_commit(&run);
        assert_failure_won(approval.wait_for_data(WaitingStatus::Started).await);
        assert_eq!(history_fetches(&api.calls()), 1);
        let op = approval
            .wait_for_data(WaitingStatus::Started)
            .await
            .expect("a later wait returns the last fold");
        assert_eq!(op.id(), "c");
        assert_eq!(history_fetches(&api.calls()), 1, "no further read");
    }

    /// A run failure recorded just before the commit also wins over every
    /// other outcome of a `LocalRunner` control that started while the run
    /// was active: a reached status, a not-reached status, a reply, a
    /// refused reply, a heartbeat, `pause` and `resume`. A control whose
    /// outcome loses changes nothing.
    #[tokio::test]
    async fn a_terminal_error_before_commit_beats_every_local_control() {
        let (run, c) = paused_callback_run().await;
        fail_at_next_commit(&run);
        assert_failure_won(c.wait_for_data(WaitingStatus::Started).await);

        let (run, c) = paused_callback_run().await;
        fail_at_next_commit(&run);
        assert_failure_won(
            c.send_callback(CallbackReply::success(&1_u32).expect("encodes"))
                .await,
        );
        assert_eq!(
            c.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started),
            "the losing reply is not applied"
        );
        assert!(!run.shared.live.lock().local.invocation_due);

        let (run, _c) = paused_callback_run().await;
        fail_at_next_commit(&run);
        assert_failure_won(
            run.operation("ghost")
                .send_callback(CallbackReply::raw("1"))
                .await,
        );

        let (run, c) = paused_callback_run().await;
        fail_at_next_commit(&run);
        assert_failure_won(c.send_callback_heartbeat().await);

        let (run, _c) = paused_callback_run().await;
        fail_at_next_commit(&run);
        assert_failure_won(run.pause().await);

        let (run, _c) = paused_callback_run().await;
        fail_at_next_commit(&run);
        assert_failure_won(run.resume().await);
        assert!(
            run.shared.live.lock().local.paused,
            "the losing resume changes nothing"
        );

        let mut run = LocalRunner::new().run(steps_named(&["s"]), ());
        let ghost = run.operation("ghost");
        (&mut run).await.expect("runner");
        fail_at_next_commit(&run);
        assert_failure_won(ghost.wait_for_data(WaitingStatus::Started).await);
    }

    /// A `LocalRunner` `pause` that waits for its invocation to end returns
    /// a run failure recorded after that end and before it returned.
    #[test]
    fn a_terminal_error_before_commit_beats_a_ready_pause() {
        let live = new_live();
        let (shared, _polls) = counting_run(&live);
        let run = RunHandle::<u32>::from_shared(shared);
        pending_control(&live, ControlKind::Run);
        assert!(live.lock().admit_invocation());
        let mut pause = std::pin::pin!(run.pause());
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(pause.as_mut().poll(&mut cx).is_pending());
        drop(live.lock().end_invocation(false));
        fail_at_next_commit(&run);
        let poll = pause.as_mut().poll(&mut cx);
        assert!(
            matches!(
                poll,
                std::task::Poll::Ready(Err(TestRunnerError::NoExecutionArn))
            ),
            "a ready pause returns the run failure: {poll:?}"
        );
    }

    /// A `CloudRunner` callback control, a `RequestResponse` wait and
    /// `pause` commit nothing, so a concurrent run failure armed at the next
    /// commit changes none of their outcomes: a reply, a refused reply, a
    /// heartbeat and the `Unsupported` errors.
    #[tokio::test(start_paused = true)]
    async fn cloud_callbacks_return_their_own_outcome_past_a_run_failure() {
        let api = fake_with_history(callback_started_history());
        let run = event_runner().run_with::<_, u32>(Arc::clone(&api), Vec::new());
        fail_at_next_commit(&run);
        run.operation("approval")
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect("the reply is sent");
        assert_eq!(
            api.calls().last(),
            Some(&FakeCall::CallbackSuccess("cb-1".to_owned(), b"1".to_vec()))
        );
        let err = run
            .operation("ghost")
            .send_callback(CallbackReply::raw("1"))
            .await
            .expect_err("no such operation");
        assert!(matches!(err, TestRunnerError::OperationNotFound), "{err:?}");
        run.operation("approval")
            .send_callback_heartbeat()
            .await
            .expect("the heartbeat is sent");
        assert_eq!(
            api.calls().last(),
            Some(&FakeCall::Heartbeat("cb-1".to_owned()))
        );
        assert_unsupported(
            &run.pause().await.expect_err("pause"),
            CLOUD_PAUSE_UNSUPPORTED,
        );
        assert!(
            run.shared.live.lock().fail_at_commit.is_some(),
            "not consumed"
        );
        assert!(run.shared.terminal_error().is_none());

        let api = fake_with_history(callback_started_history());
        let run = CloudRunner::new("f").run_with::<_, u32>(Arc::clone(&api), Vec::new());
        fail_at_next_commit(&run);
        assert_unsupported(
            &run.operation("approval")
                .wait_for_data(WaitingStatus::Started)
                .await
                .expect_err("RequestResponse"),
            WAIT_FOR_DATA_REQUEST_RESPONSE_UNSUPPORTED,
        );
        assert_eq!(api.calls(), Vec::<FakeCall>::new());
        assert!(
            run.shared.live.lock().fail_at_commit.is_some(),
            "not consumed"
        );
    }

    /// A handler that creates the callback `c` with a heartbeat timeout,
    /// then waits an hour, and never awaits the callback's result.
    async fn heartbeat_callback_then_wait((): (), ctx: DurableContext) -> Result<u32, BoxError> {
        ctx.create_callback::<u32>()
            .name("c")
            .heartbeat_timeout(std::time::Duration::from_secs(60))
            .await?;
        ctx.wait(std::time::Duration::from_secs(3600))
            .name("w")
            .await?;
        Ok(1)
    }

    /// A heartbeat to a callback that is still `Started` after the run
    /// decided to end is refused as a reply is: after a terminal envelope,
    /// and after the invocation budget ran out.
    #[tokio::test]
    async fn a_heartbeat_after_the_run_ended_is_refused() {
        for runner in [LocalRunner::new(), LocalRunner::new().max_invocations(1)] {
            let mut run = runner.run(heartbeat_callback_then_wait, ());
            let c = run.operation("c");
            let result = (&mut run).await.expect("runner");
            assert_eq!(
                c.snapshot().map(|op| op.status()),
                Some(OperationStatus::Started)
            );
            let err = c
                .send_callback_heartbeat()
                .await
                .expect_err("the run ended");
            assert!(
                matches!(err, TestRunnerError::CallbackNotPending),
                "{:?}: {err:?}",
                result.status()
            );
        }
    }

    /// The `CloudRunner` core of a run with no driver, for transition tests.
    fn cloud_core() -> (Arc<RunCore>, Arc<CloudShared<FakeApi>>) {
        let live = new_live();
        let cloud = Arc::new(CloudShared::<FakeApi>::new(
            &event_runner(),
            Arc::clone(&live),
        ));
        let core = RunCore::new(
            live,
            Engine::Cloud(Arc::clone(&cloud) as Arc<dyn CloudControl>),
            Box::pin(std::future::pending()),
        );
        (core, cloud)
    }

    /// The history event `event_type` of the step `s`.
    fn step_s_event(event_type: EventType) -> Event {
        Event::builder()
            .event_type(event_type)
            .id("s")
            .name("s")
            .event_id(1)
            .build()
    }

    /// A history read that finishes after a newer one does not replace the
    /// newer fold, and still returns its own operations to its caller.
    #[test]
    fn an_older_history_read_does_not_replace_a_newer_fold() {
        let (core, cloud) = cloud_core();
        let older = cloud.begin_request();
        let newer = cloud.begin_request();
        cloud
            .finish_history(
                None,
                newer,
                Ok(vec![step_s_event(EventType::StepSucceeded)]),
            )
            .expect("newer");
        let (_, own) = cloud
            .finish_history(None, older, Ok(vec![step_s_event(EventType::StepStarted)]))
            .expect("older");
        assert_eq!(
            own.first().map(|op| op.status),
            Some(OperationStatus::Started)
        );
        let fold = core.operations().expect("a fold");
        assert_eq!(
            fold.first().map(|op| op.status),
            Some(OperationStatus::Succeeded),
            "the newer fold stays"
        );
    }

    /// A failed callback call and a failed history read of two controls
    /// return their own errors and record no run failure. The failed read
    /// leaves the fold as it is, and later reads and calls still start.
    #[test]
    fn failed_cloud_calls_do_not_record_a_run_failure() {
        let (core, cloud) = cloud_core();
        let first = core.live.lock().add_control(ControlKind::Cloud);
        let second = core.live.lock().add_control(ControlKind::Cloud);
        let read = cloud.begin_request();
        cloud
            .finish_history(None, read, Ok(vec![step_s_event(EventType::StepStarted)]))
            .expect("a fold");
        let one = cloud.begin_request();
        let two = cloud.begin_request();
        let err = |text: &str| aws_error(Box::new(std::io::Error::other(text.to_owned())));
        let a = cloud
            .finish_call(first, one, Err(err("first")))
            .expect_err("failed");
        let b = cloud
            .finish_history(
                Some((second, ControlRole::Callback)),
                two,
                Err(err("second")),
            )
            .expect_err("failed");
        assert!(a.to_string().ends_with("first"), "{a}");
        assert!(b.to_string().ends_with("second"), "{b}");
        assert!(core.terminal_error().is_none(), "no run failure");
        let fold = core.operations().expect("the fold stays");
        assert_eq!(
            fold.first().map(|op| op.status),
            Some(OperationStatus::Started)
        );
        assert_eq!(core.live.lock().cloud.fold_request, read);
        for role in [ControlRole::Wait, ControlRole::Callback] {
            assert!(matches!(
                cloud.begin_history(role),
                Ok(HistoryStart::Read(ticket)) if ticket > two
            ));
        }
        assert!(cloud.begin_request() > two);
    }

    /// A pending wait whose Lambda call never answers is woken by a run
    /// failure, and returns it. A callback control is woken too, and keeps
    /// awaiting its call.
    #[test]
    fn a_failure_wakes_an_in_flight_call() {
        for role in [ControlRole::Wait, ControlRole::Callback] {
            let (core, cloud) = cloud_core();
            let registration = Registration::new(&core, ControlKind::Cloud);
            let request = cloud.begin_request();
            let woken = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let waker = std::task::Waker::from(Arc::new(FlagWaker(Arc::clone(&woken))));
            let mut cx = std::task::Context::from_waker(&waker);
            let mut call = std::pin::pin!(await_in_flight(
                &registration,
                role,
                request,
                std::future::pending::<()>()
            ));
            assert!(call.as_mut().poll(&mut cx).is_pending());
            assert!(matches!(
                core.live
                    .lock()
                    .controls
                    .get(&registration.id)
                    .map(|control| &control.phase),
                Some(ControlPhase::InFlight { .. })
            ));
            let (_, wakers) = core
                .live
                .lock()
                .record_failure(TestRunnerError::NoExecutionArn);
            wake(wakers);
            assert!(woken.load(Ordering::SeqCst), "{role:?}");
            let poll = call.as_mut().poll(&mut cx);
            match role {
                ControlRole::Wait => assert!(matches!(
                    poll,
                    std::task::Poll::Ready(Err(TestRunnerError::NoExecutionArn))
                )),
                ControlRole::Callback => assert!(poll.is_pending(), "{poll:?}"),
            }
        }
    }

    /// A run failure recorded while the wait polls its Lambda call, which
    /// then stays pending, ends the wait on that same poll with the run
    /// failure, and the wait is not left in flight. A callback control
    /// stays in flight.
    #[test]
    fn a_failure_recorded_during_the_call_poll_ends_a_wait() {
        for role in [ControlRole::Wait, ControlRole::Callback] {
            let (core, cloud) = cloud_core();
            let registration = Registration::new(&core, ControlKind::Cloud);
            let request = cloud.begin_request();
            let call = {
                let core = Arc::clone(&core);
                std::future::poll_fn(move |_| {
                    let (_, wakers) = core
                        .live
                        .lock()
                        .record_failure(TestRunnerError::NoExecutionArn);
                    wake(wakers);
                    std::task::Poll::<()>::Pending
                })
            };
            let waker = std::task::Waker::noop();
            let mut cx = std::task::Context::from_waker(waker);
            let mut wait = std::pin::pin!(await_in_flight(&registration, role, request, call));
            let poll = wait.as_mut().poll(&mut cx);
            let in_flight = matches!(
                core.live
                    .lock()
                    .controls
                    .get(&registration.id)
                    .map(|control| &control.phase),
                Some(ControlPhase::InFlight { .. })
            );
            match role {
                ControlRole::Wait => {
                    assert!(
                        matches!(
                            poll,
                            std::task::Poll::Ready(Err(TestRunnerError::NoExecutionArn))
                        ),
                        "{poll:?}"
                    );
                    assert!(!in_flight, "the wait is not left in flight");
                }
                ControlRole::Callback => {
                    assert!(poll.is_pending(), "{poll:?}");
                    assert!(in_flight, "the callback stays in flight");
                }
            }
        }
    }

    /// A pending wait whose Lambda call becomes ready only after a run
    /// failure was recorded returns the run failure without taking the
    /// call's output. A wait's read finished after the failure stores no
    /// fold, while a callback control's read still becomes the fold.
    #[test]
    fn a_run_failure_drops_a_wait_call_that_became_ready() {
        let (core, cloud) = cloud_core();
        let registration = Registration::new(&core, ControlKind::Cloud);
        let request = cloud.begin_request();
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let taken = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let call = {
            let ready = Arc::clone(&ready);
            let taken = Arc::clone(&taken);
            std::future::poll_fn(move |_| {
                if ready.load(Ordering::SeqCst) {
                    taken.store(true, Ordering::SeqCst);
                    std::task::Poll::Ready(())
                } else {
                    std::task::Poll::Pending
                }
            })
        };
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        let mut wait = std::pin::pin!(await_in_flight(
            &registration,
            ControlRole::Wait,
            request,
            call
        ));
        assert!(wait.as_mut().poll(&mut cx).is_pending());
        let (_, wakers) = core
            .live
            .lock()
            .record_failure(TestRunnerError::NoExecutionArn);
        wake(wakers);
        ready.store(true, Ordering::SeqCst);
        assert!(matches!(
            wait.as_mut().poll(&mut cx),
            std::task::Poll::Ready(Err(TestRunnerError::NoExecutionArn))
        ));
        assert!(!taken.load(Ordering::SeqCst), "the ready call is dropped");

        let err = cloud
            .finish_history(
                Some((registration.id, ControlRole::Wait)),
                request,
                Ok(vec![step_s_event(EventType::StepStarted)]),
            )
            .expect_err("the run failure wins");
        assert!(matches!(err, TestRunnerError::NoExecutionArn), "{err:?}");
        assert!(core.operations().is_none(), "the wait stores no fold");
        let read = cloud.begin_request();
        cloud
            .finish_history(
                Some((registration.id, ControlRole::Callback)),
                read,
                Ok(vec![step_s_event(EventType::StepStarted)]),
            )
            .expect("a callback read");
        assert!(core.operations().is_some(), "the callback read is the fold");
    }

    /// A history read starts only within the budget. A wait starts none
    /// after the run finished or failed, and returns the fold or the run
    /// failure. A callback control starts one after either, within the
    /// budget.
    #[test]
    fn a_history_read_starts_within_the_budget() {
        let (core, cloud) = cloud_core();
        core.live.lock().cloud.limit = Some(1);
        assert!(matches!(
            cloud.begin_history(ControlRole::Wait),
            Ok(HistoryStart::Read(_))
        ));
        assert!(matches!(
            cloud.begin_history(ControlRole::Callback),
            Ok(HistoryStart::Done(None))
        ));
        core.live.lock().cloud.limit = None;
        core.live.lock().end = RunEnd::Finished(None);
        assert!(matches!(
            cloud.begin_history(ControlRole::Wait),
            Ok(HistoryStart::Done(None))
        ));
        assert!(matches!(
            cloud.begin_history(ControlRole::Callback),
            Ok(HistoryStart::Read(_))
        ));

        let (core, cloud) = cloud_core();
        core.live.lock().cloud.limit = Some(2);
        core.live.lock().failure = Some(TestRunnerError::NoExecutionArn);
        assert!(matches!(
            cloud.begin_history(ControlRole::Wait),
            Err(TestRunnerError::NoExecutionArn)
        ));
        assert_eq!(core.live.lock().cloud.used, 0, "the wait spent nothing");
        assert!(matches!(
            cloud.begin_history(ControlRole::Callback),
            Ok(HistoryStart::Read(_))
        ));
        assert!(matches!(
            cloud.begin_history(ControlRole::Callback),
            Ok(HistoryStart::Read(_))
        ));
        assert!(matches!(
            cloud.begin_history(ControlRole::Callback),
            Ok(HistoryStart::Done(None))
        ));
    }

    // ---- Oversized handler results ----

    /// A handler whose serialized result is `n` bytes: `n - 2` letters and
    /// two quotes. It builds the string when it runs, so a run of one
    /// invocation builds it once.
    fn sized_handler(
        n: usize,
    ) -> impl Fn((), DurableContext) -> std::future::Ready<Result<String, BoxError>>
    + Send
    + Sync
    + 'static {
        move |(), _ctx: DurableContext| std::future::ready(Ok("a".repeat(n - 2)))
    }

    /// Asserts that `result` succeeded with an output of `n - 2` letters,
    /// comparing the lengths first so a failure does not print 6 MB.
    fn assert_full_output(result: &TestResult<String>, n: usize) {
        assert!(result.is_success(), "{:?}", result.error_message());
        let output = result.output().expect("an output");
        assert_eq!(output.len(), n - 2);
        assert!(
            output.bytes().all(|b| b == b'a'),
            "the output is the result"
        );
    }

    /// A result up to the limit is inline; above it, up to the backend's
    /// maximum, it is checkpointed and read back; above the maximum the
    /// service rejects it. No run lists an `EXECUTION` operation.
    #[tokio::test]
    async fn oversized_result_round_trips() {
        for (n, checkpointed) in [(6_291_406, false), (6_291_407, true), (6_291_456, true)] {
            let backend = Arc::new(Backend::new(QueuedReplies::default(), None));
            let result: TestResult<String> = LocalRunner::new()
                .run_on_backend(Arc::clone(&backend), sized_handler(n), "null".to_owned())
                .await
                .expect("runner");
            assert_full_output(&result, n);
            assert_eq!(
                backend.execution_result().map(|stored| stored.len()),
                checkpointed.then_some(n),
                "n = {n}"
            );
            assert!(
                result
                    .operations()
                    .iter()
                    .all(|op| op.kind() != OperationKind::Execution),
                "n = {n}"
            );
        }

        let n = 6_291_457;
        let backend = Arc::new(Backend::new(QueuedReplies::default(), None));
        let result: TestResult<String> = LocalRunner::new()
            .run_on_backend(Arc::clone(&backend), sized_handler(n), "null".to_owned())
            .await
            .expect("runner");
        assert!(result.is_failure());
        assert!(result.output().is_none());
        assert_eq!(
            result.error_type(),
            Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
        );
        let message = result.error_message().unwrap_or_default();
        assert!(
            message.starts_with("[Unrecoverable Execution] Checkpoint failed: "),
            "{message}"
        );
        assert!(
            message.contains(
                "Execution output payload size must be less than or equal to 6291456 bytes."
            ),
            "{message}"
        );
        assert_eq!(backend.execution_result(), None);
        assert!(
            result
                .operations()
                .iter()
                .all(|op| op.kind() != OperationKind::Execution)
        );
    }

    /// The backend accepts the `EXECUTION` update and the answer is lost:
    /// the invocation fails, but the run ends `SUCCEEDED` with the full
    /// output after one invocation. The end hook saw
    /// `Retrying`.
    #[tokio::test]
    async fn oversized_result_accepted_then_error_ends_execution() {
        let n = 6_291_407;
        let plugin = crate::plugin::test_support::Hooks::default();
        let runner = LocalRunner::new().options(
            crate::Options::builder()
                .plugin(plugin.clone())
                .build()
                .expect("valid options"),
        );
        let backend = Arc::new(Backend::new(QueuedReplies::default(), None));
        backend.enqueue_response(crate::client::TestResponse::AcceptThenRetryableError(
            "lost response".into(),
        ));
        let result: TestResult<String> = runner
            .run_on_backend(Arc::clone(&backend), sized_handler(n), "null".to_owned())
            .await
            .expect("runner");
        assert_full_output(&result, n);
        assert_eq!(result.invocation_count(), 1);
        let ends = plugin.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(end.status(), crate::InvocationStatus::Retrying);
        assert_eq!(
            end.execution_error().and_then(crate::WireError::error_type),
            Some(crate::error::CHECKPOINT_INVOCATION_FAILED_ERROR_TYPE)
        );
    }

    /// A handler that creates the callback `c`, with a heartbeat timeout,
    /// and then returns a result one byte over the response limit, without
    /// awaiting the callback.
    async fn callback_then_oversized((): (), ctx: DurableContext) -> Result<String, BoxError> {
        ctx.create_callback::<u32>()
            .name("c")
            .heartbeat_timeout(std::time::Duration::from_secs(60))
            .await?;
        Ok("a".repeat(crate::LAMBDA_RESPONSE_SIZE_LIMIT - 1))
    }

    /// The backend accepts the `EXECUTION` update of a live run and the
    /// answer is lost: the run ends `SUCCEEDED` and is closed, so a reply
    /// and a heartbeat to the callback still `Started` are refused.
    #[tokio::test]
    async fn oversized_result_accepted_then_error_closes_the_run() {
        let n = crate::LAMBDA_RESPONSE_SIZE_LIMIT + 1;
        let mut run = LocalRunner::new().run_configured(callback_then_oversized, (), |backend| {
            // The callback's `START` passes; the `EXECUTION` write is
            // accepted and its answer lost.
            backend.plan_checkpoint_calls(vec![PlannedCheckpoint::Pass]);
            backend.enqueue_response(crate::client::TestResponse::AcceptThenRetryableError(
                "lost response".into(),
            ));
        });
        let c = run.operation("c");
        let result = (&mut run).await.expect("runner");
        assert_full_output(&result, n);
        assert_eq!(result.invocation_count(), 1);
        assert_eq!(
            c.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started)
        );
        let err = c
            .send_callback(CallbackReply::success(&1_u32).expect("encodes"))
            .await
            .expect_err("the run ended");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        let err = c
            .send_callback_heartbeat()
            .await
            .expect_err("the run ended");
        assert!(
            matches!(err, TestRunnerError::CallbackNotPending),
            "{err:?}"
        );
        assert_eq!(
            c.snapshot().map(|op| op.status()),
            Some(OperationStatus::Started),
            "the refused reply changed nothing"
        );
    }

    /// An envelope with an empty `Result` and no checkpointed result reads
    /// the empty string, which fails to decode as the output.
    #[tokio::test]
    async fn empty_result_without_checkpoint_fails_to_decode() {
        let runner = LocalRunner::new();
        let service = |_event: lambda_runtime::LambdaEvent<crate::InvocationPayload>| {
            let response: crate::BoxedInvocationFuture = Box::pin(async {
                Ok(crate::InvocationResponse(
                    serde_json::json!({ "Status": "SUCCEEDED", "Result": "" }),
                ))
            });
            response
        };
        let err = runner
            .drive_service(Arc::new(runner.new_backend()), service, "null".to_owned())
            .await
            .expect("a run")
            .into_test_result::<serde_json::Value>()
            .expect_err("the empty string is not JSON");
        assert!(
            matches!(err, TestRunnerError::OutputDeserialization(_)),
            "{err:?}"
        );
    }

    /// The backend answers only `AcceptThenRetryableError` from a queued
    /// `TestResponse`.
    #[test]
    #[should_panic(expected = "answers only AcceptThenRetryableError")]
    fn backend_refuses_other_queued_responses() {
        let backend = Backend::new(QueuedReplies::default(), None);
        backend.enqueue_response(crate::client::TestResponse::RetryableError("x".into()));
    }
}
