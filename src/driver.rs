//!
//! The driver polls the user's handler future and controls the execution
//! lifecycle of one invocation:
//!
//! - **Suspension**: the invocation coordinator
//!   ([`crate::coordinator`]) decides when the invocation suspends. Local
//!   work that can make progress holds a permit: a polled or woken task, a
//!   step body, a `wait_for_condition` check, a serdes call, blocking work,
//!   and an admitted checkpoint write until its response is merged. An
//!   operation that waits on the service holds a wait registration. When no
//!   permit is held and at least one registration is live, the phase
//!   becomes `Suspending` and the driver ends the invocation with
//!   `PENDING`. The handler future is dropped at its current await point,
//!   so a suspension never surfaces to user code.
//!
//! - **Continuation inside the invocation**: a merged checkpoint response
//!   wakes the registrations of the operations it carries. A wait whose
//!   record is complete continues in the same invocation, before the
//!   response's checkpoint permit is released.
//!
//! - **Resume**: on a later invocation with existing checkpoint state, the
//!   engine reconstructs replay mode via [`CheckpointLog`] and re-runs the
//!   handler. Replayed operations return frozen results from the log
//!   without re-execution.
//!
//! - **Task-ownership**: each context records the `tokio::task::Id` of the
//!   task that created it. Durable operations invoked from a DIFFERENT task
//!   fail fast with a clear error. The engine's own tasks (the handler,
//!   `.spawn()`, batch items and combinator inputs) are exempt.
//!
//! Zero permits and zero registrations do not suspend: the handler awaits
//! work the SDK does not track, and the invocation runs until that work
//! finishes or the Lambda deadline passes. The driver logs a warning when
//! the deadline is reached in that state.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use crate::coordinator::{InvocationCoordinator, Phase};

// Imports used by tests: suppress dead-code warnings for the engine types
// which are consumed by operation execution.
#[cfg(test)]
use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus, EngineState};

/// The warning logged when the Lambda deadline is reached while nothing the
/// SDK tracks runs and no operation waits.
pub(crate) const UNTRACKED_DEADLINE_WARNING: &str = "the invocation reached its deadline while \
     the handler awaited work the SDK does not track; run that work in a step";

/// The outcome of driving a single invocation of the user's handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InvocationOutcome {
    /// The handler completed successfully with a serialized result.
    Complete(String),
    /// The handler suspended: an operation requires an external event
    /// before it can proceed. The engine drops the handler future and
    /// reports PENDING to the runtime.
    Pending,
    /// The handler returned an error with its wire failure record.
    Failed {
        /// The wire failure record (type, message, opaque data, stack
        /// trace) reported in the response envelope.
        error: crate::error::WireError,
    },
    /// The INVOCATION itself must fail, a Lambda runtime error rather
    /// than a `FAILED` envelope, so the durable service re-invokes, the
    /// same recovery as an interruption. Produced by a retryable
    /// checkpoint write failure that exhausted transport-level retries:
    /// the write channel is down, so nothing more is written and the
    /// execution stays alive for the retry.
    Fault {
        /// The type of the fault, which the invocation end hook receives.
        /// It does not reach Lambda.
        error_type: String,
        /// The message reported as the Lambda invocation error.
        message: String,
    },
}

/// An unrecoverable error recorded by an operation, together with the
/// scope that must die for the execution to converge.
///
/// Producers:
/// - non-determinism detection
///   ([`crate::context::DurableContext::validate_replay_identity`]): a
///   replay identity mismatch means no amount of re-invocation can make
///   progress: [`FatalSeverity::FailExecution`];
/// - checkpoint write failures, recorded where the write's response
///   arrives, under the checkpoint-token lock
///   (`crate::context::DurableContext::record_write_failure`): an
///   `Invocation`-scoped failure fails the invocation so the service
///   re-invokes ([`FatalSeverity::FailInvocation`]); an `Execution`-scoped
///   one fails the execution ([`FatalSeverity::FailExecution`]).
/// - serdes faults ([`crate::context::DurableContext::serdes_fault`]): a
///   serdes error marked with
///   [`RetryableSerdesError`](crate::serdes::RetryableSerdesError) fails the
///   invocation so the service re-invokes
///   ([`FatalSeverity::FailInvocation`]).
///
/// The invocation driver checks this slot with priority over both
/// completion and suspension.
#[derive(Debug, Clone)]
pub(crate) struct FatalError {
    /// The wire error type (e.g. `NonDeterministicExecutionError`).
    pub(crate) error_type: String,
    /// The full error message.
    pub(crate) error_message: String,
    /// Which scope dies: the execution (terminal `FAILED` envelope) or
    /// only this invocation (Lambda runtime error; the service
    /// re-invokes).
    pub(crate) severity: FatalSeverity,
}

/// The recovery scope of a [`FatalError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FatalSeverity {
    /// The execution is over: report a `FAILED` envelope carrying the
    /// error, which the service records as the execution result.
    FailExecution,
    /// Only this invocation dies: report a Lambda runtime error (no
    /// envelope), and the service re-invokes: the interruption recovery
    /// path.
    FailInvocation,
}

impl FatalError {
    /// Converts the recorded fatal into the invocation outcome its
    /// severity demands.
    pub(crate) fn into_outcome(self) -> InvocationOutcome {
        match self.severity {
            FatalSeverity::FailExecution => InvocationOutcome::Failed {
                error: crate::error::WireError::new(
                    Some(self.error_type),
                    Some(self.error_message),
                ),
            },
            FatalSeverity::FailInvocation => InvocationOutcome::Fault {
                error_type: self.error_type,
                message: self.error_message,
            },
        }
    }
}

/// Identifies the task or thread that owns a context, and tracks the
/// blessed (SDK-started) task IDs.
///
/// A context created on a Tokio task is owned by that task. A context
/// created outside any task, as `lambda_runtime` and `LocalRunner` create
/// it under `block_on`, is owned by the OS thread that created it. A caller
/// may perform durable operations on the context when it is the owner task,
/// a blessed task, or, with no task ID, the owner thread. Every other task or
/// thread is refused: a raw `tokio::spawn` task has an unblessed task ID, and
/// a `std::thread` is a different thread.
///
/// The task gateway (`crate::task`) blesses every operation task it starts:
/// the handler itself, each `.spawn()`, each map item and parallel branch,
/// and each combinator input.
#[derive(Debug)]
pub(crate) struct TaskOwnership {
    /// The owner: a task, or the creating thread when the context was
    /// created outside any task.
    owner: Owner,
    /// The IDs of the running SDK operation tasks: these are exempt from
    /// the check. The task gateway (`crate::task`) inserts a task's ID
    /// before the task's first poll and removes it before the task exits,
    /// so tokio can reuse an ID only for a task that is not in the set.
    blessed_tasks: std::sync::Mutex<std::collections::HashSet<tokio::task::Id>>,
}

/// The task or thread that owns a context.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Owner {
    /// Created on this Tokio task.
    Task(tokio::task::Id),
    /// Created outside any Tokio task, on this OS thread.
    Thread(std::thread::ThreadId),
}

impl TaskOwnership {
    /// Creates ownership tracking anchored to the calling task, or to the
    /// calling OS thread when the caller runs outside any Tokio task.
    pub(crate) fn new_current() -> Self {
        let owner = match tokio::task::try_id() {
            Some(id) => Owner::Task(id),
            None => Owner::Thread(std::thread::current().id()),
        };
        Self {
            owner,
            blessed_tasks: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Registers a task ID as blessed (engine-spawned).
    ///
    /// Blessed tasks pass the ownership check even though they are not the
    /// original owner.
    pub(crate) fn bless_task(&self, task_id: tokio::task::Id) {
        let mut blessed = self
            .blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        blessed.insert(task_id);
    }

    /// Removes a task ID from the blessed set. The task gateway calls it
    /// before a blessed task exits.
    pub(crate) fn unbless_task(&self, task_id: tokio::task::Id) {
        self.blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&task_id);
    }

    /// Returns the number of blessed task IDs.
    #[cfg(test)]
    pub(crate) fn blessed_count(&self) -> usize {
        self.blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Checks whether the caller owns this context.
    ///
    /// A caller with a task ID passes when it is the owner task or a
    /// blessed task. A caller without a task ID passes only on the owner
    /// thread. Every other caller gets `Err` with a descriptive message.
    pub(crate) fn check_current_task(&self) -> Result<(), String> {
        if let Some(current_id) = tokio::task::try_id() {
            if matches!(self.owner, Owner::Task(owner_id) if owner_id == current_id) {
                return Ok(());
            }
            let blessed = self
                .blessed_tasks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if blessed.contains(&current_id) {
                return Ok(());
            }
            Err(format!(
                "durable operation invoked from task {current_id:?}, which does not own the \
                     context (owner: {:?}). Use .spawn() instead of tokio::spawn for durable \
                     fan-out",
                self.owner
            ))
        } else {
            let current = std::thread::current().id();
            if matches!(self.owner, Owner::Thread(owner) if owner == current) {
                return Ok(());
            }
            Err(format!(
                "durable operation invoked from thread {current:?}, which does not own the \
                     context (owner: {:?}). Use .spawn() instead of std::thread::spawn for durable \
                     fan-out",
                self.owner
            ))
        }
    }
}

/// How the drive of the root task ended. It carries no panic payload: the
/// coordinator owns every invocation-failing panic.
enum DriveEnd<T> {
    /// The root task returned the handler's result.
    Completed(T),
    /// A fatal error or a panic was recorded, or the phase became
    /// `Suspending`.
    Interrupted,
}

/// Closes panic ownership when the drive is dropped before it arbitrated.
///
/// A kept payload can hold a `DurableContext`, which keeps the coordinator
/// alive, so the coordinator's own drop cannot be relied on to drop it.
/// The guard takes the payload out under the coordinator lock, which also
/// closes ownership, and drops it through [`crate::task::discard_panic`].
/// A detached task that panics after that gets its payload back from
/// `record_panic` and disposes of it. After the arbitration, `take_panic`
/// returns nothing, so the guard's drop does nothing.
struct PanicOwnershipGuard(Arc<InvocationCoordinator>);

impl Drop for PanicOwnershipGuard {
    fn drop(&mut self) {
        if let Some(payload) = self.0.take_panic() {
            crate::task::discard_panic(payload);
        }
    }
}

/// Ends the invocation and stops its operation tasks when dropped.
///
/// The drive holds one. Its normal cleanup aborts and joins every task
/// itself. When the drive future is dropped before that, for example at a
/// platform timeout, this guard's drop is what stops the operation tasks
/// whose handles were dropped.
///
/// The root task can still be in a poll on another worker thread when the
/// drive is dropped, and that poll can start an operation task after the
/// drop. So the drop moves the phase to `Finished`, which refuses every
/// later spawn, and closes the tracker, which aborts a task whose spawn
/// took its permit before that.
struct TeardownGuard(Arc<InvocationCoordinator>);

impl Drop for TeardownGuard {
    fn drop(&mut self) {
        self.0.finish();
        self.0.tasks().close();
    }
}

/// Drives a single invocation of the user's handler future.
///
/// The handler runs as the invocation's root task, started through the
/// task gateway (`crate::task::spawn_root`), and polled by its
/// permit-aware wrapper. The drive ends in one of four ways:
///
/// 1. A fatal error is recorded.
/// 2. An invocation-failing panic is recorded on the coordinator: the
///    handler panicked, the refresh scheduler panicked, a detached task
///    (the batch flusher or a direct write, which run the checkpoint
///    client) panicked, or a task's outcome holding a panic lost its
///    reader.
/// 3. The phase becomes `Suspending`.
/// 4. The handler resolves. A wait whose handle the handler dropped does
///    not change that; its record can stay `STARTED`.
///
/// The outcome is decided once, after the cleanup below, from the fatal
/// error and the panic the coordinator keeps, read one after the other.
/// Only a fatal error present when `fatal_error()` is read wins; untracked
/// plugin hook work can record one after that read, and the outcome
/// ignores it:
///
/// 1. A fatal error was recorded: the fatal outcome. A kept panic is
///    dropped through [`crate::task::discard_panic`].
/// 2. Otherwise a panic is kept: `Err` with its original payload, which
///    the caller resumes. The coordinator keeps the first panic recorded;
///    every later one is dropped.
/// 3. Otherwise the handler's result: `Complete` or `Failed`, or `Pending`
///    when the drive ended by suspension.
///
/// When `remaining` is `Some(d)`, the driver also polls a sleep of `d`
/// directly, so a timer wake takes no permit. When it fires while the
/// phase is `Running`, no permit is held and no wait is live, it logs
/// [`UNTRACKED_DEADLINE_WARNING`] once. The coordinator also counts its
/// refresh deadline down from `d` in tokio time.
///
/// While the phase is `Running`, the driver polls the refresh scheduler
/// ([`refresh_scheduler`]) beside the handler, directly, so a timer wake
/// takes no permit. The scheduler sends its refreshes through `root`, a
/// clone of the root context. When the drive ends the scheduler is
/// dropped. A refresh it started runs on a detached task, so the cleanup
/// waits for its response to be merged.
///
/// Cleanup, after every ending: the scheduler is dropped, the phase
/// becomes `Finished`, the root task is aborted and awaited, every other
/// tracked task is aborted, and the driver waits until every tracked task
/// has unwound. Dropping the futures retires their wait registrations and
/// releases their permits. A panic during the cleanup is recorded, and the
/// cleanup continues to its end. Last, every panic still unread in a task
/// slot is recorded, including one whose handle outlives the handler. So
/// when this function returns, no SDK task, permit or wait registration of
/// this invocation remains.
///
/// When this future is dropped before it decides the outcome, a guard
/// closes panic ownership and drops a kept panic (see
/// [`PanicOwnershipGuard`]), and another ends the invocation and aborts
/// every tracked operation task, including one started after the drop (see
/// [`TeardownGuard`]).
///
/// Beside the outcome it returns whether a fatal error won this
/// arbitration (`true` only for the outcome of a recorded fatal error).
/// The caller uses that value, not the coordinator's fatal slot, which
/// code after the drive can still fill.
pub(crate) async fn drive_invocation<F>(
    handler_future: F,
    task_ownership: Arc<TaskOwnership>,
    coordinator: Arc<InvocationCoordinator>,
    root: crate::DurableContext,
    remaining: Option<Duration>,
) -> Result<(InvocationOutcome, bool), Box<dyn std::any::Any + Send + 'static>>
where
    F: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
{
    // The handler runs inside the root context's frames (the invocation)
    // and the span current when the invocation started.
    let root_cx = crate::task::TaskContext::current();
    drive_with_scheduler(
        root_cx,
        handler_future,
        task_ownership,
        coordinator,
        refresh_scheduler(root),
        remaining,
    )
    .await
}

/// [`drive_invocation`] with `scheduler` polled in place of the refresh
/// scheduler. Tests pass a scheduler of their own.
async fn drive_with_scheduler<F, S>(
    root_cx: crate::task::TaskContext,
    handler_future: F,
    task_ownership: Arc<TaskOwnership>,
    coordinator: Arc<InvocationCoordinator>,
    scheduler: S,
    remaining: Option<Duration>,
) -> Result<(InvocationOutcome, bool), Box<dyn std::any::Any + Send + 'static>>
where
    F: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
    S: Future<Output = ()>,
{
    let ownership_guard = PanicOwnershipGuard(Arc::clone(&coordinator));
    // Dropping a `.spawn()` handle does not stop its task. So when this
    // future is dropped before the cleanup below, this guard ends the
    // invocation and stops every tracked operation task, including one the
    // root starts after the drop. After the cleanup no task is live, the
    // phase is already `Finished`, and its drop only closes the tracker.
    let _teardown = TeardownGuard(Arc::clone(&coordinator));
    coordinator.start_deadline(remaining);
    // Boxed, so the cleanup below can drop it before it waits for the tasks.
    let mut scheduler = Box::pin(scheduler);
    // A panic can be recorded before the drive starts, for example from the
    // destructor of a plugin factory's error. Keeping it ends the drive, so
    // the handler must never run. The check in the poll
    // below comes too late: on a multi-thread runtime another worker can
    // poll the root task before the driver polls at all. So the phase ends
    // here, before the root is spawned. The root then starts idle, it can
    // take no permit, and the gateway never polls the handler.
    if coordinator.has_panic() {
        coordinator.finish();
    }
    // The handler runs as an operation task, so every SDK poll of user code
    // runs on a Tokio worker with the gateway wrapper.
    #[expect(clippy::expect_used)]
    // reason: this function awaits Tokio timers and tasks itself, so it is
    // always polled inside a Tokio runtime.
    let (mut root, root_join) =
        crate::task::spawn_root(root_cx, &task_ownership, &coordinator, handler_future)
            .expect("the invocation driver runs inside a Tokio runtime");

    let mut deadline = remaining.map(|d| Box::pin(tokio::time::sleep(d)));
    let ended = std::future::poll_fn(|cx| {
        coordinator.register_driver_waker(cx.waker());
        if coordinator.fatal_error().is_some() || coordinator.has_panic() {
            return Poll::Ready(DriveEnd::Interrupted);
        }
        match Pin::new(&mut root).poll(cx) {
            Poll::Ready(crate::task::TaskOutcome::Value(result)) => {
                return Poll::Ready(DriveEnd::Completed(result));
            }
            Poll::Ready(crate::task::TaskOutcome::Panicked(payload)) => {
                crate::task::record_or_discard(&coordinator, payload);
                return Poll::Ready(DriveEnd::Interrupted);
            }
            Poll::Pending => {}
        }
        if coordinator.phase() == Phase::Suspending {
            return Poll::Ready(DriveEnd::Interrupted);
        }
        // A fatal error or `Suspending` ended the drive above. The phase can
        // still be `Finished` for one poll, between the root task's finish
        // and its outcome. The coordinator then refuses the scheduler's
        // permit, so the scheduler sends nothing. It loops until it is
        // dropped, so it never resolves.
        //
        // The scheduler is polled here, outside every task, so a panic in
        // it would unwind this function and skip the cleanup below. The
        // panic is caught and recorded, which ends the drive.
        if let Err(payload) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scheduler.as_mut().poll(cx)))
        {
            crate::task::record_or_discard(&coordinator, payload);
            return Poll::Ready(DriveEnd::Interrupted);
        }
        if let Some(sleep) = deadline.as_mut()
            && sleep.as_mut().poll(cx).is_ready()
        {
            deadline = None;
            if coordinator.running_and_idle() {
                tracing::warn!("{UNTRACKED_DEADLINE_WARNING}");
            }
        }
        Poll::Pending
    })
    .await;

    // Cleanup, whatever ended the drive. No task is polled from here on.
    // The scheduler stops; a refresh it started runs on a detached task,
    // which the cleanup waits for. Its drop runs inside a catch, so a panic
    // is recorded and the cleanup goes on. The root task is aborted and
    // awaited, so the handler has unwound and dropped every handle it held
    // before the other tasks are aborted. The task gateway records a panic
    // of any future it drops, and a dropped handle records a panic it did
    // not read. Then the driver waits until every tracked task has unwound,
    // so no SDK task outlives the invocation, and claims every panic still
    // unread in a slot.
    crate::task::drop_recorded(&coordinator, scheduler);
    coordinator.finish();
    root.abort();
    drop(root);
    drop(root_join.await);
    let tasks = Arc::clone(coordinator.tasks());
    tasks.abort_all();
    tasks.wait_idle().await;
    tasks.claim_unread_panics(&coordinator);

    // The arbitration. Every SDK task has stopped. Plugin hook work is
    // untracked, so it can record a fatal error concurrently with the
    // `fatal_error()` read below: a record that lands before that read is
    // used, and one that lands after it is not read. `take_panic` closes
    // panic ownership, so a later panic record is returned to its caller.
    let fatal = coordinator.fatal_error();
    let panic = coordinator.take_panic();
    drop(ownership_guard);
    #[cfg(test)]
    test_hooks::after_arbitration(&coordinator);
    if let Some(fatal) = fatal {
        if let Some(payload) = panic {
            crate::task::discard_panic(payload);
        }
        return Ok((fatal.into_outcome(), true));
    }
    if let Some(payload) = panic {
        return Err(payload);
    }
    let outcome = match ended {
        DriveEnd::Completed(Ok(result)) => InvocationOutcome::Complete(result),
        DriveEnd::Completed(Err(err)) => InvocationOutcome::Failed { error: err },
        DriveEnd::Interrupted => InvocationOutcome::Pending,
    };
    Ok((outcome, false))
}

/// The answer of an invocation whose checkpoint token was withdrawn.
///
/// The invocation wrapper calls it after the end-of-invocation flush, on
/// every exit that returns a response or a Lambda error: handler `Ok` or
/// `Err`, a transport error of either scope, a recorded fatal error and a
/// serdes fault. A panic is not such an exit, and propagates.
///
/// Without the latch it returns `response` and `end` unchanged. With it the
/// response is exactly `{"Status":"PENDING"}`, nothing is recorded as a
/// fault or an execution failure, and the end hook receives `Pending` with
/// the error `end` carried, or none.
pub(crate) fn withdrawal_answer(
    ctx: &crate::DurableContext,
    response: Result<crate::InvocationResponse, lambda_runtime::Error>,
    end: crate::InvocationEnd,
) -> (
    Result<crate::InvocationResponse, lambda_runtime::Error>,
    crate::InvocationEnd,
) {
    if !ctx.coordinator().withdrawn() {
        return (response, end);
    }
    // The end keeps the error the same path carries without withdrawal: a
    // fatal error the arbitration read, else the handler's error, else
    // none. A recorded checkpoint failure never meets a withdrawal,
    // because either latch stops the next request.
    // A suspension already answers `PENDING`, and emitted its event.
    if end.status == crate::InvocationStatus::Pending {
        return (response, end);
    }
    drop(response);
    let (pending, mut pending_end) = crate::outcome_envelope(InvocationOutcome::Pending, ctx);
    pending_end.error = end.error;
    (pending, pending_end)
}

/// [`drive_invocation`] without the fatal-arbitration flag, for callers
/// that need only the outcome.
#[cfg(test)]
pub(crate) async fn drive_outcome<F>(
    handler_future: F,
    task_ownership: Arc<TaskOwnership>,
    coordinator: Arc<InvocationCoordinator>,
    root: crate::DurableContext,
    remaining: Option<Duration>,
) -> Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>
where
    F: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
{
    drive_invocation(handler_future, task_ownership, coordinator, root, remaining)
        .await
        .map(|(outcome, _)| outcome)
}

/// Test-only hooks of the driver.
#[cfg(test)]
pub(crate) mod test_hooks {
    tokio::task_local! {
        /// When set, `drive_invocation` calls it with the coordinator after
        /// its arbitration and before it returns, as hook work that records
        /// at that point would.
        pub(crate) static AFTER_ARBITRATION: std::sync::Arc<
            dyn Fn(&super::InvocationCoordinator) + Send + Sync,
        >;
    }

    pub(crate) fn after_arbitration(coordinator: &super::InvocationCoordinator) {
        if let Ok(hook) = AFTER_ARBITRATION.try_with(std::sync::Arc::clone) {
            hook(coordinator);
        }
    }
}

/// Refreshes waiting operations on the schedule of [`crate::coordinator`],
/// until it is dropped.
///
/// It sleeps until the earliest due refresh, and re-arms when a
/// registration is added, armed or retired. When refreshes are due it
/// takes a scheduler permit, which the coordinator grants only beside an
/// activity or checkpoint permit. It sends one checkpoint with no updates
/// through `root`, whose merge wakes every operation the response
/// readied, and then schedules each due registration's next refresh. Only
/// then does it release the permit. A refused permit stops the scheduler.
/// A failed refresh is recorded as a failed checkpoint write of the
/// first due operation, and never resolves, so the scheduler sends nothing
/// more. The failure is recorded where the response arrives, not here, so
/// a refresh that is in flight when the handler returns still decides the
/// outcome (see [`crate::DurableContext::checkpoint_refresh`]).
async fn refresh_scheduler(root: crate::DurableContext) {
    use crate::coordinator::{DueRefresh, PermitKind};

    let coordinator = Arc::clone(root.coordinator());
    loop {
        match coordinator.next_refresh_at() {
            None => {
                coordinator.rearmed().await;
                continue;
            }
            Some(at) => {
                tokio::select! {
                    biased;
                    () = coordinator.rearmed() => continue,
                    () = tokio::time::sleep_until(at) => {}
                }
            }
        }
        let (keys, first_wire_id) = match coordinator.take_due(tokio::time::Instant::now()) {
            DueRefresh::Nothing | DueRefresh::UnderFloor => continue,
            DueRefresh::Due {
                keys,
                first_wire_id,
            } => (keys, first_wire_id),
        };
        let Some(permit) = coordinator.acquire(PermitKind::Scheduler) else {
            return std::future::pending().await;
        };
        root.checkpoint_refresh(&first_wire_id).await;
        let records = keys
            .iter()
            .filter_map(|(wire_id, _)| {
                root.refresh_view(wire_id)
                    .map(|view| (wire_id.clone(), view))
            })
            .collect();
        coordinator.after_refresh(&keys, &records);
        drop(permit);
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{InvocationOutcome, TaskOwnership, drive_outcome};
    use crate::client::{CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput};
    use crate::coordinator::InvocationCoordinator;
    use aws_sdk_lambda::types::{Operation, OperationStatus, OperationUpdate};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Drives `fut` through [`drive_invocation`] and resumes a handler
    /// panic, for tests that expect an outcome.
    pub(crate) async fn drive_ok<F>(
        root: crate::DurableContext,
        fut: F,
        ownership: Arc<TaskOwnership>,
        coordinator: Arc<InvocationCoordinator>,
    ) -> InvocationOutcome
    where
        F: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
    {
        match drive_outcome(fut, ownership, coordinator, root, None).await {
            Ok(outcome) => outcome,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Drives an operation future through the real invocation driver and
    /// returns the invocation outcome. The operation's own result is
    /// discarded: a waiting operation yields
    /// [`InvocationOutcome::Pending`] exactly as in production, and a
    /// completing one yields `Complete`. Used by operation unit tests to
    /// drive a waiting future to an outcome instead of awaiting it directly
    /// (a direct await would wait forever).
    ///
    /// `ownership` is the task-ownership tracker of the context the
    /// operation runs on: the driver blesses its root task there.
    pub(crate) async fn outcome_of<F>(
        root: crate::DurableContext,
        ownership: Arc<TaskOwnership>,
        coordinator: Arc<InvocationCoordinator>,
        fut: F,
    ) -> InvocationOutcome
    where
        F: IntoFuture + Send + 'static,
        F::IntoFuture: Send + 'static,
    {
        // Box the operation future: `drive_invocation` stack-pins its
        // argument, so a large operation future would otherwise inflate
        // every test future that awaits this helper (clippy
        // `large_futures`). Test-only; production handler futures are
        // driven unboxed.
        let fut = Box::pin(async move {
            let _ = fut.await;
            Ok::<_, crate::error::WireError>("ok".to_owned())
        });
        match drive_outcome(fut, ownership, coordinator, root, None).await {
            Ok(outcome) => outcome,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Drives `fut` to its invocation outcome and asserts that it ended the
    /// invocation with a fault whose message contains `needle`.
    #[expect(clippy::panic)] // reason: test assertion on an unexpected outcome
    pub(crate) async fn expect_serdes_fault<F>(
        root: crate::DurableContext,
        ownership: Arc<TaskOwnership>,
        coordinator: Arc<InvocationCoordinator>,
        fut: F,
        needle: &str,
    ) where
        F: IntoFuture + Send + 'static,
        F::IntoFuture: Send + 'static,
    {
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            outcome_of(root, ownership, coordinator, fut),
        )
        .await
        .expect("the operation must reach an invocation outcome within 5 s");
        let InvocationOutcome::Fault { message, .. } = outcome else {
            panic!("expected an invocation fault, got {outcome:?}");
        };
        assert!(
            message.contains(needle),
            "fault message {message:?} must contain {needle:?}"
        );
    }

    /// Decides the operations one checkpoint response carries, from the
    /// updates of the call and every update received so far (this call's
    /// included).
    type Responder = dyn Fn(&[OperationUpdate], &[OperationUpdate]) -> Vec<Operation> + Send + Sync;

    /// A test `ExecutionClient` that answers each checkpoint call with the
    /// operations its responder returns, optionally after a delay of tokio
    /// time, and records every call.
    pub(crate) struct ScriptedClient {
        respond: Box<Responder>,
        delay: Option<Duration>,
        /// Notified before the first call returns, when set.
        first_call_gate: Option<Arc<tokio::sync::Notify>>,
        calls: Mutex<Vec<(String, Vec<OperationUpdate>)>>,
        tokens: Mutex<u32>,
    }

    impl ScriptedClient {
        /// A client whose responses carry what `respond` returns.
        pub(crate) fn new(
            respond: impl Fn(&[OperationUpdate], &[OperationUpdate]) -> Vec<Operation>
            + Send
            + Sync
            + 'static,
        ) -> Self {
            Self {
                respond: Box::new(respond),
                delay: None,
                first_call_gate: None,
                calls: Mutex::new(Vec::new()),
                tokens: Mutex::new(0),
            }
        }

        /// A client whose responses carry nothing.
        pub(crate) fn empty() -> Self {
            Self::new(|_, _| Vec::new())
        }

        /// Answers each call after `delay` of tokio time.
        pub(crate) fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = Some(delay);
            self
        }

        /// Holds the first call's response until `gate` is notified.
        pub(crate) fn with_first_call_gate(mut self, gate: Arc<tokio::sync::Notify>) -> Self {
            self.first_call_gate = Some(gate);
            self
        }

        /// Every call's checkpoint token and updates, in call order.
        pub(crate) fn calls(&self) -> Vec<(String, Vec<OperationUpdate>)> {
            self.calls.lock().expect("lock").clone()
        }

        /// Every update received, in call order.
        pub(crate) fn updates(&self) -> Vec<OperationUpdate> {
            self.calls().into_iter().flat_map(|(_, u)| u).collect()
        }

        /// The number of checkpoint calls received.
        pub(crate) fn call_count(&self) -> usize {
            self.calls.lock().expect("lock").len()
        }

        /// Whether an update of `name` with `action` was received.
        pub(crate) fn received(
            &self,
            name: &str,
            action: &aws_sdk_lambda::types::OperationAction,
        ) -> bool {
            self.updates()
                .iter()
                .any(|u| u.name() == Some(name) && u.action() == action)
        }
    }

    impl ExecutionClient for ScriptedClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            let (operations, first) = {
                let mut calls = self.calls.lock().expect("lock");
                let mut history: Vec<OperationUpdate> =
                    calls.iter().flat_map(|(_, u)| u.iter().cloned()).collect();
                history.extend(updates.iter().cloned());
                let operations = (self.respond)(&updates, &history);
                calls.push((checkpoint_token.to_owned(), updates));
                (operations, calls.len() == 1)
            };
            let token = {
                let mut tokens = self.tokens.lock().expect("lock");
                *tokens += 1;
                format!("t{tokens}")
            };
            let delay = self.delay;
            let gate = if first {
                self.first_call_gate.clone()
            } else {
                None
            };
            Box::pin(async move {
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                Ok(CheckpointOutput::new(Some(token), operations, None))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(GetStateOutput::new(Vec::new(), None)) })
        }
    }

    /// Polls `f` once with the calling task's context, and returns.
    pub(crate) async fn poll_once<F: Future>(mut f: Pin<&mut F>) {
        std::future::poll_fn(|cx| {
            // The handle stays pending: the tests that call this await it
            // later.
            let _ = f.as_mut().poll(cx);
            std::task::Poll::Ready(())
        })
        .await;
    }

    /// One request a [`RefreshClient`] answers.
    pub(crate) struct RefreshRequest<'a> {
        /// The request's updates.
        pub(crate) updates: &'a [OperationUpdate],
        /// Every update received so far, this request's included.
        pub(crate) history: &'a [OperationUpdate],
        /// The 1-based number of this request among the requests with no
        /// updates. 0 for a request with updates.
        pub(crate) empty_number: usize,
        /// The tokio time since the client was created, when the response
        /// is built.
        pub(crate) now: Duration,
        /// The wall-clock time when the client was created.
        pub(crate) base: std::time::SystemTime,
    }

    impl RefreshRequest<'_> {
        /// The timestamp `in_` after the request's tokio time, in the
        /// wall-clock time the SDK reads under paused tokio time: the
        /// client's creation time plus the tokio time since.
        pub(crate) fn timestamp_in(&self, in_: Duration) -> aws_smithy_types::DateTime {
            aws_smithy_types::DateTime::from(self.base + self.now + in_)
        }
    }

    /// Decides the response to one request.
    type RefreshResponder =
        dyn Fn(&RefreshRequest<'_>) -> Result<Vec<Operation>, ExecutionClientError> + Send + Sync;

    /// A test `ExecutionClient` for the refresh scheduler. It records the
    /// tokio time of every call and of every answer to a call with no
    /// updates, delays the answer to such a call by a fixed time, and
    /// builds each response when it answers.
    pub(crate) struct RefreshClient {
        respond: Box<RefreshResponder>,
        empty_delay: Duration,
        start: tokio::time::Instant,
        base: std::time::SystemTime,
        calls: Mutex<Vec<(Duration, Vec<OperationUpdate>)>>,
        empty_answers: Mutex<Vec<Duration>>,
        tokens: Mutex<u32>,
    }

    impl RefreshClient {
        /// A client whose responses `respond` builds. Create it inside the
        /// test's runtime: its clock starts here.
        pub(crate) fn new(
            respond: impl Fn(&RefreshRequest<'_>) -> Result<Vec<Operation>, ExecutionClientError>
            + Send
            + Sync
            + 'static,
        ) -> Self {
            Self {
                respond: Box::new(respond),
                empty_delay: Duration::ZERO,
                start: tokio::time::Instant::now(),
                base: std::time::SystemTime::now(),
                calls: Mutex::new(Vec::new()),
                empty_answers: Mutex::new(Vec::new()),
                tokens: Mutex::new(0),
            }
        }

        /// Answers each call with no updates after `delay` of tokio time.
        pub(crate) fn with_empty_delay(mut self, delay: Duration) -> Self {
            self.empty_delay = delay;
            self
        }

        /// The tokio time since the client was created.
        pub(crate) fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        /// The tokio time of each call with no updates, in call order.
        pub(crate) fn empty_calls(&self) -> Vec<Duration> {
            self.calls
                .lock()
                .expect("lock")
                .iter()
                .filter(|(_, updates)| updates.is_empty())
                .map(|(at, _)| *at)
                .collect()
        }

        /// The tokio time of each answer to a call with no updates.
        pub(crate) fn empty_answers(&self) -> Vec<Duration> {
            self.empty_answers.lock().expect("lock").clone()
        }

        /// The tokio time of the first call that carried an update of
        /// `name` with `action`.
        pub(crate) fn time_of(
            &self,
            name: &str,
            action: &aws_sdk_lambda::types::OperationAction,
        ) -> Option<Duration> {
            self.calls
                .lock()
                .expect("lock")
                .iter()
                .find(|(_, updates)| named(updates, name, action).is_some())
                .map(|(at, _)| *at)
        }

        /// Every update received, in call order.
        pub(crate) fn updates(&self) -> Vec<OperationUpdate> {
            self.calls
                .lock()
                .expect("lock")
                .iter()
                .flat_map(|(_, u)| u.iter().cloned())
                .collect()
        }

        /// Whether an update of `name` with `action` was received.
        pub(crate) fn received(
            &self,
            name: &str,
            action: &aws_sdk_lambda::types::OperationAction,
        ) -> bool {
            self.time_of(name, action).is_some()
        }
    }

    impl ExecutionClient for RefreshClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            let (history, empty_number) = {
                let mut calls = self.calls.lock().expect("lock");
                calls.push((self.start.elapsed(), updates.clone()));
                let history: Vec<OperationUpdate> =
                    calls.iter().flat_map(|(_, u)| u.iter().cloned()).collect();
                let empty_number = if updates.is_empty() {
                    calls.iter().filter(|(_, u)| u.is_empty()).count()
                } else {
                    0
                };
                (history, empty_number)
            };
            let token = {
                let mut tokens = self.tokens.lock().expect("lock");
                *tokens += 1;
                format!("t{tokens}")
            };
            Box::pin(async move {
                if empty_number > 0 && !self.empty_delay.is_zero() {
                    tokio::time::sleep(self.empty_delay).await;
                }
                let now = self.start.elapsed();
                let operations = (self.respond)(&RefreshRequest {
                    updates: &updates,
                    history: &history,
                    empty_number,
                    now,
                    base: self.base,
                })?;
                if empty_number > 0 {
                    self.empty_answers.lock().expect("lock").push(now);
                }
                Ok(CheckpointOutput::new(Some(token), operations, None))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(GetStateOutput::new(Vec::new(), None)) })
        }
    }

    /// A root context on `client`, with an empty checkpoint log, whose
    /// Lambda deadline is 0 (none).
    pub(crate) fn refresh_ctx(client: &Arc<RefreshClient>) -> crate::DurableContext {
        crate::DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
        )
    }

    /// The record of a callback whose START is `start`, with `status`, its
    /// callback ID and `result`.
    pub(crate) fn callback_record(
        start: &OperationUpdate,
        status: OperationStatus,
        result: Option<&str>,
    ) -> Operation {
        let mut details = aws_sdk_lambda::types::CallbackDetails::builder().callback_id("cb-1");
        if let Some(result) = result {
            details = details.result(result);
        }
        operation_like(start, status)
            .callback_details(details.build())
            .build()
            .expect("all required fields set")
    }

    /// Answers the START of every callback among `updates` with `STARTED`
    /// and its callback ID.
    pub(crate) fn callback_starts(updates: &[OperationUpdate]) -> Vec<Operation> {
        updates
            .iter()
            .filter(|u| {
                u.r#type() == &aws_sdk_lambda::types::OperationType::Callback
                    && u.action() == &aws_sdk_lambda::types::OperationAction::Start
            })
            .map(|start| callback_record(start, OperationStatus::Started, None))
            .collect()
    }

    /// Asserts that `actual` is `expected` within 50 ms. The end times a
    /// response carries are wall-clock times taken when the response is
    /// built, so a delay computed from one can be off by the real time a
    /// test takes between two steps.
    #[track_caller]
    pub(crate) fn assert_near(actual: Duration, expected: Duration) {
        let diff = actual.abs_diff(expected);
        assert!(
            diff <= Duration::from_millis(50),
            "expected {expected:?}, got {actual:?}"
        );
    }

    /// Asserts that `actual` holds one time per entry of `expected`, each
    /// within 50 ms (see [`assert_near`]).
    #[track_caller]
    pub(crate) fn assert_times(actual: &[Duration], expected: &[Duration]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "expected times {expected:?}, got {actual:?}"
        );
        for (a, e) in actual.iter().zip(expected) {
            assert_near(*a, *e);
        }
    }

    /// A root context on `client`, with an empty checkpoint log.
    pub(crate) fn scripted_ctx(client: &Arc<ScriptedClient>) -> crate::DurableContext {
        crate::DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
        )
    }

    /// An operation record with the identity of `update` (its ID, type,
    /// sub-type, name and parent) and `status`, as a checkpoint response
    /// carries it.
    pub(crate) fn operation_like(
        update: &OperationUpdate,
        status: OperationStatus,
    ) -> aws_sdk_lambda::types::builders::OperationBuilder {
        let mut builder = Operation::builder()
            .id(update.id())
            .r#type(update.r#type().clone())
            .status(status)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0));
        if let Some(sub_type) = update.sub_type() {
            builder = builder.sub_type(sub_type);
        }
        if let Some(name) = update.name() {
            builder = builder.name(name);
        }
        if let Some(parent) = update.parent_id() {
            builder = builder.parent_id(parent);
        }
        builder
    }

    /// The first update in `updates` named `name` with `action`.
    pub(crate) fn named<'a>(
        updates: &'a [OperationUpdate],
        name: &str,
        action: &aws_sdk_lambda::types::OperationAction,
    ) -> Option<&'a OperationUpdate> {
        updates
            .iter()
            .find(|u| u.name() == Some(name) && u.action() == action)
    }

    /// Drives `handler` on `ctx` through [`drive_invocation`], with
    /// `remaining` as the time left.
    pub(crate) async fn drive_handler<F, Fut>(
        ctx: crate::DurableContext,
        remaining: Option<Duration>,
        handler: F,
    ) -> Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>
    where
        F: FnOnce(crate::DurableContext) -> Fut,
        Fut: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
    {
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let handler = Box::pin(handler(ctx));
        drive_outcome(handler, ownership, coordinator, root, remaining).await
    }

    /// Records the level and message of every event while installed as
    /// the thread's default subscriber.
    #[derive(Clone, Default)]
    pub(crate) struct EventMessages(Arc<Mutex<Vec<(tracing::Level, String)>>>);

    struct MessageVisitor(String);

    impl tracing::field::Visit for MessageVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for EventMessages {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.0
                .lock()
                .expect("lock")
                .push((*event.metadata().level(), visitor.0));
        }
    }

    impl EventMessages {
        /// Installs a capture as the thread's default subscriber until the
        /// returned guard drops.
        pub(crate) fn install() -> (Self, tracing::subscriber::DefaultGuard) {
            use tracing_subscriber::layer::SubscriberExt as _;
            let capture = Self::default();
            let subscriber = tracing_subscriber::registry().with(capture.clone());
            (capture, tracing::subscriber::set_default(subscriber))
        }

        /// The messages of the events at `level`, in emission order.
        pub(crate) fn at(&self, level: tracing::Level) -> Vec<String> {
            self.0
                .lock()
                .expect("lock")
                .iter()
                .filter(|(l, _)| *l == level)
                .map(|(_, m)| m.clone())
                .collect()
        }

        /// The number of `operation_replayed` events.
        pub(crate) fn replayed(&self) -> usize {
            self.at(tracing::Level::DEBUG)
                .iter()
                .filter(|m| *m == crate::observability::event_names::OPERATION_REPLAYED)
                .count()
        }
    }

    /// Turns an operation error into the handler's wire error.
    #[expect(clippy::needless_pass_by_value)]
    // reason: it is passed to `map_err`, which hands over the error.
    pub(crate) fn wire_err(err: crate::OperationError) -> crate::error::WireError {
        crate::error::WireError::new(None::<String>, Some(format!("{err:#}")))
    }

    /// One checkpoint request a [`LatencyClient`] received.
    #[derive(Clone)]
    pub(crate) struct LatencyRequest {
        /// The tokio time since the client was created, when it received
        /// the request.
        pub(crate) received: Duration,
        /// The tokio time when it answered the request. `None` until then.
        pub(crate) answered: Option<Duration>,
        /// The request's updates.
        pub(crate) updates: Vec<OperationUpdate>,
    }

    /// A test `ExecutionClient` that answers each checkpoint call
    /// `latency` of tokio time after it receives it, and records the time
    /// of each request and of each answer.
    ///
    /// It answers a `START` with a `STARTED` record of the operation, so
    /// the checkpoint log holds the record the service would return. With
    /// `fail_start`, it answers a request that carries a `START` with a
    /// non-retryable error instead.
    pub(crate) struct LatencyClient {
        latency: Duration,
        fail_start: bool,
        start: tokio::time::Instant,
        requests: Arc<Mutex<Vec<LatencyRequest>>>,
        tokens: Mutex<u32>,
    }

    impl LatencyClient {
        /// A client that answers after `latency`. Create it inside the
        /// test's runtime: its clock starts here.
        pub(crate) fn new(latency: Duration) -> Self {
            Self {
                latency,
                fail_start: false,
                start: tokio::time::Instant::now(),
                requests: Arc::new(Mutex::new(Vec::new())),
                tokens: Mutex::new(0),
            }
        }

        /// Answers a request that carries a `START` with a non-retryable
        /// error.
        pub(crate) fn failing_start(mut self) -> Self {
            self.fail_start = true;
            self
        }

        /// The tokio time since the client was created.
        pub(crate) fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        /// Every request received, in order.
        pub(crate) fn requests(&self) -> Vec<LatencyRequest> {
            self.requests.lock().expect("lock").clone()
        }

        /// The first request that carries an update with `action`.
        pub(crate) fn request_with(
            &self,
            action: &aws_sdk_lambda::types::OperationAction,
        ) -> Option<LatencyRequest> {
            self.requests()
                .into_iter()
                .find(|r| r.updates.iter().any(|u| u.action() == action))
        }

        /// Whether any request carried a `SUCCEED`, `RETRY` or `FAIL`
        /// update.
        pub(crate) fn received_outcome(&self) -> bool {
            use aws_sdk_lambda::types::OperationAction;
            self.requests()
                .iter()
                .flat_map(|r| r.updates.clone())
                .any(|u| {
                    matches!(
                        u.action(),
                        OperationAction::Succeed | OperationAction::Retry | OperationAction::Fail
                    )
                })
        }
    }

    impl ExecutionClient for LatencyClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            use aws_sdk_lambda::types::OperationAction;
            let index = {
                let mut requests = self.requests.lock().expect("lock");
                requests.push(LatencyRequest {
                    received: self.start.elapsed(),
                    answered: None,
                    updates: updates.clone(),
                });
                requests.len() - 1
            };
            let token = {
                let mut tokens = self.tokens.lock().expect("lock");
                *tokens += 1;
                format!("t{tokens}")
            };
            Box::pin(async move {
                tokio::time::sleep(self.latency).await;
                if let Some(request) = self.requests.lock().expect("lock").get_mut(index) {
                    request.answered = Some(self.start.elapsed());
                }
                let starts: Vec<&OperationUpdate> = updates
                    .iter()
                    .filter(|u| u.action() == &OperationAction::Start)
                    .collect();
                if self.fail_start && !starts.is_empty() {
                    return Err(ExecutionClientError::new_non_retryable("START rejected"));
                }
                let operations = starts
                    .into_iter()
                    .map(|u| {
                        operation_like(u, OperationStatus::Started)
                            .build()
                            .expect("all required fields set")
                    })
                    .collect();
                Ok(CheckpointOutput::new(Some(token), operations, None))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(GetStateOutput::new(Vec::new(), None)) })
        }
    }

    /// A root context on `client`, with an empty checkpoint log.
    /// `buffer_window` is `None` for direct writes (batching off), and
    /// `Some(delay)` for batching with `checkpoint_delay(delay)`.
    pub(crate) fn latency_ctx(
        client: &Arc<LatencyClient>,
        buffer_window: Option<Duration>,
    ) -> crate::DurableContext {
        crate::DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
            buffer_window,
            0,
            None,
        )
    }
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::test_support::drive_ok;
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    // Suspension drops the future at the await point

    /// A guard that sets a flag when dropped: proves the future was
    /// dropped at the await point.
    struct DropGuard {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    /// A root context whose checkpoint calls carry nothing back, so a wait
    /// started on it stays waiting.
    fn waiting_ctx() -> crate::DurableContext {
        let client = Arc::new(test_support::ScriptedClient::empty());
        test_support::scripted_ctx(&client)
    }

    #[tokio::test]
    async fn suspension_drops_future_at_await_point() {
        let ctx = waiting_ctx();
        let drop_observed = Arc::new(AtomicBool::new(false));
        let after_await_ran = Arc::new(AtomicBool::new(false));

        let drop_clone = Arc::clone(&drop_observed);
        let after_clone = Arc::clone(&after_await_ran);
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let _guard = DropGuard {
                    dropped: drop_clone,
                };

                // A wait that nothing completes: the invocation suspends
                // and the driver drops this future.
                let _ = ctx.wait(Duration::from_secs(30)).await;

                // This code must NEVER run: the future is dropped above.
                after_clone.store(true, Ordering::SeqCst);
                Ok("should not reach".to_owned())
            },
            ownership,
            coordinator,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            drop_observed.load(Ordering::SeqCst),
            "future was not dropped at the await point"
        );
        assert!(
            !after_await_ran.load(Ordering::SeqCst),
            "code after the suspension point must not run"
        );
    }

    // Suspension is unswallowable (no catchable error)

    #[tokio::test]
    async fn suspension_unswallowable_by_catch_all() {
        let ctx = waiting_ctx();
        let after_catch_ran = Arc::new(AtomicBool::new(false));
        let after_clone = Arc::clone(&after_catch_ran);
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                // User tries a "catch-all" pattern: wrapping in a closure
                // that catches panics. This cannot catch future-drop.
                let result: Result<String, crate::error::WireError> = async {
                    let _ = ctx.wait(Duration::from_secs(30)).await;
                    Ok("survived".to_owned())
                }
                .await;

                // If somehow the suspension was catchable, this would run:
                after_clone.store(true, Ordering::SeqCst);
                result
            },
            ownership,
            coordinator,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !after_catch_ran.load(Ordering::SeqCst),
            "catch-all must not intercept suspension — future is dropped"
        );
    }

    // PENDING outcome surfaces to the driver caller

    #[tokio::test]
    async fn pending_outcome_surfaces_to_caller() {
        let ctx = waiting_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let _ = ctx.wait(Duration::from_secs(30)).await;
                Ok("unreachable".to_owned())
            },
            ownership,
            coordinator,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
    }

    // Normal completion

    #[tokio::test]
    async fn driver_returns_complete_on_success() {
        let ctx = crate::DurableContext::test_context();
        let signal = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            ctx.clone(),
            async move { Ok::<_, crate::error::WireError>("done".to_owned()) },
            Arc::clone(ctx.task_ownership()),
            Arc::clone(&signal),
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    #[tokio::test]
    async fn driver_returns_failed_on_error() {
        let ctx = crate::DurableContext::test_context();
        let signal = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                Err(crate::error::WireError::new(
                    Some("Error".to_owned()),
                    Some("boom".to_owned()),
                ))
            },
            Arc::clone(ctx.task_ownership()),
            Arc::clone(&signal),
        )
        .await;

        let InvocationOutcome::Failed { error } = outcome else {
            unreachable!("expected Failed, got {outcome:?}");
        };
        assert_eq!(error.error_type(), Some("Error"));
        assert_eq!(error.error_message(), Some("boom"));
    }

    // Resume: replayed operations return frozen results

    #[tokio::test]
    async fn resume_replays_frozen_results_without_re_executing() {
        // Simulate a resume invocation: checkpoint log has a result for
        // operation "1" (keyed by its wire ID), so the engine should
        // return the frozen value without calling the step body.
        let execution_count = Arc::new(AtomicU32::new(0));
        let exec_clone = Arc::clone(&execution_count);

        let wire_key = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire_key.clone(),
            CheckpointRecord {
                id: wire_key,
                status: CheckpointStatus::Succeeded,
                result: Some(r#""frozen_value""#.to_owned()),
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
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
            },
        )]));

        let engine = EngineState::new_root(Arc::clone(&log));

        // Simulate what the engine does on resume: check if replaying at
        // the operation's position.
        let op_id = engine.mint_id();
        let is_replay = engine.is_replaying_at(op_id.positional());

        let result = if is_replay {
            // Replay path: return frozen result without executing.
            let wire = crate::engine::compute_wire_id_public(op_id.positional());
            let record = log.get(&wire);
            record.unwrap().result.clone()
        } else {
            // Live path: execute the body (should NOT happen here).
            exec_clone.fetch_add(1, Ordering::SeqCst);
            Some(r#""live_value""#.to_owned())
        };

        assert_eq!(result, Some(r#""frozen_value""#.to_owned()));
        assert_eq!(
            execution_count.load(Ordering::SeqCst),
            0,
            "step body must NOT execute during replay"
        );
    }

    // Task ownership: foreign task fails fast

    #[tokio::test]
    async fn ownership_check_rejects_foreign_task() {
        // Must create ownership inside a spawned task where try_id()
        // returns Some: the #[tokio::test] root runs in block_on which
        // has no task ID.
        let result = tokio::spawn(async {
            let ownership = Arc::new(TaskOwnership::new_current());
            let ownership_clone = Arc::clone(&ownership);

            // Spawn a DIFFERENT task and try the ownership check there.
            let handle = tokio::spawn(async move { ownership_clone.check_current_task() });

            handle.await.unwrap()
        })
        .await;

        let inner_result = result.unwrap();
        assert!(inner_result.is_err());
        let msg = inner_result.unwrap_err();
        assert!(
            msg.contains("Use .spawn()"),
            "error message should guide users: {msg}"
        );
    }

    // Task ownership: owning task succeeds

    #[tokio::test]
    async fn ownership_check_allows_owning_task() {
        // Run inside a spawned task so try_id() returns Some.
        let result = tokio::spawn(async {
            let ownership = TaskOwnership::new_current();
            ownership.check_current_task()
        })
        .await;

        let inner = result.unwrap();
        assert!(inner.is_ok(), "owning task should pass: {inner:?}");
    }

    // Task ownership: .spawn() exemption (blessed tasks)

    #[tokio::test]
    async fn ownership_spawn_exemption_blesses_task() {
        // Create ownership in a spawned task, then spawn a child and bless it.
        let result = tokio::spawn(async {
            let ownership = Arc::new(TaskOwnership::new_current());
            let ownership_clone = Arc::clone(&ownership);

            // Simulate what .spawn() does: launch a task and bless it.
            let handle = tokio::spawn(async move {
                let task_id = tokio::task::id();
                ownership_clone.bless_task(task_id);
                ownership_clone.check_current_task()
            });

            handle.await.unwrap()
        })
        .await;

        let inner = result.unwrap();
        assert!(
            inner.is_ok(),
            "blessed task should pass ownership check: {inner:?}"
        );
    }

    // Task ownership: production topology (inline under block_on)

    /// Reproduces the REAL production topology: handler awaited inline
    /// under `block_on` (NOT inside `tokio::spawn`). `#[tokio::test]`
    /// runs in `block_on`, so `try_id()` is `None`: matching production.
    #[tokio::test]
    async fn ownership_root_context_allows_inline_operations() {
        // Context created inline (try_id() == None): production path.
        let ownership = TaskOwnership::new_current();
        assert!(
            matches!(ownership.owner, Owner::Thread(_)),
            "sanity: #[tokio::test] root runs in block_on with no task ID"
        );
        // Operation invoked inline on the owner thread: should pass.
        let result = ownership.check_current_task();
        assert!(
            result.is_ok(),
            "inline operation on the owner thread should pass: {result:?}"
        );
        // Operation invoked on another thread with no task ID: refused.
        let ownership = Arc::new(ownership);
        let other = Arc::clone(&ownership);
        let result = std::thread::spawn(move || other.check_current_task())
            .join()
            .unwrap();
        assert!(
            result.is_err_and(|msg| msg.contains("does not own the context")),
            "a thread that does not own the context must be refused"
        );
    }

    /// A context created on a task refuses a caller with no task ID, even
    /// on the thread that ran the owner task.
    #[test]
    fn ownership_task_context_refuses_callers_without_a_task() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let ownership = rt.block_on(async {
            tokio::spawn(async { Arc::new(TaskOwnership::new_current()) })
                .await
                .unwrap()
        });
        assert!(matches!(ownership.owner, Owner::Task(_)));
        assert!(ownership.check_current_task().is_err());
    }

    /// Production topology: a durable operation invoked from a bare user
    /// `tokio::spawn` (NOT blessed) MUST be rejected.
    #[tokio::test]
    async fn ownership_root_context_rejects_unblessed_spawn() {
        // Context created inline (try_id() == None): production path.
        let ownership = Arc::new(TaskOwnership::new_current());
        assert!(
            matches!(ownership.owner, Owner::Thread(_)),
            "sanity: root runs in block_on with no task ID"
        );
        let ownership_clone = Arc::clone(&ownership);

        // User tokio::spawn, NOT blessed.
        let handle = tokio::spawn(async move { ownership_clone.check_current_task() });

        let inner = handle.await.unwrap();
        assert!(
            inner.is_err(),
            "unblessed spawned task must be rejected in root-context mode"
        );
        let msg = inner.unwrap_err();
        assert!(
            msg.contains("Use .spawn()"),
            "error message should guide users: {msg}"
        );
    }

    /// Production topology: SDK-blessed spawns (`.spawn()`, combinators)
    /// still work in root-context mode.
    #[tokio::test]
    async fn ownership_root_context_allows_blessed_spawn() {
        // Context created inline (try_id() == None): production path.
        let ownership = Arc::new(TaskOwnership::new_current());
        assert!(
            matches!(ownership.owner, Owner::Thread(_)),
            "sanity: root runs in block_on with no task ID"
        );
        let ownership_clone = Arc::clone(&ownership);

        // Simulate what .spawn() / combinators do: spawn and bless.
        let handle = tokio::spawn(async move {
            let task_id = tokio::task::id();
            ownership_clone.bless_task(task_id);
            ownership_clone.check_current_task()
        });

        let inner = handle.await.unwrap();
        assert!(
            inner.is_ok(),
            "blessed task in root-context mode should pass: {inner:?}"
        );
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test module
mod suspension_containment_and_task_ownership {
    use super::InvocationOutcome;
    use super::test_support::drive_ok;
    use crate::client::{ExecutionClient, InMemoryExecutionClient, TestResponse};
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    fn succeeded_record(wire: &str) -> CheckpointRecord {
        CheckpointRecord {
            id: wire.to_owned(),
            status: CheckpointStatus::Succeeded,
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        }
    }

    // 1. Suspension never lets following user code run
    // Each op is awaited and its (never-produced) result ignored; the side
    // effect after the await MUST NOT run and the invocation MUST be Pending.

    #[tokio::test]
    async fn wait_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let _ = ctx_h.wait(Duration::from_secs(30)).await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended wait ran"
        );
    }

    #[tokio::test]
    async fn invoke_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let _ = ctx_h
                    .invoke::<serde_json::Value, _>("target-fn", "input".to_owned())
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended invoke ran"
        );
    }

    #[tokio::test]
    async fn step_retry_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                // Body fails; the default retry strategy schedules a retry,
                // which suspends (parks) on the backend timer.
                let _ = ctx_h
                    .step(|_| async { Err::<i32, crate::BoxError>("boom".into()) })
                    .name("retrying")
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended step retry ran"
        );
    }

    #[tokio::test]
    async fn wait_for_condition_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let _ = ctx_h
                    .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
                    .wait_strategy_fn(|_state: i32, _attempt| {
                        crate::wait_for_condition::WaitDecision::continue_with(Duration::from_secs(
                            1,
                        ))
                    })
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended wfc ran"
        );
    }

    #[tokio::test]
    async fn callback_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let Ok(cb) = ctx_h.create_callback::<serde_json::Value>().await else {
                    return Err(crate::error::WireError::new(
                        Some("E".to_owned()),
                        Some("create failed".to_owned()),
                    ));
                };
                // Awaiting a pending callback result suspends.
                let _ = cb.result().await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended callback ran"
        );
    }

    // 2. A replayed completed wait returns Ok and following code runs

    #[tokio::test]
    async fn replayed_completed_wait_returns_ok_and_continues() {
        let wire = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire.clone(),
            succeeded_record(&wire),
        )]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                ctx_h
                    .wait(Duration::from_secs(30))
                    .await
                    .expect("replayed completed wait returns Ok");
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            ran.load(Ordering::SeqCst),
            "code after a replayed wait must run"
        );
    }

    // 3. Validation errors remain catchable; checkpoint failures do not

    #[tokio::test]
    async fn validation_error_is_catchable_err() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                // An out-of-range failure percentage is a validation
                // error, returned as Err. (`max_concurrency(0)` fails the
                // execution instead, as JS does.)
                let r = ctx_h
                    .parallel(vec![crate::Branch::new("b", |_| async {
                        Ok("b".to_owned())
                    })])
                    .completion(
                        crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_percentage(
                            150.0,
                        ),
                    )
                    .await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, crate::error::WireError>("handled".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("handled".to_owned()));
        assert!(
            caught.load(Ordering::SeqCst),
            "validation error must be catchable"
        );
    }

    /// A checkpoint API failure is never yielded to the handler as a
    /// catchable error (issue #43): the handler is dropped at its await
    /// point exactly as `suspend_now` drops it, so a catch around the
    /// operation cannot observe the failure or branch on it. A
    /// non-retryable failure fails the EXECUTION with the
    /// checkpoint-failure error type.
    #[tokio::test]
    async fn non_retryable_checkpoint_failure_uncatchable_fails_execution() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError("kaboom".to_owned()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                // The live wait's START checkpoint fails non-retryably.
                // Pre-#43 this surfaced as a catchable Err; now the future
                // parks and the driver drops the handler, so neither branch
                // below runs.
                let r = ctx_h.wait(Duration::from_secs(5)).await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, crate::error::WireError>("handled".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert!(
            !caught.load(Ordering::SeqCst),
            "a checkpoint API failure must not be observable by a catch"
        );
        match outcome {
            InvocationOutcome::Failed { error } => {
                assert_eq!(
                    error.error_type(),
                    Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE),
                    "a non-retryable checkpoint failure fails the execution \
                     with the unrecoverable-execution type"
                );
                assert!(
                    error.error_message().is_some_and(|m| m.contains("kaboom")),
                    "the service's rejection reason is preserved: {error:?}"
                );
            }
            other => panic!("expected a FAILED execution outcome, got {other:?}"),
        }
    }

    /// The retryable counterpart: a checkpoint failure that exhausted the
    /// client's internal retries fails the INVOCATION (a Lambda runtime
    /// error, so the service re-invokes: the interruption recovery path)
    /// rather than the execution, and is equally unobservable by a catch.
    #[tokio::test]
    async fn retryable_checkpoint_failure_uncatchable_fails_invocation() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::RetryableError("throttled".to_owned()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let r = ctx_h.wait(Duration::from_secs(5)).await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, crate::error::WireError>("handled".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert!(
            !caught.load(Ordering::SeqCst),
            "a checkpoint API failure must not be observable by a catch"
        );
        match outcome {
            InvocationOutcome::Fault { message, .. } => {
                assert!(
                    message.contains("throttled"),
                    "the exhausted failure's reason is preserved: {message}"
                );
            }
            other => panic!("expected an invocation Fault outcome, got {other:?}"),
        }
    }

    // 5. Dropping a .spawn() handle loses only the result: the task keeps
    // running until the tracker aborts it.

    #[tokio::test]
    async fn dropping_spawn_handle_keeps_task_running() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let started = Arc::new(AtomicBool::new(false));
        let reached_end = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        {
            let s = Arc::clone(&started);
            let e = Arc::clone(&reached_end);
            let d = Arc::clone(&dropped);
            let handle = ctx
                .step(move |_| {
                    let s = Arc::clone(&s);
                    let e = Arc::clone(&e);
                    let d = Arc::clone(&d);
                    async move {
                        struct G(Arc<AtomicBool>);
                        impl Drop for G {
                            fn drop(&mut self) {
                                self.0.store(true, Ordering::SeqCst);
                            }
                        }
                        let _g = G(d);
                        s.store(true, Ordering::SeqCst);
                        std::future::pending::<()>().await;
                        e.store(true, Ordering::SeqCst);
                        Ok(1i32)
                    }
                })
                .name("bg")
                .spawn();
            // Let the spawned task run to its park point.
            for _ in 0..3 {
                tokio::task::yield_now().await;
            }
            assert!(
                started.load(Ordering::SeqCst),
                "spawned task should have started"
            );
            drop(handle);
        }
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        assert!(
            !dropped.load(Ordering::SeqCst),
            "a dropped .spawn() handle does not cancel the task"
        );
        // The drive's teardown still stops it.
        let tasks = Arc::clone(ctx.coordinator().tasks());
        tasks.abort_all();
        tasks.wait_idle().await;
        assert!(dropped.load(Ordering::SeqCst), "abort_all stops the task");
        assert!(
            !reached_end.load(Ordering::SeqCst),
            "the stopped task does not reach its end"
        );
    }

    // 4 + 6. A spawned straggler (non-durable `pending()` body) holds the
    // invocation until the Lambda timeout fires. The timeout drops the
    // `drive_invocation` future, so the driver's cleanup never runs. Dropping
    // it drops the root task's `TrackedTask` handle, which aborts the root
    // task, and the drive's teardown guard, which aborts every tracked
    // operation task. Dropping the straggler's own handle would not stop
    // it. No SUCCEED checkpoint is made for the aborted straggler. The
    // bounded timeout simulates the Lambda platform's execution timeout
    // backstop.

    #[tokio::test]
    async fn straggler_is_aborted_by_timeout_no_succeed_checkpoint() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());

        let sib_started = Arc::new(AtomicBool::new(false));
        let sib_completed = Arc::new(AtomicBool::new(false));
        let sib_dropped = Arc::new(AtomicBool::new(false));

        let ctx_h = ctx.clone();
        let s = Arc::clone(&sib_started);
        let c = Arc::clone(&sib_completed);
        let d = Arc::clone(&sib_dropped);

        // Wrap in a timeout to simulate the Lambda execution timeout. The
        // straggler (a step body awaiting a non-durable pending future)
        // holds an activity permit, so the invocation does NOT report
        // Pending on its own. The Lambda timeout is the backstop that aborts
        // the straggler.
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            drive_ok(
                ctx.clone(),
                async move {
                    // Sibling task: starts, then blocks on non-durable
                    // pending(). Its body never returns, so it is a
                    // "straggler".
                    let sib = ctx_h
                        .step(move |_| {
                            let s = Arc::clone(&s);
                            let c = Arc::clone(&c);
                            let d = Arc::clone(&d);
                            async move {
                                struct G(Arc<AtomicBool>);
                                impl Drop for G {
                                    fn drop(&mut self) {
                                        self.0.store(true, Ordering::SeqCst);
                                    }
                                }
                                let _g = G(d);
                                s.store(true, Ordering::SeqCst);
                                std::future::pending::<()>().await;
                                c.store(true, Ordering::SeqCst);
                                Ok(1i32)
                            }
                        })
                        .name("sibling")
                        .spawn();
                    // Let the sibling run to its park point.
                    for _ in 0..3 {
                        tokio::task::yield_now().await;
                    }
                    // The owner waits. The straggler still holds a permit, so
                    // the handler stays at this await point until the timeout
                    // fires and drops us.
                    let _ = ctx_h.wait(Duration::from_secs(30)).await;
                    let _ = sib.await;
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await;

        // The timeout fired: the invocation never completed on its own
        // because the straggler held its permit.
        assert!(
            result.is_err(),
            "drive_invocation must NOT terminate on its own while a step body \
             runs; the Lambda timeout is the backstop"
        );

        // Let the runtime process the abort of the sibling task.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        // Give tokio a moment to run the drop paths.
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(
            sib_started.load(Ordering::SeqCst),
            "sibling should have started"
        );
        assert!(
            !sib_completed.load(Ordering::SeqCst),
            "sibling must not complete (its body is non-durable pending)"
        );
        assert!(
            sib_dropped.load(Ordering::SeqCst),
            "sibling task must be aborted when the timeout drops the handler"
        );
        let recorded = client.recorded_updates();
        assert!(
            recorded
                .iter()
                .all(|u| u.action() != &OperationAction::Succeed),
            "no operation should have checkpointed SUCCEED — the straggler \
             was aborted before reaching its terminal"
        );
    }

    // The drive is dropped while the root task is inside a poll on another
    // worker thread. Aborting the root does not interrupt that poll, and
    // the tracker has no live task, so the root can still start an
    // operation task after the drop. The teardown must stop it: the task
    // never runs, or it is aborted, and no work of the invocation remains.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_spawn_after_drive_drop_is_stopped() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let coordinator = Arc::clone(ctx.coordinator());
        let (in_poll_tx, in_poll_rx) = std::sync::mpsc::channel::<()>();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel::<()>();
        let body_started = Arc::new(AtomicBool::new(false));
        let body_dropped = Arc::new(AtomicBool::new(false));
        let spawned = Arc::new(AtomicBool::new(false));

        let ctx_h = ctx.clone();
        let started = Arc::clone(&body_started);
        let dropped = Arc::clone(&body_dropped);
        let after_spawn = Arc::clone(&spawned);
        let handler = async move {
            // Hold this worker inside the root's poll until the test has
            // dropped the drive.
            in_poll_tx.send(()).expect("the test waits for this signal");
            resume_rx.recv().expect("the test resumes the root");
            let handle = ctx_h
                .step(move |_| {
                    let started = Arc::clone(&started);
                    let dropped = Arc::clone(&dropped);
                    async move {
                        struct G(Arc<AtomicBool>);
                        impl Drop for G {
                            fn drop(&mut self) {
                                self.0.store(true, Ordering::SeqCst);
                            }
                        }
                        let _g = G(dropped);
                        started.store(true, Ordering::SeqCst);
                        std::future::pending::<()>().await;
                        Ok(1i32)
                    }
                })
                .name("late")
                .spawn();
            drop(handle);
            after_spawn.store(true, Ordering::SeqCst);
            std::future::pending::<Result<String, crate::error::WireError>>().await
        };

        let mut drive = Box::pin(super::drive_invocation(
            handler,
            Arc::clone(ctx.task_ownership()),
            Arc::clone(&coordinator),
            ctx.clone(),
            None,
        ));
        // One poll spawns the root, which a worker then polls.
        std::future::poll_fn(|cx| {
            assert!(drive.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        in_poll_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the root is inside its poll");
        assert_eq!(coordinator.phase(), super::Phase::Running);
        assert!(!coordinator.tasks().is_closed());
        drop(drive);
        // The teardown refuses every later spawn, and aborts a task whose
        // spawn took its permit before that.
        assert_eq!(coordinator.phase(), super::Phase::Finished);
        assert!(coordinator.tasks().is_closed());
        resume_tx.send(()).expect("the root waits for this signal");

        tokio::time::timeout(Duration::from_secs(5), async {
            while !spawned.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the root reaches its spawn after the drop");
        // Give a spawned task time to be polled, if it was started.
        tokio::time::sleep(Duration::from_millis(50)).await;
        tokio::time::timeout(Duration::from_secs(5), coordinator.tasks().wait_idle())
            .await
            .expect("no operation task outlives the dropped drive");
        assert_eq!(coordinator.tasks().live(), 0);
        assert!(
            !body_started.load(Ordering::SeqCst) || body_dropped.load(Ordering::SeqCst),
            "a late operation task that ran was aborted"
        );
    }
}

#[cfg(test)]
mod scoped_suspension {
    use super::InvocationOutcome;
    use super::test_support::drive_ok;
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use crate::future::Branch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// Counts `WaitStarted` checkpoints (each carries `wait_options`).
    fn wait_starts(client: &InMemoryExecutionClient) -> usize {
        client
            .recorded_updates()
            .iter()
            .filter(|u| u.wait_options().is_some())
            .count()
    }

    // 1. Branch A waits while branch B has runnable work: B COMPLETES before
    //    the invocation reports Pending.
    #[tokio::test]
    async fn sibling_with_runnable_work_completes_before_invocation_suspends() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran_c = Arc::clone(&b_ran);
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let flag = b_ran_c;
                let branches = vec![
                    Branch::new("A-wait", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }),
                    Branch::new("B-step", move |cc: DurableContext| {
                        Box::pin(async move {
                            let v = cc.step(|_| async { Ok(7_i32) }).await?;
                            flag.store(true, Ordering::SeqCst);
                            Ok(v)
                        })
                    }),
                ];
                let _ = ctx_h.parallel(branches).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            b_ran.load(Ordering::SeqCst),
            "sibling B must complete its runnable work before the invocation suspends"
        );
    }

    // 2. Two branches suspend: Pending only after BOTH parked (both reached
    //    WaitStarted).
    #[tokio::test]
    async fn two_branches_suspend_pending_after_both_parked() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let branches = vec![
                    Branch::new("a", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }),
                    Branch::new("b", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }),
                ];
                let _ = ctx_h.parallel(branches).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(
            wait_starts(&client),
            2,
            "both branches must checkpoint WaitStarted before the invocation suspends"
        );
    }

    // 3. A suspended branch retains its slot: with max_concurrency=2 and four
    //    waiting branches, peak concurrent bodies equals the cap (not 4), and
    //    only cap-many branches ever start.
    #[tokio::test]
    async fn suspended_branches_retain_slots_peak_bounded_by_cap() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let ctx_h = ctx.clone();
        let active_h = Arc::clone(&active);
        let peak_h = Arc::clone(&peak);

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let mut branches = Vec::new();
                for i in 0..4 {
                    let active = Arc::clone(&active_h);
                    let peak = Arc::clone(&peak_h);
                    branches.push(Branch::new(format!("w{i}"), move |cc: DurableContext| {
                        Box::pin(async move {
                            let cur = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(cur, Ordering::SeqCst);
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }));
                }
                let _ = ctx_h.parallel(branches).max_concurrency(2).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "peak concurrent branches must equal the cap — suspended branches hold their slots"
        );
        assert!(
            active.load(Ordering::SeqCst) <= 2,
            "no more than cap-many branches may enter while others hold suspended slots"
        );
        assert_eq!(
            wait_starts(&client),
            2,
            "only the two slot-holding branches start; the other two never do"
        );
    }

    // 4. An eligible unstarted branch STARTS once a terminal completion frees
    //    a slot (contrast with test 3, where suspended branches hold slots).
    #[tokio::test]
    async fn slot_freed_by_completion_lets_next_branch_start() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let second_ran = Arc::new(AtomicBool::new(false));
        let second_h = Arc::clone(&second_ran);
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let flag = second_h;
                let branches = vec![
                    Branch::new("first", |cc: DurableContext| async move {
                        let v = cc.step(|_| async { Ok(1_i32) }).await?;
                        Ok(v)
                    }),
                    Branch::new("second", move |cc: DurableContext| {
                        Box::pin(async move {
                            let v = cc.step(|_| async { Ok(2_i32) }).await?;
                            flag.store(true, Ordering::SeqCst);
                            Ok(v)
                        })
                    }),
                ];
                let r = ctx_h.parallel(branches).max_concurrency(1).await;
                Ok::<_, crate::error::WireError>(format!("{}", r.is_ok()))
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Complete("true".to_owned()));
        assert!(
            second_ran.load(Ordering::SeqCst),
            "second branch must start after the first frees its slot"
        );
    }

    // 5 + 8. Nested map/parallel/child suspension propagates through child
    //         scopes to the root and reaches Pending WITHOUT hanging.
    #[tokio::test]
    async fn nested_coordinator_suspension_propagates_to_root_without_hang() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let ctx_h = ctx.clone();

        let fut = drive_ok(
            ctx.clone(),
            async move {
                let outer = vec![Branch::new("outer", |cc: DurableContext| {
                    Box::pin(async move {
                        // A child context whose step feeds a nested map whose
                        // single item waits: exercises child + coordinator
                        // scope propagation in one branch.
                        let inner = vec![Branch::new("inner-wait", |icc: DurableContext| {
                            Box::pin(async move {
                                icc.wait(Duration::from_secs(30)).await?;
                                Ok(0_i32)
                            })
                        })];
                        cc.parallel(inner)
                            .await
                            .map(|v: Vec<i32>| v.into_iter().sum::<i32>())
                            .map_err(Into::into)
                    })
                })];
                let _ = ctx_h.parallel(outer).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        );

        let outcome = tokio::time::timeout(Duration::from_secs(5), fut)
            .await
            .expect("nested suspension must reach Pending, not hang");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(
            wait_starts(&client),
            1,
            "the innermost wait must checkpoint WaitStarted"
        );
    }

    // 6. Early-completion threshold (min_successful) excludes a parked branch:
    //    the batch completes and the invocation does NOT suspend.
    #[tokio::test]
    async fn min_successful_completion_excludes_parked_branch() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                let branches = vec![
                    Branch::new("ok", |cc: DurableContext| async move {
                        let v = cc.step(|_| async { Ok(1_i32) }).await?;
                        Ok(v)
                    }),
                    Branch::new("waiter", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(2_i32)
                        })
                    }),
                ];
                let r = ctx_h
                    .parallel(branches)
                    .completion(
                        crate::builders::map_parallel::CompletionConfig::builder()
                            .min_successful(1)
                            .build()
                            .expect("valid completion config"),
                    )
                    .await;
                Ok::<_, crate::error::WireError>(format!("ok={}", r.is_ok()))
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        // Threshold met by "ok" → batch completes; the parked "waiter" is
        // excluded and the invocation does NOT suspend.
        assert_eq!(outcome, InvocationOutcome::Complete("ok=true".to_owned()));
    }

    // 7. No regression: a ROOT-level wait suspends the invocation immediately
    //    and following code does not run.
    #[tokio::test]
    async fn root_level_wait_suspends_invocation_immediately() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let after = Arc::new(AtomicBool::new(false));
        let after_c = Arc::clone(&after);
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            ctx.clone(),
            async move {
                ctx_h.wait(Duration::from_secs(30)).await.map_err(|e| {
                    crate::error::WireError::new(Some("W".to_owned()), Some(e.to_string()))
                })?;
                after_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !after.load(Ordering::SeqCst),
            "code after a root-level wait must not run"
        );
    }
}

//
// `.spawn()` used to let a waiting operation inside a top-level `.spawn()`
// suspend the invocation while a sibling spawned task was still running, which
// aborted the sibling after it had already checkpointed STARTED. The
// invocation coordinator now suspends only when no SDK work can make progress.
//
// These tests encode the CORRECT behaviour and run as part of `make check`.
#[cfg(test)]
mod spawn_scope_regressions {
    use super::InvocationOutcome;
    use super::test_support::drive_ok;
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Upper bound on either test: a suspension-scoping defect must show up as
    /// a FAILED assertion, never as a hung test run.
    const BOUND: Duration = Duration::from_secs(5);

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// True if the client recorded a terminal (Succeed) checkpoint for the
    /// operation named `name`.
    fn succeeded(client: &InMemoryExecutionClient, name: &str) -> bool {
        client
            .recorded_updates()
            .iter()
            .any(|u| matches!(u.action(), OperationAction::Succeed) && u.name() == Some(name))
    }

    /// `wait.spawn()` beside `step.spawn()`: the spawned wait parks, but the
    /// spawned step is still runnable, so the invocation must stay alive until
    /// the step reaches its terminal checkpoint and only then report Pending.
    #[tokio::test]
    async fn spawned_step_reaches_terminal_checkpoint_before_pending() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let body_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&body_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                ctx.clone(),
                async move {
                    // Parks: no timer result is available on a first invocation.
                    let wait = ctx_h.wait(Duration::from_secs(10)).spawn();
                    // Runnable, and slower than the wait's park so the ordering
                    // under test is deterministic.
                    let work = ctx_h
                        .step(move |_| {
                            let done = Arc::clone(&done);
                            async move {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                                done.store(true, Ordering::SeqCst);
                                Ok(7_i32)
                            }
                        })
                        .name("work")
                        .spawn();
                    let _ = work.await;
                    let _ = wait.await;
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            body_done.load(Ordering::SeqCst),
            "the spawned step body must run to completion before the invocation \
             suspends; a parked spawned sibling must not abort it"
        );
        assert!(
            succeeded(&client, "work"),
            "the spawned step must reach its terminal Succeed checkpoint before \
             the invocation reports Pending; otherwise its STARTED checkpoint \
             re-executes the body on resume and duplicates side effects"
        );
    }

    /// The documented composition: `tokio::join!` over a spawned wait and a
    /// spawned step. Same guarantee as above, reached through `join!`.
    #[tokio::test]
    async fn joined_spawned_wait_and_step_lets_the_step_finish() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let body_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&body_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                ctx.clone(),
                async move {
                    let wait = ctx_h.wait(Duration::from_secs(10)).spawn();
                    let work = ctx_h
                        .step(move |_| {
                            let done = Arc::clone(&done);
                            async move {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                                done.store(true, Ordering::SeqCst);
                                Ok(7_i32)
                            }
                        })
                        .name("work")
                        .spawn();
                    let (_timer, _result) = tokio::join!(wait, work);
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            body_done.load(Ordering::SeqCst),
            "join! over a spawned wait and a spawned step must not abort the step"
        );
        assert!(
            succeeded(&client, "work"),
            "the joined spawned step must reach its terminal Succeed checkpoint \
             before the invocation reports Pending"
        );
    }

    /// A spawned wait waits, then the owner performs a non-spawned sequential
    /// step. The sequential step must complete and checkpoint Succeed. The
    /// handler then resolves, so the invocation completes with its result
    /// while the dropped wait's record stays `STARTED`.
    #[tokio::test]
    async fn spawned_wait_parks_then_owner_does_sequential_step() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let step_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&step_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                ctx.clone(),
                async move {
                    // Spawn a wait: it parks immediately (no timer result on
                    // first invocation). Hold the handle so dropping it
                    // does not abort the task, but never await it.
                    let _wait = ctx_h.wait(Duration::from_secs(10)).spawn();

                    // Yield once so the spawned wait task has a chance to run
                    // and register its wait. The yield runs under a held
                    // activity permit: a bare yield while an operation waits
                    // is a point where the invocation may suspend.
                    ctx_h.hold_activity(tokio::task::yield_now()).await;

                    // Now do a NON-SPAWNED sequential step. If the spawned
                    // wait already flagged ROOT, the driver will drop us here.
                    let _result = ctx_h
                        .step(move |_| {
                            let done = Arc::clone(&done);
                            async move {
                                done.store(true, Ordering::SeqCst);
                                Ok(42_i32)
                            }
                        })
                        .name("sequential_work")
                        .await;

                    // Handler completes normally: the invocation completes
                    // with its result.
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            step_done.load(Ordering::SeqCst),
            "the non-spawned step must complete before Pending; a parked spawn \
             must not pre-empt the owner's sequential work"
        );
        assert!(
            succeeded(&client, "sequential_work"),
            "the sequential step must reach its terminal Succeed checkpoint \
             before the invocation reports Pending"
        );
    }

    /// A spawned `try_join_all` containing a waiting wait input beside a
    /// runnable step sibling. The combinator is itself spawned. The waiting
    /// input keeps the combinator pending, the runnable sibling spawned
    /// alongside must still complete, and only then does the invocation
    /// suspend, without a deadlock.
    #[tokio::test]
    async fn spawned_combinator_with_parking_input_does_not_deadlock() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let sibling_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&sibling_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                ctx.clone(),
                async move {
                    // Parking input: a spawned wait (will park on first
                    // invocation since no timer has elapsed).
                    let parking_wait = ctx_h.wait(Duration::from_mins(1)).spawn();
                    // The combinator contains the parking input alongside a
                    // step that resolves immediately.
                    let instant_step = ctx_h.step(|_| async { Ok(()) }).name("inner").future();
                    // Spawn the combinator: this registers the combinator on
                    // root scope. The defect made the parking input also park
                    // root, causing a deadlock.
                    let combo = ctx_h.try_join_all([parking_wait, instant_step]).spawn();

                    // Runnable sibling spawned on root beside the combinator.
                    let sibling = ctx_h
                        .step(move |_| {
                            let d = Arc::clone(&done);
                            async move {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                                d.store(true, Ordering::SeqCst);
                                Ok(42_i32)
                            }
                        })
                        .name("sibling")
                        .spawn();

                    let _ = tokio::join!(combo, sibling);
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect(
            "drive_invocation must terminate within the bound; \
             a deadlock from the spawned combinator defect would hang here",
        );

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            sibling_done.load(Ordering::SeqCst),
            "the runnable sibling must complete before the invocation \
             suspends; the combinator's suspension must not abort it"
        );
        assert!(
            succeeded(&client, "sibling"),
            "the sibling must reach its terminal Succeed checkpoint \
             before the invocation reports Pending"
        );
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: the handler panics under test and test assertions
mod task_gateway {
    use super::{InvocationOutcome, drive_outcome};
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use crate::future::DurableFuture;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn live_ctx() -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::new(InMemoryExecutionClient::new(Vec::new())) as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// Sets its flag when dropped.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Spawns a step whose body holds a drop guard and awaits `pending()`.
    fn spawn_holding_step(ctx: &DurableContext, dropped: &Arc<AtomicBool>) -> DurableFuture<()> {
        let dropped = Arc::clone(dropped);
        ctx.step(move |_| {
            let dropped = Arc::clone(&dropped);
            async move {
                let _guard = DropFlag(dropped);
                std::future::pending::<()>().await;
                Ok(())
            }
        })
        .name("hold")
        .spawn()
    }

    async fn yield_three_times() {
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
    }

    /// A spawned step body still mid-await when the handler resolves is
    /// dropped by the drive's cleanup before `drive_invocation` returns, and
    /// the invocation completes with the handler's result.
    #[tokio::test]
    async fn aborted_spawned_body_drops_before_invocation_returns() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let dropped = Arc::new(AtomicBool::new(false));
        let handler_dropped = Arc::clone(&dropped);
        let outcome = drive_outcome(
            async move {
                let _wait = ctx.wait(Duration::from_secs(10)).name("timer").spawn();
                let _step = spawn_holding_step(&ctx, &handler_dropped);
                yield_three_times().await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            ownership,
            signal,
            root,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            dropped.load(Ordering::SeqCst),
            "the step body must be dropped before the invocation returns"
        );
    }

    /// A handler panic returns its payload only after every tracked task
    /// has unwound.
    #[tokio::test]
    async fn handler_panic_returns_payload_after_cleanup() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let dropped = Arc::new(AtomicBool::new(false));
        let handler_dropped = Arc::clone(&dropped);
        let outcome = drive_outcome(
            async move {
                let _step = spawn_holding_step(&ctx, &handler_dropped);
                yield_three_times().await;
                panic!("handler boom");
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"handler boom"));
        assert!(dropped.load(Ordering::SeqCst), "the step body was dropped");
        assert_eq!(tasks.live(), 0);
    }

    /// After a handler panic, the driver also aborts a task whose handle
    /// the handler moved out of its own future.
    #[tokio::test]
    async fn handler_panic_aborts_task_whose_handle_left_the_handler() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let dropped = Arc::new(AtomicBool::new(false));
        let handler_dropped = Arc::clone(&dropped);
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let handler_escaped = Arc::clone(&escaped);
        let outcome = drive_outcome(
            async move {
                let step = spawn_holding_step(&ctx, &handler_dropped);
                *handler_escaped.lock().expect("lock") = Some(step);
                yield_three_times().await;
                panic!("handler boom");
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        assert!(outcome.is_err(), "expected the handler panic");
        assert!(
            escaped.lock().expect("lock").is_some(),
            "the test still holds the handle"
        );
        assert!(dropped.load(Ordering::SeqCst), "the step body was dropped");
        assert_eq!(tasks.live(), 0);
    }

    /// A handler that panics while a `.spawn()`ed wait is registered returns
    /// the panic payload, not `Pending`.
    #[tokio::test]
    async fn handler_panic_with_parked_spawn_returns_payload() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let parked_signal = Arc::clone(&signal);
        let outcome = drive_outcome(
            async move {
                let _wait = ctx.wait(Duration::from_secs(10)).name("timer").spawn();
                // Held activity keeps the handler runnable while it waits
                // for the registration.
                ctx.hold_activity(async {
                    for _ in 0..1000 {
                        if parked_signal.live_waits() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await;
                assert_eq!(parked_signal.live_waits(), 1, "the wait is registered");
                panic!("handler boom");
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"handler boom"));
        assert_eq!(tasks.live(), 0);
    }

    /// A handler that panics while a wait registration is live returns the
    /// panic payload.
    #[tokio::test]
    async fn handler_panic_after_suspend_request_returns_payload() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let handler_signal = Arc::clone(&signal);
        let outcome = drive_outcome(
            async move {
                let _ctx = ctx;
                let _registration =
                    handler_signal.register_test_wait("op", std::task::Waker::noop());
                panic!("handler boom");
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"handler boom"));
        assert_eq!(tasks.live(), 0);
    }

    /// A recorded fatal error takes precedence over a handler panic.
    #[tokio::test]
    async fn fatal_error_wins_over_handler_panic() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let handler_signal = Arc::clone(&signal);
        let outcome = drive_outcome(
            async move {
                let _ctx = ctx;
                handler_signal.record_fatal("Fatal".to_owned(), "mismatch".to_owned());
                panic!("handler boom");
            },
            ownership,
            signal,
            root,
            None,
        )
        .await
        .expect("the fatal error takes precedence over the panic");
        assert_eq!(
            outcome,
            InvocationOutcome::Failed {
                error: crate::error::WireError::new(
                    Some("Fatal".to_owned()),
                    Some("mismatch".to_owned()),
                ),
            }
        );
    }

    /// Panics with an [`InnerPanicPayload`] when dropped.
    pub(super) struct NestedPanicPayload;

    impl Drop for NestedPanicPayload {
        fn drop(&mut self) {
            std::panic::panic_any(InnerPanicPayload);
        }
    }

    /// Panics when dropped.
    pub(super) struct InnerPanicPayload;

    impl Drop for InnerPanicPayload {
        fn drop(&mut self) {
            panic!("inner payload drop");
        }
    }

    /// Records a fatal error on the invocation's coordinator when it is
    /// dropped.
    struct FatalOnDrop(Arc<crate::coordinator::InvocationCoordinator>);

    impl Drop for FatalOnDrop {
        fn drop(&mut self) {
            self.0
                .record_fatal("Fatal".to_owned(), "mismatch during cleanup".to_owned());
        }
    }

    /// A fatal error that a tracked task records while the driver cleans up
    /// after a handler panic still takes precedence over that panic. The
    /// handler moves the step's handle out of its own future, so only the
    /// driver's `abort_all` drops the step body, after the drive observed
    /// the panic. Dropping the body records the fatal error.
    #[tokio::test(flavor = "current_thread")]
    async fn fatal_recorded_during_panic_cleanup_wins_over_panic() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let body_signal = Arc::clone(&signal);
        let started = Arc::new(AtomicBool::new(false));
        let body_started = Arc::clone(&started);
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let handler_escaped = Arc::clone(&escaped);
        let check_signal = Arc::clone(&signal);
        let outcome = drive_outcome(
            async move {
                let step = ctx
                    .step(move |_| {
                        let guard = FatalOnDrop(Arc::clone(&body_signal));
                        let started = Arc::clone(&body_started);
                        async move {
                            let _guard = guard;
                            started.store(true, Ordering::SeqCst);
                            std::future::pending::<()>().await;
                            Ok(())
                        }
                    })
                    .name("hold")
                    .spawn();
                *handler_escaped.lock().expect("lock") = Some(step);
                yield_three_times().await;
                assert!(check_signal.fatal_error().is_none(), "no fatal yet");
                panic!("handler boom");
            },
            ownership,
            signal,
            root,
            None,
        )
        .await
        .expect("the fatal error recorded during cleanup takes precedence over the panic");
        assert!(started.load(Ordering::SeqCst), "the step body started");
        assert_eq!(
            outcome,
            InvocationOutcome::Failed {
                error: crate::error::WireError::new(
                    Some("Fatal".to_owned()),
                    Some("mismatch during cleanup".to_owned()),
                ),
            }
        );
        assert_eq!(tasks.live(), 0);
    }

    /// Runs its closure on the invocation's coordinator when it is dropped.
    struct OnDrop(
        Arc<crate::coordinator::InvocationCoordinator>,
        Arc<dyn Fn(&crate::coordinator::InvocationCoordinator) + Send + Sync>,
    );

    impl Drop for OnDrop {
        fn drop(&mut self) {
            (self.1)(&self.0);
        }
    }

    /// Drives a handler that panics with a [`NestedPanicPayload`] while a
    /// step body holds an [`OnDrop`] running `on_cleanup`. Only the
    /// driver's `abort_all` drops the body, after the drive observed the
    /// handler panic, as in `fatal_recorded_during_panic_cleanup_wins_over_panic`.
    async fn drive_nested_handler_panic_with_cleanup(
        on_cleanup: Arc<dyn Fn(&crate::coordinator::InvocationCoordinator) + Send + Sync>,
    ) -> (
        Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>,
        usize,
    ) {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let body_signal = Arc::clone(&signal);
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let handler_escaped = Arc::clone(&escaped);
        let outcome = drive_outcome(
            async move {
                let step = ctx
                    .step(move |_| {
                        let guard = OnDrop(Arc::clone(&body_signal), Arc::clone(&on_cleanup));
                        async move {
                            let _guard = guard;
                            std::future::pending::<()>().await;
                            Ok(())
                        }
                    })
                    .name("hold")
                    .spawn();
                *handler_escaped.lock().expect("lock") = Some(step);
                yield_three_times().await;
                std::panic::panic_any(NestedPanicPayload);
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        (outcome, tasks.live())
    }

    /// A fatal error recorded during the cleanup takes precedence over a
    /// handler panic whose payload panics on drop with a payload that
    /// panics on drop again. The driver drops the handler's payload inside
    /// a catch and leaks the nested payload, so no panic escapes.
    #[tokio::test(flavor = "current_thread")]
    async fn fatal_during_cleanup_wins_over_nested_panicking_handler_payload() {
        let (outcome, live) = drive_nested_handler_panic_with_cleanup(Arc::new(|coordinator| {
            coordinator.record_fatal("Fatal".to_owned(), "during cleanup".to_owned());
        }))
        .await;
        let Ok(InvocationOutcome::Failed { error }) = &outcome else {
            panic!("expected the fatal outcome, got {outcome:?}");
        };
        assert_eq!(error.error_message(), Some("during cleanup"));
        assert_eq!(live, 0);
    }

    /// A handler that returns a value while a wait registration is live
    /// completes with that value: the resolved handler decides the outcome.
    #[tokio::test]
    async fn handler_value_with_live_registration_completes() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let handler_signal = Arc::clone(&signal);
        let outcome = drive_outcome(
            async move {
                let _ctx = ctx;
                let _registration =
                    handler_signal.register_test_wait("op", std::task::Waker::noop());
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            ownership,
            signal,
            root,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    /// The handler runs on a blessed task of its own, and its blessing is
    /// removed once the drive ends.
    #[tokio::test]
    async fn handler_runs_as_blessed_operation_task() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let outcome = drive_outcome(
            async move {
                let on_task = tokio::task::try_id().is_some();
                let value = ctx
                    .step(|_| async { Ok(1_i32) })
                    .name("s")
                    .await
                    .map_err(|e| {
                        crate::error::WireError::new(None::<String>, Some(e.to_string()))
                    })?;
                Ok::<_, crate::error::WireError>(format!("{on_task}:{value}"))
            },
            Arc::clone(&ownership),
            signal,
            root,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("true:1".to_owned()));
        assert_eq!(ownership.blessed_count(), 0);
    }

    /// The root task is not counted on the tracker, so a handler that waits
    /// for idle does not wait for itself.
    #[tokio::test]
    async fn wait_idle_in_handler_does_not_wait_for_handler() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let tasks = Arc::clone(signal.tasks());
        let outcome = drive_outcome(
            async move {
                let step = ctx.step(|_| async { Ok(1_i32) }).name("quick").spawn();
                let _ = step.await;
                let idle = tokio::time::timeout(Duration::from_secs(1), tasks.wait_idle()).await;
                Ok::<_, crate::error::WireError>(idle.is_ok().to_string())
            },
            ownership,
            signal,
            root,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("true".to_owned()));
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions
mod coordinator_suspension {
    use super::test_support::{
        EventMessages, ScriptedClient, drive_handler, named, operation_like, scripted_ctx, wire_err,
    };
    use super::{InvocationOutcome, UNTRACKED_DEADLINE_WARNING};
    use crate::coordinator::{PermitKind, Phase};
    use crate::engine::CheckpointStatus;
    use aws_sdk_lambda::types::{OperationAction, OperationStatus};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Waits, inside the handler's held activity, until `condition` holds,
    /// polling every 10 ms of tokio time, at most 1,000 times.
    async fn until(ctx: &crate::DurableContext, condition: impl Fn() -> bool) -> bool {
        ctx.hold_activity(async {
            for _ in 0..1000 {
                if condition() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            false
        })
        .await
    }

    /// A client that, on the call carrying the step named `step`'s SUCCEED,
    /// also carries the operation named `waiting` with `status`.
    fn completes_on_step_succeed(
        step: &'static str,
        waiting: &'static str,
        status: OperationStatus,
    ) -> ScriptedClient {
        ScriptedClient::new(move |updates, history| {
            if named(updates, step, &OperationAction::Succeed).is_none() {
                return Vec::new();
            }
            named(history, waiting, &OperationAction::Start)
                .map(|start| {
                    operation_like(start, status.clone())
                        .build()
                        .expect("all required fields set")
                })
                .into_iter()
                .collect()
        })
    }

    /// A step body that returns once the client received the START of the
    /// operation named `waiting`, so the response that completes it comes
    /// after its registration.
    async fn after_start_of(
        client: &Arc<ScriptedClient>,
        waiting: &str,
    ) -> Result<u32, crate::BoxError> {
        for _ in 0..1000 {
            if client.received(waiting, &OperationAction::Start) {
                return Ok(2);
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        Err("the waiting operation never started".into())
    }

    #[tokio::test(start_paused = true)]
    async fn lone_wait_suspends_at_once() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            ctx.wait(Duration::from_secs(1)).await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(client.call_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn untracked_await_does_not_suspend_and_warns() {
        let (capture, _guard) = EventMessages::install();
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let drive = drive_handler(ctx, Some(Duration::from_secs(2)), |_ctx| async move {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok("done".to_owned())
        });
        let timed = tokio::time::timeout(Duration::from_secs(5), drive).await;
        assert!(timed.is_err(), "the untracked await does not suspend");
        assert_eq!(
            capture.at(tracing::Level::WARN),
            vec![UNTRACKED_DEADLINE_WARNING.to_owned()]
        );
    }

    /// A deadline that passes while SDK work runs logs nothing.
    #[tokio::test(start_paused = true)]
    async fn deadline_during_tracked_work_does_not_warn() {
        let (capture, _guard) = EventMessages::install();
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let outcome = drive_handler(ctx, Some(Duration::from_secs(1)), |ctx| async move {
            ctx.step(|_| async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(1_u32)
            })
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(capture.at(tracing::Level::WARN), Vec::<String>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn sibling_response_completes_wait_in_invocation() {
        let (events, _guard) = EventMessages::install();
        let client = Arc::new(completes_on_step_succeed(
            "s",
            "w",
            OperationStatus::Succeeded,
        ));
        let ctx = scripted_ctx(&client);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            ctx.step(move |_| async move { after_start_of(&body_client, "w").await })
                .name("s")
                .await
                .map_err(wire_err)?;
            w.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(events.replayed(), 0, "a resumed wait emits no replay event");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_checkpoint_holds_permit_until_merged() {
        let client = Arc::new(completes_on_step_succeed(
            "s",
            "w",
            OperationStatus::Succeeded,
        ));
        let ctx = crate::DurableContext::new_root_with_client_and_coalescer(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::from_millis(50)),
        );
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            ctx.step(move |_| async move { after_start_of(&body_client, "w").await })
                .name("s")
                .await
                .map_err(wire_err)?;
            w.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    /// A root context on `client` whose writes wait out a coalescing window
    /// of one hour, with `log` as the checkpoint log.
    fn hour_window_ctx(
        client: &Arc<ScriptedClient>,
        log: crate::engine::CheckpointLog,
    ) -> crate::DurableContext {
        crate::DurableContext::new_root_with_client_and_coalescer(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::clone(client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::from_secs(3600)),
        )
    }

    #[tokio::test(start_paused = true)]
    async fn completion_preempts_window_of_dropped_contributor() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = hour_window_ctx(&client, crate::engine::CheckpointLog::empty());
        let coordinator = Arc::clone(ctx.coordinator());
        let started = tokio::time::Instant::now();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(10)).name("w").spawn();
            let buffered = ctx
                .hold_activity(async {
                    for _ in 0..1000 {
                        if ctx.coordinator().permits(PermitKind::Checkpoint) == 1 {
                            return true;
                        }
                        tokio::task::yield_now().await;
                    }
                    false
                })
                .await;
            if !buffered {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("the START of w was never buffered"),
                ));
            }
            drop(w);
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "completion waited {:?} of the window",
            started.elapsed()
        );
        assert!(
            client.received("w", &OperationAction::Start),
            "the buffered START is flushed"
        );
        assert_eq!(coordinator.tasks().live(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn waiting_operation_preempts_window() {
        let client = Arc::new(ScriptedClient::empty());
        let wait_id = crate::engine::compute_wire_id_public("1");
        let log = crate::engine::CheckpointLog::from_records(vec![(
            wait_id.clone(),
            crate::engine::CheckpointRecord::with_status(&wait_id, CheckpointStatus::Started),
        )]);
        let ctx = hour_window_ctx(&client, log);
        let started = tokio::time::Instant::now();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(10)).name("w").spawn();
            ctx.step(|_| async { Ok(1_u32) })
                .name("s")
                .await
                .map_err(wire_err)?;
            w.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the step's writes waited {:?} of the window",
            started.elapsed()
        );
        assert!(client.received("s", &OperationAction::Succeed));
    }

    #[tokio::test(start_paused = true)]
    async fn wait_resumed_maps_terminal_status() {
        let client = Arc::new(completes_on_step_succeed("s", "w", OperationStatus::Failed));
        let ctx = scripted_ctx(&client);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            ctx.step(move |_| async move { after_start_of(&body_client, "w").await })
                .name("s")
                .await
                .map_err(wire_err)?;
            let err = w.await.expect_err("the wait record failed");
            let crate::error::OperationErrorKind::Wait(wait_err) = err.kind() else {
                panic!("expected a wait error, got {err:?}");
            };
            let crate::error::WaitErrorKind::UnexpectedStatus(status) = wait_err.kind();
            Ok(status.status().to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("FAILED".to_owned()));
    }

    #[tokio::test(start_paused = true)]
    async fn resolved_handler_completes_with_dropped_spawned_wait() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let handler_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(10)).name("w").spawn();
            let registered = ctx
                .hold_activity(async {
                    for _ in 0..1000 {
                        if handler_client.received("w", &OperationAction::Start)
                            && ctx.coordinator().live_waits() == 1
                        {
                            return true;
                        }
                        tokio::task::yield_now().await;
                    }
                    false
                })
                .await;
            if !registered {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("the wait never registered"),
                ));
            }
            // The dropped wait keeps waiting until the drive ends, which
            // stops it. The handler's result decides the outcome.
            drop(w);
            Ok(serde_json::to_string("done").expect("a string serializes"))
        })
        .await;
        assert!(
            matches!(&outcome, Ok(InvocationOutcome::Complete(v)) if v == "\"done\""),
            "unexpected outcome: {outcome:?}"
        );
        let w_updates: Vec<_> = client
            .updates()
            .into_iter()
            .filter(|u| u.name() == Some("w"))
            .collect();
        assert_eq!(w_updates.len(), 1, "only the START of w was written");
        assert_eq!(
            w_updates
                .first()
                .map(aws_sdk_lambda::types::OperationUpdate::action),
            Some(&OperationAction::Start)
        );
        assert_eq!(coordinator.tasks().live(), 0);
        assert_eq!(coordinator.live_waits(), 0);
    }

    /// The coordinator's counts after a drive: no task, permit or wait
    /// registration remains.
    fn assert_clean(coordinator: &crate::coordinator::InvocationCoordinator) {
        assert_eq!(coordinator.permits(PermitKind::Activity), 0);
        assert_eq!(coordinator.permits(PermitKind::Checkpoint), 0);
        assert_eq!(coordinator.live_waits(), 0);
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// Spawns a wait of an hour and a step whose body never returns.
    fn spawn_wait_and_stuck_step(
        ctx: &crate::DurableContext,
    ) -> (crate::DurableFuture<()>, crate::DurableFuture<()>) {
        let wait = ctx.wait(Duration::from_secs(3600)).name("w").spawn();
        let step = ctx
            .step(|_| async {
                std::future::pending::<()>().await;
                Ok(())
            })
            .name("stuck")
            .spawn();
        (wait, step)
    }

    #[tokio::test(start_paused = true)]
    async fn invocation_end_leaves_no_task_permit_or_registration() {
        // (a) The handler returns Ok.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ops = spawn_wait_and_stuck_step(&ctx);
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            Ok("a".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("a".to_owned()));
        assert_clean(&coordinator);

        // (b) The handler drops both handles and returns Ok. The detached
        // step and wait keep running until the drive ends, which stops
        // them.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            drop(spawn_wait_and_stuck_step(&ctx));
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            Ok("b".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("b".to_owned()));
        assert_clean(&coordinator);

        // (c) The handler returns Err.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ops = spawn_wait_and_stuck_step(&ctx);
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            Err(crate::error::WireError::new(Some("E"), Some("c")))
        })
        .await
        .expect("no panic");
        assert!(matches!(outcome, InvocationOutcome::Failed { .. }));
        assert_clean(&coordinator);

        // (d) The handler panics.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ops = spawn_wait_and_stuck_step(&ctx);
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            panic!("d");
        })
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"d"));
        assert_clean(&coordinator);
    }

    #[tokio::test(start_paused = true)]
    async fn unawaited_finished_spawn_does_not_block_suspension() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let start = tokio::time::Instant::now();
        let handler_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, Some(Duration::from_mins(15)), |ctx| async move {
            let _h = ctx.step(|_| async { Ok(1_u32) }).name("h").spawn();
            let tasks = Arc::clone(ctx.coordinator().tasks());
            let exited = until(&ctx, || {
                handler_client.received("h", &OperationAction::Succeed) && tasks.live() == 0
            })
            .await;
            if !exited {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("h never ended"),
                ));
            }
            ctx.wait(Duration::from_secs(3600))
                .await
                .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the invocation suspended at once, not at the deadline"
        );
    }

    #[test]
    fn exhausted_budget_does_not_suspend_runnable_handler() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let client = Arc::new(ScriptedClient::empty());
            let ctx = scripted_ctx(&client);
            let outcome = drive_handler(ctx, None, |ctx| async move {
                let _wait = ctx.wait(Duration::from_secs(3600)).name("w").spawn();
                let coordinator = Arc::clone(ctx.coordinator());
                let registered = until(&ctx, || coordinator.live_waits() == 1).await;
                if !registered {
                    return Err(crate::error::WireError::new(
                        None::<String>,
                        Some("no wait"),
                    ));
                }
                let sibling = ctx.step(|_| async { Ok(1_u32) }).name("sibling").spawn();
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
                for value in 0..20_000 {
                    tx.send(value).expect("the receiver is alive");
                }
                drop(tx);
                let mut received = 0_u32;
                while rx.recv().await.is_some() {
                    received += 1;
                }
                sibling.await.map_err(wire_err)?;
                ctx.step(|_| async { Ok(2_u32) })
                    .name("after")
                    .await
                    .map_err(wire_err)?;
                Ok(received.to_string())
            })
            .await
            .expect("no panic");
            assert_eq!(outcome, InvocationOutcome::Complete("20000".to_owned()));
            assert!(client.received("after", &OperationAction::Succeed));
        });
    }

    #[tokio::test(start_paused = true)]
    async fn detached_write_in_flight_at_finish_completes() {
        let client = Arc::new(
            ScriptedClient::new(|updates, _| {
                named(updates, "quick", &OperationAction::Succeed)
                    .map(|succeed| {
                        operation_like(succeed, OperationStatus::Succeeded)
                            .build()
                            .expect("all required fields set")
                    })
                    .into_iter()
                    .collect()
            })
            .with_delay(Duration::from_millis(500)),
        );
        let ctx = scripted_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let log_ctx = ctx.clone();
        let handler_client = Arc::clone(&client);
        let start = tokio::time::Instant::now();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let h = ctx.step(|_| async { Ok(1_u32) }).name("quick").spawn();
            let sent = until(&ctx, || {
                handler_client.received("quick", &OperationAction::Succeed)
            })
            .await;
            if !sent {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("no SUCCEED"),
                ));
            }
            drop(h);
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(start.elapsed() >= Duration::from_millis(500));
        assert_eq!(
            log_ctx.checkpoint_status_view("1").map(|v| v.status),
            Some(CheckpointStatus::Succeeded),
            "the in-flight write's response was merged"
        );
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// Runs the scenario of `root_yield_now_is_a_boundary_on_every_runtime`
    /// once and returns its outcome and whether `after` ran.
    async fn yield_with_live_wait() -> (InvocationOutcome, bool) {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let after_ran = Arc::new(AtomicBool::new(false));
        let handler_ran = Arc::clone(&after_ran);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _wait = ctx.wait(Duration::from_secs(3600)).name("w").spawn();
            let coordinator = Arc::clone(ctx.coordinator());
            let ready = ctx
                .hold_activity(async {
                    for _ in 0..10_000 {
                        if coordinator.live_waits() == 1
                            && coordinator.permits(PermitKind::Activity) == 2
                            && coordinator.permits(PermitKind::Checkpoint) == 0
                        {
                            return true;
                        }
                        tokio::task::yield_now().await;
                    }
                    false
                })
                .await;
            if !ready {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("not ready"),
                ));
            }
            tokio::task::yield_now().await;
            ctx.step(move |_| async move {
                handler_ran.store(true, Ordering::SeqCst);
                Ok(1_u32)
            })
            .name("after")
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        (outcome, after_ran.load(Ordering::SeqCst))
    }

    #[test]
    fn root_yield_now_is_a_boundary_on_every_runtime() {
        let current = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let multi = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("runtime");
        for _ in 0..100 {
            let (outcome, after_ran) = current.block_on(yield_with_live_wait());
            assert_eq!(
                outcome,
                InvocationOutcome::Pending,
                "current-thread runtime"
            );
            assert!(!after_ran, "current-thread runtime");

            let (outcome, after_ran) = multi.block_on(yield_with_live_wait());
            assert_eq!(outcome, InvocationOutcome::Pending, "multi-thread block_on");
            assert!(!after_ran, "multi-thread block_on");

            let (outcome, after_ran) = multi
                .block_on(async { spawned(yield_with_live_wait()).await })
                .expect("the spawned scenario does not panic");
            assert_eq!(outcome, InvocationOutcome::Pending, "multi-thread task");
            assert!(!after_ran, "multi-thread task");
        }
    }

    /// Runs `future` on its own tokio task.
    #[expect(clippy::disallowed_methods)]
    // reason: the scenario must run on a task of the multi-thread runtime,
    // outside the SDK's gateway.
    fn spawned<F>(future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        tokio::spawn(future)
    }

    #[tokio::test(start_paused = true)]
    async fn phase_is_finished_after_every_drive() {
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |_ctx| async move { Ok("done".to_owned()) })
            .await
            .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(coordinator.phase(), Phase::Finished);
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions
mod refresh_scheduler {
    use super::InvocationOutcome;
    use super::test_support::{
        RefreshClient, RefreshRequest, assert_near, assert_times, callback_record, callback_starts,
        drive_handler, named, operation_like, poll_once, refresh_ctx, wire_err,
    };
    use crate::BoxError;
    use crate::builders::wait_for_condition::WaitDecision;
    use crate::client::ExecutionClientError;
    use crate::coordinator::PermitKind;
    use crate::engine::CheckpointStatus;
    use aws_sdk_lambda::types::{
        ChainedInvokeDetails, Operation, OperationAction, OperationStatus, OperationUpdate,
        StepDetails, WaitDetails,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    const fn millis(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    /// A wait record of `start` with `status`, whose scheduled end is
    /// `end_in` from now when `end_in` is `Some`.
    fn wait_record(
        req: &RefreshRequest<'_>,
        start: &OperationUpdate,
        status: OperationStatus,
        end_in: Option<Duration>,
    ) -> Operation {
        let mut builder = operation_like(start, status);
        if let Some(end_in) = end_in {
            builder = builder.wait_details(
                WaitDetails::builder()
                    .scheduled_end_timestamp(req.timestamp_in(end_in))
                    .build(),
            );
        }
        builder.build().expect("all required fields set")
    }

    /// An invoke record of `start` with `status` and `result`.
    fn invoke_record(
        start: &OperationUpdate,
        status: OperationStatus,
        result: Option<&str>,
    ) -> Operation {
        let mut details = ChainedInvokeDetails::builder();
        if let Some(result) = result {
            details = details.result(result);
        }
        operation_like(start, status)
            .chained_invoke_details(details.build())
            .build()
            .expect("all required fields set")
    }

    /// A `wait_for_condition` record of `update` with `status`, carrying
    /// the update's payload as the state and, when `next_in` is `Some`, a
    /// next attempt `next_in` from now.
    fn condition_record(
        req: &RefreshRequest<'_>,
        update: &OperationUpdate,
        status: OperationStatus,
        payload: Option<&str>,
        next_in: Option<Duration>,
    ) -> Operation {
        let mut details = StepDetails::builder().attempt(1);
        if let Some(payload) = payload {
            details = details.result(payload);
        }
        if let Some(next_in) = next_in {
            details = details.next_attempt_timestamp(req.timestamp_in(next_in));
        }
        operation_like(update, status)
            .step_details(details.build())
            .build()
            .expect("all required fields set")
    }

    /// The first update of `name` with `action` among the request's
    /// updates.
    fn this<'a>(
        req: &'a RefreshRequest<'_>,
        name: &str,
        action: &OperationAction,
    ) -> Option<&'a OperationUpdate> {
        named(req.updates, name, action)
    }

    /// The first update of `name` with `action` received so far.
    fn ever<'a>(
        req: &'a RefreshRequest<'_>,
        name: &str,
        action: &OperationAction,
    ) -> Option<&'a OperationUpdate> {
        named(req.history, name, action)
    }

    /// A step that sleeps `d` of tokio time.
    fn sleeping_step(
        ctx: &crate::DurableContext,
        name: &str,
        d: Duration,
    ) -> crate::DurableFuture<u32> {
        ctx.step(move |_| async move {
            tokio::time::sleep(d).await;
            Ok(1_u32)
        })
        .name(name)
        .future()
    }

    /// A spawned step that sleeps `d` of tokio time.
    fn spawned_sleeping_step(
        ctx: &crate::DurableContext,
        name: &str,
        d: Duration,
    ) -> crate::DurableFuture<u32> {
        ctx.step(move |_| async move {
            tokio::time::sleep(d).await;
            Ok(1_u32)
        })
        .name(name)
        .spawn()
    }

    /// The test-3 handler: branch A waits 5 s and runs step `a`, branch B
    /// runs a step of 60 s.
    async fn parallel_wait_then_step(ctx: crate::DurableContext) -> Result<String, BoxError> {
        let branches: Vec<crate::Branch<u32>> = vec![
            crate::Branch::new("A", |child: crate::DurableContext| async move {
                child.wait(secs(5)).name("w").await?;
                let a = child.step(|_| async { Ok(1_u32) }).name("a").await?;
                Ok(a)
            }),
            crate::Branch::new("B", |child: crate::DurableContext| async move {
                let b = child
                    .step(|_| async {
                        tokio::time::sleep(secs(60)).await;
                        Ok(2_u32)
                    })
                    .name("b")
                    .await?;
                Ok(b)
            }),
        ];
        let results = ctx.parallel(branches).name("p").await?;
        Ok(format!("{results:?}"))
    }

    /// The client of test 3: the wait's START carries an end 5 s ahead,
    /// and the first refresh finds it `SUCCEEDED`. Every other update is
    /// answered with the status it records, so the log holds every
    /// operation's final status.
    fn test3_client() -> RefreshClient {
        RefreshClient::new(|req| {
            let mut ops: Vec<Operation> = req
                .updates
                .iter()
                .map(|u| {
                    let status = match u.action() {
                        OperationAction::Succeed => OperationStatus::Succeeded,
                        OperationAction::Fail => OperationStatus::Failed,
                        _ => OperationStatus::Started,
                    };
                    let end_in = (u.name() == Some("w")).then_some(secs(5));
                    wait_record(req, u, status, end_in)
                })
                .collect();
            if req.empty_number == 1
                && let Some(start) = ever(req, "w", &OperationAction::Start)
            {
                ops.push(wait_record(req, start, OperationStatus::Succeeded, None));
            }
            Ok(ops)
        })
    }

    #[tokio::test(start_paused = true)]
    async fn parallel_wait_then_step_finishes_in_one_invocation() {
        let client = Arc::new(test3_client());
        let ctx = refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            parallel_wait_then_step(ctx)
                .await
                .map_err(|e| crate::error::WireError::new(None::<String>, Some(e.to_string())))
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("[1, 2]".to_owned()));
        assert_times(&client.empty_calls(), &[secs(5)]);
        assert_near(
            client
                .time_of("a", &OperationAction::Start)
                .expect("a started"),
            secs(5),
        );
    }

    /// Runs the callback scenario of test 4: a callback result started with
    /// `spawn_result()` and polled once, and a step of `step` beside it.
    /// Returns the outcome; the handler returns the callback's value.
    async fn callback_beside_step(
        client: &Arc<RefreshClient>,
        remaining: Option<Duration>,
        step: Duration,
    ) -> InvocationOutcome {
        let ctx = refresh_ctx(client);
        drive_handler(ctx, remaining, move |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            sleeping_step(&ctx, "s", step).await.map_err(wire_err)?;
            h.await.map_err(wire_err)
        })
        .await
        .expect("no panic")
    }

    /// A client that answers the callback START, and completes the callback
    /// with `"ok"` on the first refresh when `complete` is set.
    fn callback_client(complete: bool) -> RefreshClient {
        RefreshClient::new(move |req| {
            let mut ops = callback_starts(req.updates);
            if complete
                && req.empty_number == 1
                && let Some(start) = ever(req, "cb", &OperationAction::Start)
            {
                ops.push(callback_record(
                    start,
                    OperationStatus::Succeeded,
                    Some("\"ok\""),
                ));
            }
            Ok(ops)
        })
    }

    #[tokio::test(start_paused = true)]
    async fn callback_refresh_continues_in_invocation() {
        let client = Arc::new(callback_client(true));
        let outcome = callback_beside_step(&client, None, secs(3)).await;
        assert_eq!(outcome, InvocationOutcome::Complete("ok".to_owned()));
        assert_times(&client.empty_calls(), &[secs(1)]);
    }

    /// Runs a spawned invoke, polled once, beside a step of 5 s, and returns
    /// the outcome; the handler returns the invoke's value.
    async fn invoke_beside_step(client: &Arc<RefreshClient>) -> InvocationOutcome {
        let ctx = refresh_ctx(client);
        drive_handler(ctx, None, |ctx| async move {
            let mut h = std::pin::pin!(ctx.invoke::<u32, _>("target", 1_u32).name("inv").spawn());
            poll_once(h.as_mut()).await;
            sleeping_step(&ctx, "s", secs(5)).await.map_err(wire_err)?;
            let value = h.await.map_err(wire_err)?;
            Ok(value.to_string())
        })
        .await
        .expect("no panic")
    }

    #[tokio::test(start_paused = true)]
    async fn invoke_refresh_backs_off() {
        let client = Arc::new(RefreshClient::new(|req| {
            let Some(start) = ever(req, "inv", &OperationAction::Start) else {
                return Ok(Vec::new());
            };
            Ok(match req.empty_number {
                1 | 2 => vec![invoke_record(start, OperationStatus::Started, None)],
                3 => vec![invoke_record(start, OperationStatus::Succeeded, Some("7"))],
                _ => Vec::new(),
            })
        }));
        let outcome = invoke_beside_step(&client).await;
        assert_eq!(outcome, InvocationOutcome::Complete("7".to_owned()));
        assert_times(&client.empty_calls(), &[secs(1), secs(2), secs(4)]);
    }

    #[tokio::test(start_paused = true)]
    async fn changed_status_restarts_schedule() {
        let client = Arc::new(RefreshClient::new(|req| {
            let Some(start) = ever(req, "inv", &OperationAction::Start) else {
                return Ok(Vec::new());
            };
            Ok(match req.empty_number {
                1 => vec![invoke_record(start, OperationStatus::Pending, None)],
                2 => vec![invoke_record(start, OperationStatus::Succeeded, Some("7"))],
                _ => Vec::new(),
            })
        }));
        let outcome = invoke_beside_step(&client).await;
        assert_eq!(outcome, InvocationOutcome::Complete("7".to_owned()));
        assert_times(&client.empty_calls(), &[secs(1), secs(2)]);
    }

    /// The check future of [`condition`].
    type CheckFuture = std::future::Ready<Result<i32, BoxError>>;

    /// A `wait_for_condition` that continues with 1 s until its state is at
    /// least 2. Each check records the tokio time it ran at in `checks`.
    fn condition(
        ctx: &crate::DurableContext,
        start: tokio::time::Instant,
        checks: &Arc<Mutex<Vec<Duration>>>,
    ) -> crate::builders::WaitForConditionBuilder<
        i32,
        impl Fn(crate::StepContext, i32) -> CheckFuture + Send + Sync + 'static,
        CheckFuture,
    > {
        let checks = Arc::clone(checks);
        ctx.wait_for_condition(
            move |_ctx, state: i32| {
                checks.lock().expect("lock").push(start.elapsed());
                std::future::ready(Ok::<i32, BoxError>(state + 1))
            },
            0,
        )
        .wait_strategy_fn(|state: i32, _attempt| {
            if state >= 2 {
                WaitDecision::complete()
            } else {
                WaitDecision::continue_with(secs(1))
            }
        })
        .name("cond")
    }

    /// Answers a condition RETRY with `PENDING` and a next attempt 1 s
    /// ahead, and the refresh numbered `ready_on` with `READY`.
    fn condition_client(ready_on: usize) -> RefreshClient {
        RefreshClient::new(move |req| {
            let mut ops = Vec::new();
            if let Some(retry) = this(req, "cond", &OperationAction::Retry) {
                ops.push(condition_record(
                    req,
                    retry,
                    OperationStatus::Pending,
                    retry.payload(),
                    Some(secs(1)),
                ));
            }
            if req.empty_number == ready_on
                && let Some(retry) = ever(req, "cond", &OperationAction::Retry)
            {
                ops.push(condition_record(
                    req,
                    retry,
                    OperationStatus::Ready,
                    retry.payload(),
                    None,
                ));
            }
            Ok(ops)
        })
    }

    #[tokio::test(start_paused = true)]
    async fn condition_check_refreshes_at_next_attempt() {
        let client = Arc::new(condition_client(1));
        let ctx = refresh_ctx(&client);
        let start = tokio::time::Instant::now();
        let checks = Arc::new(Mutex::new(Vec::new()));
        let handler_checks = Arc::clone(&checks);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut h = std::pin::pin!(condition(&ctx, start, &handler_checks).spawn());
            poll_once(h.as_mut()).await;
            sleeping_step(&ctx, "s", secs(3)).await.map_err(wire_err)?;
            let state = h.await.map_err(wire_err)?;
            Ok(state.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("2".to_owned()));
        let checks = checks.lock().expect("lock").clone();
        assert_times(&checks, &[secs(0), secs(1)]);
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_wakes_ready_operations_in_sorted_order() {
        let end = std::time::SystemTime::now() + secs(1);
        let client = Arc::new(RefreshClient::new(move |req| {
            let mut ops = Vec::new();
            for name in ["w1", "w2"] {
                if let Some(start) = this(req, name, &OperationAction::Start) {
                    ops.push(
                        operation_like(start, OperationStatus::Started)
                            .wait_details(
                                WaitDetails::builder()
                                    .scheduled_end_timestamp(aws_smithy_types::DateTime::from(end))
                                    .build(),
                            )
                            .build()
                            .expect("all required fields set"),
                    );
                }
                if req.empty_number == 1
                    && let Some(start) = ever(req, name, &OperationAction::Start)
                {
                    ops.push(wait_record(req, start, OperationStatus::Succeeded, None));
                }
            }
            Ok(ops)
        }));
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w1 = ctx.wait(secs(1)).name("w1").spawn();
            let w2 = ctx.wait(secs(1)).name("w2").spawn();
            let s = spawned_sleeping_step(&ctx, "s", secs(3));
            let (a, b) = tokio::join!(w1, w2);
            a.map_err(wire_err)?;
            b.map_err(wire_err)?;
            s.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(client.empty_calls().len(), 1);
        let updates = client.updates();
        let mut expected: Vec<String> = ["w1", "w2"]
            .iter()
            .map(|n| {
                named(&updates, n, &OperationAction::Start)
                    .expect("the wait started")
                    .id()
                    .to_owned()
            })
            .collect();
        expected.sort();
        assert_eq!(coordinator.wake_log(), expected);
    }

    #[tokio::test(start_paused = true)]
    async fn lone_wait_starts_no_refresh() {
        let client = Arc::new(RefreshClient::new(|_| Ok(Vec::new())));
        let ctx = refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            ctx.wait(secs(1)).await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        tokio::time::sleep(secs(60)).await;
        assert_eq!(client.empty_calls(), Vec::<Duration>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_in_flight_finishes_and_readies_first() {
        let client = Arc::new(
            RefreshClient::new(|req| {
                let mut ops = Vec::new();
                if let Some(start) = this(req, "w", &OperationAction::Start) {
                    ops.push(wait_record(
                        req,
                        start,
                        OperationStatus::Started,
                        Some(secs(1)),
                    ));
                }
                if req.empty_number == 1
                    && let Some(start) = ever(req, "w", &OperationAction::Start)
                {
                    ops.push(wait_record(req, start, OperationStatus::Succeeded, None));
                }
                Ok(ops)
            })
            .with_empty_delay(secs(1)),
        );
        let ctx = refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(secs(1)).name("w").spawn();
            let s = spawned_sleeping_step(&ctx, "s", millis(1500));
            w.await.map_err(wire_err)?;
            s.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_times(&client.empty_answers(), &[secs(2)]);
    }

    #[tokio::test(start_paused = true)]
    async fn no_refresh_within_one_second_of_deadline() {
        let client = Arc::new(callback_client(true));
        let outcome = callback_beside_step(&client, Some(millis(1500)), secs(3)).await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(client.empty_calls(), Vec::<Duration>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_stops_under_one_second_left() {
        let client = Arc::new(callback_client(false));
        let outcome = callback_beside_step(&client, Some(millis(2500)), secs(3)).await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_times(&client.empty_calls(), &[secs(1)]);
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_failure_keeps_classification() {
        // A retryable failure fails the invocation.
        let client = Arc::new(RefreshClient::new(|req| {
            if req.empty_number > 0 {
                return Err(ExecutionClientError::from_retryable(
                    "throttled refresh".to_owned(),
                ));
            }
            Ok(callback_starts(req.updates))
        }));
        let outcome = callback_beside_step(&client, None, secs(3)).await;
        let InvocationOutcome::Fault { message, .. } = &outcome else {
            panic!("expected a fault, got {outcome:?}");
        };
        assert!(message.contains("throttled refresh"), "got {message}");
        // The refresh sends no update, so the failure names the wire ID of
        // its first due registration, the callback's.
        let cb = crate::engine::compute_wire_id_public("1");
        assert!(
            message.contains(&format!("operation {cb}:")),
            "got {message}"
        );
        assert_eq!(client.empty_calls().len(), 1, "no refresh after a failure");

        // A non-retryable failure fails the execution.
        let rejected = ExecutionClientError::new_non_retryable("rejected refresh");
        let expected = crate::error::checkpoint_execution_failure_wire(&rejected);
        let client = Arc::new(RefreshClient::new(move |req| {
            if req.empty_number > 0 {
                return Err(rejected.clone());
            }
            Ok(callback_starts(req.updates))
        }));
        let outcome = callback_beside_step(&client, None, secs(3)).await;
        let InvocationOutcome::Failed { error } = &outcome else {
            panic!("expected a failed execution, got {outcome:?}");
        };
        assert_eq!(error.error_type(), expected.error_type());
        assert_eq!(client.empty_calls().len(), 1, "no refresh after a failure");
    }

    /// A pending callback whose result is spawned and polled once, beside
    /// 3 s of activity that writes no checkpoint, then awaited.
    async fn callback_beside_activity(ctx: crate::DurableContext) -> InvocationOutcome {
        drive_handler(ctx, None, |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            ctx.hold_activity(tokio::time::sleep(secs(3))).await;
            h.await.map_err(wire_err)
        })
        .await
        .expect("no panic")
    }

    /// A refresh failure under checkpoint batching is recorded where the
    /// response arrives, on the detached flush task, for the refresh's
    /// first due registration. No later write observes the failed batch,
    /// so without that record the invocation would suspend on the callback
    /// instead.
    #[tokio::test(start_paused = true)]
    async fn batched_refresh_failure_keeps_classification() {
        let batched_ctx = |client: &Arc<RefreshClient>| {
            crate::DurableContext::new_root_with_client_and_coalescer(
                "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(crate::engine::CheckpointLog::empty()),
                Arc::clone(client) as Arc<dyn crate::client::ExecutionClient>,
                "t0".to_owned(),
                crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::ZERO),
            )
        };

        // A retryable failure fails the invocation.
        let client = Arc::new(RefreshClient::new(|req| {
            if req.empty_number > 0 {
                return Err(ExecutionClientError::from_retryable(
                    "throttled refresh".to_owned(),
                ));
            }
            Ok(callback_starts(req.updates))
        }));
        let outcome = callback_beside_activity(batched_ctx(&client)).await;
        let InvocationOutcome::Fault { message, .. } = &outcome else {
            panic!("expected a fault, got {outcome:?}");
        };
        assert!(message.contains("throttled refresh"), "got {message}");
        let cb = crate::engine::compute_wire_id_public("1");
        assert!(
            message.contains(&format!("operation {cb}:")),
            "got {message}"
        );

        // A non-retryable failure fails the execution.
        let rejected = ExecutionClientError::new_non_retryable("rejected refresh");
        let expected = crate::error::checkpoint_execution_failure_wire(&rejected);
        let client = Arc::new(RefreshClient::new(move |req| {
            if req.empty_number > 0 {
                return Err(rejected.clone());
            }
            Ok(callback_starts(req.updates))
        }));
        let outcome = callback_beside_activity(batched_ctx(&client)).await;
        let InvocationOutcome::Failed { error } = &outcome else {
            panic!("expected a failed execution, got {outcome:?}");
        };
        assert_eq!(error.error_type(), expected.error_type());
    }

    /// The payload of a `panic!` or a failed `assert!` with a message.
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> Option<&str> {
        payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
    }

    /// The checkpoint client panics during a refresh under checkpoint
    /// batching, the default. The refresh is written by a detached flush
    /// task, whose outcome has no reader, while the scheduler waits for
    /// the batch holding its permit. The task gateway records the panic,
    /// which ends the drive although the batch is never published. So the
    /// drive ends, cleans up, and returns the original payload. The
    /// virtual-time timeout bounds the test if the drive hangs.
    #[tokio::test(start_paused = true)]
    async fn batched_client_panic_fails_invocation() {
        let client = Arc::new(RefreshClient::new(|req| {
            assert!(req.empty_number == 0, "batched refresh client panic");
            Ok(callback_starts(req.updates))
        }));
        let ctx = crate::DurableContext::new_root_with_client_and_coalescer(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::ZERO),
        );
        let coordinator = Arc::clone(ctx.coordinator());
        let ended = tokio::time::timeout(
            secs(600),
            drive_handler(ctx, None, |ctx| async move {
                let cb = ctx
                    .create_callback::<String>()
                    .name("cb")
                    .await
                    .map_err(wire_err)?;
                let mut h = std::pin::pin!(cb.spawn_result());
                poll_once(h.as_mut()).await;
                ctx.hold_activity(tokio::time::sleep(secs(3))).await;
                h.await.map_err(wire_err)
            }),
        )
        .await
        .expect("the drive ends");
        let Err(payload) = ended else {
            panic!("expected the client panic, got {ended:?}");
        };
        assert_eq!(
            panic_message(payload.as_ref()),
            Some("batched refresh client panic")
        );
        assert_eq!(coordinator.tasks().live(), 0);
        assert_eq!(coordinator.live_waits(), 0);
        assert_times(&client.empty_calls(), &[secs(1)]);
    }

    /// A kept panic ends a drive whose handler holds an activity permit
    /// and never resolves, and the driver returns its payload.
    #[tokio::test(start_paused = true)]
    async fn kept_panic_ends_the_drive() {
        let client = Arc::new(callback_client(false));
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let ended = tokio::time::timeout(
            secs(600),
            drive_handler(ctx, None, |ctx| async move {
                assert!(
                    ctx.coordinator()
                        .record_panic(Box::new("detached boom"))
                        .is_none()
                );
                ctx.hold_activity(std::future::pending::<()>()).await;
                Ok("unreachable".to_owned())
            }),
        )
        .await
        .expect("the drive ends");
        let Err(payload) = ended else {
            panic!("expected the kept panic, got {ended:?}");
        };
        assert_eq!(panic_message(payload.as_ref()), Some("detached boom"));
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// A fatal error takes precedence over a panic recorded after it, as
    /// it does over every other ending. The panic payload is dropped, and a
    /// panic from its destructor is discarded.
    #[tokio::test(start_paused = true)]
    async fn fatal_takes_precedence_over_kept_panic() {
        struct PanicsOnDrop;
        impl Drop for PanicsOnDrop {
            #[expect(clippy::panic)]
            // reason: the test checks that the driver discards a panic
            // from the payload's destructor.
            fn drop(&mut self) {
                panic!("payload drop");
            }
        }
        let client = Arc::new(callback_client(false));
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let ended = drive_handler(ctx, None, |ctx| async move {
            ctx.coordinator()
                .record_invocation_fault("FatalFirst", "fatal first".to_owned());
            assert!(
                ctx.coordinator()
                    .record_panic(Box::new(PanicsOnDrop))
                    .is_none()
            );
            Ok("done".to_owned())
        })
        .await;
        let Ok(InvocationOutcome::Fault { message, .. }) = &ended else {
            panic!("expected the fatal outcome, got {ended:?}");
        };
        assert!(message.contains("fatal first"), "{message}");
        assert!(!coordinator.has_panic());
    }

    /// The kept panic's payload panics on drop with a payload that panics
    /// on drop again. The driver discards the first destructor panic and
    /// leaks its payload, so the fatal outcome still returns.
    #[tokio::test(start_paused = true)]
    async fn fatal_takes_precedence_over_nested_panicking_kept_payload() {
        let client = Arc::new(callback_client(false));
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let ended = drive_handler(ctx, None, |ctx| async move {
            ctx.coordinator()
                .record_invocation_fault("FatalFirst", "fatal first".to_owned());
            assert!(
                ctx.coordinator()
                    .record_panic(Box::new(super::task_gateway::NestedPanicPayload))
                    .is_none()
            );
            Ok("done".to_owned())
        })
        .await;
        let Ok(InvocationOutcome::Fault { message, .. }) = &ended else {
            panic!("expected the fatal outcome, got {ended:?}");
        };
        assert!(message.contains("fatal first"), "{message}");
        assert!(!coordinator.has_panic());
    }

    /// The handler returns while a refresh is in flight, without
    /// checkpoint batching. The driver drops the scheduler at once, so
    /// only the write task can record the failure. A retryable failure
    /// must still fail the invocation, and a non-retryable one the
    /// execution, instead of the handler's `Complete`.
    #[tokio::test(start_paused = true)]
    async fn refresh_failure_in_flight_when_handler_returns_keeps_classification() {
        async fn run(failure: ExecutionClientError) -> (InvocationOutcome, Arc<RefreshClient>) {
            let client = Arc::new(
                RefreshClient::new(move |req| {
                    if req.empty_number > 0 {
                        return Err(failure.clone());
                    }
                    Ok(callback_starts(req.updates))
                })
                .with_empty_delay(millis(500)),
            );
            let ctx = refresh_ctx(&client);
            let coordinator = Arc::clone(ctx.coordinator());
            let outcome = drive_handler(ctx, None, |ctx| async move {
                let cb = ctx
                    .create_callback::<String>()
                    .name("cb")
                    .await
                    .map_err(wire_err)?;
                let mut h = std::pin::pin!(cb.spawn_result());
                poll_once(h.as_mut()).await;
                ctx.hold_activity(tokio::time::sleep(millis(1200))).await;
                Ok("done".to_owned())
            })
            .await
            .expect("no panic");
            assert_eq!(coordinator.tasks().live(), 0);
            (outcome, client)
        }

        let (outcome, client) = run(ExecutionClientError::from_retryable(
            "throttled refresh".to_owned(),
        ))
        .await;
        let InvocationOutcome::Fault { message, .. } = &outcome else {
            panic!("expected a fault, got {outcome:?}");
        };
        assert!(message.contains("throttled refresh"), "got {message}");
        assert_times(&client.empty_calls(), &[secs(1)]);

        let rejected = ExecutionClientError::new_non_retryable("rejected refresh");
        let expected = crate::error::checkpoint_execution_failure_wire(&rejected);
        let (outcome, client) = run(rejected).await;
        let InvocationOutcome::Failed { error } = &outcome else {
            panic!("expected a failed execution, got {outcome:?}");
        };
        assert_eq!(error.error_type(), expected.error_type());
        assert_times(&client.empty_calls(), &[secs(1)]);
    }

    /// The panic payload of the unbatched refresh tests below.
    struct RefreshPanic(u32);

    /// Drives a handler that returns while an unbatched refresh is in
    /// flight, after recording a fatal error when `fatal` is set. The
    /// checkpoint client panics with `RefreshPanic(7)` after the handler
    /// returned, so the driver has already dropped the scheduler, and with
    /// it the refresh's receiver.
    async fn drive_unbatched_refresh_panic_after_return(
        fatal: bool,
    ) -> (
        Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>,
        Arc<crate::coordinator::InvocationCoordinator>,
    ) {
        let client = Arc::new(
            RefreshClient::new(|req| {
                if req.empty_number > 0 {
                    std::panic::panic_any(RefreshPanic(7));
                }
                Ok(callback_starts(req.updates))
            })
            .with_empty_delay(millis(500)),
        );
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            ctx.hold_activity(tokio::time::sleep(millis(1200))).await;
            if fatal {
                ctx.coordinator()
                    .record_invocation_fault("FatalFirst", "fatal first".to_owned());
            }
            Ok("done".to_owned())
        })
        .await;
        (outcome, coordinator)
    }

    /// The checkpoint client panics during an unbatched refresh after the
    /// handler returned, so the driver has already dropped the scheduler.
    /// The task gateway records the panic of the detached write task on
    /// the coordinator, and the drive returns that payload instead of the
    /// handler's `Complete`.
    #[tokio::test(start_paused = true)]
    async fn unbatched_refresh_panic_after_handler_returns() {
        let (outcome, coordinator) = drive_unbatched_refresh_panic_after_return(false).await;
        let Err(payload) = outcome else {
            panic!("expected the refresh panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<RefreshPanic>().map(|p| p.0), Some(7));
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// A fatal error recorded before the unbatched refresh panic still
    /// takes precedence over it.
    #[tokio::test(start_paused = true)]
    async fn fatal_wins_over_unbatched_refresh_panic_after_handler_returns() {
        let (outcome, coordinator) = drive_unbatched_refresh_panic_after_return(true).await;
        let Ok(InvocationOutcome::Fault { message, .. }) = &outcome else {
            panic!("expected the fatal outcome, got {outcome:?}");
        };
        assert!(message.contains("fatal first"), "{message}");
        assert!(!coordinator.has_panic());
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// The handler panics while a batched refresh is in flight. The driver
    /// drops the scheduler, and so the refresh's caller, when the handler
    /// panics, and the end-of-invocation flush point never runs after a
    /// panic. So only the flusher that publishes the batch's failure can
    /// record it. Its classification must win over the handler's panic.
    #[tokio::test(start_paused = true)]
    async fn batched_refresh_failure_when_handler_panics_keeps_classification() {
        async fn run(failure: ExecutionClientError) -> InvocationOutcome {
            let client = Arc::new(
                RefreshClient::new(move |req| {
                    if req.empty_number > 0 {
                        return Err(failure.clone());
                    }
                    Ok(callback_starts(req.updates))
                })
                .with_empty_delay(millis(500)),
            );
            let ctx = crate::DurableContext::new_root_with_client_and_coalescer(
                "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(crate::engine::CheckpointLog::empty()),
                Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
                "t0".to_owned(),
                crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::ZERO),
            );
            let coordinator = Arc::clone(ctx.coordinator());
            let outcome = drive_handler(ctx, None, |ctx| async move {
                let cb = ctx
                    .create_callback::<String>()
                    .name("cb")
                    .await
                    .map_err(wire_err)?;
                let mut h = std::pin::pin!(cb.spawn_result());
                poll_once(h.as_mut()).await;
                ctx.hold_activity(tokio::time::sleep(millis(1200))).await;
                panic!("handler panic beside a refresh");
            })
            .await
            .expect("the refresh failure wins over the handler panic");
            assert_eq!(coordinator.tasks().live(), 0);
            assert_times(&client.empty_calls(), &[secs(1)]);
            outcome
        }

        let outcome = run(ExecutionClientError::from_retryable(
            "throttled refresh".to_owned(),
        ))
        .await;
        let InvocationOutcome::Fault { message, .. } = &outcome else {
            panic!("expected a fault, got {outcome:?}");
        };
        assert!(message.contains("throttled refresh"), "got {message}");

        let rejected = ExecutionClientError::new_non_retryable("rejected refresh");
        let expected = crate::error::checkpoint_execution_failure_wire(&rejected);
        let outcome = run(rejected).await;
        let InvocationOutcome::Failed { error } = &outcome else {
            panic!("expected a failed execution, got {outcome:?}");
        };
        assert_eq!(error.error_type(), expected.error_type());
    }

    /// The checkpoint client panics during an unbatched refresh while the
    /// handler runs. The task gateway records the panic of the detached
    /// write task, and the scheduler waits on a channel that never carries
    /// it. The driver must still run its cleanup, so no task outlives the
    /// drive, and then return the original payload.
    #[tokio::test(start_paused = true)]
    async fn refresh_client_panic_cleans_up_and_propagates() {
        let client = Arc::new(RefreshClient::new(|req| {
            assert!(req.empty_number == 0, "refresh client panic");
            Ok(callback_starts(req.updates))
        }));
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let ended = drive_handler(ctx, None, |ctx| async move {
            let step = spawned_sleeping_step(&ctx, "s", secs(3));
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            step.await.map_err(wire_err)?;
            h.await.map_err(wire_err)
        })
        .await;
        let Err(payload) = ended else {
            panic!("expected the client panic, got {ended:?}");
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        assert_eq!(message, Some("refresh client panic"));
        assert_eq!(coordinator.tasks().live(), 0);
        assert_eq!(coordinator.live_waits(), 0);
        assert_times(&client.empty_calls(), &[secs(1)]);
    }

    #[tokio::test(start_paused = true)]
    async fn no_refresh_after_invocation_ends() {
        let client = Arc::new(callback_client(false));
        let ctx = refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            sleeping_step(&ctx, "s", millis(2500))
                .await
                .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        tokio::time::sleep(secs(120)).await;
        assert_times(&client.empty_calls(), &[secs(1), secs(2)]);
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_in_flight_when_handler_returns_completes() {
        let client = Arc::new(
            RefreshClient::new(|req| {
                let mut ops = callback_starts(req.updates);
                if req.empty_number > 0
                    && req.now >= millis(1400)
                    && let Some(start) = ever(req, "cb", &OperationAction::Start)
                {
                    ops.push(callback_record(
                        start,
                        OperationStatus::Succeeded,
                        Some("\"ok\""),
                    ));
                }
                Ok(ops)
            })
            .with_empty_delay(millis(500)),
        );
        let ctx = refresh_ctx(&client);
        let log_ctx = ctx.clone();
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            ctx.hold_activity(tokio::time::sleep(millis(1200))).await;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        let returned = client.elapsed();
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_times(&client.empty_answers(), &[millis(1500)]);
        assert!(returned >= millis(1500), "returned at {returned:?}");
        let cb_id = named(&client.updates(), "cb", &OperationAction::Start)
            .expect("the callback started")
            .id()
            .to_owned();
        assert_eq!(
            log_ctx.refresh_view(&cb_id).map(|(status, _)| status),
            Some(CheckpointStatus::Succeeded)
        );
        assert_eq!(coordinator.tasks().live(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn new_earlier_registration_rearms_scheduler() {
        let client = Arc::new(RefreshClient::new(|req| {
            let mut ops = callback_starts(req.updates);
            if let Some(start) = this(req, "w", &OperationAction::Start) {
                ops.push(wait_record(
                    req,
                    start,
                    OperationStatus::Started,
                    Some(millis(300)),
                ));
            }
            if req.empty_number == 1
                && let Some(start) = ever(req, "w", &OperationAction::Start)
            {
                ops.push(wait_record(req, start, OperationStatus::Succeeded, None));
            }
            Ok(ops)
        }));
        let ctx = refresh_ctx(&client);
        let waited = Arc::new(Mutex::new(None));
        let handler_waited = Arc::clone(&waited);
        let handler_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let s = spawned_sleeping_step(&ctx, "s", secs(10));
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let mut h = std::pin::pin!(cb.spawn_result());
            poll_once(h.as_mut()).await;
            tokio::time::sleep(millis(200)).await;
            let result = ctx.wait(secs(1)).name("w").await;
            *handler_waited.lock().expect("lock") =
                Some((result.is_ok(), handler_client.elapsed()));
            s.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let first = *client.empty_calls().first().expect("a refresh was sent");
        assert_near(first, millis(500));
        let (ok, at) = waited.lock().expect("lock").expect("the wait returned");
        assert!(ok, "the wait returns Ok(())");
        assert_near(at, millis(500));
    }

    #[tokio::test(start_paused = true)]
    async fn retired_earliest_registration_rearms_scheduler() {
        let client = Arc::new(RefreshClient::new(|req| {
            let mut ops = callback_starts(req.updates);
            if let Some(start) = this(req, "w", &OperationAction::Start) {
                ops.push(wait_record(
                    req,
                    start,
                    OperationStatus::Started,
                    Some(secs(3)),
                ));
            }
            if req.empty_number == 1
                && let Some(start) = ever(req, "w", &OperationAction::Start)
            {
                ops.push(wait_record(req, start, OperationStatus::Succeeded, None));
            }
            Ok(ops)
        }));
        let ctx = refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let s = spawned_sleeping_step(&ctx, "s", secs(10));
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            // A lazy result future, polled once, registers the wait.
            // Dropping it retires the registration. (Dropping the handle of
            // `spawn_result()` would not: the wait keeps running.)
            let mut cb_h = cb.result();
            poll_once(std::pin::Pin::new(&mut cb_h)).await;
            let w = ctx.wait(secs(3)).name("w").spawn();
            let (waited, ()) = tokio::join!(w, async move {
                tokio::time::sleep(millis(500)).await;
                drop(cb_h);
            });
            waited.map_err(wire_err)?;
            s.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_times(&client.empty_calls(), &[secs(3)]);
    }

    /// The kinds of operation tests 21 and 22 start with `.spawn()` or
    /// `spawn_result()`.
    #[derive(Debug, Clone, Copy)]
    enum Spawned {
        Wait,
        Callback,
        Invoke,
        Condition,
    }

    impl Spawned {
        /// The name the operation gets.
        fn name(self) -> &'static str {
            match self {
                Self::Wait => "w",
                Self::Callback => "cb",
                Self::Invoke => "inv",
                Self::Condition => "cond",
            }
        }

        /// A client whose responses start the operation, and make it ready
        /// on the first refresh.
        fn client(self) -> RefreshClient {
            RefreshClient::new(move |req| {
                let mut ops = callback_starts(req.updates);
                if let Some(start) = this(req, "w", &OperationAction::Start) {
                    ops.push(wait_record(
                        req,
                        start,
                        OperationStatus::Started,
                        Some(secs(1)),
                    ));
                }
                if let Some(retry) = this(req, "cond", &OperationAction::Retry) {
                    ops.push(condition_record(
                        req,
                        retry,
                        OperationStatus::Pending,
                        retry.payload(),
                        Some(secs(1)),
                    ));
                }
                if req.empty_number == 1 {
                    let ready = match self {
                        Self::Wait => ever(req, "w", &OperationAction::Start)
                            .map(|s| wait_record(req, s, OperationStatus::Succeeded, None)),
                        Self::Callback => ever(req, "cb", &OperationAction::Start).map(|s| {
                            callback_record(s, OperationStatus::Succeeded, Some("\"ok\""))
                        }),
                        Self::Invoke => ever(req, "inv", &OperationAction::Start)
                            .map(|s| invoke_record(s, OperationStatus::Succeeded, Some("7"))),
                        Self::Condition => ever(req, "cond", &OperationAction::Retry).map(|r| {
                            condition_record(req, r, OperationStatus::Ready, r.payload(), None)
                        }),
                    };
                    ops.extend(ready);
                }
                Ok(ops)
            })
        }

        /// Starts the operation on `ctx` and returns its handle, whose
        /// value is rendered as a string.
        async fn start(
            self,
            ctx: &crate::DurableContext,
            checks: &Arc<Mutex<Vec<Duration>>>,
        ) -> Result<crate::DurableFuture<String>, crate::error::WireError> {
            Ok(match self {
                Self::Wait => {
                    let h = ctx.wait(secs(1)).name("w").spawn();
                    crate::DurableFuture::from_async(
                        async move { h.await.map(|()| "()".to_owned()) },
                    )
                }
                Self::Callback => {
                    let cb = ctx
                        .create_callback::<String>()
                        .name("cb")
                        .await
                        .map_err(wire_err)?;
                    cb.spawn_result()
                }
                Self::Invoke => {
                    let h = ctx.invoke::<u32, _>("target", 1_u32).name("inv").spawn();
                    crate::DurableFuture::from_async(async move { h.await.map(|v| v.to_string()) })
                }
                Self::Condition => {
                    let h = condition(ctx, tokio::time::Instant::now(), checks).spawn();
                    crate::DurableFuture::from_async(async move { h.await.map(|v| v.to_string()) })
                }
            })
        }

        /// The value the handle resolves with once the operation is ready.
        fn value(self) -> &'static str {
            match self {
                Self::Wait => "()",
                Self::Callback => "ok",
                Self::Invoke => "7",
                Self::Condition => "2",
            }
        }
    }

    /// A wait, an invoke and a callback result start refreshing only after
    /// their handle is first polled.
    #[tokio::test(start_paused = true)]
    async fn unawaited_spawned_operation_is_not_refreshed() {
        for kind in [Spawned::Wait, Spawned::Callback, Spawned::Invoke] {
            let client = Arc::new(kind.client());
            let ctx = refresh_ctx(&client);
            let coordinator = Arc::clone(ctx.coordinator());
            let seen = Arc::new(Mutex::new(None));
            let handler_seen = Arc::clone(&seen);
            let handler_client = Arc::clone(&client);
            let checks = Arc::new(Mutex::new(Vec::new()));
            let outcome = drive_handler(ctx, None, move |ctx| async move {
                // The wrapper future is never polled, so its handle is not.
                let _h = kind.start(&ctx, &checks).await?;
                let live = ctx
                    .hold_activity(async {
                        for _ in 0..1000 {
                            if handler_client.received(kind.name(), &OperationAction::Start)
                                && ctx.coordinator().live_waits() == 1
                            {
                                return Some(ctx.coordinator().live_waits());
                            }
                            tokio::task::yield_now().await;
                        }
                        None
                    })
                    .await;
                *handler_seen.lock().expect("lock") = Some(live);
                sleeping_step(&ctx, "s", secs(3)).await.map_err(wire_err)?;
                Ok("done".to_owned())
            })
            .await
            .expect("no panic");
            let live = seen.lock().expect("lock").expect("the handler ran");
            assert_eq!(live, Some(1), "{kind:?}: the operation is a live wait");
            assert_eq!(
                client.empty_calls(),
                Vec::<Duration>::new(),
                "{kind:?}: no refresh"
            );
            assert_eq!(
                outcome,
                InvocationOutcome::Complete("done".to_owned()),
                "{kind:?}"
            );
            assert_eq!(coordinator.live_waits(), 0, "{kind:?}");
        }
    }

    /// A `wait_for_condition` started with `.spawn()` arms its refresh when
    /// it registers its wait, so its next check runs when the delay ends
    /// although nothing polls its handle.
    #[tokio::test(start_paused = true)]
    async fn unawaited_spawned_condition_is_refreshed() {
        let client = Arc::new(Spawned::Condition.client());
        let ctx = refresh_ctx(&client);
        let checks = Arc::new(Mutex::new(Vec::new()));
        let handler_checks = Arc::clone(&checks);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            // The handle is kept and never polled.
            let _h = Spawned::Condition.start(&ctx, &handler_checks).await?;
            sleeping_step(&ctx, "s", secs(3)).await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_times(&checks.lock().expect("lock"), &[secs(0), secs(1)]);
        assert!(
            !client.empty_calls().is_empty(),
            "the condition is refreshed"
        );
    }

    /// Dropping the handle of `Callback::spawn_result()` keeps the result
    /// wait registered.
    #[tokio::test(start_paused = true)]
    async fn dropped_callback_spawn_result_keeps_waiting() {
        let client = Arc::new(Spawned::Callback.client());
        let ctx = refresh_ctx(&client);
        let seen = Arc::new(Mutex::new(None));
        let handler_seen = Arc::clone(&seen);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            drop(cb.spawn_result());
            let live = ctx
                .hold_activity(async {
                    for _ in 0..1000 {
                        if ctx.coordinator().live_waits() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    ctx.coordinator().live_waits()
                })
                .await;
            *handler_seen.lock().expect("lock") = Some(live);
            sleeping_step(&ctx, "s", secs(1)).await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(*seen.lock().expect("lock"), Some(1), "the wait stays live");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    /// A wait, an invoke and a callback result start refreshing at the
    /// first poll of their handle. A condition refreshes from its
    /// registration (see `unawaited_spawned_condition_is_refreshed`).
    #[tokio::test(start_paused = true)]
    async fn spawned_operation_refresh_starts_at_first_poll() {
        for kind in [Spawned::Wait, Spawned::Callback, Spawned::Invoke] {
            let client = Arc::new(kind.client());
            let ctx = refresh_ctx(&client);
            let resolved = Arc::new(Mutex::new(None));
            let handler_resolved = Arc::clone(&resolved);
            let handler_client = Arc::clone(&client);
            let checks = Arc::new(Mutex::new(Vec::new()));
            let outcome = drive_handler(ctx, None, move |ctx| async move {
                let h = kind.start(&ctx, &checks).await?;
                let s = spawned_sleeping_step(&ctx, "s", secs(4));
                tokio::time::sleep(secs(2)).await;
                let value = h.await.map_err(wire_err)?;
                *handler_resolved.lock().expect("lock") = Some(handler_client.elapsed());
                s.await.map_err(wire_err)?;
                Ok(value)
            })
            .await
            .expect("no panic");
            let expected_at = match kind {
                Spawned::Wait | Spawned::Condition => secs(2),
                Spawned::Callback | Spawned::Invoke => secs(3),
            };
            assert_eq!(
                outcome,
                InvocationOutcome::Complete(kind.value().to_owned()),
                "{kind:?}"
            );
            let first = *client.empty_calls().first().expect("a refresh was sent");
            assert_near(first, expected_at);
            let at = resolved.lock().expect("lock").expect("the handle resolved");
            assert_near(at, expected_at);
        }
    }

    /// The scheduler of a drive whose only work holds no permit sends
    /// nothing. A recorded fatal error keeps the phase `Running` with a due
    /// registration and no permit, the one state in which the coordinator
    /// refuses the scheduler's permit while the scheduler is polled.
    #[tokio::test(start_paused = true)]
    async fn refused_scheduler_permit_sends_nothing() {
        let client = Arc::new(RefreshClient::new(|_| Ok(Vec::new())));
        let ctx = refresh_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        coordinator.record_fatal("Fatal".to_owned(), "stop".to_owned());
        let _wait = coordinator
            .register_test_wait("op", std::task::Waker::noop())
            .expect("running");
        let mut scheduler = std::pin::pin!(super::refresh_scheduler(ctx));
        let polled = tokio::time::timeout(secs(30), scheduler.as_mut()).await;
        assert!(polled.is_err(), "the scheduler stays pending");
        assert_eq!(client.empty_calls(), Vec::<Duration>::new());
        assert_eq!(coordinator.permits(PermitKind::Scheduler), 0);
    }

    #[cfg(feature = "test-util")]
    #[tokio::test(start_paused = true)]
    async fn replay_after_suspension_matches_in_process() {
        // In process: the wait continues inside the one invocation.
        let client = Arc::new(test3_client());
        let ctx = refresh_ctx(&client);
        let log_ctx = ctx.clone();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            parallel_wait_then_step(ctx)
                .await
                .map_err(|e| crate::error::WireError::new(None::<String>, Some(e.to_string())))
        })
        .await
        .expect("no panic");
        let InvocationOutcome::Complete(in_process) = outcome else {
            panic!("expected Complete, got {outcome:?}");
        };
        let mut seen: Vec<String> = Vec::new();
        let mut in_process_ops = Vec::new();
        for update in client.updates() {
            if seen.contains(&update.id().to_owned()) {
                continue;
            }
            seen.push(update.id().to_owned());
            let (status, _) = log_ctx.refresh_view(update.id()).expect("recorded");
            in_process_ops.push((
                update.name().map(str::to_owned),
                update.r#type().as_str().to_owned(),
                status,
            ));
        }

        // Through LocalRunner, which suspends at the wait and replays.
        let result = crate::test_util::LocalRunner::new()
            .run(
                |(): (), ctx: crate::DurableContext| parallel_wait_then_step(ctx),
                (),
            )
            .await
            .expect("the run finishes");
        assert!(result.invocation_count() > 1, "LocalRunner suspended");
        assert_eq!(
            result.output().map(String::as_str),
            Some(in_process.as_str())
        );
        let replayed_ops: Vec<_> = result
            .operations()
            .iter()
            .map(|op| {
                let status = match op.status() {
                    crate::OperationStatus::Succeeded => CheckpointStatus::Succeeded,
                    crate::OperationStatus::Failed => CheckpointStatus::Failed,
                    other => CheckpointStatus::Unknown(format!("{other:?}")),
                };
                (
                    op.name().map(str::to_owned),
                    op.kind().wire_str().to_owned(),
                    status,
                )
            })
            .collect();
        assert_eq!(replayed_ops, in_process_ops);
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: handlers and destructors panic under test, and test assertions
mod panic_owner {
    //! The invocation coordinator owns every invocation-failing panic, and
    //! the driver resumes the first one after its cleanup.

    use super::{InvocationOutcome, drive_outcome, drive_with_scheduler};
    use crate::client::{
        CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput,
        InMemoryExecutionClient,
    };
    use crate::context::DurableContext;
    use crate::coordinator::InvocationCoordinator;
    use crate::engine::CheckpointLog;
    use crate::future::DurableFuture;
    use crate::task::TrackedJoinSet;
    use aws_sdk_lambda::types::OperationUpdate;
    use std::any::Any;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};
    use std::time::Duration;

    type Payload = Box<dyn Any + Send + 'static>;
    type Driven = Result<InvocationOutcome, Payload>;

    fn ctx_with(client: Arc<dyn ExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token0".to_owned(),
        )
    }

    fn live_ctx() -> DurableContext {
        ctx_with(Arc::new(InMemoryExecutionClient::new(Vec::new())))
    }

    /// A panic payload identified by its number. It counts its drops, so a
    /// test can check that a payload was dropped through `discard_panic`.
    struct Tagged(u32, Arc<AtomicUsize>);

    impl Drop for Tagged {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Panics with `Tagged(n)`, whose drops `drops` counts.
    fn panic_tagged(n: u32, drops: &Arc<AtomicUsize>) -> ! {
        std::panic::panic_any(Tagged(n, Arc::clone(drops)))
    }

    /// A new drop counter.
    fn counter() -> Arc<AtomicUsize> {
        Arc::new(AtomicUsize::new(0))
    }

    /// The number of a [`Tagged`] payload, or `None` for another type.
    fn tag(payload: &(dyn Any + Send)) -> Option<u32> {
        payload.downcast_ref::<Tagged>().map(|p| p.0)
    }

    /// The tag of the panic the drive returned. Fails on any other outcome.
    fn returned_tag(driven: &Driven) -> Option<u32> {
        let Err(payload) = driven else {
            panic!("expected a panic, got {driven:?}");
        };
        tag(payload.as_ref())
    }

    /// Panics with `Tagged(n)` when dropped.
    struct PanicsOnDrop(u32, Arc<AtomicUsize>);

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            panic_tagged(self.0, &self.1);
        }
    }

    /// Sets its flag when dropped.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Yields, holding an activity permit, until `done` holds. Fails after
    /// 1,000 yields.
    async fn until(ctx: &DurableContext, done: impl Fn() -> bool) {
        ctx.hold_activity(async {
            for _ in 0..1000 {
                if done() {
                    return;
                }
                tokio::task::yield_now().await;
            }
            panic!("the condition did not hold within 1,000 yields");
        })
        .await;
    }

    /// Drives `handler` on `ctx` and returns the drive's result and the
    /// number of live tasks after it.
    async fn drive<F, Fut>(ctx: DurableContext, handler: F) -> (Driven, usize)
    where
        F: FnOnce(DurableContext) -> Fut,
        Fut: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
    {
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(coordinator.tasks());
        let root = ctx.clone();
        let driven = drive_outcome(handler(ctx), ownership, coordinator, root, None).await;
        (driven, tasks.live())
    }

    /// Spawns a step whose body holds `guard` and stays pending, and sets
    /// `started` once the body runs.
    fn spawn_holding<G: Send + 'static>(
        ctx: &DurableContext,
        name: &str,
        guard: G,
        started: &Arc<AtomicBool>,
    ) -> DurableFuture<()> {
        let guard = Arc::new(Mutex::new(Some(guard)));
        let started = Arc::clone(started);
        ctx.step(move |_| {
            let guard = guard.lock().expect("lock").take();
            let started = Arc::clone(&started);
            async move {
                let _guard = guard;
                started.store(true, Ordering::SeqCst);
                std::future::pending::<()>().await;
                Ok(())
            }
        })
        .name(name)
        .spawn()
    }

    /// Spawns a step whose body panics with `Tagged(n)`.
    fn spawn_panicking(
        ctx: &DurableContext,
        n: u32,
        drops: &Arc<AtomicUsize>,
    ) -> DurableFuture<()> {
        let drops = Arc::clone(drops);
        ctx.step(move |_| {
            let drops = Arc::clone(&drops);
            async move { panic_tagged(n, &drops) }
        })
        .name("boom")
        .spawn()
    }

    /// Test 6. The handler drops the handle of a step that panicked and
    /// returns `Ok`. The handle's drop records the unread panic.
    #[tokio::test]
    async fn dropped_handle_of_panicked_task_fails_invocation() {
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let step = spawn_panicking(&ctx, 6, &counter());
            let tasks = Arc::clone(ctx.coordinator().tasks());
            until(&ctx, || tasks.live() == 0).await;
            drop(step);
            Ok("done".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(6));
        assert_eq!(live, 0);
    }

    /// Test 7. The handler moves the handle of a step that panicked into a
    /// value the test keeps, and returns `Ok`. The cleanup claims the
    /// unread panic.
    #[tokio::test]
    async fn escaped_handle_with_unread_panic_fails_invocation() {
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let kept = Arc::clone(&escaped);
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let step = spawn_panicking(&ctx, 7, &counter());
            let tasks = Arc::clone(ctx.coordinator().tasks());
            until(&ctx, || tasks.live() == 0).await;
            *kept.lock().expect("lock") = Some(step);
            Ok("done".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(7));
        assert!(
            escaped.lock().expect("lock").is_some(),
            "the test still holds the handle"
        );
        assert_eq!(live, 0);
    }

    /// Test 8. The handler drops the handle of a pending step whose body
    /// panics when it is dropped, and returns. Dropping the handle does not
    /// stop the step. The drive's cleanup aborts it, which drops the body,
    /// and the gateway records that panic.
    #[tokio::test]
    async fn panic_while_aborting_dropped_task_fails_invocation() {
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let started = Arc::new(AtomicBool::new(false));
            let step = spawn_holding(&ctx, "hold", PanicsOnDrop(8, counter()), &started);
            until(&ctx, || started.load(Ordering::SeqCst)).await;
            drop(step);
            Ok("done".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(8));
        assert_eq!(live, 0);
    }

    /// Test 9. A join set's item has delivered its panic to the queue, and
    /// the set drops without joining it.
    #[tokio::test]
    async fn queued_join_set_panic_fails_invocation() {
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let drops = counter();
            let mut set = TrackedJoinSet::<()>::new(Arc::clone(ctx.coordinator()));
            set.spawn(
                crate::task::TaskContext::none(),
                ctx.task_ownership(),
                async move { panic_tagged(9, &drops) },
            )
            .expect("runtime");
            let tasks = Arc::clone(ctx.coordinator().tasks());
            until(&ctx, || tasks.live() == 0).await;
            drop(set);
            Ok("done".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(9));
        assert_eq!(live, 0);
    }

    /// Stays pending, sets its flag at its first poll, and panics with
    /// `Tagged(n)` when dropped.
    struct PendingThenDropPanics {
        started: Arc<AtomicBool>,
        _guard: PanicsOnDrop,
    }

    impl Future for PendingThenDropPanics {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            self.started.store(true, Ordering::SeqCst);
            Poll::Pending
        }
    }

    /// Test 10. A join set drops while its item is pending, and the item's
    /// future panics when the abort drops it.
    #[tokio::test]
    async fn join_set_panic_after_drop_fails_invocation() {
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let started = Arc::new(AtomicBool::new(false));
            let mut set = TrackedJoinSet::<()>::new(Arc::clone(ctx.coordinator()));
            set.spawn(
                crate::task::TaskContext::none(),
                ctx.task_ownership(),
                PendingThenDropPanics {
                    started: Arc::clone(&started),
                    _guard: PanicsOnDrop(10, counter()),
                },
            )
            .expect("runtime");
            until(&ctx, || started.load(Ordering::SeqCst)).await;
            drop(set);
            Ok("done".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(10));
        assert_eq!(live, 0);
    }

    /// Records a fatal error when dropped.
    struct FatalOnDrop(Arc<InvocationCoordinator>);

    impl Drop for FatalOnDrop {
        fn drop(&mut self) {
            self.0
                .record_fatal("Fatal".to_owned(), "during cleanup".to_owned());
        }
    }

    /// Test 4. The handler panics; during the cleanup a task records a
    /// fatal error. The fatal outcome wins, and the handler's payload is
    /// dropped through `discard_panic`.
    #[tokio::test(flavor = "current_thread")]
    async fn fatal_after_panic_wins() {
        let drops = counter();
        let handler_drops = Arc::clone(&drops);
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let kept = Arc::clone(&escaped);
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let started = Arc::new(AtomicBool::new(false));
            let guard = FatalOnDrop(Arc::clone(ctx.coordinator()));
            *kept.lock().expect("lock") = Some(spawn_holding(&ctx, "hold", guard, &started));
            until(&ctx, || started.load(Ordering::SeqCst)).await;
            assert!(ctx.coordinator().fatal_error().is_none(), "no fatal yet");
            panic_tagged(4, &handler_drops)
        })
        .await;
        let Ok(InvocationOutcome::Failed { error }) = &driven else {
            panic!("expected the fatal outcome, got {driven:?}");
        };
        assert_eq!(error.error_message(), Some("during cleanup"));
        assert_eq!(drops.load(Ordering::SeqCst), 1, "the payload was dropped");
        assert_eq!(live, 0);
    }

    /// Test 5. A fatal error is recorded, then the handler panics. The
    /// fatal outcome wins, and the payload is dropped through
    /// `discard_panic`.
    #[tokio::test]
    async fn fatal_before_panic_wins() {
        let drops = counter();
        let handler_drops = Arc::clone(&drops);
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            ctx.coordinator()
                .record_fatal("Fatal".to_owned(), "first".to_owned());
            panic_tagged(5, &handler_drops)
        })
        .await;
        let Ok(InvocationOutcome::Failed { error }) = &driven else {
            panic!("expected the fatal outcome, got {driven:?}");
        };
        assert_eq!(error.error_message(), Some("first"));
        assert_eq!(drops.load(Ordering::SeqCst), 1, "the payload was dropped");
        assert_eq!(live, 0);
    }

    /// Test 18. The handler panics with A. The cleanup then aborts a
    /// pending task whose future panics with B when dropped. A is
    /// resumed, and B is dropped through `discard_panic`.
    #[tokio::test(flavor = "current_thread")]
    async fn first_recorded_panic_wins() {
        let a_drops = counter();
        let b_drops = counter();
        let (handler_a, handler_b) = (Arc::clone(&a_drops), Arc::clone(&b_drops));
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let kept = Arc::clone(&escaped);
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let started = Arc::new(AtomicBool::new(false));
            let guard = PanicsOnDrop(182, handler_b);
            *kept.lock().expect("lock") = Some(spawn_holding(&ctx, "hold", guard, &started));
            until(&ctx, || started.load(Ordering::SeqCst)).await;
            panic_tagged(181, &handler_a)
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(181));
        assert_eq!(b_drops.load(Ordering::SeqCst), 1, "B was dropped");
        assert_eq!(
            a_drops.load(Ordering::SeqCst),
            0,
            "A is returned, not dropped"
        );
        assert_eq!(live, 0);
    }

    /// Test 19. The handler returns `Ok`. One task the cleanup aborts
    /// panics when dropped; another holds a drop flag. The cleanup goes on
    /// after the panic, so the second task is dropped too.
    #[tokio::test]
    async fn cleanup_continues_after_a_panic() {
        let flag = Arc::new(AtomicBool::new(false));
        let handler_flag = Arc::clone(&flag);
        let escaped: Arc<Mutex<Vec<DurableFuture<()>>>> = Arc::new(Mutex::new(Vec::new()));
        let kept = Arc::clone(&escaped);
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let first = Arc::new(AtomicBool::new(false));
            let second = Arc::new(AtomicBool::new(false));
            let a = spawn_holding(&ctx, "panics", PanicsOnDrop(19, counter()), &first);
            let b = spawn_holding(&ctx, "flag", DropFlag(handler_flag), &second);
            kept.lock().expect("lock").extend([a, b]);
            until(&ctx, || {
                first.load(Ordering::SeqCst) && second.load(Ordering::SeqCst)
            })
            .await;
            Ok("done".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(19));
        assert!(flag.load(Ordering::SeqCst), "the second task was dropped");
        assert_eq!(live, 0);
    }

    /// Test 20. The handler future holds a value that panics when the
    /// cleanup drops the root task, after the drive ended by suspension.
    #[tokio::test]
    async fn root_drop_panic_during_cleanup_fails_invocation() {
        let (driven, live) = drive(live_ctx(), |ctx| async move {
            let _guard = PanicsOnDrop(20, counter());
            ctx.wait(Duration::from_secs(10))
                .name("timer")
                .await
                .map_err(|err| {
                    crate::error::WireError::new(None::<String>, Some(err.to_string()))
                })?;
            Ok("unreachable".to_owned())
        })
        .await;
        assert_eq!(returned_tag(&driven), Some(20));
        assert_eq!(live, 0);
    }

    /// A test scheduler that panics with `Tagged(n)` at its first poll.
    struct PanickingScheduler(u32);

    impl Future for PanickingScheduler {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            panic_tagged(self.0, &counter())
        }
    }

    /// A test scheduler that stays pending and panics with `Tagged(n)` when
    /// dropped.
    struct DropPanickingScheduler {
        _guard: PanicsOnDrop,
    }

    impl Future for DropPanickingScheduler {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }

    /// Drives a handler that holds a step and returns once the step runs,
    /// with `scheduler` polled in place of the refresh scheduler.
    async fn drive_with<S: Future<Output = ()>>(
        scheduler: S,
        wait_forever: bool,
    ) -> (Driven, usize, bool) {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(coordinator.tasks());
        let dropped = Arc::new(AtomicBool::new(false));
        let body_dropped = Arc::clone(&dropped);
        let handler = async move {
            let started = Arc::new(AtomicBool::new(false));
            let _step = spawn_holding(&ctx, "hold", DropFlag(body_dropped), &started);
            if wait_forever {
                ctx.hold_activity(std::future::pending::<()>()).await;
            }
            until(&ctx, || started.load(Ordering::SeqCst)).await;
            Ok("done".to_owned())
        };
        let driven = drive_with_scheduler(
            crate::task::TaskContext::none(),
            handler,
            ownership,
            coordinator,
            scheduler,
            None,
        )
        .await
        .map(|(outcome, _)| outcome);
        (driven, tasks.live(), dropped.load(Ordering::SeqCst))
    }

    /// Test 21. (a) The scheduler panics at its first poll while the
    /// handler's work is live; (b) a pending scheduler panics when the
    /// cleanup drops it after the handler returned. Each fails the
    /// invocation after the cleanup.
    #[tokio::test]
    async fn scheduler_poll_or_drop_panic_fails_after_cleanup() {
        let (driven, live, _) = drive_with(PanickingScheduler(211), true).await;
        assert_eq!(returned_tag(&driven), Some(211));
        assert_eq!(live, 0);

        let (driven, live, dropped) = drive_with(
            DropPanickingScheduler {
                _guard: PanicsOnDrop(212, counter()),
            },
            false,
        )
        .await;
        assert_eq!(returned_tag(&driven), Some(212));
        assert!(dropped, "the cleanup dropped the step after the panic");
        assert_eq!(live, 0);
    }

    /// A payload that holds a context, counts its drops, and panics when
    /// dropped with a payload whose destructor panics again.
    struct SelfReferential(Option<DurableContext>, Arc<AtomicUsize>);

    impl Drop for SelfReferential {
        fn drop(&mut self) {
            drop(self.0.take());
            self.1.fetch_add(1, Ordering::SeqCst);
            std::panic::panic_any(super::task_gateway::NestedPanicPayload);
        }
    }

    /// Polls `drive` until the handler was dropped and the drive is in its
    /// cleanup, waiting for a detached task.
    async fn poll_into_cleanup(
        drive: &mut Pin<Box<impl Future<Output = Driven>>>,
        coordinator: &InvocationCoordinator,
        handler_dropped: &AtomicBool,
    ) {
        for _ in 0..1000 {
            let polled = std::future::poll_fn(|cx| Poll::Ready(drive.as_mut().poll(cx))).await;
            assert!(polled.is_pending(), "the drive waits for the detached task");
            if handler_dropped.load(Ordering::SeqCst)
                && coordinator.phase() == crate::coordinator::Phase::Finished
            {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the drive did not reach its cleanup");
    }

    /// Test 3. A payload holds the only remaining `DurableContext` clone,
    /// so it keeps the coordinator alive. (a) It is recorded, and the drive
    /// is dropped before the arbitration: the guard drops it. (b) A
    /// detached task is in flight when the drive is dropped, and panics
    /// with it afterwards: the payload comes back to the task, which drops
    /// it. In both cases the payload is dropped once, no panic escapes, and
    /// the coordinator keeps nothing.
    #[tokio::test]
    async fn cancelled_drive_discards_self_referential_panic() {
        for late in [false, true] {
            let ctx = live_ctx();
            let ownership = Arc::clone(ctx.task_ownership());
            let coordinator = Arc::clone(ctx.coordinator());
            let root = ctx.clone();
            let drops = counter();
            let handler_drops = Arc::clone(&drops);
            let handler_dropped = Arc::new(AtomicBool::new(false));
            let flag = DropFlag(Arc::clone(&handler_dropped));
            let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
            let handler = async move {
                let _flag = flag;
                if late {
                    // The detached task takes the only context clone left
                    // after the drive is dropped.
                    let payload_ctx = ctx.clone();
                    let detached = async move {
                        drop(release_rx.await);
                        std::panic::panic_any(SelfReferential(Some(payload_ctx), handler_drops));
                    };
                    crate::task::spawn_detached(ctx.coordinator(), detached).expect("runtime");
                } else {
                    // A detached task keeps the cleanup waiting until the
                    // test releases it.
                    let detached = async move {
                        drop(release_rx.await);
                    };
                    crate::task::spawn_detached(ctx.coordinator(), detached).expect("runtime");
                    let payload = SelfReferential(Some(ctx.clone()), handler_drops);
                    assert!(ctx.coordinator().record_panic(Box::new(payload)).is_none());
                }
                drop(ctx);
                if late {
                    // The handler returns, so the cleanup waits for the
                    // detached task.
                    return Ok("done".to_owned());
                }
                std::future::pending::<Result<String, crate::error::WireError>>().await
            };
            let mut drive = Box::pin(drive_outcome(
                handler,
                ownership,
                Arc::clone(&coordinator),
                root,
                None,
            ));
            poll_into_cleanup(&mut drive, &coordinator, &handler_dropped).await;
            let dropped =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(drive)));
            assert!(dropped.is_ok(), "no panic escapes the drop");
            release_tx.send(()).expect("the detached task waits");
            coordinator.tasks().wait_idle().await;
            assert_eq!(drops.load(Ordering::SeqCst), 1, "late = {late}");
            assert!(!coordinator.has_panic(), "late = {late}");
            assert!(coordinator.take_panic().is_none());
        }
    }

    /// A client whose checkpoint future panics with `Tagged(n)`, at its
    /// poll or, with `on_drop`, when it is dropped after it returned.
    #[derive(Debug)]
    struct PanickingClient {
        n: u32,
        on_drop: bool,
    }

    /// Returns a checkpoint output and panics with `Tagged(n)` when
    /// dropped.
    struct ReadyThenDropPanics(u32);

    impl Future for ReadyThenDropPanics {
        type Output = Result<CheckpointOutput, ExecutionClientError>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(Ok(CheckpointOutput {
                checkpoint_token: Some("token1".to_owned()),
                updated_operations: Vec::new(),
                next_marker: None,
            }))
        }
    }

    impl Drop for ReadyThenDropPanics {
        fn drop(&mut self) {
            panic_tagged(self.0, &counter());
        }
    }

    impl ExecutionClient for PanickingClient {
        fn checkpoint(
            &self,
            _arn: &str,
            _token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            let n = self.n;
            if self.on_drop {
                return Box::pin(ReadyThenDropPanics(n));
            }
            Box::pin(async move { panic_tagged(n, &counter()) })
        }

        fn get_state(
            &self,
            _arn: &str,
            _token: &str,
            _marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(GetStateOutput {
                    operations: Vec::new(),
                    next_marker: None,
                })
            })
        }
    }

    /// Test 15. With checkpoint batching, the client's checkpoint future
    /// returns and then panics when the flusher drops it. The gateway
    /// records the flusher's panic.
    #[tokio::test]
    async fn detached_future_drop_panic_fails_invocation() {
        let ctx = DurableContext::new_root_with_client_and_coalescer(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::new(PanickingClient {
                n: 15,
                on_drop: true,
            }) as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::ZERO),
        );
        let (driven, live) = tokio::time::timeout(
            Duration::from_secs(10),
            drive(ctx, |ctx| async move {
                ctx.step(|_| async { Ok(1_u32) })
                    .name("s")
                    .await
                    .map_err(|err| {
                        crate::error::WireError::new(None::<String>, Some(err.to_string()))
                    })?;
                Ok("done".to_owned())
            }),
        )
        .await
        .expect("the drive ends");
        assert_eq!(returned_tag(&driven), Some(15));
        assert_eq!(live, 0);
    }

    /// Test 17. Without batching, an operation's write future is polled
    /// after its client panicked: it stays pending, because the write
    /// channel carries no payload. The drive returns the client's payload.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn write_waiter_stays_pending_when_client_panics() {
        let ctx = ctx_with(Arc::new(PanickingClient {
            n: 17,
            on_drop: false,
        }));
        let second_poll: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
        let observed = Arc::clone(&second_poll);
        let (driven, live) = tokio::time::timeout(
            Duration::from_secs(10),
            drive(ctx, |ctx| async move {
                let update = OperationUpdate::builder()
                    .id("w")
                    .r#type(aws_sdk_lambda::types::OperationType::Step)
                    .action(aws_sdk_lambda::types::OperationAction::Start)
                    .build()
                    .expect("valid update");
                let mut write = std::pin::pin!(ctx.checkpoint_updates(vec![update]));
                std::future::poll_fn(|cx| {
                    assert!(
                        write.as_mut().poll(cx).is_pending(),
                        "the write runs on its task"
                    );
                    // The write task runs on the other worker. Wait for its
                    // panic inside this poll, so the driver cannot end the
                    // drive between the two polls.
                    // `block_in_place` hands this worker's other tasks,
                    // the new write task among them, to another worker.
                    tokio::task::block_in_place(|| {
                        let start = std::time::Instant::now();
                        while !ctx.coordinator().has_panic() {
                            assert!(
                                start.elapsed() < Duration::from_secs(5),
                                "the client panics"
                            );
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    });
                    *observed.lock().expect("lock") = Some(write.as_mut().poll(cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                std::future::pending::<Result<String, crate::error::WireError>>().await
            }),
        )
        .await
        .expect("the drive ends");
        assert_eq!(
            *second_poll.lock().expect("lock"),
            Some(true),
            "the waiter stays pending"
        );
        assert_eq!(returned_tag(&driven), Some(17));
        assert_eq!(live, 0);
    }

    /// A test scheduler that stays pending. When the cleanup drops it, it
    /// waits until a probe task, spawned after the root task, has run.
    struct ProbeOnDrop(Arc<AtomicBool>);

    impl Future for ProbeOnDrop {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }

    impl Drop for ProbeOnDrop {
        // The probe is spawned outside the SDK's gateway, so it queues
        // behind the root task on the runtime's inject queue.
        fn drop(&mut self) {
            let (tx, rx) = std::sync::mpsc::channel();
            drop(tokio::runtime::Handle::current().spawn(async move {
                tx.send(()).ok();
            }));
            // A timeout would leave the flag unset, which fails the test.
            if rx.recv_timeout(Duration::from_secs(10)).is_ok() {
                self.0.store(true, Ordering::SeqCst);
            }
        }
    }

    /// A panic recorded before the drive starts ends it before the handler
    /// runs. The runtime has one worker, and the driver runs on the
    /// `block_on` thread. So the root task and the probe both go to the
    /// runtime's inject queue, in that order, and the worker takes them in
    /// that order. The cleanup drops the scheduler before it aborts the
    /// root, and the scheduler's drop waits for the probe. So the worker
    /// has started the root task before the abort, and the handler stays
    /// unpolled only because the root cannot take a permit.
    #[test]
    fn panic_recorded_before_the_drive_leaves_the_handler_unpolled() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("runtime");
        let drops = counter();
        let polled = Arc::new(AtomicBool::new(false));
        let probe_ran = Arc::new(AtomicBool::new(false));
        let handler_polled = Arc::clone(&polled);
        let scheduler = ProbeOnDrop(Arc::clone(&probe_ran));
        let driven = runtime.block_on(async {
            let ctx = live_ctx();
            let ownership = Arc::clone(ctx.task_ownership());
            let coordinator = Arc::clone(ctx.coordinator());
            assert!(
                coordinator
                    .record_panic(Box::new(Tagged(119, Arc::clone(&drops))))
                    .is_none(),
                "the coordinator keeps the panic"
            );
            let handler = async move {
                handler_polled.store(true, Ordering::SeqCst);
                std::future::pending::<Result<String, crate::error::WireError>>().await
            };
            drive_with_scheduler(
                crate::task::TaskContext::none(),
                handler,
                ownership,
                coordinator,
                scheduler,
                None,
            )
            .await
            .map(|(outcome, _)| outcome)
        });
        assert!(probe_ran.load(Ordering::SeqCst), "the probe ran");
        assert!(
            !polled.load(Ordering::SeqCst),
            "the handler is never polled"
        );
        assert_eq!(returned_tag(&driven), Some(119));
    }
}

/// Tests of the answer to a checkpoint response without a token.
#[cfg(test)]
mod withdrawal_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use aws_sdk_lambda::types::{
        Operation, OperationAction, OperationStatus as WireStatus, OperationType, OperationUpdate,
    };

    use super::test_support::EventMessages;
    use crate::client::{
        CheckpointOutput, ExecutionClient, ExecutionClientError, ExecutionClientErrorScope,
        GetStateOutput, InMemoryExecutionClient,
    };
    use crate::coordinator::{PermitKind, Phase, WITHDRAWAL_WARNING};
    use crate::engine::CheckpointLog;
    use crate::plugin::{
        HookFuture, HookWork, InvocationEndInfo, InvocationInfo, OperationChangeInfo,
        OperationInfo, Plugin, PluginFactory,
    };
    use crate::{BoxError, DurableContext, InvocationStatus, Options, WireError};

    const ARN: &str = "arn:aws:lambda:us-east-1:1:function:test";
    const REQUEST_ID: &str = "req-1";

    /// How [`WithdrawingClient`] answers one checkpoint request.
    #[derive(Clone)]
    enum Answer {
        /// As [`InMemoryExecutionClient`] answers it, with a token.
        Pass,
        /// `CheckpointOutput::new(None, vec![], None)`.
        Tokenless,
        /// No token, with these operations and this marker.
        TokenlessWith(Vec<Operation>, String),
        /// A transport error of this scope.
        Fail(ExecutionClientErrorScope),
    }

    /// Decides the answer to request `n` (1-based) with `updates`, and the
    /// tokio time to wait before it.
    type Script = dyn Fn(usize, &[OperationUpdate]) -> (Answer, Duration) + Send + Sync;

    /// A scripted `ExecutionClient` that records every call.
    struct WithdrawingClient {
        inner: InMemoryExecutionClient,
        script: Box<Script>,
        calls: Mutex<Vec<Vec<OperationUpdate>>>,
        get_state_calls: AtomicUsize,
        /// When set, every `get_state` call fails.
        get_state_fails: std::sync::atomic::AtomicBool,
    }

    impl WithdrawingClient {
        fn new(
            script: impl Fn(usize, &[OperationUpdate]) -> (Answer, Duration) + Send + Sync + 'static,
        ) -> Arc<Self> {
            Arc::new(Self {
                inner: InMemoryExecutionClient::new(Vec::new()),
                script: Box::new(script),
                calls: Mutex::new(Vec::new()),
                get_state_calls: AtomicUsize::new(0),
                get_state_fails: std::sync::atomic::AtomicBool::new(false),
            })
        }

        /// Answers request `n` without a token, and every other one as
        /// [`Answer::Pass`].
        fn tokenless_at(n: usize) -> Arc<Self> {
            Self::new(move |call, _| {
                if call == n {
                    (Answer::Tokenless, Duration::ZERO)
                } else {
                    (Answer::Pass, Duration::ZERO)
                }
            })
        }

        fn calls(&self) -> Vec<Vec<OperationUpdate>> {
            self.calls.lock().unwrap().clone()
        }

        fn updates(&self) -> Vec<OperationUpdate> {
            self.calls().into_iter().flatten().collect()
        }
    }

    impl ExecutionClient for WithdrawingClient {
        fn checkpoint(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            let number = {
                let mut calls = self.calls.lock().unwrap();
                calls.push(updates.clone());
                calls.len()
            };
            let (answer, delay) = (self.script)(number, &updates);
            let pass = matches!(answer, Answer::Pass).then(|| {
                self.inner
                    .checkpoint(execution_arn, checkpoint_token, updates)
            });
            Box::pin(async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                match answer {
                    Answer::Pass => match pass {
                        Some(pass) => pass.await,
                        None => unreachable!("a pass answer has a future"),
                    },
                    Answer::Tokenless => Ok(CheckpointOutput::new(None, vec![], None)),
                    Answer::TokenlessWith(operations, marker) => {
                        Ok(CheckpointOutput::new(None, operations, Some(marker)))
                    }
                    Answer::Fail(scope) => Err(ExecutionClientError::new(
                        scope,
                        "scripted checkpoint failure",
                    )),
                }
            })
        }

        fn get_state(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            self.get_state_calls.fetch_add(1, Ordering::SeqCst);
            if self.get_state_fails.load(Ordering::SeqCst) {
                return Box::pin(async {
                    Err(ExecutionClientError::new_non_retryable(
                        "scripted get_state failure",
                    ))
                });
            }
            self.inner
                .get_state(execution_arn, checkpoint_token, marker)
        }
    }

    /// What a [`RecordingPlugin`] saw.
    #[derive(Default)]
    struct Seen {
        ends: Vec<(InvocationStatus, Option<WireError>)>,
        operation_starts: usize,
        operation_ends: usize,
        changes: usize,
    }

    #[derive(Clone, Default)]
    struct RecordingFactory(Arc<Mutex<Seen>>);

    impl RecordingFactory {
        /// The end hook's status and error. Exactly one end hook ran.
        fn end(&self) -> (InvocationStatus, Option<WireError>) {
            let seen = self.0.lock().unwrap();
            assert_eq!(seen.ends.len(), 1, "one end hook");
            seen.ends.first().cloned().unwrap()
        }
    }

    impl PluginFactory for RecordingFactory {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(RecordingPlugin(Arc::clone(&self.0)))))
        }
    }

    struct RecordingPlugin(Arc<Mutex<Seen>>);

    impl Plugin for RecordingPlugin {
        fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
            self.0.lock().unwrap().operation_starts += 1;
            Ok(None)
        }

        fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
            self.0.lock().unwrap().operation_ends += 1;
            Ok(None)
        }

        fn on_operation_change(&self, _info: &OperationChangeInfo) -> HookWork {
            self.0.lock().unwrap().changes += 1;
            Ok(None)
        }

        fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
            self.0
                .lock()
                .unwrap()
                .ends
                .push((info.status(), info.execution_error().cloned()));
            Box::pin(std::future::ready(Ok(())))
        }
    }

    /// Runs `handler` behind `wrap` over `client` on a new execution, with
    /// checkpoint batching `batching` and `factory`, and returns the
    /// response.
    async fn run<F, Fut>(
        client: &Arc<WithdrawingClient>,
        batching: bool,
        factory: &RecordingFactory,
        handler: F,
    ) -> serde_json::Value
    where
        F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<u32, BoxError>> + Send,
    {
        run_on(client, batching, factory, &[], handler).await
    }

    /// [`run`] on an execution whose state also holds `records`.
    async fn run_on<F, Fut>(
        client: &Arc<WithdrawingClient>,
        batching: bool,
        factory: &RecordingFactory,
        records: &[serde_json::Value],
        handler: F,
    ) -> serde_json::Value
    where
        F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<u32, BoxError>> + Send,
    {
        run_result(client, batching, factory, records, handler)
            .await
            .expect("the invocation answers a response")
    }

    /// [`run_on`] that returns a Lambda error instead of failing.
    async fn run_result<F, Fut>(
        client: &Arc<WithdrawingClient>,
        batching: bool,
        factory: &RecordingFactory,
        records: &[serde_json::Value],
        handler: F,
    ) -> Result<serde_json::Value, lambda_runtime::Error>
    where
        F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<u32, BoxError>> + Send,
    {
        let mut operations = vec![serde_json::json!({
            "Id": "root",
            "Type": "EXECUTION",
            "Status": "STARTED",
            "ExecutionDetails": { "InputPayload": "{}" }
        })];
        operations.extend_from_slice(records);
        let options = Options::builder()
            .execution_client(Arc::clone(client) as Arc<dyn ExecutionClient>)
            .checkpoint_batching(batching)
            .plugin(factory.clone())
            .build()
            .unwrap();
        let service = crate::wrap(handler, options);
        let mut lambda_ctx = lambda_runtime::Context::default();
        lambda_ctx.request_id = REQUEST_ID.to_owned();
        let event = lambda_runtime::LambdaEvent::new(
            crate::InvocationPayload(serde_json::json!({
                "DurableExecutionArn": ARN,
                "CheckpointToken": "t0",
                "InitialExecutionState": { "Operations": operations }
            })),
            lambda_ctx,
        );
        tokio::time::timeout(Duration::from_secs(60), service(event))
            .await
            .expect("the invocation ends")
            .map(|response| response.0)
    }

    /// Asserts that the last request `client` received carries `name`'s
    /// update with `action`: no request follows the failed one.
    fn last_request_has(client: &WithdrawingClient, name: &str, action: &OperationAction) {
        let calls = client.calls();
        let last = calls.last().expect("a request was sent");
        assert!(has(last, name, action), "{calls:?}");
    }

    /// Asserts the answer and the end of an invocation whose checkpoint
    /// write failed with a transport error of `scope`: a Lambda error and
    /// a `Retrying` end for `Invocation`, and `FAILED` with a `Failed` end
    /// for `Execution`, each with the failure's error.
    fn assert_failure_answer(
        scope: ExecutionClientErrorScope,
        response: Result<serde_json::Value, lambda_runtime::Error>,
        factory: &RecordingFactory,
    ) {
        let (status, error) = factory.end();
        let error = error.expect("the end carries the failure");
        if scope == ExecutionClientErrorScope::Invocation {
            let err = response.expect_err("the invocation fails");
            assert_eq!(status, InvocationStatus::Retrying);
            assert_eq!(
                error.error_type(),
                Some(crate::error::CHECKPOINT_INVOCATION_FAILED_ERROR_TYPE)
            );
            assert_eq!(error.error_message(), Some(err.to_string().as_str()));
        } else {
            let response = response.expect("an envelope");
            assert_eq!(
                response
                    .pointer("/Status")
                    .and_then(serde_json::Value::as_str),
                Some("FAILED"),
                "{response}"
            );
            assert_eq!(status, InvocationStatus::Failed);
            assert_eq!(
                error.error_type(),
                Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
            );
            assert_eq!(
                response
                    .pointer("/Error/ErrorType")
                    .and_then(serde_json::Value::as_str),
                Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
            );
        }
        assert!(
            error
                .error_message()
                .is_some_and(|m| m.contains("scripted checkpoint failure")),
            "{error:?}"
        );
    }

    fn pending() -> serde_json::Value {
        serde_json::json!({ "Status": "PENDING" })
    }

    /// Whether `updates` hold an update of `name` with `action`.
    fn has(updates: &[OperationUpdate], name: &str, action: &OperationAction) -> bool {
        updates
            .iter()
            .any(|u| u.name() == Some(name) && u.action() == action)
    }

    fn no_fail(client: &WithdrawingClient) -> bool {
        client
            .updates()
            .iter()
            .all(|u| u.action() != &OperationAction::Fail)
    }

    /// A root context on `client`, without buffering.
    fn direct_ctx(client: &Arc<WithdrawingClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            ARN.to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
        )
    }

    fn step_start(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Step)
            .action(OperationAction::Start)
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn tokenless_direct_checkpoint_answers_pending() {
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            Ok(ctx.step(|_| async { Ok(1_u32) }).name("a").await?)
        })
        .await;
        assert_eq!(response, pending());
        assert_eq!(client.calls().len(), 1);
        assert!(no_fail(&client));
        assert_eq!(factory.end(), (InvocationStatus::Pending, None));
    }

    #[tokio::test]
    async fn tokenless_buffered_checkpoint_answers_pending() {
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, true, &factory, |_event, ctx| async move {
            let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
            let b = ctx.step(|_| async { Ok(2_u32) }).name("b").spawn();
            Ok(a.await? + b.await?)
        })
        .await;
        assert_eq!(response, pending());
        let calls = client.calls();
        assert_eq!(calls.len(), 1, "{calls:?}");
        let first = calls.first().unwrap();
        assert!(has(first, "a", &OperationAction::Start), "{first:?}");
        assert!(has(first, "b", &OperationAction::Start), "{first:?}");
        assert!(no_fail(&client));
        assert_eq!(factory.end(), (InvocationStatus::Pending, None));
    }

    #[tokio::test]
    async fn tokenless_latch_blocks_later_checkpoints() {
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
            let b = ctx.step(|_| async { Ok(2_u32) }).name("b").spawn();
            Ok(a.await? + b.await?)
        })
        .await;
        assert_eq!(response, pending());
        let calls = client.calls();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert!(has(calls.first().unwrap(), "a", &OperationAction::Start));
        assert!(!has(&client.updates(), "b", &OperationAction::Start));
    }

    #[tokio::test]
    async fn tokenless_emits_lifecycle_events_not_plugin_hooks() {
        let (messages, _guard) = EventMessages::install();
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            Ok(ctx
                .step(|_| async { Ok(1_u32) })
                .name("a")
                .semantics(crate::StepSemantics::AtMostOncePerRetry)
                .await?)
        })
        .await;
        assert_eq!(response, pending());
        let started = messages
            .at(tracing::Level::INFO)
            .into_iter()
            .chain(messages.at(tracing::Level::DEBUG))
            .filter(|m| m == crate::observability::event_names::OPERATION_STARTED)
            .count();
        assert_eq!(started, 1, "the accepted START emits its event");
        let seen = factory.0.lock().unwrap();
        assert_eq!(seen.changes, 0, "no on_operation_change");
        assert_eq!(seen.operation_starts, 0, "no operation start hook");
        assert_eq!(seen.operation_ends, 0, "no operation end hook");
    }

    /// One recorded event: its level, target and fields.
    type Recorded = (tracing::Level, String, Vec<(String, String)>);

    /// Records every event's level, target and fields.
    #[derive(Clone, Default)]
    struct FieldsLayer(Arc<Mutex<Vec<Recorded>>>);

    struct FieldsVisitor(Vec<(String, String)>);

    impl tracing::field::Visit for FieldsVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.push((field.name().to_owned(), format!("{value:?}")));
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for FieldsLayer {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = FieldsVisitor(Vec::new());
            event.record(&mut visitor);
            let meta = event.metadata();
            self.0
                .lock()
                .unwrap()
                .push((*meta.level(), meta.target().to_owned(), visitor.0));
        }
    }

    #[tokio::test]
    async fn tokenless_warns_once_with_fields() {
        use tracing_subscriber::layer::SubscriberExt as _;
        let layer = FieldsLayer::default();
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(layer.clone()));
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            drop(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn());
            Ok(1)
        })
        .await;
        assert_eq!(response, pending());
        let warns: Vec<Recorded> = layer
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|(level, _, _)| *level == tracing::Level::WARN)
            .cloned()
            .collect();
        assert_eq!(warns.len(), 1, "{warns:?}");
        let (_, target, fields) = warns.first().unwrap();
        assert_eq!(target, "aws_durable_execution_sdk::coordinator");
        let field = |name: &str| {
            fields
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(field("executionArn").as_deref(), Some(ARN));
        assert_eq!(field("requestId").as_deref(), Some(REQUEST_ID));
        assert_eq!(field("message").as_deref(), Some(WITHDRAWAL_WARNING));
    }

    #[tokio::test(start_paused = true)]
    async fn tokenless_discards_new_state() {
        let state = Operation::builder()
            .id("other")
            .r#type(OperationType::Step)
            .status(WireStatus::Succeeded)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .build()
            .unwrap();
        let client = WithdrawingClient::new(move |_, _| {
            (
                Answer::TokenlessWith(vec![state.clone()], "marker".to_owned()),
                Duration::ZERO,
            )
        });
        let ctx = direct_ctx(&client);
        let write = ctx.submit_checkpoint_updates(vec![step_start("x")]);
        assert!(
            tokio::time::timeout(Duration::from_secs(5), write)
                .await
                .is_err(),
            "the caller never resolves"
        );
        assert!(ctx.coordinator().withdrawn());
        assert!(
            ctx.checkpoint_status_view("other").is_none()
                && ctx.checkpoint_status_view("x").is_none(),
            "the state is not merged"
        );
        assert_eq!(client.get_state_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn tokenless_leaves_queued_contributors_unresolved() {
        let client = WithdrawingClient::new(|n, _| {
            if n == 1 {
                (Answer::Tokenless, Duration::from_secs(1))
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let ctx = DurableContext::new_root_with_client_and_coalescer(
            ARN.to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::ZERO),
        );
        let first = ctx.submit_checkpoint_updates(vec![step_start("x")]);
        while client.calls().is_empty() {
            tokio::task::yield_now().await;
        }
        let second = ctx.submit_checkpoint_updates(vec![step_start("y")]);
        let (first, second) = tokio::join!(
            tokio::time::timeout(Duration::from_secs(5), first),
            tokio::time::timeout(Duration::from_secs(5), second),
        );
        assert!(first.is_err(), "the in-flight caller never resolves");
        assert!(second.is_err(), "the queued caller never resolves");
        assert!(ctx.coordinator().withdrawn());
        assert_eq!(client.calls().len(), 1, "the queued update is never sent");
    }

    /// The withdrawal drops the queued writes itself: a reserved write,
    /// which no writer would ever take while it is held, is dropped with
    /// its permit, unbuffered and buffered alike. It is never sent, even
    /// after its reservation is released.
    #[tokio::test(start_paused = true)]
    async fn withdrawal_drops_queued_and_held_writes() {
        for buffered in [false, true] {
            let client = WithdrawingClient::new(|n, _| {
                if n == 1 {
                    (Answer::Tokenless, Duration::from_secs(1))
                } else {
                    (Answer::Pass, Duration::ZERO)
                }
            });
            let ctx = if buffered {
                DurableContext::new_root_with_client_and_coalescer(
                    ARN.to_owned(),
                    lambda_runtime::Context::default(),
                    Arc::new(CheckpointLog::empty()),
                    Arc::clone(&client) as Arc<dyn ExecutionClient>,
                    "t0".to_owned(),
                    crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::ZERO),
                )
            } else {
                direct_ctx(&client)
            };
            let first = ctx.submit_checkpoint_updates(vec![step_start("x")]);
            while client.calls().is_empty() {
                tokio::task::yield_now().await;
            }
            let held = ctx.reserve_checkpoint_updates(vec![step_start("y")]);
            assert_eq!(ctx.coordinator().permits(PermitKind::Checkpoint), 2);
            assert!(
                tokio::time::timeout(Duration::from_secs(5), first)
                    .await
                    .is_err(),
                "the in-flight caller never resolves"
            );
            assert!(ctx.coordinator().withdrawn());
            assert_eq!(ctx.coordinator().phase(), Phase::Suspending);
            assert_eq!(
                ctx.coordinator().permits(PermitKind::Checkpoint),
                0,
                "buffered={buffered}: the held write was dropped with its permit"
            );
            let held = held.release();
            assert!(
                tokio::time::timeout(Duration::from_secs(5), held)
                    .await
                    .is_err(),
                "the held caller never resolves"
            );
            assert_eq!(client.calls().len(), 1, "the held write is never sent");
        }
    }

    /// Runs a handler that spawns steps `a` and `b` together, so both
    /// STARTs are queued before either is written. `a`'s START fails at
    /// once with a transport error of `scope`, and `b`'s START, queued
    /// behind it, would be answered without a token.
    async fn queued_client_error(
        scope: ExecutionClientErrorScope,
    ) -> (
        Arc<WithdrawingClient>,
        Result<serde_json::Value, lambda_runtime::Error>,
        RecordingFactory,
    ) {
        let client = WithdrawingClient::new(move |_, updates| {
            if has(updates, "a", &OperationAction::Start) {
                (Answer::Fail(scope), Duration::ZERO)
            } else if has(updates, "b", &OperationAction::Start) {
                (Answer::Tokenless, Duration::ZERO)
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let factory = RecordingFactory::default();
        let response = run_result(&client, false, &factory, &[], |_event, ctx| async move {
            let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
            let b = ctx.step(|_| async { Ok(2_u32) }).name("b").spawn();
            Ok(a.await? + b.await?)
        })
        .await;
        (client, response, factory)
    }

    /// A failed write drops the write queued behind it: `b`'s START is
    /// never sent, and the failure is the answer.
    #[tokio::test]
    async fn queued_write_behind_invocation_client_error_is_not_sent() {
        let scope = ExecutionClientErrorScope::Invocation;
        let (client, response, factory) = queued_client_error(scope).await;
        assert_eq!(client.calls().len(), 1, "only a's START");
        assert_failure_answer(scope, response, &factory);
    }

    #[tokio::test]
    async fn queued_write_behind_execution_client_error_is_not_sent() {
        let scope = ExecutionClientErrorScope::Execution;
        let (client, response, factory) = queued_client_error(scope).await;
        assert_eq!(client.calls().len(), 1, "only a's START");
        assert!(no_fail(&client), "no terminal FAIL is sent");
        assert_failure_answer(scope, response, &factory);
    }

    /// Answers the request that holds `a`'s START after a second, without
    /// a token.
    fn tokenless_start_of_a_after_a_second() -> Arc<WithdrawingClient> {
        WithdrawingClient::new(|_, updates| {
            if has(updates, "a", &OperationAction::Start) {
                (Answer::Tokenless, Duration::from_secs(1))
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        })
    }

    #[tokio::test]
    async fn pending_wins_over_handler_ok() {
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            drop(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn());
            Ok(1)
        })
        .await;
        assert_eq!(response, pending());
        assert!(
            client
                .updates()
                .iter()
                .all(|u| u.r#type() != &OperationType::Execution),
            "no EXECUTION update is sent"
        );
        assert_eq!(factory.end(), (InvocationStatus::Pending, None));
    }

    #[tokio::test]
    async fn pending_wins_over_handler_err() {
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            drop(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn());
            Err::<u32, BoxError>("boom".into())
        })
        .await;
        assert_eq!(response, pending());
        let (status, error) = factory.end();
        assert_eq!(status, InvocationStatus::Pending);
        assert_eq!(error.unwrap().error_message(), Some("boom"));
    }

    /// Runs `handler_err` (or a handler that returns `Ok(1)`) after it
    /// spawns steps `a` and `b` and drops both handles, without buffering.
    /// `a`'s START fails with an invocation-scoped transport error after a
    /// second, after the handler returned, and `b`'s START, queued behind
    /// it, would be answered without a token.
    async fn failed_after_handler(
        handler_err: bool,
    ) -> (
        Arc<WithdrawingClient>,
        Result<serde_json::Value, lambda_runtime::Error>,
        RecordingFactory,
    ) {
        let client = WithdrawingClient::new(|_, updates| {
            if has(updates, "a", &OperationAction::Start) {
                (
                    Answer::Fail(ExecutionClientErrorScope::Invocation),
                    Duration::from_secs(1),
                )
            } else if has(updates, "b", &OperationAction::Start) {
                (Answer::Tokenless, Duration::ZERO)
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let factory = RecordingFactory::default();
        let response = run_result(
            &client,
            false,
            &factory,
            &[],
            move |_event, ctx| async move {
                drop(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn());
                drop(ctx.step(|_| async { Ok(2_u32) }).name("b").spawn());
                if handler_err {
                    return Err::<u32, BoxError>("boom".into());
                }
                Ok(1)
            },
        )
        .await;
        (client, response, factory)
    }

    #[tokio::test(start_paused = true)]
    async fn failure_after_handler_ok_decides_and_stops_later_writes() {
        let (client, response, factory) = failed_after_handler(false).await;
        assert_eq!(client.calls().len(), 1, "only a's START");
        assert_failure_answer(ExecutionClientErrorScope::Invocation, response, &factory);
    }

    #[tokio::test(start_paused = true)]
    async fn failure_after_handler_err_decides_and_stops_later_writes() {
        let (client, response, factory) = failed_after_handler(true).await;
        assert_eq!(client.calls().len(), 1, "only a's START");
        assert_failure_answer(ExecutionClientErrorScope::Invocation, response, &factory);
    }

    /// The failure of a write whose caller was stopped before the response
    /// came decides the outcome: the wait's START queued behind it is never
    /// sent.
    #[tokio::test(start_paused = true)]
    async fn failure_of_stopped_caller_decides_and_stops_later_writes() {
        let client = WithdrawingClient::new(|_, updates| {
            if has(updates, "a", &OperationAction::Start) {
                (
                    Answer::Fail(ExecutionClientErrorScope::Invocation),
                    Duration::from_secs(1),
                )
            } else if has(updates, "w", &OperationAction::Start) {
                (Answer::Tokenless, Duration::ZERO)
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let factory = RecordingFactory::default();
        let response = run_result(&client, false, &factory, &[], |_event, ctx| async move {
            // The child's body returns while `a`'s START is in flight, so
            // its scope closes and stops `a` before the response.
            ctx.run_in_child_context(|child| async move {
                drop(child.step(|_| async { Ok(1_u32) }).name("a").spawn());
                tokio::time::sleep(Duration::from_millis(500)).await;
                Ok(0_u32)
            })
            .name("c")
            .await?;
            ctx.wait(Duration::from_secs(60)).name("w").await?;
            Ok(1)
        })
        .await;
        assert!(!has(&client.updates(), "w", &OperationAction::Start));
        last_request_has(&client, "a", &OperationAction::Start);
        assert_failure_answer(ExecutionClientErrorScope::Invocation, response, &factory);
    }

    /// Runs a handler that spawns step `a`, drops its handle, waits 1.5 s
    /// and returns `Err("boom")` when `handler_err`, else `Ok(1)`, without
    /// buffering. `a`'s SUCCEED fails with an execution-scoped transport
    /// error after a second.
    async fn stopped_write_failure(
        handler_err: bool,
    ) -> (
        Arc<WithdrawingClient>,
        Result<serde_json::Value, lambda_runtime::Error>,
        RecordingFactory,
    ) {
        let client = WithdrawingClient::new(|_, updates| {
            if has(updates, "a", &OperationAction::Succeed) {
                (
                    Answer::Fail(ExecutionClientErrorScope::Execution),
                    Duration::from_secs(1),
                )
            } else if has(updates, "a", &OperationAction::Fail) {
                (Answer::Tokenless, Duration::from_secs(2))
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let factory = RecordingFactory::default();
        let response = run_result(
            &client,
            false,
            &factory,
            &[],
            move |_event, ctx| async move {
                drop(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn());
                tokio::time::sleep(Duration::from_millis(1500)).await;
                if handler_err {
                    return Err::<u32, BoxError>("boom".into());
                }
                Ok(1)
            },
        )
        .await;
        (client, response, factory)
    }

    #[tokio::test(start_paused = true)]
    async fn failed_outcome_write_decides_over_handler_ok_and_sends_no_fail() {
        let (client, response, factory) = stopped_write_failure(false).await;
        assert!(no_fail(&client), "a's terminal FAIL is not sent");
        last_request_has(&client, "a", &OperationAction::Succeed);
        assert_failure_answer(ExecutionClientErrorScope::Execution, response, &factory);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_outcome_write_decides_over_handler_err_and_sends_no_fail() {
        let (client, response, factory) = stopped_write_failure(true).await;
        assert!(no_fail(&client), "a's terminal FAIL is not sent");
        last_request_has(&client, "a", &OperationAction::Succeed);
        assert_failure_answer(ExecutionClientErrorScope::Execution, response, &factory);
    }

    /// Runs a handler whose step `a` SUCCEED fails with a transport error
    /// of `scope` after a second, while step `b`'s START waits behind it
    /// and would be answered without a token. Returns the client, the
    /// answer and the end.
    async fn earlier_client_error(
        scope: ExecutionClientErrorScope,
    ) -> (
        Arc<WithdrawingClient>,
        Result<serde_json::Value, lambda_runtime::Error>,
        RecordingFactory,
    ) {
        let client = WithdrawingClient::new(move |_, updates| {
            if has(updates, "a", &OperationAction::Succeed) {
                (Answer::Fail(scope), Duration::from_secs(1))
            } else if has(updates, "b", &OperationAction::Start) {
                (Answer::Tokenless, Duration::from_secs(1))
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let factory = RecordingFactory::default();
        let response = run_result(&client, false, &factory, &[], |_event, ctx| async move {
            let a = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
            tokio::time::sleep(Duration::from_millis(500)).await;
            let b = ctx.step(|_| async { Ok(2_u32) }).name("b").spawn();
            Ok(a.await? + b.await?)
        })
        .await;
        (client, response, factory)
    }

    #[tokio::test(start_paused = true)]
    async fn invocation_client_error_stops_the_write_waiting_behind_it() {
        let scope = ExecutionClientErrorScope::Invocation;
        let (client, response, factory) = earlier_client_error(scope).await;
        assert!(!has(&client.updates(), "b", &OperationAction::Start));
        last_request_has(&client, "a", &OperationAction::Succeed);
        assert_failure_answer(scope, response, &factory);
    }

    #[tokio::test(start_paused = true)]
    async fn execution_client_error_stops_the_write_waiting_behind_it() {
        let scope = ExecutionClientErrorScope::Execution;
        let (client, response, factory) = earlier_client_error(scope).await;
        assert!(!has(&client.updates(), "b", &OperationAction::Start));
        last_request_has(&client, "a", &OperationAction::Succeed);
        assert!(no_fail(&client), "no terminal FAIL is sent");
        assert_failure_answer(scope, response, &factory);
    }

    #[tokio::test(start_paused = true)]
    async fn pending_wins_over_context_validation_fatal() {
        let client = tokenless_start_of_a_after_a_second();
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            let cloned = ctx.clone();
            Ok(ctx
                .step(move |_| async move {
                    let inner = cloned.step(|_| async { Ok(2_u32) }).name("inner").await?;
                    Ok(inner)
                })
                .name("a")
                .await?)
        })
        .await;
        assert_eq!(response, pending());
        let (status, error) = factory.end();
        assert_eq!(status, InvocationStatus::Pending);
        let error = error.unwrap();
        assert_eq!(error.error_type(), Some("Error"));
        assert!(
            error
                .error_message()
                .is_some_and(|m| m.starts_with("context usage error")),
            "{error:?}"
        );
    }

    /// A serdes whose calls fail with a retryable error.
    struct FailingSerdes;

    impl crate::serdes::Serdes<u32> for FailingSerdes {
        fn serialize(
            &self,
            _value: u32,
            _context: crate::serdes::SerdesContext,
        ) -> impl Future<Output = Result<String, BoxError>> + Send {
            std::future::ready(Err(injected_serdes_failure()))
        }

        fn deserialize(
            &self,
            _wire: String,
            _context: crate::serdes::SerdesContext,
        ) -> impl Future<Output = Result<u32, BoxError>> + Send {
            std::future::ready(Err(injected_serdes_failure()))
        }
    }

    fn injected_serdes_failure() -> BoxError {
        Box::new(crate::serdes::RetryableSerdesError::new(
            "injected serdes failure",
        ))
    }

    /// Step `r`'s replayed result fails to deserialize while step `a`'s
    /// START is in flight. A step serializes its result only after its
    /// START is answered, so the fault comes from the replay.
    #[tokio::test(start_paused = true)]
    async fn pending_wins_over_serdes_fault() {
        let client = tokenless_start_of_a_after_a_second();
        let factory = RecordingFactory::default();
        let replayed = serde_json::json!({
            "Id": "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
            "Type": "STEP",
            "SubType": "Step",
            "Name": "r",
            "Status": "SUCCEEDED",
            "StepDetails": { "Attempt": 1, "Result": "1" }
        });
        let response = run_on(
            &client,
            false,
            &factory,
            &[replayed],
            |_event, ctx| async move {
                let replayed = ctx
                    .step(|_| async { Ok(1_u32) })
                    .name("r")
                    .serdes(FailingSerdes)
                    .future();
                drop(ctx.step(|_| async { Ok(2_u32) }).name("a").spawn());
                Ok(replayed.await?)
            },
        )
        .await;
        assert!(has(&client.updates(), "a", &OperationAction::Start));
        assert_eq!(response, pending());
        let (status, error) = factory.end();
        assert_eq!(status, InvocationStatus::Pending);
        let error = error.unwrap();
        assert_eq!(
            error.error_type(),
            Some(crate::error::SERDES_FAILED_ERROR_TYPE)
        );
        assert!(
            error
                .error_message()
                .is_some_and(|m| m.contains("injected serdes failure")),
            "{error:?}"
        );
    }

    #[tokio::test]
    async fn pure_withdrawal_end_hook_has_no_error() {
        let client = WithdrawingClient::tokenless_at(1);
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
            std::future::pending::<()>().await;
            Ok(1)
        })
        .await;
        assert_eq!(response, pending());
        assert_eq!(factory.end(), (InvocationStatus::Pending, None));
    }

    /// An `EXECUTION` update of the root execution with `action`.
    fn execution_update(action: &OperationAction) -> OperationUpdate {
        let builder = OperationUpdate::builder()
            .id("root")
            .r#type(OperationType::Execution)
            .action(action.clone());
        let builder = if *action == OperationAction::Fail {
            builder.error(
                aws_sdk_lambda::types::ErrorObject::builder()
                    .error_type("Boom")
                    .error_message("boom")
                    .build(),
            )
        } else {
            builder.payload("1")
        };
        builder.build().unwrap()
    }

    #[tokio::test]
    async fn execution_update_exempt_from_withdrawal() {
        for (action, status) in [
            (OperationAction::Succeed, WireStatus::Succeeded),
            (OperationAction::Fail, WireStatus::Failed),
        ] {
            let (messages, _guard) = EventMessages::install();
            let terminal = Operation::builder()
                .id("root")
                .r#type(OperationType::Execution)
                .status(status.clone())
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .build()
                .unwrap();
            let client = WithdrawingClient::new(move |_, _| {
                (
                    Answer::TokenlessWith(vec![terminal.clone()], String::new()),
                    Duration::ZERO,
                )
            });
            let ctx = direct_ctx(&client);
            let output = tokio::time::timeout(
                Duration::from_secs(5),
                ctx.checkpoint_updates_direct(vec![execution_update(&action)]),
            )
            .await
            .expect("the call resolves")
            .expect("the call returns Ok");
            assert_eq!(output.checkpoint_token(), None);
            assert!(
                output
                    .updated_operations
                    .iter()
                    .any(|op| op.r#type() == &OperationType::Execution && op.status() == &status),
                "the execution is terminal"
            );
            assert!(!ctx.coordinator().withdrawn(), "no latch is set");
            assert_eq!(ctx.coordinator().phase(), Phase::Running);
            assert_eq!(messages.at(tracing::Level::WARN), Vec::<String>::new());
        }
    }

    /// A tokenless response to an `EXECUTION` update returns `Ok` without
    /// fetching its pages, because the spent token is still in place: the
    /// response carries a marker, and `get_state` would fail.
    #[tokio::test]
    async fn execution_update_exempt_skips_pagination() {
        for action in [OperationAction::Succeed, OperationAction::Fail] {
            let client = WithdrawingClient::new(move |_, _| {
                (
                    Answer::TokenlessWith(Vec::new(), "marker".to_owned()),
                    Duration::ZERO,
                )
            });
            client.get_state_fails.store(true, Ordering::SeqCst);
            let ctx = direct_ctx(&client);
            let output = tokio::time::timeout(
                Duration::from_secs(5),
                ctx.checkpoint_updates_direct(vec![execution_update(&action)]),
            )
            .await
            .expect("the call resolves")
            .expect("the call returns Ok");
            assert_eq!(output.checkpoint_token(), None);
            assert!(!ctx.coordinator().withdrawn(), "no latch is set");
            assert_eq!(ctx.coordinator().phase(), Phase::Running);
            assert_eq!(
                client.get_state_calls.load(Ordering::SeqCst),
                0,
                "{action:?}: no get_state call"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_checkpoint_withdrawn() {
        let client = WithdrawingClient::new(|_, updates| {
            if updates.is_empty() {
                (Answer::Tokenless, Duration::ZERO)
            } else {
                (Answer::Pass, Duration::ZERO)
            }
        });
        let factory = RecordingFactory::default();
        let response = run(&client, false, &factory, |_event, ctx| async move {
            let wait = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    Ok(1_u32)
                })
                .name("s")
                .spawn();
            let (waited, value) = tokio::join!(wait, step);
            waited?;
            Ok(value?)
        })
        .await;
        assert_eq!(response, pending());
        let calls = client.calls();
        let refresh = calls
            .iter()
            .position(Vec::is_empty)
            .expect("a refresh was sent");
        assert_eq!(
            refresh + 1,
            calls.len(),
            "no checkpoint after the refresh: {calls:?}"
        );
        assert!(!has(&client.updates(), "s", &OperationAction::Succeed));
        assert_eq!(factory.end(), (InvocationStatus::Pending, None));
    }
}
