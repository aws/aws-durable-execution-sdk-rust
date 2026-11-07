//! The task gateway: every tokio task the SDK starts goes through here.
//!
//! The SDK starts tasks for `.spawn()`, for each map item and parallel
//! branch, for each combinator input, for the checkpoint batch flusher, for
//! detached checkpoint writes, for blocking serdes work, and for the
//! handler itself. Each of these tasks goes through one of the functions
//! below:
//!
//! - [`spawn_root`] starts the handler of one invocation;
//! - [`spawn_tracked`] and [`TrackedJoinSet::spawn`] start operation tasks;
//! - [`spawn_detached`] starts internal work that must not be aborted;
//! - [`run_blocking`] runs work on the blocking pool.
//!
//! Every task except the root holds one count on the invocation's
//! [`TaskTracker`] until it exits. The invocation driver aborts every
//! abortable task at the end of the drive and waits until the count is
//! zero. So no SDK task outlives its invocation.
//!
//! Dropping the handle of an operation task does not abort the task: the
//! operation loses only its result. A task stops early only when its scope
//! closes (see [`crate::checkpoint_coalescer::Admission`]), when the join
//! set that owns it is dropped or aborts it, or when the drive aborts every
//! task. Dropping the root's handle aborts the root task.
//!
//! The wrapper around each task catches a panic of the task's future and
//! hands it to the consumer as [`TaskOutcome::Panicked`]. The consumer
//! calls [`TaskOutcome::into_value`], which resumes the panic with its
//! original payload. So tokio never sees a panic from an SDK task, and no
//! consumer handles a `JoinError`.
//!
//! An outcome can have no reader: the task is detached, or its handle, its
//! join set or its blocking waiter was dropped. A panic in such an outcome
//! still fails the invocation. So it is recorded on the invocation's
//! [`InvocationCoordinator`], which keeps the first invocation-failing
//! panic for the driver. A value without a reader is dropped inside a
//! catch, and a panic from its destructor is recorded the same way. A panic
//! of a task's future while the task is aborted is recorded too. At the end
//! of the drive, [`TaskTracker::claim_unread_panics`] records every panic
//! that still sits unread in a slot whose handle is alive.
//!
//! The wrapper of an operation task registers the task with the
//! task-ownership check before the first poll of its future, and removes
//! it before the task exits. So every task that can run a durable
//! operation passes the check, and tokio can reuse a task ID only for a
//! task that is not in the set.
//!
//! # Permits
//!
//! The wrapper of the root task and of every operation task holds an
//! activity permit of the invocation's
//! [`InvocationCoordinator`](crate::coordinator::InvocationCoordinator)
//! while its task can make progress (see [`Schedule`]):
//!
//! 1. A spawn takes the permit on the spawning task, so the new task counts
//!    before tokio polls it. A spawn after the phase left `Running` starts
//!    nothing and returns a handle that never resolves.
//! 2. The wrapper polls its future with a waker of its own. A wake while the
//!    task is idle takes a permit; a wake during a poll keeps the poll's
//!    permit. A poll that returns `Pending` without a wake releases it.
//! 3. A task that finishes hands its outcome over, which wakes the consumer
//!    and so gives the consumer its permit, and only then releases its own.
//! 4. Once the phase is not `Running`, the wrapper no longer polls its
//!    future.
//!
//! Detached tasks run the user's checkpoint client, but no handler code.
//! They take no activity permit and ignore the phase.

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::task::{Context, Poll, Wake, Waker};

use tokio::task::{AbortHandle, Id, JoinSet};

use crate::coordinator::{InvocationCoordinator, Permit, PermitKind, Phase};
use crate::driver::TaskOwnership;
use crate::error::{OperationError, OperationErrorKind, StepError, StepErrorKind};
use crate::plugin::Frames;

/// A captured panic payload.
type Payload = Box<dyn Any + Send + 'static>;

/// The message of the error an operation returns when it needs a task and
/// no Tokio runtime is running on the calling thread.
pub(crate) const NO_RUNTIME_MESSAGE: &str = "durable operation started with `.spawn()` while no \
     Tokio runtime is running; call it from the handler or from an SDK operation";

tokio::task_local! {
    /// The coordinator of the invocation whose task is being polled. The
    /// gateway wrapper sets it for each poll of its future, so blocking
    /// work started from any SDK task holds a permit of that invocation
    /// and is counted on its tracker.
    static CURRENT_COORDINATOR: Arc<InvocationCoordinator>;
}

/// How a tracked task ended.
pub(crate) enum TaskOutcome<T> {
    /// The future returned a value.
    Value(T),
    /// The future, or its destructor, panicked with this payload.
    Panicked(Payload),
}

impl<T> TaskOutcome<T> {
    /// Returns the value, or resumes the panic with its original payload.
    pub(crate) fn into_value(self) -> T {
        match self {
            Self::Value(value) => value,
            Self::Panicked(payload) => std::panic::resume_unwind(payload),
        }
    }
}

impl<T> std::fmt::Debug for TaskOutcome<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(_) => f.write_str("Value(..)"),
            Self::Panicked(_) => f.write_str("Panicked(..)"),
        }
    }
}

/// Counts live tracked tasks of one invocation, and keeps an abort handle
/// for each operation task, which [`TaskTracker::abort_all`] aborts.
#[derive(Debug)]
pub(crate) struct TaskTracker {
    /// The number of tasks that have not exited yet.
    live: Mutex<usize>,
    /// The abort handles of the tasks [`Self::abort_all`] aborts, and
    /// whether `abort_all` has run.
    abortable: Mutex<Abortable>,
    /// Woken when `live` reaches zero.
    idle: tokio::sync::Notify,
    /// The slots that received an outcome while their reader was alive
    /// (see [`Self::claim_unread_panics`]).
    unread: Mutex<Vec<Weak<dyn UnreadPanic>>>,
}

/// A slot that can hold an outcome nobody has read yet.
trait UnreadPanic: Send + Sync {
    /// Takes the panic payload the slot holds, if it holds one. Leaves any
    /// other outcome in the slot.
    fn claim_panic(&self) -> Option<Payload>;
}

impl TaskTracker {
    /// Creates a tracker with no live task.
    pub(crate) fn new() -> Self {
        Self {
            live: Mutex::new(0),
            abortable: Mutex::new(Abortable {
                handles: BTreeMap::new(),
                aborting: false,
                closed: false,
            }),
            idle: tokio::sync::Notify::new(),
            unread: Mutex::new(Vec::new()),
        }
    }

    /// Remembers a slot that received an outcome, so the cleanup can claim
    /// a panic in it that no reader takes.
    fn watch_unread(&self, slot: Weak<dyn UnreadPanic>) {
        self.unread
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(slot);
    }

    /// Records on `coordinator` every panic that is still unread in a
    /// slot, including a slot whose handle is alive outside the handler.
    ///
    /// The driver calls it after every task has stopped, so no task
    /// delivers an outcome after it. A handle that is polled afterwards
    /// reads an unread value. It finds a claimed panic's slot empty and
    /// stays pending.
    pub(crate) fn claim_unread_panics(&self, coordinator: &InvocationCoordinator) {
        let slots =
            std::mem::take(&mut *self.unread.lock().unwrap_or_else(PoisonError::into_inner));
        for slot in slots {
            if let Some(payload) = slot.upgrade().and_then(|slot| slot.claim_panic()) {
                record_or_discard(coordinator, payload);
            }
        }
    }

    /// Adds one live task. The returned permit removes it on drop.
    pub(crate) fn acquire(self: &Arc<Self>) -> TaskPermit {
        *self.live.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        TaskPermit {
            tasks: Some(Arc::clone(self)),
        }
    }

    /// Returns the number of live tasks.
    pub(crate) fn live(&self) -> usize {
        *self.live.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Aborts every task started by `spawn_tracked` or
    /// `TrackedJoinSet::spawn` that has not unwound yet. Detached tasks keep
    /// running.
    ///
    /// A task that a live task starts after this call is aborted as soon
    /// as it is registered. Teardown needs this rule. A tracked task can be
    /// in the middle of a poll on another worker thread while this call
    /// runs, and that poll can start a nested operation task. The nested
    /// task holds a permit from before its spawn. So if it were not
    /// aborted, [`Self::wait_idle`] would wait for it forever.
    pub(crate) fn abort_all(&self) {
        // Set the flag and take the handles under one lock. A registration
        // takes the same lock. So each registration either lands in the
        // taken map or sees the flag. The `live` lock is taken first, in
        // the order `release` takes both. With no live task, no task can
        // register later, so the flag stays clear and the tracker stays
        // usable. The locks are not held while tokio processes the aborts.
        let handles = {
            let live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
            let mut abortable = self
                .abortable
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            abortable.aborting = *live > 0;
            drop(live);
            std::mem::take(&mut abortable.handles)
        };
        for handle in handles.into_values() {
            handle.abort();
        }
    }

    /// Aborts every tracked task, as [`Self::abort_all`] does, and aborts
    /// every task registered later, for good.
    ///
    /// The drive calls it when it is dropped before its cleanup. The root
    /// task is not counted here, and aborting it does not interrupt a poll
    /// that runs on another worker thread. So the tracker can have no live
    /// task while the root, still in that poll, starts a new operation
    /// task. `abort_all` would let that task run, because it leaves its
    /// flag clear when no task is live. A closed tracker aborts it as soon
    /// as it is registered, and [`Self::release`] never reopens it.
    pub(crate) fn close(&self) {
        let handles = {
            let mut abortable = self
                .abortable
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            abortable.closed = true;
            std::mem::take(&mut abortable.handles)
        };
        for handle in handles.into_values() {
            handle.abort();
        }
    }

    /// Returns whether [`Self::close`] has run.
    #[cfg(test)]
    pub(crate) fn is_closed(&self) -> bool {
        self.abortable
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closed
    }

    /// Returns the number of stored abort handles.
    #[cfg(test)]
    fn abortable_count(&self) -> usize {
        self.abortable
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .handles
            .len()
    }

    /// Returns once no tracked task is live.
    pub(crate) async fn wait_idle(&self) {
        loop {
            // Register for the wakeup BEFORE the check, so a permit dropped
            // between the check and the await still wakes this waiter.
            let notified = self.idle.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            if self.live() == 0 {
                return;
            }
            notified.await;
        }
    }

    /// Stores the abort handle of a task that [`Self::abort_all`] aborts.
    /// After `abort_all` has run, or once the tracker is closed, aborts the
    /// task instead.
    fn register_abort(&self, handle: AbortHandle) {
        let late = {
            let mut abortable = self
                .abortable
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if abortable.aborting || abortable.closed {
                Some(handle)
            } else {
                abortable.handles.insert(handle.id(), handle);
                None
            }
        };
        if let Some(handle) = late {
            handle.abort();
        }
    }

    /// Removes one live task. At zero, clears the abort handles and the
    /// aborting flag, and wakes
    /// [`Self::wait_idle`].
    fn release(&self) {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        *live = live.saturating_sub(1);
        if *live == 0 {
            let mut abortable = self
                .abortable
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            abortable.handles.clear();
            abortable.aborting = false;
            drop(abortable);
            self.idle.notify_waiters();
        }
    }
}

/// The abort side of a [`TaskTracker`].
#[derive(Debug)]
struct Abortable {
    /// The abort handles of tasks that have not been aborted. Cleared when
    /// no task is live.
    handles: BTreeMap<Id, AbortHandle>,
    /// Set by [`TaskTracker::abort_all`]. While it is set, a new
    /// registration aborts its task at once. Cleared when no task is live:
    /// only a live task can start a tracked task, so after that point no
    /// task of the teardown can register.
    aborting: bool,
    /// Set by [`TaskTracker::close`] and never cleared. While it is set, a
    /// new registration aborts its task at once.
    closed: bool,
}

/// One live task. Dropping it subtracts 1. At zero the tracker clears its
/// abort handles and wakes `wait_idle`. A permit without a tracker does
/// nothing.
#[derive(Debug)]
pub(crate) struct TaskPermit {
    tasks: Option<Arc<TaskTracker>>,
}

impl Drop for TaskPermit {
    fn drop(&mut self) {
        if let Some(tasks) = self.tasks.take() {
            tasks.release();
        }
    }
}

/// No Tokio runtime is running on the calling thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NoRuntime;

impl NoRuntime {
    /// The error an operation returns when it needs a task and no Tokio
    /// runtime is running. Its kind is the one the task-ownership check
    /// returns.
    pub(crate) fn operation_error(_no_runtime: Self) -> OperationError {
        OperationError::from_kind(OperationErrorKind::Step(StepError::new(
            StepErrorKind::ExecutionFailed,
            Some(NO_RUNTIME_MESSAGE.into()),
        )))
    }
}

impl std::fmt::Display for NoRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(NO_RUNTIME_MESSAGE)
    }
}

impl std::error::Error for NoRuntime {}

/// Returns `Err(NoRuntime)` when no Tokio runtime is running on this thread.
pub(crate) fn ensure_runtime() -> Result<(), NoRuntime> {
    tokio::runtime::Handle::try_current()
        .map(drop)
        .map_err(|_| NoRuntime)
}

/// Returns the ID of the task being polled.
fn current_task_id() -> Id {
    #[expect(clippy::expect_used)]
    // reason: only the gateway wrapper calls this, while tokio polls it as
    // the future of a task that `tokio::spawn` or `JoinSet::spawn` started,
    // and such a future always runs with a task ID.
    tokio::task::try_id().expect("a future that tokio spawned runs with a task ID")
}

/// Drops a panic payload that cannot become the invocation's outcome.
///
/// It is used only for a payload the coordinator rejected, a kept payload
/// that a cancelled drive disposes of, a payload that a recorded fatal
/// error outranks, and a second panic a task's gateway catches after it
/// chose the task's first panic as the outcome.
///
/// A payload is a value of any type, and its destructor can panic too. So
/// the payload is dropped inside a catch. When that destructor panics, the
/// payload of that second panic is leaked with `mem::forget` instead of
/// dropped, because its destructor can panic again. So no panic escapes.
pub(crate) fn discard_panic(payload: Payload) {
    if let Err(nested) = catch_unwind(AssertUnwindSafe(move || drop(payload))) {
        std::mem::forget(nested);
    }
}

/// Hands an invocation-failing panic to `coordinator`, and drops a payload
/// it does not keep through [`discard_panic`].
pub(crate) fn record_or_discard(coordinator: &InvocationCoordinator, payload: Payload) {
    if let Some(rejected) = coordinator.record_panic(payload) {
        discard_panic(rejected);
    }
}

/// Drops `value`, which no reader receives, inside a catch. A panic from
/// its destructor is invocation-failing, so it is recorded on
/// `coordinator`.
pub(crate) fn drop_recorded<T>(coordinator: &InvocationCoordinator, value: T) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(move || drop(value))) {
        record_or_discard(coordinator, payload);
    }
}

/// Drops `value` after the gateway chose an earlier panic as the task's
/// outcome. A panic from its destructor is that task's second panic, so
/// it goes to [`discard_panic`].
pub(crate) fn drop_after_first_panic<T>(value: T) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(move || drop(value))) {
        discard_panic(payload);
    }
}

/// Disposes of an outcome that no reader receives: a panic is recorded on
/// `coordinator`, and a value is dropped by [`drop_recorded`].
fn dispose<T>(coordinator: &InvocationCoordinator, outcome: TaskOutcome<T>) {
    match outcome {
        TaskOutcome::Panicked(payload) => record_or_discard(coordinator, payload),
        TaskOutcome::Value(value) => drop_recorded(coordinator, value),
    }
}

/// The scheduling state of one permit-aware task.
#[derive(Debug)]
enum ScheduleState {
    /// Not runnable: no permit is held.
    Idle,
    /// Woken (or just spawned) and waiting for its next poll: holds the
    /// permit that poll uses.
    Queued(Permit),
    /// Being polled. `woken` records a wake during the poll.
    Polling { permit: Permit, woken: bool },
    /// The task finished or was dropped. Wakes take nothing.
    Done,
}

/// The permit state of one permit-aware task, and the waker the wrapper
/// polls its future with.
///
/// A registration or a resource the future waits on holds this waker. So
/// a wake that makes the task runnable takes the task's activity permit
/// before the waker of the task itself runs.
pub(crate) struct Schedule {
    state: Mutex<ScheduleState>,
    coordinator: Arc<InvocationCoordinator>,
    /// The waker tokio polled the task with last. Every wake also wakes it.
    outer: Mutex<Option<Waker>>,
}

impl std::fmt::Debug for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Schedule").finish_non_exhaustive()
    }
}

impl Schedule {
    fn with_state(coordinator: &Arc<InvocationCoordinator>, state: ScheduleState) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(state),
            coordinator: Arc::clone(coordinator),
            outer: Mutex::new(None),
        })
    }

    /// A schedule that starts `Queued` with an activity permit taken now.
    /// Returns `None` when the phase is not `Running`.
    fn queued(coordinator: &Arc<InvocationCoordinator>) -> Option<Arc<Self>> {
        let permit = coordinator.acquire(PermitKind::Activity)?;
        Some(Self::with_state(coordinator, ScheduleState::Queued(permit)))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ScheduleState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stores the waker tokio polled the task with.
    fn set_outer(&self, waker: &Waker) {
        let mut outer = self.outer.lock().unwrap_or_else(PoisonError::into_inner);
        match &*outer {
            Some(existing) if existing.will_wake(waker) => {}
            _ => *outer = Some(waker.clone()),
        }
    }

    /// Starts a poll: `Queued` becomes `Polling` with the same permit, and
    /// `Idle` takes a fresh permit. Returns `false` when no permit can be
    /// taken (the phase left `Running`) or the task is done.
    fn begin_poll(&self) -> bool {
        let mut state = self.lock();
        match std::mem::replace(&mut *state, ScheduleState::Done) {
            ScheduleState::Queued(permit) => {
                *state = ScheduleState::Polling {
                    permit,
                    woken: false,
                };
                true
            }
            ScheduleState::Idle => {
                if let Some(permit) = self.coordinator.acquire(PermitKind::Activity) {
                    *state = ScheduleState::Polling {
                        permit,
                        woken: false,
                    };
                    true
                } else {
                    *state = ScheduleState::Idle;
                    false
                }
            }
            polling @ ScheduleState::Polling { .. } => {
                *state = polling;
                true
            }
            ScheduleState::Done => false,
        }
    }

    /// Ends a poll that returned `Pending`. With a wake during the poll, or
    /// with `requeue`, the task stays `Queued` with the poll's permit.
    /// Otherwise it becomes `Idle`, and the permit is released after the
    /// lock is.
    fn end_pending(&self, requeue: bool) {
        let released = {
            let mut state = self.lock();
            match std::mem::replace(&mut *state, ScheduleState::Done) {
                ScheduleState::Polling { permit, woken } if woken || requeue => {
                    *state = ScheduleState::Queued(permit);
                    None
                }
                ScheduleState::Polling { permit, .. } => {
                    *state = ScheduleState::Idle;
                    Some(permit)
                }
                other => {
                    *state = other;
                    None
                }
            }
        };
        drop(released);
    }

    /// Ends the task's last poll: the task is done, and the poll's permit is
    /// returned so the caller releases it after it handed the outcome over.
    fn end_ready(&self) -> Option<Permit> {
        let mut state = self.lock();
        match std::mem::replace(&mut *state, ScheduleState::Done) {
            ScheduleState::Polling { permit, .. } | ScheduleState::Queued(permit) => Some(permit),
            ScheduleState::Idle | ScheduleState::Done => None,
        }
    }

    /// Marks the task done and releases any permit it holds.
    fn close(&self) {
        let released = std::mem::replace(&mut *self.lock(), ScheduleState::Done);
        drop(released);
    }

    /// The waker the wrapper polls its future with.
    fn waker(self: &Arc<Self>) -> Waker {
        Waker::from(Arc::clone(self))
    }

    fn on_wake(&self) {
        {
            let mut state = self.lock();
            match &mut *state {
                ScheduleState::Idle => {
                    if let Some(permit) = self.coordinator.acquire(PermitKind::Activity) {
                        *state = ScheduleState::Queued(permit);
                    }
                }
                ScheduleState::Polling { woken, .. } => *woken = true,
                ScheduleState::Queued(_) | ScheduleState::Done => {}
            }
        }
        let outer = self
            .outer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(outer) = outer {
            outer.wake();
        }
    }
}

impl Wake for Schedule {
    fn wake(self: Arc<Self>) {
        self.on_wake();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.on_wake();
    }
}

/// The state an outcome slot shares between a task and its handle.
struct SlotState<T> {
    outcome: Option<TaskOutcome<T>>,
    waker: Option<Waker>,
    /// Set when the reader is gone. An outcome delivered after that has no
    /// reader, so it is disposed of at once.
    closed: bool,
}

/// The slot a task stores its outcome in, shared with its handle.
struct Slot<T> {
    state: Mutex<SlotState<T>>,
    /// The coordinator that records an outcome without a reader. `None` for
    /// blocking work outside an invocation and for a refused spawn.
    coordinator: Option<Arc<InvocationCoordinator>>,
}

impl<T> Slot<T> {
    fn new(coordinator: Option<Arc<InvocationCoordinator>>) -> Self {
        Self {
            state: Mutex::new(SlotState {
                outcome: None,
                waker: None,
                closed: false,
            }),
            coordinator,
        }
    }

    /// Disposes of an outcome that no reader receives. Without a
    /// coordinator nothing owns the outcome, so it is dropped.
    fn dispose(&self, outcome: TaskOutcome<T>) {
        match &self.coordinator {
            Some(coordinator) => dispose(coordinator, outcome),
            None => drop(outcome),
        }
    }

    /// Marks the reader gone, and disposes of an outcome it did not read.
    fn close(&self) {
        let unread = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.closed = true;
            state.waker = None;
            state.outcome.take()
        };
        if let Some(outcome) = unread {
            self.dispose(outcome);
        }
    }

    /// Takes the outcome, or registers the consumer's waker.
    fn poll_take(&self, cx: &Context<'_>) -> Poll<TaskOutcome<T>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(outcome) = state.outcome.take() {
            return Poll::Ready(outcome);
        }
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

impl<T: Send + 'static> Slot<T> {
    /// Stores the outcome and wakes the consumer that registered a waker.
    /// After the reader is gone, disposes of the outcome instead. A stored
    /// outcome is watched, so the cleanup records it if it is a panic that
    /// no reader takes.
    fn deliver(self: &Arc<Self>, outcome: TaskOutcome<T>) {
        let (waker, unread) = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if state.closed {
                (None, Some(outcome))
            } else {
                state.outcome = Some(outcome);
                (state.waker.take(), None)
            }
        };
        if let Some(outcome) = unread {
            self.dispose(outcome);
            return;
        }
        if let Some(coordinator) = &self.coordinator {
            let slot: Weak<dyn UnreadPanic> = Arc::downgrade(self) as Weak<dyn UnreadPanic>;
            coordinator.tasks().watch_unread(slot);
        }
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl<T: Send + 'static> UnreadPanic for Slot<T> {
    /// Takes an unread panic. An unread value stays in the slot, so a
    /// handle that outlives the drive still reads it.
    fn claim_panic(&self) -> Option<Payload> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match state.outcome.take() {
            Some(TaskOutcome::Panicked(payload)) => Some(payload),
            value => {
                state.outcome = value;
                None
            }
        }
    }
}

/// Waits for the outcome blocking work stores in its slot. Dropping it
/// closes the slot, so an outcome the work stores later is disposed of.
struct SlotWait<T>(Arc<Slot<T>>);

impl<T> Future for SlotWait<T> {
    type Output = TaskOutcome<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.poll_take(cx)
    }
}

impl<T> Drop for SlotWait<T> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// The queue a join set's tasks store their outcomes in.
struct JoinState<T> {
    outcomes: VecDeque<(Id, TaskOutcome<T>)>,
    waker: Option<Waker>,
    /// Set when the join set is dropped. An outcome delivered after that is
    /// disposed of at once.
    closed: bool,
}

/// Where the gateway wrapper stores its task's outcome.
enum Sink<T> {
    /// The slot of a [`TrackedTask`].
    Slot(Arc<Slot<T>>),
    /// The queue of a [`TrackedJoinSet`].
    Queue(Arc<Mutex<JoinState<T>>>),
    /// The coordinator: a detached task's outcome has no reader, so a panic
    /// is recorded there.
    Record,
}

impl<T: Send + 'static> Sink<T> {
    fn deliver(&self, outcome: TaskOutcome<T>, coordinator: &InvocationCoordinator) {
        match self {
            Self::Slot(slot) => slot.deliver(outcome),
            Self::Queue(queue) => {
                let id = current_task_id();
                let (waker, unread) = {
                    let mut state = queue.lock().unwrap_or_else(PoisonError::into_inner);
                    if state.closed {
                        (None, Some(outcome))
                    } else {
                        state.outcomes.push_back((id, outcome));
                        (state.waker.take(), None)
                    }
                };
                if let Some(outcome) = unread {
                    dispose(coordinator, outcome);
                }
                if let Some(waker) = waker {
                    waker.wake();
                }
            }
            Self::Record => dispose(coordinator, outcome),
        }
    }
}

/// Removes a blessed task ID from the ownership set on drop.
struct BlessGuard {
    ownership: Arc<TaskOwnership>,
    id: Id,
}

impl Drop for BlessGuard {
    fn drop(&mut self) {
        self.ownership.unbless_task(self.id);
    }
}

/// How a wrapper takes part in the suspension decision.
enum Gate {
    /// An operation task or the root: polled under its [`Schedule`] while
    /// the phase is `Running`, and not polled afterwards.
    Scheduled {
        schedule: Arc<Schedule>,
        waker: Waker,
        /// The root sets the phase to `Finished` before it releases its
        /// permit, when the handler resolves or panics.
        root: bool,
    },
    /// A detached task: no permit, polled whatever the phase.
    Detached,
}

/// What a tracked task runs inside: the plugin frames of the context that
/// created its operation, and the `tracing` span that was current when it
/// was created.
///
/// A [`DurableFuture`](crate::DurableFuture) captures one when it is
/// created. The gateway runs every poll and every drop of the task's
/// future through [`TaskContext::catch`]: inside the span, then inside the
/// frames. So a task runs with the context of the code that created it,
/// as JS's `AsyncLocalStorage` gives a promise the context it was created
/// in, and a span a plugin's wrap creates is a child of the captured span.
#[derive(Debug, Clone)]
pub(crate) struct TaskContext {
    frames: Frames,
    span: tracing::Span,
}

impl TaskContext {
    /// The context `frames` and `span`.
    pub(crate) fn new(frames: Frames, span: tracing::Span) -> Self {
        Self { frames, span }
    }

    /// The frames and the span current on this thread: what a task
    /// created now runs inside.
    pub(crate) fn current() -> Self {
        Self::new(Frames::current(), tracing::Span::current())
    }

    /// No frames and no span.
    pub(crate) fn none() -> Self {
        Self::new(Frames::none(), tracing::Span::none())
    }

    /// Runs `f` on this thread inside these frames and this span.
    pub(crate) fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        let _span = self.span.enter();
        self.frames.enter(f)
    }

    /// Runs `action` inside the span, then inside the frames, and catches
    /// every panic, for the task gateway:
    ///
    /// ```text
    /// catch the span's enter and exit
    ///   enter the span
    ///     enter the frames
    ///       catch the action
    ///     leave the frames
    ///   leave the span
    /// ```
    ///
    /// No panic unwinds through a plugin frame. The action's result is kept
    /// until the span has exited. The first panic wins: a panic entering
    /// the span prevents the action, and a panic of the action wins over a
    /// later panic exiting the span, which is discarded. A panic exiting
    /// the span after the action returned wins over its value, which is
    /// dropped with its own panic contained. An action that a panic
    /// entering the span prevented is dropped the same way, after the
    /// catch, so it never drops while that panic unwinds.
    fn catch<R>(&self, action: impl FnOnce() -> R) -> std::thread::Result<R> {
        catch_in_span(&self.span, || {
            self.frames.enter(|| catch_unwind(AssertUnwindSafe(action)))
        })
    }

    /// The frames, for tests that check what a task runs inside.
    #[cfg(test)]
    pub(crate) fn frames(&self) -> &Frames {
        &self.frames
    }

    /// The span, for tests that check what a task runs inside.
    #[cfg(test)]
    pub(crate) fn span(&self) -> &tracing::Span {
        &self.span
    }
}

/// Runs `action` inside `span` and catches every panic of the span's enter,
/// the action and the span's exit:
///
/// ```text
/// catch the span's enter and exit
///   enter the span
///     catch the action
///   leave the span
/// ```
///
/// The action's result is kept until the span has exited, so the span's
/// exit never runs while a panic unwinds. The first panic wins: a panic
/// entering the span prevents the action, and a panic of the action wins
/// over a later panic exiting the span, which is discarded. A panic exiting
/// the span after the action returned wins over its value, which is
/// dropped with its own panic contained. An action that a panic entering
/// the span prevented is dropped the same way, after the catch, so it
/// never drops while that panic unwinds.
pub(crate) fn catch_in_span<R>(
    span: &tracing::Span,
    action: impl FnOnce() -> std::thread::Result<R>,
) -> std::thread::Result<R> {
    let mut action = Some(action);
    let mut inner = None;
    let outer = catch_unwind(AssertUnwindSafe(|| {
        let _entered = span.enter();
        #[expect(clippy::expect_used)]
        // reason: this closure runs once, and only the line below takes
        // the action, so it is still present after the span's enter.
        let action = action
            .take()
            .expect("the action is taken only once, after the span's enter");
        inner = Some(catch_unwind(AssertUnwindSafe(action)).unwrap_or_else(Err));
    }));
    drop_after_first_panic(action);
    match (inner, outer) {
        (Some(result), Ok(())) => result,
        (Some(Ok(value)), Err(exit)) => {
            drop_after_first_panic(value);
            Err(exit)
        }
        (Some(Err(first)), Err(exit)) => {
            discard_panic(exit);
            Err(first)
        }
        (None, outer) => {
            #[expect(clippy::expect_used)]
            // reason: the action's panic is caught inside the closure, so
            // without a result the span's enter panicked.
            Err(outer.expect_err("the action runs unless the span's enter panicked"))
        }
    }
}

/// Drops `task_cx` inside a catch. Its span's close and its frames'
/// destruction run user code (a subscriber's `on_close`, a plugin's
/// destructor), so a panic there is returned instead of unwinding through
/// the gateway.
fn dispose_task_cx(task_cx: TaskContext) -> std::thread::Result<()> {
    catch_unwind(AssertUnwindSafe(move || drop(task_cx)))
}

/// The future of every gateway task: polls the inner future under
/// `catch_unwind`, blesses the task before the first poll, and hands the
/// outcome to the consumer before it releases its permits.
///
/// The fields drop in declaration order, so after `Drop::drop` has dropped
/// the inner future and closed the schedule, the bless guard drops before
/// the task count.
struct Wrapped<F: Future> {
    /// The inner future. `None` once it finished or was dropped.
    inner: Option<Pin<Box<F>>>,
    /// The span and the frames every poll and every drop of `inner` run
    /// inside, through [`TaskContext::catch`]. Replaced with
    /// [`TaskContext::none`] and dropped inside a catch once `inner` is
    /// gone, because dropping the last clone of its span closes it.
    task_cx: TaskContext,
    /// Present from the first poll of an operation task until it exits.
    bless: Option<BlessGuard>,
    /// The ownership handle to bless the task in, taken at the first poll.
    ownership: Option<Arc<TaskOwnership>>,
    /// The coordinator set as `CURRENT_COORDINATOR` during each poll.
    coordinator: Arc<InvocationCoordinator>,
    gate: Gate,
    /// Where the outcome goes.
    sink: Sink<F::Output>,
    /// The task's count, dropped after the outcome is stored.
    permit: Option<TaskPermit>,
}

impl<F: Future> Wrapped<F> {
    fn new(
        task_cx: TaskContext,
        future: F,
        ownership: Option<Arc<TaskOwnership>>,
        coordinator: Arc<InvocationCoordinator>,
        gate: Gate,
        sink: Sink<F::Output>,
        permit: Option<TaskPermit>,
    ) -> Self {
        Self {
            inner: Some(Box::pin(future)),
            task_cx,
            bless: None,
            ownership,
            coordinator,
            gate,
            sink,
            permit,
        }
    }
}

/// A [`Gate::Scheduled`] for `schedule`.
fn scheduled(schedule: Arc<Schedule>, root: bool) -> Gate {
    let waker = schedule.waker();
    Gate::Scheduled {
        schedule,
        waker,
        root,
    }
}

impl<F> Future for Wrapped<F>
where
    F: Future,
    F::Output: Send + 'static,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if let Gate::Scheduled { schedule, .. } = &this.gate {
            schedule.set_outer(cx.waker());
            if this.coordinator.phase() != Phase::Running || !schedule.begin_poll() {
                return Poll::Pending;
            }
        }
        // Bless before the inner future can run any code, so a durable
        // operation it calls on its first poll passes the ownership check.
        if let Some(ownership) = this.ownership.take() {
            let id = current_task_id();
            ownership.bless_task(id);
            this.bless = Some(BlessGuard { ownership, id });
        }
        #[expect(clippy::expect_used)]
        // reason: the inner future is taken only when this poll returns
        // `Ready`, and tokio never polls a task's future after `Ready`.
        let inner = this
            .inner
            .as_mut()
            .expect("tokio does not poll a finished task");
        // `TaskContext::catch` enters the span, then the frames, and
        // catches innermost: a panic of the task's future, or of its span's
        // enter and exit, never unwinds through a plugin frame.
        let task_cx = &this.task_cx;
        let coordinator = &this.coordinator;
        let polled = match &this.gate {
            Gate::Scheduled { waker, .. } => {
                let mut inner_cx = Context::from_waker(waker);
                task_cx.catch(|| {
                    CURRENT_COORDINATOR.sync_scope(Arc::clone(coordinator), || {
                        inner.as_mut().poll(&mut inner_cx)
                    })
                })
            }
            Gate::Detached => task_cx.catch(|| {
                CURRENT_COORDINATOR.sync_scope(Arc::clone(coordinator), || inner.as_mut().poll(cx))
            }),
        };
        let outcome = match polled {
            Ok(Poll::Pending) => {
                if let Gate::Scheduled { schedule, .. } = &this.gate {
                    // After the cooperative budget is spent, a tokio
                    // resource returns `Pending` and wakes the task only
                    // after this poll returned. Without the requeue the
                    // task would look idle while it is runnable.
                    let exhausted = !tokio::task::coop::has_budget_remaining();
                    schedule.end_pending(exhausted);
                    if exhausted {
                        cx.waker().wake_by_ref();
                    }
                }
                return Poll::Pending;
            }
            Ok(Poll::Ready(value)) => {
                let inner = this.inner.take();
                match this.task_cx.catch(move || drop(inner)) {
                    Ok(()) => TaskOutcome::Value(value),
                    Err(payload) => {
                        // The destructor's panic is the outcome. A panic
                        // while dropping the value is the task's second
                        // panic, so it is discarded.
                        drop_after_first_panic(value);
                        TaskOutcome::Panicked(payload)
                    }
                }
            }
            Err(payload) => {
                // The poll's panic is the outcome. A destructor panic after
                // it is the task's second panic, so it is discarded.
                let inner = this.inner.take();
                if let Err(later) = this.task_cx.catch(move || drop(inner)) {
                    discard_panic(later);
                }
                TaskOutcome::Panicked(payload)
            }
        };
        // The task context is disposed of before the outcome is handed
        // over: the last clone of its span closes here, and a subscriber's
        // `on_close` is user code. Its panic is the task's panic only when
        // the task had none.
        let task_cx = std::mem::replace(&mut this.task_cx, TaskContext::none());
        let outcome = match (dispose_task_cx(task_cx), outcome) {
            (Ok(()), outcome) => outcome,
            (Err(payload), TaskOutcome::Value(value)) => {
                drop_after_first_panic(value);
                TaskOutcome::Panicked(payload)
            }
            (Err(later), first @ TaskOutcome::Panicked(_)) => {
                discard_panic(later);
                first
            }
        };
        drop(this.bless.take());
        // The poll's permit is held while the outcome is handed over, so
        // the consumer's wake takes its own permit first.
        let permit = match &this.gate {
            Gate::Scheduled { schedule, .. } => schedule.end_ready(),
            Gate::Detached => None,
        };
        this.sink.deliver(outcome, &this.coordinator);
        if let Gate::Scheduled { root: true, .. } = &this.gate {
            this.coordinator.finish();
        }
        drop(permit);
        drop(this.permit.take());
        Poll::Ready(())
    }
}

impl<F: Future> Drop for Wrapped<F> {
    fn drop(&mut self) {
        // An aborted task: its future is dropped before it finished, and
        // nobody reads an outcome. A panic from the future's destructor is
        // invocation-failing, so it is recorded.
        let dropped = match self.inner.take() {
            Some(inner) => self.task_cx.catch(move || drop(inner)),
            None => Ok(()),
        };
        // Then the task context, inside a catch: the last clone of its span
        // closes here, and a subscriber's `on_close` is user code. A panic
        // of the future's destructor came first, so it wins.
        let task_cx = std::mem::replace(&mut self.task_cx, TaskContext::none());
        match (dropped, dispose_task_cx(task_cx)) {
            (Ok(()), Ok(())) => {}
            (Err(payload), Ok(())) | (Ok(()), Err(payload)) => {
                record_or_discard(&self.coordinator, payload);
            }
            (Err(first), Err(later)) => {
                discard_panic(later);
                record_or_discard(&self.coordinator, first);
            }
        }
        // A waker that outlives the task takes no permit from now on.
        if let Gate::Scheduled { schedule, .. } = &self.gate {
            schedule.close();
        }
    }
}

/// The three raw tokio spawn calls. Everything else in the crate starts a
/// task through the gateway functions of this module.
#[expect(clippy::disallowed_methods)]
// reason: the gateway is the one place that calls tokio's spawn functions;
// `clippy.toml` disallows them everywhere else and points here.
mod raw {
    use std::future::Future;

    use tokio::task::{AbortHandle, JoinHandle, JoinSet};

    pub(super) fn spawn<F>(future: F) -> JoinHandle<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        tokio::spawn(future)
    }

    pub(super) fn join_set_spawn<F>(set: &mut JoinSet<()>, future: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        set.spawn(future)
    }

    pub(super) fn spawn_blocking<F>(f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        drop(tokio::task::spawn_blocking(f));
    }
}

/// The handle of a tracked task. Holds the outcome slot it shares with its
/// task, and the task's `AbortHandle`. A handle without an abort handle
/// belongs to a spawn the coordinator refused: it never resolves.
///
/// Dropping the handle closes the slot. An outcome the task stored before
/// and did not hand over, or stores after, is disposed of: a panic is
/// recorded on the coordinator, and a value is dropped inside a catch.
///
/// Dropping the root's handle also aborts the root task. Dropping the
/// handle of an operation task loses only its result: the task runs until
/// it ends, until the scope it was started in closes (see
/// [`crate::checkpoint_coalescer::Admission`]), or until the drive aborts
/// every tracked task.
pub(crate) struct TrackedTask<T> {
    slot: Arc<Slot<T>>,
    abort: Option<AbortHandle>,
    /// Set for the root, whose handle aborts the task on drop.
    abort_on_drop: bool,
}

impl<T> std::fmt::Debug for TrackedTask<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedTask")
            .field("id", &self.abort.as_ref().map(AbortHandle::id))
            .finish_non_exhaustive()
    }
}

impl<T> TrackedTask<T> {
    /// A handle of a spawn that started nothing: it never resolves.
    fn refused() -> Self {
        Self {
            slot: Arc::new(Slot::new(None)),
            abort: None,
            abort_on_drop: false,
        }
    }

    /// Aborts the task. Aborting a finished task does nothing, so an outcome
    /// the task already stored stays in the slot.
    pub(crate) fn abort(&self) {
        if let Some(abort) = &self.abort {
            abort.abort();
        }
    }

    /// The task's abort handle. `None` for a spawn that started nothing.
    pub(crate) fn abort_handle(&self) -> Option<AbortHandle> {
        self.abort.clone()
    }
}

impl<T> Future for TrackedTask<T> {
    type Output = TaskOutcome<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.slot.poll_take(cx)
    }
}

impl<T> Drop for TrackedTask<T> {
    fn drop(&mut self) {
        if self.abort_on_drop {
            self.abort();
        }
        self.slot.close();
    }
}

/// Starts an operation task, counted on the coordinator's tracker and
/// aborted by [`TaskTracker::abort_all`]. Dropping the returned handle does
/// not abort the task.
///
/// Blesses the task in `ownership` before the first poll of `future`. Takes
/// an activity permit for the task on the calling task. When the phase is
/// not `Running`, starts nothing and returns a handle that never resolves.
pub(crate) fn spawn_tracked<F>(
    task_cx: TaskContext,
    coordinator: &Arc<InvocationCoordinator>,
    ownership: &Arc<TaskOwnership>,
    future: F,
) -> Result<TrackedTask<F::Output>, NoRuntime>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    ensure_runtime()?;
    let Some(schedule) = Schedule::queued(coordinator) else {
        return Ok(TrackedTask::refused());
    };
    let tasks = coordinator.tasks();
    let permit = tasks.acquire();
    let slot = Arc::new(Slot::new(Some(Arc::clone(coordinator))));
    let handle = raw::spawn(Wrapped::new(
        task_cx,
        future,
        Some(Arc::clone(ownership)),
        Arc::clone(coordinator),
        scheduled(schedule, false),
        Sink::Slot(Arc::clone(&slot)),
        Some(permit),
    ));
    tasks.register_abort(handle.abort_handle());
    Ok(TrackedTask {
        slot,
        abort: Some(handle.abort_handle()),
        abort_on_drop: false,
    })
}

/// Starts the handler of one invocation as an operation task. Blessed in
/// `ownership` before its first poll and polled by the same wrapper as
/// `spawn_tracked`, with `coordinator` as the current coordinator. Not
/// counted in the tracker and not reached by `abort_all`:
/// `drive_invocation` aborts it and awaits the returned `JoinHandle` before
/// it aborts the other tasks. When the handler resolves or panics, its
/// wrapper sets the phase to `Finished` before it releases its permit.
pub(crate) fn spawn_root<F>(
    task_cx: TaskContext,
    ownership: &Arc<TaskOwnership>,
    coordinator: &Arc<InvocationCoordinator>,
    future: F,
) -> Result<(TrackedTask<F::Output>, tokio::task::JoinHandle<()>), NoRuntime>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    ensure_runtime()?;
    // A coordinator that is no longer `Running` refuses the permit; the
    // root then starts idle and is never polled.
    let schedule = Schedule::queued(coordinator)
        .unwrap_or_else(|| Schedule::with_state(coordinator, ScheduleState::Idle));
    let slot = Arc::new(Slot::new(Some(Arc::clone(coordinator))));
    let handle = raw::spawn(Wrapped::new(
        task_cx,
        future,
        Some(Arc::clone(ownership)),
        Arc::clone(coordinator),
        scheduled(schedule, true),
        Sink::Slot(Arc::clone(&slot)),
        None,
    ));
    let task = TrackedTask {
        slot,
        abort: Some(handle.abort_handle()),
        abort_on_drop: true,
    };
    Ok((task, handle))
}

/// Not aborted on drop or by `abort_all`, still counted. For the batch
/// flusher and for detached checkpoint writes. Takes no ownership handle
/// and never blesses its task, because this work never creates a durable
/// operation. Takes no activity permit and is polled whatever the phase.
/// The task's outcome has no reader: a panic of its future, while it is
/// polled or dropped, is recorded on the coordinator.
/// While the phase is `Running`, a permit keeps the invocation open until
/// the write's response is merged: the checkpoint permit of a buffered or
/// direct write, or the refresh scheduler's permit for an unbatched
/// refresh, which the scheduler holds until it gets the result. The
/// scheduler is dropped when the drive ends, and its permit with it. From
/// then on only the task count keeps the write alive: the cleanup's
/// `wait_idle` waits for it.
pub(crate) fn spawn_detached<F>(
    coordinator: &Arc<InvocationCoordinator>,
    future: F,
) -> Result<(), NoRuntime>
where
    F: Future<Output = ()> + Send + 'static,
{
    ensure_runtime()?;
    let permit = coordinator.tasks().acquire();
    drop(raw::spawn(Wrapped::new(
        TaskContext::none(),
        future,
        None,
        Arc::clone(coordinator),
        Gate::Detached,
        Sink::Record,
        Some(permit),
    )));
    Ok(())
}

/// Runs plugin hook work on its own task: a raw `tokio::spawn`.
///
/// The task is not counted, takes no permit, is not polled by the
/// permit-aware wrapper, blesses nothing and enters no plugin frame. So
/// it holds neither the task cleanup nor the suspension decision. The
/// plugins' invocation-end drain joins it, and aborts it at the drain
/// deadline. `future` catches its own panics.
pub(crate) fn spawn_plugin_work(
    future: impl Future<Output = ()> + Send + 'static,
) -> Result<tokio::task::JoinHandle<()>, NoRuntime> {
    ensure_runtime()?;
    Ok(raw::spawn(future))
}

/// A set of operation tasks whose outcomes are joined in completion order.
///
/// The outcomes of finished tasks wait in a queue it shares with its tasks;
/// the `JoinSet<()>` is used only to abort. Dropping the set aborts every
/// task in it and closes the queue. An outcome still in the queue, or
/// delivered after the drop, is disposed of: a panic is recorded on the
/// coordinator, and a value is dropped inside a catch.
pub(crate) struct TrackedJoinSet<T> {
    outcomes: Arc<Mutex<JoinState<T>>>,
    tasks: JoinSet<()>,
    coordinator: Arc<InvocationCoordinator>,
    /// Tasks whose outcome has not been joined yet.
    pending: usize,
}

impl<T> std::fmt::Debug for TrackedJoinSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedJoinSet")
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> TrackedJoinSet<T> {
    /// Creates an empty set whose tasks are counted on the coordinator's
    /// tracker.
    pub(crate) fn new(coordinator: Arc<InvocationCoordinator>) -> Self {
        Self {
            outcomes: Arc::new(Mutex::new(JoinState {
                outcomes: VecDeque::new(),
                waker: None,
                closed: false,
            })),
            tasks: JoinSet::new(),
            coordinator,
            pending: 0,
        }
    }

    /// Starts an operation task in the set, and returns its ID.
    ///
    /// Blesses the task in `ownership` before the first poll of `future`,
    /// and takes an activity permit for it on the calling task. When the
    /// phase is not `Running`, starts nothing, returns `None`, and the set
    /// never yields another outcome.
    pub(crate) fn spawn<F>(
        &mut self,
        task_cx: TaskContext,
        ownership: &Arc<TaskOwnership>,
        future: F,
    ) -> Result<Option<Id>, NoRuntime>
    where
        F: Future<Output = T> + Send + 'static,
    {
        ensure_runtime()?;
        // A refused task still counts as pending, so the set never reports
        // that every task was joined.
        self.pending += 1;
        let Some(schedule) = Schedule::queued(&self.coordinator) else {
            return Ok(None);
        };
        let tracker = self.coordinator.tasks();
        let permit = tracker.acquire();
        let abort = raw::join_set_spawn(
            &mut self.tasks,
            Wrapped::new(
                task_cx,
                future,
                Some(Arc::clone(ownership)),
                Arc::clone(&self.coordinator),
                scheduled(schedule, false),
                Sink::Queue(Arc::clone(&self.outcomes)),
                Some(permit),
            ),
        );
        let id = abort.id();
        tracker.register_abort(abort);
        Ok(Some(id))
    }

    /// Returns the outcome of the next task to finish, or `None` when every
    /// task's outcome was joined.
    pub(crate) async fn join_next_with_id(&mut self) -> Option<(Id, TaskOutcome<T>)> {
        std::future::poll_fn(|cx| {
            if self.pending == 0 {
                return Poll::Ready(None);
            }
            let mut state = self.outcomes.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(joined) = state.outcomes.pop_front() {
                self.pending -= 1;
                return Poll::Ready(Some(joined));
            }
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        })
        .await
    }

    /// Aborts every task in the set.
    pub(crate) fn abort_all(&mut self) {
        self.tasks.abort_all();
    }
}

impl<T> Drop for TrackedJoinSet<T> {
    fn drop(&mut self) {
        let unread = {
            let mut state = self.outcomes.lock().unwrap_or_else(PoisonError::into_inner);
            state.closed = true;
            state.waker = None;
            std::mem::take(&mut state.outcomes)
        };
        for (_, outcome) in unread {
            dispose(&self.coordinator, outcome);
        }
    }
}

/// Runs `f` on the blocking pool.
///
/// With a current coordinator (the `CURRENT_COORDINATOR` task-local), the
/// work holds an activity permit and a count on the coordinator's tracker
/// from before the spawn until after it handed its outcome over. So the
/// invocation does not suspend while the work runs, and the consumer's
/// wake takes the consumer's permit before the work's permit is released.
/// When the coordinator refuses the permit, nothing runs and the future
/// never resolves. Without a current coordinator the work is not counted.
///
/// Dropping the returned future before it resolves closes the work's
/// slot. Inside an invocation, an outcome the work stored and nobody read,
/// or stores later, is then disposed of: a panic is recorded on the
/// coordinator, and a value is dropped inside a catch.
///
/// Takes no ownership handle and never blesses, because blocking serdes
/// work never creates a durable operation.
pub(crate) async fn run_blocking<F, R>(f: F) -> Result<TaskOutcome<R>, NoRuntime>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    ensure_runtime()?;
    let coordinator = CURRENT_COORDINATOR.try_with(Arc::clone).ok();
    let permits = match &coordinator {
        Some(coordinator) => {
            let Some(activity) = coordinator.acquire(PermitKind::Activity) else {
                return std::future::pending().await;
            };
            Some((activity, coordinator.tasks().acquire()))
        }
        None => None,
    };
    let slot = Arc::new(Slot::new(coordinator));
    let task_slot = Arc::clone(&slot);
    raw::spawn_blocking(move || {
        // `f` is consumed by the call, so a capture whose destructor panics
        // panics inside `catch_unwind`, and that panic is the outcome.
        let outcome = match catch_unwind(AssertUnwindSafe(f)) {
            Ok(value) => TaskOutcome::Value(value),
            Err(payload) => TaskOutcome::Panicked(payload),
        };
        task_slot.deliver(outcome);
        drop(permits);
    });
    Ok(SlotWait(slot).await)
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Helpers for tests in other modules.

    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    use super::{CURRENT_COORDINATOR, ScheduleState as Inner, Slot, TaskOutcome};
    use crate::coordinator::{InvocationCoordinator, Permit};

    /// Calls `f` with `coordinator` as the current coordinator.
    pub(crate) fn with_current_coordinator<R>(
        coordinator: &Arc<InvocationCoordinator>,
        f: impl FnOnce() -> R,
    ) -> R {
        CURRENT_COORDINATOR.sync_scope(Arc::clone(coordinator), f)
    }

    /// The observable state of a [`Schedule`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum ScheduleState {
        Idle,
        Queued,
        Polling,
        Done,
    }

    /// A permit-aware schedule driven by hand, as the gateway wrapper
    /// drives it. Dropping it closes the schedule.
    pub(crate) struct Schedule(Arc<super::Schedule>);

    impl Schedule {
        /// A schedule in `Idle`.
        pub(crate) fn idle(coordinator: &Arc<InvocationCoordinator>) -> Self {
            Self(super::Schedule::with_state(coordinator, Inner::Idle))
        }

        pub(crate) fn waker(&self) -> Waker {
            self.0.waker()
        }

        pub(crate) fn state(&self) -> ScheduleState {
            match &*self.0.lock() {
                Inner::Idle => ScheduleState::Idle,
                Inner::Queued(_) => ScheduleState::Queued,
                Inner::Polling { .. } => ScheduleState::Polling,
                Inner::Done => ScheduleState::Done,
            }
        }

        pub(crate) fn begin_poll(&self) -> bool {
            self.0.begin_poll()
        }

        pub(crate) fn end_pending(&self, requeue: bool) {
            self.0.end_pending(requeue);
        }

        pub(crate) fn end_ready(&self) -> Option<Permit> {
            self.0.end_ready()
        }

        /// Polls `future` once as the gateway wrapper does: under the
        /// schedule's permit and waker, with the schedule's coordinator as
        /// the current coordinator. A `Ready` poll releases the permit.
        pub(crate) fn poll<F: Future>(&self, future: Pin<&mut F>) -> Poll<F::Output> {
            if !self.0.begin_poll() {
                return Poll::Pending;
            }
            let waker = self.0.waker();
            let mut cx = Context::from_waker(&waker);
            let coordinator = Arc::clone(&self.0.coordinator);
            let polled = CURRENT_COORDINATOR.sync_scope(coordinator, || future.poll(&mut cx));
            match polled {
                Poll::Pending => {
                    self.0.end_pending(false);
                    Poll::Pending
                }
                Poll::Ready(value) => {
                    drop(self.0.end_ready());
                    Poll::Ready(value)
                }
            }
        }
    }

    impl Drop for Schedule {
        fn drop(&mut self) {
            self.0.close();
        }
    }

    /// An outcome slot whose consumer registered `waker`.
    pub(crate) struct TestSlot<T>(Arc<Slot<T>>);

    impl<T: Send + 'static> TestSlot<T> {
        /// Stores `value` as the outcome and wakes the consumer.
        pub(crate) fn deliver_value(&self, value: T) {
            self.0.deliver(TaskOutcome::Value(value));
        }
    }

    /// A slot whose consumer is `waker`.
    pub(crate) fn slot_with_waker<T>(waker: &Waker) -> TestSlot<T> {
        let slot = Arc::new(Slot::new(None));
        let cx = Context::from_waker(waker);
        assert!(slot.poll_take(&cx).is_pending());
        TestSlot(slot)
    }

    /// Builds a future with `make` on a new thread with no Tokio runtime,
    /// polls it once there with a no-op waker, and returns the result of
    /// that poll.
    pub(crate) fn poll_once_without_runtime<F, Fut>(make: F) -> Poll<Fut::Output>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future,
        Fut::Output: Send + 'static,
    {
        std::thread::spawn(move || {
            assert!(
                tokio::runtime::Handle::try_current().is_err(),
                "the thread must have no Tokio runtime"
            );
            let fut = std::pin::pin!(make());
            fut.poll(&mut Context::from_waker(Waker::noop()))
        })
        .join()
        .expect("polling on the plain thread must not panic")
    }

    /// The payload of `panic!("literal")`: a `&'static str`. Returns
    /// `None` for a payload of any other type, so a test that compares the
    /// result also checks that the payload kept its original type.
    pub(crate) fn str_payload(payload: &(dyn std::any::Any + Send)) -> Option<&'static str> {
        payload.downcast_ref::<&'static str>().copied()
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: futures that panic on purpose, and test assertions
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    use super::test_support::{poll_once_without_runtime, str_payload, with_current_coordinator};
    use super::{
        NoRuntime, TaskOutcome, TrackedJoinSet, run_blocking, spawn_detached, spawn_root,
        spawn_tracked,
    };
    use crate::coordinator::{InvocationCoordinator, PermitKind, Phase};
    use crate::driver::TaskOwnership;

    fn coordinator() -> Arc<InvocationCoordinator> {
        InvocationCoordinator::new()
    }

    /// Ownership tracking anchored to the test's root future, which has no
    /// task ID, as in production.
    fn ownership() -> Arc<TaskOwnership> {
        Arc::new(TaskOwnership::new_current())
    }

    /// Sets its flag when dropped.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Panics with "drop boom" when dropped.
    struct DropPanics;

    impl Drop for DropPanics {
        fn drop(&mut self) {
            panic!("drop boom");
        }
    }

    /// A panic payload whose destructor panics with a `DropPanics`
    /// payload, whose destructor panics again.
    struct PayloadDropPanics;

    impl Drop for PayloadDropPanics {
        fn drop(&mut self) {
            std::panic::panic_any(DropPanics);
        }
    }

    /// Counts its drops.
    struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// `discard_panic` drops the payload it is given.
    #[test]
    fn discard_panic_drops_the_payload() {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        super::discard_panic(Box::new(DropCounter(Arc::clone(&drops))));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    /// The payload's destructor panics with a payload whose destructor
    /// panics too. `discard_panic` catches the first destructor's panic and
    /// leaks its payload, so it returns normally.
    #[test]
    fn discard_panic_forgets_a_nested_panicking_payload() {
        let returned = std::panic::catch_unwind(|| {
            super::discard_panic(Box::new(PayloadDropPanics));
        });
        assert!(
            returned.is_ok(),
            "a nested payload's destructor panic escaped"
        );
    }

    /// Returns `Ready(7)` on its first poll and panics when dropped.
    struct ReadyThenDropPanics;

    impl Future for ReadyThenDropPanics {
        type Output = i32;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<i32> {
            Poll::Ready(7)
        }
    }

    impl Drop for ReadyThenDropPanics {
        fn drop(&mut self) {
            panic!("drop boom");
        }
    }

    /// Panics with "poll boom" when polled and with "drop boom" when dropped.
    struct PollPanicsThenDropPanics;

    impl Future for PollPanicsThenDropPanics {
        type Output = i32;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<i32> {
            panic!("poll boom");
        }
    }

    impl Drop for PollPanicsThenDropPanics {
        fn drop(&mut self) {
            panic!("drop boom");
        }
    }

    /// Stores its waker in a shared slot, stays pending, and panics when
    /// dropped.
    struct StoresWakerThenDropPanics(Arc<Mutex<Option<Waker>>>);

    impl Future for StoresWakerThenDropPanics {
        type Output = ();

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            *self.0.lock().expect("lock") = Some(cx.waker().clone());
            Poll::Pending
        }
    }

    impl Drop for StoresWakerThenDropPanics {
        fn drop(&mut self) {
            panic!("drop boom");
        }
    }

    /// The `&'static str` payload of a panic outcome, or `None` when the
    /// payload has another type.
    fn panicked_with<T>(outcome: TaskOutcome<T>) -> Option<&'static str> {
        match outcome {
            TaskOutcome::Panicked(payload) => str_payload(payload.as_ref()),
            TaskOutcome::Value(_) => panic!("expected a panic outcome"),
        }
    }

    #[tokio::test]
    async fn tracked_task_returns_value_with_permit() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let task = spawn_tracked(super::TaskContext::none(), &coord, &ownership(), async {
            7
        })
        .expect("runtime");
        let outcome = task.await;
        assert_eq!(tasks.live(), 0, "the task released its permit on exit");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
    }

    #[tokio::test]
    async fn tracked_task_catches_panic() {
        let coord = coordinator();
        let task = spawn_tracked(super::TaskContext::none(), &coord, &ownership(), async {
            panic!("boom");
        })
        .expect("runtime");
        let outcome: TaskOutcome<()> = task.await;
        assert_eq!(panicked_with(outcome), Some("boom"));
    }

    #[tokio::test]
    async fn dropped_tracked_task_counts_until_unwound() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&dropped));
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &ownership(),
            async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            },
        )
        .expect("runtime");
        tokio::task::yield_now().await;
        task.abort();
        drop(task);
        assert_eq!(tasks.live(), 1, "the aborted task still counts");
        assert!(
            !dropped.load(Ordering::SeqCst),
            "tokio has not unwound it yet"
        );
        tasks.wait_idle().await;
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(tasks.live(), 0);
    }

    /// Dropping the handle of an operation task loses only its result: the
    /// task runs to its end.
    #[tokio::test]
    async fn dropped_tracked_task_keeps_running() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &ownership(),
            async move {
                assert!(tx.send(()).is_ok(), "the test waits for the signal");
            },
        )
        .expect("runtime");
        drop(task);
        assert!(rx.await.is_ok(), "the task ran after its handle dropped");
        tasks.wait_idle().await;
        assert_eq!(tasks.live(), 0);
    }

    /// Dropping the root's handle still aborts the root task.
    #[tokio::test]
    async fn dropped_root_handle_aborts_the_root_task() {
        let coord = coordinator();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&dropped));
        let (root, join) = spawn_root(
            super::TaskContext::none(),
            &ownership(),
            &coord,
            async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            },
        )
        .expect("runtime");
        tokio::task::yield_now().await;
        drop(root);
        let joined = tokio::time::timeout(std::time::Duration::from_secs(5), join)
            .await
            .expect("the root task ends within 5 s");
        assert!(
            joined.is_err_and(|err| err.is_cancelled()),
            "the root task was aborted"
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "the root future was dropped"
        );
    }

    #[tokio::test]
    async fn blocking_work_counts_until_closure_returns() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let work_barrier = Arc::clone(&barrier);
        let mut work = Box::pin(run_blocking(move || {
            work_barrier.wait();
        }));
        let polled = with_current_coordinator(&coord, || {
            work.as_mut().poll(&mut Context::from_waker(Waker::noop()))
        });
        assert!(polled.is_pending());
        drop(work);
        assert_eq!(tasks.live(), 1, "the blocking work counts after the drop");
        barrier.wait();
        tasks.wait_idle().await;
        assert_eq!(tasks.live(), 0);
    }

    #[tokio::test]
    async fn blocking_work_without_tracker_is_not_counted() {
        let outcome = run_blocking(|| 5).await.expect("runtime");
        assert!(matches!(outcome, TaskOutcome::Value(5)));
    }

    /// Outside an invocation no coordinator owns the outcome. A value the
    /// work returns after its waiter dropped is still dropped.
    #[tokio::test]
    async fn blocking_value_after_waiter_drop_outside_invocation_is_dropped() {
        let dropped = Arc::new(AtomicBool::new(false));
        let value = DropFlag(Arc::clone(&dropped));
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let mut work = Box::pin(run_blocking(move || {
            release_rx.recv().expect("the test releases the closure");
            let _done = done_tx;
            value
        }));
        let polled = work.as_mut().poll(&mut Context::from_waker(Waker::noop()));
        assert!(polled.is_pending());
        drop(work);
        release_tx.send(()).expect("the closure waits");
        drop(done_rx.await);
        spin_until("the value is dropped", || dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn detached_task_counts_until_done() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        spawn_detached(&coord, async move {
            let _ = rx.await;
        })
        .expect("runtime");
        assert_eq!(tasks.live(), 1);
        tx.send(()).expect("the detached task waits");
        tasks.wait_idle().await;
        assert_eq!(tasks.live(), 0);
    }

    #[tokio::test]
    async fn abort_all_aborts_tracked_tasks_only() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let tracked_dropped = Arc::new(AtomicBool::new(false));
        let joined_dropped = Arc::new(AtomicBool::new(false));
        let detached_done = Arc::new(AtomicBool::new(false));

        let guard = DropFlag(Arc::clone(&tracked_dropped));
        let _task = spawn_tracked(super::TaskContext::none(), &coord, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        let guard = DropFlag(Arc::clone(&joined_dropped));
        set.spawn(super::TaskContext::none(), &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let done = Arc::clone(&detached_done);
        spawn_detached(&coord, async move {
            let _ = rx.await;
            done.store(true, Ordering::SeqCst);
        })
        .expect("runtime");
        tokio::task::yield_now().await;

        tasks.abort_all();
        tx.send(()).expect("the detached task was not aborted");
        tasks.wait_idle().await;
        assert!(tracked_dropped.load(Ordering::SeqCst));
        assert!(joined_dropped.load(Ordering::SeqCst));
        assert!(
            detached_done.load(Ordering::SeqCst),
            "the detached task ran to its end"
        );
        assert_eq!(tasks.live(), 0);
    }

    /// A task registered after `abort_all` is aborted at once, through
    /// both gateways that store an abort handle. Once no task is live, the
    /// tracker accepts tasks again.
    #[tokio::test]
    async fn abort_all_aborts_late_registrations() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        // A live task, as during teardown: the parent of the late tasks.
        let parent = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            std::future::pending::<()>(),
        )
        .expect("runtime");
        tokio::task::yield_now().await;
        tasks.abort_all();

        let tracked_dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&tracked_dropped));
        let late_task = spawn_tracked(super::TaskContext::none(), &coord, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let joined_dropped = Arc::new(AtomicBool::new(false));
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        let guard = DropFlag(Arc::clone(&joined_dropped));
        set.spawn(super::TaskContext::none(), &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");

        tokio::time::timeout(std::time::Duration::from_secs(5), tasks.wait_idle())
            .await
            .expect("the late tasks were aborted, so the tracker goes idle");
        assert!(tracked_dropped.load(Ordering::SeqCst));
        assert!(joined_dropped.load(Ordering::SeqCst));
        assert_eq!(tasks.live(), 0);
        assert_eq!(tasks.abortable_count(), 0);
        // The handles are still held, so only `abort_all` aborted the tasks.
        drop((parent, late_task, set));

        let task =
            spawn_tracked(super::TaskContext::none(), &coord, &own, async { 7 }).expect("runtime");
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("an idle tracker no longer aborts new tasks");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
    }

    /// `abort_all` on a tracker with no live task leaves it usable: no task
    /// of a teardown can register afterwards, so new tasks are not aborted.
    #[tokio::test]
    async fn abort_all_on_idle_tracker_keeps_it_usable() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        tasks.abort_all();
        let task = spawn_tracked(super::TaskContext::none(), &coord, &ownership(), async {
            7
        })
        .expect("runtime");
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("the task was not aborted");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
    }

    /// A closed tracker aborts every task registered later, through both
    /// gateways, even across an interval with no live task: a release does
    /// not reopen it. It also aborts the tasks that were live when it
    /// closed.
    #[tokio::test]
    async fn closed_tracker_aborts_every_later_registration() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let live_dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&live_dropped));
        let live = spawn_tracked(super::TaskContext::none(), &coord, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        tokio::task::yield_now().await;
        assert!(!tasks.is_closed());
        tasks.close();
        assert!(tasks.is_closed());
        tokio::time::timeout(std::time::Duration::from_secs(5), tasks.wait_idle())
            .await
            .expect("close aborts the live task");
        assert!(live_dropped.load(Ordering::SeqCst));
        assert_eq!(tasks.live(), 0);
        drop(live);

        for _ in 0..2 {
            let tracked_dropped = Arc::new(AtomicBool::new(false));
            let guard = DropFlag(Arc::clone(&tracked_dropped));
            let late_task = spawn_tracked(super::TaskContext::none(), &coord, &own, async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            })
            .expect("runtime");
            let joined_dropped = Arc::new(AtomicBool::new(false));
            let mut set = TrackedJoinSet::new(Arc::clone(&coord));
            let guard = DropFlag(Arc::clone(&joined_dropped));
            set.spawn(super::TaskContext::none(), &own, async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            })
            .expect("runtime");
            tokio::time::timeout(std::time::Duration::from_secs(5), tasks.wait_idle())
                .await
                .expect("a closed tracker aborts a late task, so it goes idle");
            assert!(tracked_dropped.load(Ordering::SeqCst));
            assert!(joined_dropped.load(Ordering::SeqCst));
            assert_eq!(tasks.live(), 0);
            drop((late_task, set));
        }
    }

    /// A tracked task keeps starting nested tracked tasks on a worker
    /// thread while the test thread tears the tracker down. Each nested
    /// handle escapes into a shared list, so only `abort_all` can stop the
    /// nested task. A nested task registered after `abort_all` took the
    /// handles must still be aborted, or `wait_idle` never returns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn abort_all_races_nested_spawns() {
        for _ in 0..50 {
            let coord = coordinator();
            let tasks = Arc::clone(coord.tasks());
            let own = ownership();
            let escaped = Arc::new(Mutex::new(Vec::new()));
            let started = Arc::new(tokio::sync::Notify::new());
            let spawner = {
                let coord = Arc::clone(&coord);
                let own = Arc::clone(&own);
                let escaped = Arc::clone(&escaped);
                let started = Arc::clone(&started);
                spawn_tracked(
                    super::TaskContext::none(),
                    &coord.clone(),
                    &own.clone(),
                    async move {
                        loop {
                            let nested = spawn_tracked(
                                super::TaskContext::none(),
                                &coord,
                                &own,
                                std::future::pending::<()>(),
                            )
                            .expect("runtime");
                            escaped.lock().expect("lock").push(nested);
                            started.notify_one();
                            tokio::task::yield_now().await;
                        }
                    },
                )
                .expect("runtime")
            };
            started.notified().await;
            tasks.abort_all();
            tokio::time::timeout(std::time::Duration::from_secs(10), tasks.wait_idle())
                .await
                .expect("every nested task was aborted");
            assert_eq!(tasks.live(), 0);
            drop(spawner);
            drop(escaped);
        }
    }

    #[test]
    fn spawn_without_runtime_returns_no_runtime() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let ran = Arc::new(AtomicBool::new(false));
        let thread_coord = Arc::clone(&coord);
        let thread_ran = Arc::clone(&ran);
        let polled = poll_once_without_runtime(move || {
            let own = ownership();
            let tracked =
                spawn_tracked(super::TaskContext::none(), &thread_coord, &own, async {}).map(drop);
            let detached = spawn_detached(&thread_coord, async {});
            let mut set = TrackedJoinSet::<()>::new(Arc::clone(&thread_coord));
            let joined = set
                .spawn(super::TaskContext::none(), &own, async {})
                .map(drop);
            let blocking = run_blocking(move || thread_ran.store(true, Ordering::SeqCst));
            async move {
                let blocking = blocking.await.map(drop);
                (tracked, detached, joined, blocking)
            }
        });
        let Poll::Ready(results) = polled else {
            panic!("run_blocking must resolve on its first poll");
        };
        assert_eq!(
            results,
            (
                Err(NoRuntime),
                Err(NoRuntime),
                Err(NoRuntime),
                Err(NoRuntime)
            )
        );
        assert!(!ran.load(Ordering::SeqCst), "the closure did not run");
        assert_eq!(tasks.live(), 0);
    }

    #[tokio::test]
    async fn operation_tasks_are_blessed_before_first_poll() {
        let coord = coordinator();
        let own = ownership();
        let task_own = Arc::clone(&own);
        let tracked = spawn_tracked(super::TaskContext::none(), &coord, &own, async move {
            task_own.check_current_task()
        })
        .expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        let set_own = Arc::clone(&own);
        set.spawn(super::TaskContext::none(), &own, async move {
            set_own.check_current_task()
        })
        .expect("runtime");

        assert!(matches!(tracked.await, TaskOutcome::Value(Ok(()))));
        let (_, joined) = set.join_next_with_id().await.expect("one task");
        assert!(matches!(joined, TaskOutcome::Value(Ok(()))));
    }

    #[tokio::test]
    async fn detached_task_is_not_blessed() {
        let coord = coordinator();
        let own = ownership();
        let task_own = Arc::clone(&own);
        let (tx, rx) = tokio::sync::oneshot::channel();
        spawn_detached(&coord, async move {
            let _ = tx.send(task_own.check_current_task());
        })
        .expect("runtime");
        let checked = rx.await.expect("the detached task sends");
        let err = checked.expect_err("a detached task is not blessed");
        assert!(err.contains("Use .spawn()"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn finished_operation_task_is_unblessed_before_it_exits() {
        let coord = coordinator();
        let own = ownership();
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        for _ in 0..100 {
            let task_own = Arc::clone(&own);
            set.spawn(super::TaskContext::none(), &own, async move {
                task_own.check_current_task()
            })
            .expect("runtime");
        }
        let mut joined = 0;
        while let Some((_, outcome)) = set.join_next_with_id().await {
            assert!(matches!(outcome, TaskOutcome::Value(Ok(()))));
            joined += 1;
        }
        assert_eq!(joined, 100);
        assert_eq!(own.blessed_count(), 0);
    }

    #[tokio::test]
    async fn aborted_operation_task_is_unblessed() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let task = spawn_tracked(super::TaskContext::none(), &coord, &own, async move {
            let _ = tx.send(());
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        rx.await.expect("the task signals");
        assert_eq!(own.blessed_count(), 1);
        task.abort();
        drop(task);
        tasks.wait_idle().await;
        assert_eq!(own.blessed_count(), 0);
    }

    #[tokio::test]
    async fn finished_task_with_escaped_handle_does_not_block_idle() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let task = spawn_tracked(super::TaskContext::none(), &coord, &ownership(), async {
            1
        })
        .expect("runtime");
        let held = Arc::new(Mutex::new(Some(task)));
        for _ in 0..100 {
            if tasks.live() == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), tasks.wait_idle())
            .await
            .expect("wait_idle returns while the handle is still held");
        assert!(held.lock().expect("lock").is_some());
        assert_eq!(tasks.live(), 0);
    }

    #[tokio::test]
    async fn ready_then_drop_panic_is_the_outcome() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            ReadyThenDropPanics,
        )
        .expect("runtime");
        assert_eq!(panicked_with(task.await), Some("drop boom"));
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(super::TaskContext::none(), &own, ReadyThenDropPanics)
            .expect("runtime");
        let (_, joined) = set.join_next_with_id().await.expect("one task");
        assert_eq!(panicked_with(joined), Some("drop boom"));
        tasks.wait_idle().await;
        assert_eq!(tasks.live(), 0);
    }

    /// Panics with "value boom" when dropped.
    struct ValueDropPanics;

    impl Drop for ValueDropPanics {
        fn drop(&mut self) {
            panic!("value boom");
        }
    }

    /// Returns a value that panics when dropped, and panics itself when
    /// dropped.
    struct ReadyValueThenDropPanics;

    impl Future for ReadyValueThenDropPanics {
        type Output = ValueDropPanics;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<ValueDropPanics> {
            Poll::Ready(ValueDropPanics)
        }
    }

    impl Drop for ReadyValueThenDropPanics {
        fn drop(&mut self) {
            panic!("drop boom");
        }
    }

    /// When the destructor of a finished future panics, the value is
    /// dropped, and a panic while dropping it is discarded.
    #[tokio::test]
    async fn value_drop_panic_after_drop_panic_is_discarded() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &ownership(),
            ReadyValueThenDropPanics,
        )
        .expect("runtime");
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("the outcome is delivered");
        assert_eq!(panicked_with(outcome), Some("drop boom"));
        tasks.wait_idle().await;
        assert_eq!(tasks.live(), 0);
    }

    /// The tracker clears its abort handles once no task is live.
    #[tokio::test]
    async fn idle_tracker_clears_abort_handles() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let task =
            spawn_tracked(super::TaskContext::none(), &coord, &own, async {}).expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(super::TaskContext::none(), &own, async {})
            .expect("runtime");
        assert_eq!(tasks.abortable_count(), 2);
        let _ = task.await;
        let _ = set.join_next_with_id().await;
        tasks.wait_idle().await;
        assert_eq!(tasks.abortable_count(), 0);
    }

    #[tokio::test]
    async fn poll_panic_wins_over_drop_panic() {
        let coord = coordinator();
        let own = ownership();
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            PollPanicsThenDropPanics,
        )
        .expect("runtime");
        assert_eq!(panicked_with(task.await), Some("poll boom"));
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(super::TaskContext::none(), &own, PollPanicsThenDropPanics)
            .expect("runtime");
        let (_, joined) = set.join_next_with_id().await.expect("one task");
        assert_eq!(panicked_with(joined), Some("poll boom"));
    }

    #[tokio::test]
    async fn blocking_capture_drop_panic_is_the_outcome() {
        let captured = DropPanics;
        let outcome = run_blocking(move || {
            let _keep = &captured;
            1
        })
        .await
        .expect("runtime");
        assert_eq!(panicked_with(outcome), Some("drop boom"));
    }

    #[tokio::test]
    async fn aborted_task_drop_panic_releases_permit() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let waker_slot = Arc::new(Mutex::new(None));
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            StoresWakerThenDropPanics(Arc::clone(&waker_slot)),
        )
        .expect("runtime");
        for _ in 0..100 {
            if waker_slot.lock().expect("lock").is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
        let waker = waker_slot
            .lock()
            .expect("lock")
            .clone()
            .expect("the task was polled");
        waker.wake_by_ref();
        task.abort();
        drop(task);
        tasks.wait_idle().await;
        assert_eq!(tasks.live(), 0);
        assert_eq!(own.blessed_count(), 0);
        assert_eq!(
            coord.permits(PermitKind::Activity),
            0,
            "the aborted task released its activity permit while its waker is held"
        );
        waker.wake_by_ref();
        assert_eq!(
            coord.permits(PermitKind::Activity),
            0,
            "a wake after the task ended takes no permit"
        );
        drop(waker);
    }

    /// A panic payload, identified by its tag, that counts its drops.
    struct TaggedPayload(&'static str, Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for TaggedPayload {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A value whose destructor panics with the `&'static str` it holds.
    struct PanicsWith(&'static str);

    impl Drop for PanicsWith {
        fn drop(&mut self) {
            std::panic::panic_any(self.0);
        }
    }

    /// The tag of a [`TaggedPayload`], or `None` for another payload type.
    fn tag(payload: &(dyn std::any::Any + Send)) -> Option<&'static str> {
        payload.downcast_ref::<TaggedPayload>().map(|p| p.0)
    }

    /// Blocking work whose closure waits for `release` and then panics
    /// with "blocking boom" or, with `value`, returns a value whose
    /// destructor panics with "value boom". The work is started inside an
    /// invocation by one poll with `coord` as the current coordinator.
    fn started_blocking(
        coord: &Arc<InvocationCoordinator>,
        release: std::sync::mpsc::Receiver<()>,
        value: bool,
    ) -> Pin<Box<impl Future<Output = Result<TaskOutcome<PanicsWith>, NoRuntime>>>> {
        let mut work = Box::pin(run_blocking(move || {
            release.recv().expect("the test releases the closure");
            if !value {
                std::panic::panic_any("blocking boom");
            }
            PanicsWith("value boom")
        }));
        let polled = with_current_coordinator(coord, || {
            work.as_mut().poll(&mut Context::from_waker(Waker::noop()))
        });
        assert!(polled.is_pending());
        work
    }

    /// The future of `run_blocking` is dropped while the closure runs, and
    /// the closure then panics. Nobody reads the outcome, so the panic is
    /// recorded on the coordinator.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_panic_after_waiter_drop_is_recorded() {
        let coord = coordinator();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work = started_blocking(&coord, release_rx, false);
        drop(work);
        release_tx.send(()).expect("the closure waits");
        coord.tasks().wait_idle().await;
        let payload = coord.take_panic().expect("the closure's panic is recorded");
        assert_eq!(str_payload(payload.as_ref()), Some("blocking boom"));
    }

    /// The closure panics, and the future of `run_blocking` is dropped
    /// before it is polled again. The panic sits unread in the slot, so the
    /// drop records it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_panic_before_waiter_drop_is_recorded() {
        let coord = coordinator();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work = started_blocking(&coord, release_rx, false);
        release_tx.send(()).expect("the closure waits");
        coord.tasks().wait_idle().await;
        assert!(!coord.has_panic(), "the slot holds the panic");
        drop(work);
        let payload = coord.take_panic().expect("the closure's panic is recorded");
        assert_eq!(str_payload(payload.as_ref()), Some("blocking boom"));
    }

    /// Returns `Ready` with a [`PanicsWith`] value.
    struct ReadyPanicsWith(&'static str);

    impl Future for ReadyPanicsWith {
        type Output = PanicsWith;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<PanicsWith> {
            Poll::Ready(PanicsWith(self.0))
        }
    }

    /// Takes the coordinator's kept panic and checks that it is the
    /// destructor's payload `expected`.
    fn assert_recorded(coord: &InvocationCoordinator, expected: &str) {
        let payload = coord
            .take_panic()
            .expect("the destructor's panic is recorded");
        assert_eq!(str_payload(payload.as_ref()), Some(expected));
    }

    /// A value that a dropped reader would have received is dropped inside
    /// a catch, and its destructor's panic is recorded. Six cases: a
    /// tracked slot, a join-set queue and a blocking slot, each (a) with
    /// the value delivered before the reader drops, and (b) after.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn undelivered_value_drop_panic_is_recorded() {
        let own = ownership();

        // Tracked slot, (a): the task stores the value, then the handle drops.
        let coord = coordinator();
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            ReadyPanicsWith("tracked a"),
        )
        .expect("runtime");
        coord.tasks().wait_idle().await;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(task))).is_ok());
        assert_recorded(&coord, "tracked a");

        // Tracked slot, (b): the handle drops, then the value arrives.
        let coord = coordinator();
        let slot = Arc::new(super::Slot::new(Some(Arc::clone(&coord))));
        drop(super::TrackedTask {
            slot: Arc::clone(&slot),
            abort: None,
            abort_on_drop: false,
        });
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                slot.deliver(TaskOutcome::Value(PanicsWith("tracked b")));
            }))
            .is_ok()
        );
        assert_recorded(&coord, "tracked b");

        // Join-set queue, (a): the item stores the value, then the set drops.
        let coord = coordinator();
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(super::TaskContext::none(), &own, ReadyPanicsWith("queue a"))
            .expect("runtime");
        coord.tasks().wait_idle().await;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(set))).is_ok());
        assert_recorded(&coord, "queue a");

        // Join-set queue, (b): the set drops, then the value arrives.
        let coord = coordinator();
        let set = TrackedJoinSet::<PanicsWith>::new(Arc::clone(&coord));
        let queue = Arc::clone(&set.outcomes);
        drop(set);
        let deliver_coord = Arc::clone(&coord);
        super::raw::spawn(async move {
            super::Sink::Queue(queue)
                .deliver(TaskOutcome::Value(PanicsWith("queue b")), &deliver_coord);
        })
        .await
        .expect("the delivery does not panic");
        assert_recorded(&coord, "queue b");

        // Blocking slot, (a): the closure returns the value, then the
        // waiter drops.
        let coord = coordinator();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work = started_blocking(&coord, release_rx, true);
        release_tx.send(()).expect("the closure waits");
        coord.tasks().wait_idle().await;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(work))).is_ok());
        assert_recorded(&coord, "value boom");

        // Blocking slot, (b): the waiter drops, then the closure returns
        // the value.
        let coord = coordinator();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work = started_blocking(&coord, release_rx, true);
        drop(work);
        release_tx.send(()).expect("the closure waits");
        coord.tasks().wait_idle().await;
        assert_recorded(&coord, "value boom");
    }

    /// Panics with `TaggedPayload("A")` when polled and with
    /// `TaggedPayload("B")` when dropped.
    struct PollPanicsADropPanicsB(Arc<std::sync::atomic::AtomicUsize>);

    impl Future for PollPanicsADropPanicsB {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            std::panic::panic_any(TaggedPayload(
                "A",
                Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            ));
        }
    }

    impl Drop for PollPanicsADropPanicsB {
        fn drop(&mut self) {
            std::panic::panic_any(TaggedPayload("B", Arc::clone(&self.0)));
        }
    }

    /// A value whose destructor panics with `TaggedPayload("B")`.
    struct ValuePanicsB(Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for ValuePanicsB {
        fn drop(&mut self) {
            std::panic::panic_any(TaggedPayload("B", Arc::clone(&self.0)));
        }
    }

    /// Returns a [`ValuePanicsB`] and panics with `TaggedPayload("A")` when
    /// dropped.
    struct ReadyDropPanicsA(Arc<std::sync::atomic::AtomicUsize>);

    impl Future for ReadyDropPanicsA {
        type Output = ValuePanicsB;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<ValuePanicsB> {
            Poll::Ready(ValuePanicsB(Arc::clone(&self.0)))
        }
    }

    impl Drop for ReadyDropPanicsA {
        fn drop(&mut self) {
            std::panic::panic_any(TaggedPayload(
                "A",
                Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            ));
        }
    }

    /// The gateway keeps a task's first panic when disposing of the task
    /// panics again, and drops the second panic's payload through
    /// `discard_panic`. Two cases: (a) the poll panics with A and dropping
    /// the future panics with B; (b) the poll returns, dropping the future
    /// panics with A, and dropping the returned value panics with B.
    #[tokio::test]
    async fn task_keeps_its_first_panic_when_disposal_panics() {
        let coord = coordinator();
        let own = ownership();

        let b_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            PollPanicsADropPanicsB(Arc::clone(&b_drops)),
        )
        .expect("runtime");
        let TaskOutcome::Panicked(payload) = task.await else {
            panic!("expected the poll's panic");
        };
        assert_eq!(tag(payload.as_ref()), Some("A"));
        assert_eq!(b_drops.load(Ordering::SeqCst), 1, "B was dropped");

        let b_drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &own,
            ReadyDropPanicsA(Arc::clone(&b_drops)),
        )
        .expect("runtime");
        let TaskOutcome::Panicked(payload) = task.await else {
            panic!("expected the destructor's panic");
        };
        assert_eq!(tag(payload.as_ref()), Some("A"));
        assert_eq!(b_drops.load(Ordering::SeqCst), 1, "B was dropped");
        assert!(!coord.has_panic(), "a second panic is not recorded");
    }

    /// The cleanup claim records a panic that is still unread in a slot
    /// whose handle is alive. A panic its handle already read, and a slot
    /// that no longer exists, give nothing.
    #[tokio::test]
    async fn claim_unread_panics_records_only_unread_panics() {
        let coord = coordinator();
        let own = ownership();
        let read = spawn_tracked(super::TaskContext::none(), &coord, &own, async {
            panic!("read")
        })
        .expect("runtime");
        assert_eq!(panicked_with(read.await), Some("read"));
        let gone = Arc::new(super::Slot::<()>::new(Some(Arc::clone(&coord))));
        gone.deliver(TaskOutcome::Panicked(Box::new("gone")));
        drop(gone);
        let unread: super::TrackedTask<()> =
            spawn_tracked(super::TaskContext::none(), &coord, &own, async {
                panic!("unread")
            })
            .expect("runtime");
        coord.tasks().wait_idle().await;
        assert!(!coord.has_panic());
        coord.tasks().claim_unread_panics(&coord);
        let payload = coord.take_panic().expect("the unread panic is recorded");
        assert_eq!(str_payload(payload.as_ref()), Some("unread"));
        // A second claim finds nothing: the claim took the slots.
        coord.tasks().claim_unread_panics(&coord);
        assert!(coord.take_panic().is_none());
        drop(unread);
    }

    /// The cleanup claim leaves an unread value in its slot. So a handle
    /// that outlives the drive still reads the task's value, and the claim
    /// records nothing.
    #[tokio::test]
    async fn claim_unread_panics_keeps_an_unread_value() {
        let coord = coordinator();
        let own = ownership();
        let task = spawn_tracked(super::TaskContext::none(), &coord, &own, async { 7_u32 })
            .expect("runtime");
        coord.tasks().wait_idle().await;
        coord.tasks().claim_unread_panics(&coord);
        assert!(coord.take_panic().is_none(), "a value is not recorded");
        let mut task = std::pin::pin!(task);
        let polled = task.as_mut().poll(&mut Context::from_waker(Waker::noop()));
        let Poll::Ready(TaskOutcome::Value(value)) = polled else {
            panic!("the handle reads the value the claim left in the slot");
        };
        assert_eq!(value, 7);
    }

    /// Returns `Ready(())` and panics with the `&'static str` it holds when
    /// dropped.
    struct ReadyUnitDropPanics(&'static str);

    impl Future for ReadyUnitDropPanics {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    impl Drop for ReadyUnitDropPanics {
        fn drop(&mut self) {
            std::panic::panic_any(self.0);
        }
    }

    /// A detached task's panic, while its future is polled or when the
    /// finished future is dropped, is recorded on the coordinator: its
    /// outcome has no reader.
    #[tokio::test]
    async fn detached_task_panic_is_recorded() {
        let coord = coordinator();
        spawn_detached(&coord, async { std::panic::panic_any("detached poll") }).expect("runtime");
        coord.tasks().wait_idle().await;
        let payload = coord.take_panic().expect("recorded");
        assert_eq!(str_payload(payload.as_ref()), Some("detached poll"));

        let coord = coordinator();
        spawn_detached(&coord, ReadyUnitDropPanics("detached drop")).expect("runtime");
        coord.tasks().wait_idle().await;
        let payload = coord.take_panic().expect("recorded");
        assert_eq!(str_payload(payload.as_ref()), Some("detached drop"));
    }

    /// Waits, polling the condition, until it holds or 5 s passed.
    fn spin_until(what: &str, condition: impl Fn() -> bool) {
        let start = std::time::Instant::now();
        while !condition() {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "timed out waiting for {what}"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn blocking_work_wakes_consumer_before_release() {
        use crate::task::test_support::{Schedule, ScheduleState};

        let coord = coordinator();
        let _wait = coord
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let consumer = Schedule::idle(&coord);
        let mut work = Box::pin(run_blocking(move || {
            release_rx.recv().expect("the test releases the closure");
        }));
        assert!(consumer.poll(work.as_mut()).is_pending());
        assert_eq!(consumer.state(), ScheduleState::Idle);
        assert_eq!(
            coord.permits(PermitKind::Activity),
            1,
            "the blocking work holds a permit while it waits"
        );
        assert_eq!(coord.phase(), Phase::Running);

        release_tx.send(()).expect("the closure waits");
        let tasks = Arc::clone(coord.tasks());
        spin_until("the blocking work to exit", || tasks.live() == 0);
        assert_eq!(consumer.state(), ScheduleState::Queued);
        assert_eq!(
            coord.permits(PermitKind::Activity),
            1,
            "the consumer holds its own permit before the work released its own"
        );
        assert_eq!(coord.phase(), Phase::Running);

        let polled = consumer.poll(work.as_mut());
        assert!(matches!(polled, Poll::Ready(Ok(TaskOutcome::Value(())))));
        assert_eq!(coord.phase(), Phase::Suspending);
    }

    #[tokio::test]
    async fn spawn_after_suspension_spawns_nothing() {
        let coord = coordinator();
        let _wait = coord
            .register_test_wait("op", Waker::noop())
            .expect("running");
        drop(coord.acquire(PermitKind::Activity));
        assert_eq!(coord.phase(), Phase::Suspending);

        let ran = Arc::new(AtomicBool::new(false));
        let task_ran = Arc::clone(&ran);
        let mut task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &ownership(),
            async move {
                task_ran.store(true, Ordering::SeqCst);
            },
        )
        .expect("runtime");
        let polled = Pin::new(&mut task).poll(&mut Context::from_waker(Waker::noop()));
        assert!(polled.is_pending());
        tokio::task::yield_now().await;
        assert!(!ran.load(Ordering::SeqCst), "nothing was started");
        assert_eq!(coord.tasks().live(), 0);

        let mut set = TrackedJoinSet::<()>::new(Arc::clone(&coord));
        let id = set
            .spawn(super::TaskContext::none(), &ownership(), async {})
            .expect("runtime");
        assert!(id.is_none(), "the join set started nothing");
        let joined = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            set.join_next_with_id(),
        )
        .await;
        assert!(joined.is_err(), "a refused join set never yields");
        assert_eq!(coord.tasks().live(), 0);
    }

    #[tokio::test]
    async fn blocking_work_after_suspension_never_runs() {
        let coord = coordinator();
        let _wait = coord
            .register_test_wait("op", Waker::noop())
            .expect("running");
        drop(coord.acquire(PermitKind::Activity));
        let ran = Arc::new(AtomicBool::new(false));
        let work_ran = Arc::clone(&ran);
        let mut work = Box::pin(run_blocking(move || work_ran.store(true, Ordering::SeqCst)));
        let polled = with_current_coordinator(&coord, || {
            work.as_mut().poll(&mut Context::from_waker(Waker::noop()))
        });
        assert!(polled.is_pending());
        assert!(!ran.load(Ordering::SeqCst));
        assert_eq!(coord.tasks().live(), 0);
    }

    /// A task that is woken while it is not polled holds a permit until
    /// its next poll ends.
    #[tokio::test]
    async fn woken_task_counts_until_polled() {
        let coord = coordinator();
        let (tx, rx) = tokio::sync::oneshot::channel::<u32>();
        let task = spawn_tracked(
            super::TaskContext::none(),
            &coord,
            &ownership(),
            async move { rx.await.unwrap_or(0) },
        )
        .expect("runtime");
        assert_eq!(
            coord.permits(PermitKind::Activity),
            1,
            "a spawned task counts before its first poll"
        );
        let tasks = Arc::clone(coord.tasks());
        for _ in 0..100 {
            if coord.permits(PermitKind::Activity) == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            coord.permits(PermitKind::Activity),
            0,
            "an idle task holds nothing"
        );
        tx.send(5).expect("the task waits");
        assert_eq!(
            coord.permits(PermitKind::Activity),
            1,
            "the wake took a permit for the next poll"
        );
        assert!(matches!(task.await, TaskOutcome::Value(5)));
        tasks.wait_idle().await;
        assert_eq!(coord.permits(PermitKind::Activity), 0);
    }

    /// The root wrapper sets `Finished` before it releases its permit, so
    /// an unwinding registration cannot turn the release into a
    /// suspension.
    #[tokio::test]
    async fn root_sets_finished_before_release() {
        let coord = coordinator();
        let _wait = coord
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let (root, join) = spawn_root(super::TaskContext::none(), &ownership(), &coord, async {
            3
        })
        .expect("runtime");
        assert!(matches!(root.await, TaskOutcome::Value(3)));
        drop(join.await);
        assert_eq!(coord.phase(), Phase::Finished);
    }

    /// A detached task is polled after the phase left `Running`.
    #[tokio::test]
    async fn detached_task_ignores_the_phase() {
        let coord = coordinator();
        coord.finish();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        spawn_detached(&coord, async move {
            let _ = tx.send(());
        })
        .expect("runtime");
        tokio::time::timeout(std::time::Duration::from_secs(5), rx)
            .await
            .expect("the detached task ran")
            .expect("the detached task sent");
    }

    /// After the phase left `Running`, an idle schedule cannot start a poll
    /// and a wake takes no permit.
    #[test]
    fn idle_schedule_takes_nothing_after_suspension() {
        use crate::task::test_support::{Schedule, ScheduleState};

        let coord = coordinator();
        let schedule = Schedule::idle(&coord);
        coord.finish();
        assert!(!schedule.begin_poll(), "no permit after the phase ended");
        assert_eq!(schedule.state(), ScheduleState::Idle);
        schedule.waker().wake_by_ref();
        assert_eq!(schedule.state(), ScheduleState::Idle);
        assert_eq!(coord.permits(PermitKind::Activity), 0);
    }

    /// A root spawned after the phase ended is never polled.
    #[tokio::test]
    async fn root_after_finish_is_never_polled() {
        let coord = coordinator();
        coord.finish();
        let ran = Arc::new(AtomicBool::new(false));
        let root_ran = Arc::clone(&ran);
        let (root, join) = spawn_root(
            super::TaskContext::none(),
            &ownership(),
            &coord,
            async move {
                root_ran.store(true, Ordering::SeqCst);
            },
        )
        .expect("runtime");
        let waited = tokio::time::timeout(std::time::Duration::from_millis(100), root).await;
        assert!(waited.is_err(), "the root never resolves");
        assert!(!ran.load(Ordering::SeqCst));
        join.abort();
        let _ = join.await;
    }

    thread_local! {
        /// Set to make [`Scopes`] panic at the next exit of a `captured`
        /// span on this thread.
        static EXIT_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn arm_exit() {
        EXIT_ARMED.with(|armed| armed.set(true));
    }

    thread_local! {
        /// Set to make [`Scopes`] panic at the next enter of a `captured`
        /// span on this thread.
        static ENTER_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    thread_local! {
        /// Set to make [`Scopes`] panic `"close boom"` at the next close of
        /// a `captured` span on this thread.
        static CLOSE_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn arm_close() {
        CLOSE_ARMED.with(|armed| armed.set(true));
    }

    /// A plugin whose invocation wrap enters a `frame` span around `next()`.
    #[derive(Clone)]
    struct FrameSpan;

    impl crate::plugin::Plugin for FrameSpan {
        fn wrap_invocation(&self, _info: &crate::plugin::InvocationInfo, next: &mut dyn FnMut()) {
            tracing::info_span!("frame").in_scope(next);
        }
    }

    impl crate::plugin::PluginFactory for FrameSpan {
        fn create_plugin(
            &self,
            _info: &crate::plugin::InvocationInfo,
        ) -> Result<Option<Box<dyn crate::plugin::Plugin>>, crate::BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// A task context with a `captured` span and an invocation frame whose
    /// wrap enters a `frame` span.
    fn captured_cx() -> super::TaskContext {
        super::TaskContext::new(
            crate::plugin::Frames::invocation(Some(crate::plugin::test_support::plugins_of(
                FrameSpan,
            ))),
            tracing::info_span!("captured"),
        )
    }

    /// One record a [`Scopes`] layer saw: its message and its stored span
    /// chain, innermost first.
    type Seen = Arc<Mutex<Vec<(String, Vec<&'static str>)>>>;

    /// A layer that keeps every event's message and stored span chain, and
    /// panics `"exit boom"` at an armed exit of a `captured` span.
    struct Scopes(Seen);

    impl<S> tracing_subscriber::Layer<S> for Scopes
    where
        S: tracing::Subscriber + for<'l> tracing_subscriber::registry::LookupSpan<'l>,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            struct Message(String);
            impl tracing::field::Visit for Message {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }
            }
            let mut message = Message(String::new());
            event.record(&mut message);
            let chain = ctx
                .event_scope(event)
                .map(|scope| scope.map(|span| span.name()).collect())
                .unwrap_or_default();
            self.0.lock().unwrap().push((message.0, chain));
        }

        fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
            let captured = ctx.span(id).is_some_and(|span| span.name() == "captured");
            let fire = captured && ENTER_ARMED.with(|armed| armed.replace(false));
            assert!(!fire, "enter boom");
        }

        fn on_exit(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
            let captured = ctx.span(id).is_some_and(|span| span.name() == "captured");
            let fire = captured && EXIT_ARMED.with(|armed| armed.replace(false));
            assert!(!fire, "exit boom");
        }

        fn on_close(&self, id: tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
            let captured = ctx.span(&id).is_some_and(|span| span.name() == "captured");
            let fire = captured && CLOSE_ARMED.with(|armed| armed.replace(false));
            assert!(!fire, "close boom");
        }
    }

    /// Installs a [`Scopes`] layer on this thread.
    fn scopes() -> (Seen, tracing::subscriber::DefaultGuard) {
        use tracing_subscriber::layer::SubscriberExt as _;
        let seen = Seen::default();
        let guard = test_tracing::install_test_subscriber(
            tracing_subscriber::registry().with(Scopes(Arc::clone(&seen))),
        );
        (seen, guard)
    }

    /// Asserts that `seen` holds exactly `messages`, each under the
    /// `frame` span whose stored parent is the `captured` span.
    fn assert_under_frame_in_captured(seen: &Seen, messages: &[&str]) {
        let seen = seen.lock().unwrap().clone();
        let got: Vec<&str> = seen.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(got, messages, "{seen:?}");
        for (message, chain) in &seen {
            assert_eq!(chain, &["frame", "captured"], "{message}");
        }
    }

    /// Logs `"<label> poll"` when polled and returns `Ready`, and logs
    /// `"<label> drop"` when dropped.
    struct Probe(&'static str);

    impl Future for Probe {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            tracing::info!("{} poll", self.0);
            Poll::Ready(())
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            tracing::info!("{} drop", self.0);
        }
    }

    /// The gateway runs a poll and the destruction of a finished future
    /// inside the captured frames, inside the captured span, for a
    /// scheduled and a detached task.
    #[tokio::test]
    async fn gateway_scope_orders_captured_span_frames_and_action() {
        let (seen, _guard) = scopes();
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let task = spawn_tracked(captured_cx(), &coord, &ownership(), Probe("scheduled"))
            .expect("runtime");
        assert!(matches!(task.await, TaskOutcome::Value(())));
        drop(super::raw::spawn(super::Wrapped::new(
            captured_cx(),
            Probe("detached"),
            None,
            Arc::clone(&coord),
            super::Gate::Detached,
            super::Sink::Record,
            Some(tasks.acquire()),
        )));
        tasks.wait_idle().await;
        assert!(!coord.has_panic());
        assert_under_frame_in_captured(
            &seen,
            &[
                "scheduled poll",
                "scheduled drop",
                "detached poll",
                "detached drop",
            ],
        );
    }

    /// Arms the exit panic, then panics `"poll boom"`, in its poll.
    struct PollPanicsAfterArming;

    impl Future for PollPanicsAfterArming {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            arm_exit();
            panic!("poll boom");
        }
    }

    /// Returns `Ready`; arms the exit panic, then panics `"drop boom"`, in
    /// its destructor.
    struct DropPanicsAfterArming;

    impl Future for DropPanicsAfterArming {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    impl Drop for DropPanicsAfterArming {
        fn drop(&mut self) {
            arm_exit();
            panic!("drop boom");
        }
    }

    /// The first panic of a gateway action wins over a later panic of the
    /// captured span's exit; an exit panic after an action that returned
    /// is the task's panic.
    #[tokio::test]
    async fn gateway_scope_keeps_the_first_panic() {
        let (_seen, _guard) = scopes();
        let coord = coordinator();
        let own = ownership();
        let poll =
            spawn_tracked(captured_cx(), &coord, &own, PollPanicsAfterArming).expect("runtime");
        assert_eq!(panicked_with(poll.await), Some("poll boom"));
        let drop =
            spawn_tracked(captured_cx(), &coord, &own, DropPanicsAfterArming).expect("runtime");
        assert_eq!(panicked_with(drop.await), Some("drop boom"));
        let exit = spawn_tracked(captured_cx(), &coord, &own, async {
            arm_exit();
            7
        })
        .expect("runtime");
        assert_eq!(panicked_with(exit.await), Some("exit boom"));
        assert!(
            !EXIT_ARMED.with(std::cell::Cell::get),
            "every armed exit fired"
        );
    }

    /// Logs `"<label> drop"` when dropped, then panics `"drop boom"` when
    /// `panics`.
    struct LogsOnDrop {
        label: &'static str,
        panics: bool,
    }

    impl Drop for LogsOnDrop {
        fn drop(&mut self) {
            tracing::info!("{} drop", self.label);
            assert!(!self.panics, "drop boom");
        }
    }

    /// Panics `"poll boom"` when polled; holds a guard its own drop drops.
    struct PollPanics {
        _guard: LogsOnDrop,
    }

    impl Future for PollPanics {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            panic!("poll boom");
        }
    }

    /// The drop of a finished future, the cleanup after a poll panic and
    /// the drop of an aborted pending future run inside the captured span
    /// and frames. A cleanup panic after a poll panic is discarded; a
    /// destructor panic of an aborted future is recorded.
    #[tokio::test]
    async fn gateway_scope_covers_ready_failed_and_aborted_drops() {
        let (seen, _guard) = scopes();
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let guard = |label, panics| LogsOnDrop { label, panics };
        let ready = guard("ready", false);
        let task = spawn_tracked(captured_cx(), &coord, &own, async move {
            let _guard = ready;
            std::future::ready(()).await;
        })
        .expect("runtime");
        assert!(matches!(task.await, TaskOutcome::Value(())));
        let task = spawn_tracked(
            captured_cx(),
            &coord,
            &own,
            PollPanics {
                _guard: guard("failed", true),
            },
        )
        .expect("runtime");
        assert_eq!(panicked_with(task.await), Some("poll boom"));
        assert!(!coord.has_panic(), "the cleanup panic was discarded");
        let aborted = guard("aborted", true);
        let task = spawn_tracked(captured_cx(), &coord, &own, async move {
            let _guard = aborted;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        tokio::task::yield_now().await;
        task.abort();
        drop(task);
        tasks.wait_idle().await;
        let recorded = coord
            .take_panic()
            .expect("the aborted drop's panic is recorded");
        assert_eq!(str_payload(recorded.as_ref()), Some("drop boom"));
        assert_under_frame_in_captured(&seen, &["ready drop", "failed drop", "aborted drop"]);
        // A panic entering the span prevents the drop inside it: the future
        // is dropped after the catch, and its destructor's later panic is
        // discarded. A fresh coordinator, because one keeps only its first
        // panic.
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let unentered = guard("unentered", true);
        let task = spawn_tracked(captured_cx(), &coord, &own, async move {
            let _guard = unentered;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        tokio::task::yield_now().await;
        ENTER_ARMED.with(|armed| armed.set(true));
        task.abort();
        drop(task);
        tasks.wait_idle().await;
        let recorded = coord.take_panic().expect("the enter panic is recorded");
        assert_eq!(str_payload(recorded.as_ref()), Some("enter boom"));
        // The drop ran outside the frames. (The registry pushed the span
        // before the layer panicked, so the span itself stays current.)
        let seen = seen.lock().unwrap().clone();
        let last = seen
            .last()
            .map(|(m, chain)| (m.as_str(), chain.contains(&"frame")));
        assert_eq!(last, Some(("unentered drop", false)), "{seen:?}");
    }

    /// Returns `Pending` forever; arms the close panic, then panics
    /// `"drop boom"`, in its destructor.
    struct PendingDropPanicsAfterArmingClose;

    impl Future for PendingDropPanicsAfterArmingClose {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }

    impl Drop for PendingDropPanicsAfterArmingClose {
        fn drop(&mut self) {
            arm_close();
            panic!("drop boom");
        }
    }

    /// The gateway disposes of a task's context, whose span closes with
    /// it, inside a catch, after the future and before the outcome is
    /// handed over. A close panic is the task's panic only when nothing
    /// panicked first, on the completed and the aborted path.
    #[tokio::test]
    async fn gateway_disposes_task_context_with_the_first_panic() {
        let (_seen, _guard) = scopes();
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let own = ownership();
        let completed = spawn_tracked(captured_cx(), &coord, &own, async {
            arm_close();
            7
        })
        .expect("runtime");
        assert_eq!(panicked_with(completed.await), Some("close boom"));
        let failed = spawn_tracked(captured_cx(), &coord, &own, async {
            arm_close();
            panic!("poll boom");
        })
        .expect("runtime");
        assert_eq!(panicked_with(failed.await), Some("poll boom"));
        assert!(!CLOSE_ARMED.with(std::cell::Cell::get), "the close fired");
        assert!(!coord.has_panic(), "the later close panic was discarded");
        // An aborted pending task: the close panic is recorded.
        let aborted = spawn_tracked(captured_cx(), &coord, &own, std::future::pending::<()>())
            .expect("runtime");
        tokio::task::yield_now().await;
        arm_close();
        aborted.abort();
        drop(aborted);
        tasks.wait_idle().await;
        let recorded = coord.take_panic().expect("the close panic is recorded");
        assert_eq!(str_payload(recorded.as_ref()), Some("close boom"));
        // An aborted task whose destructor panics first: that panic is
        // recorded, and the close panic after it is discarded.
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let aborted = spawn_tracked(
            captured_cx(),
            &coord,
            &own,
            PendingDropPanicsAfterArmingClose,
        )
        .expect("runtime");
        tokio::task::yield_now().await;
        aborted.abort();
        drop(aborted);
        tasks.wait_idle().await;
        assert!(!CLOSE_ARMED.with(std::cell::Cell::get), "the close fired");
        let recorded = coord
            .take_panic()
            .expect("the destructor panic is recorded");
        assert_eq!(str_payload(recorded.as_ref()), Some("drop boom"));
    }
}
