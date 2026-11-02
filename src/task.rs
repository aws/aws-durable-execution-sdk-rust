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
//! The wrapper around each task catches a panic of the task's future and
//! hands it to the consumer as [`TaskOutcome::Panicked`]. The consumer
//! calls [`TaskOutcome::into_value`], which resumes the panic with its
//! original payload. So tokio never sees a panic from an SDK task, and no
//! consumer handles a `JoinError`.
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
//! Detached tasks run no user code, take no activity permit and ignore the
//! phase.

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Wake, Waker};

use tokio::task::{AbortHandle, Id, JoinSet};

use crate::coordinator::{InvocationCoordinator, Permit, PermitKind, Phase};
use crate::driver::TaskOwnership;
use crate::error::{OperationError, OperationErrorKind, StepError, StepErrorKind};

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
/// for each task that is aborted on drop.
#[derive(Debug)]
pub(crate) struct TaskTracker {
    /// The number of tasks that have not exited yet.
    live: Mutex<usize>,
    /// The abort handles of the tasks [`Self::abort_all`] aborts, and
    /// whether `abort_all` has run.
    abortable: Mutex<Abortable>,
    /// Woken when `live` reaches zero.
    idle: tokio::sync::Notify,
}

impl TaskTracker {
    /// Creates a tracker with no live task.
    pub(crate) fn new() -> Self {
        Self {
            live: Mutex::new(0),
            abortable: Mutex::new(Abortable {
                handles: BTreeMap::new(),
                aborting: false,
            }),
            idle: tokio::sync::Notify::new(),
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
    /// After `abort_all` has run, aborts the task instead.
    fn register_abort(&self, handle: AbortHandle) {
        let late = {
            let mut abortable = self
                .abortable
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if abortable.aborting {
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
fn ensure_runtime() -> Result<(), NoRuntime> {
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

/// Runs `f` and discards a panic from it.
fn discard_panic(f: impl FnOnce()) {
    drop(catch_unwind(AssertUnwindSafe(f)));
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
}

/// The slot a task stores its outcome in, shared with its handle.
struct Slot<T> {
    state: Mutex<SlotState<T>>,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new(SlotState {
                outcome: None,
                waker: None,
            }),
        }
    }

    /// Stores the outcome and wakes the consumer that registered a waker.
    fn deliver(&self, outcome: TaskOutcome<T>) {
        let waker = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.outcome = Some(outcome);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
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

/// Waits for the outcome a task stores in its slot.
struct SlotWait<T>(Arc<Slot<T>>);

impl<T> Future for SlotWait<T> {
    type Output = TaskOutcome<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.poll_take(cx)
    }
}

/// The queue a join set's tasks store their outcomes in.
struct JoinState<T> {
    outcomes: VecDeque<(Id, TaskOutcome<T>)>,
    waker: Option<Waker>,
}

/// Where the gateway wrapper stores its task's outcome.
enum Sink<T> {
    /// The slot of a [`TrackedTask`].
    Slot(Arc<Slot<T>>),
    /// The queue of a [`TrackedJoinSet`].
    Queue(Arc<Mutex<JoinState<T>>>),
    /// Nowhere: a detached task's outcome has no consumer.
    Discard,
}

impl<T> Sink<T> {
    fn deliver(&self, outcome: TaskOutcome<T>) {
        match self {
            Self::Slot(slot) => slot.deliver(outcome),
            Self::Queue(queue) => {
                let id = current_task_id();
                let waker = {
                    let mut state = queue.lock().unwrap_or_else(PoisonError::into_inner);
                    state.outcomes.push_back((id, outcome));
                    state.waker.take()
                };
                if let Some(waker) = waker {
                    waker.wake();
                }
            }
            Self::Discard => {}
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
        future: F,
        ownership: Option<Arc<TaskOwnership>>,
        coordinator: Arc<InvocationCoordinator>,
        gate: Gate,
        sink: Sink<F::Output>,
        permit: Option<TaskPermit>,
    ) -> Self {
        Self {
            inner: Some(Box::pin(future)),
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

impl<F: Future> Future for Wrapped<F> {
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
        let polled = match &this.gate {
            Gate::Scheduled { waker, .. } => {
                let mut inner_cx = Context::from_waker(waker);
                CURRENT_COORDINATOR.sync_scope(Arc::clone(&this.coordinator), || {
                    catch_unwind(AssertUnwindSafe(|| inner.as_mut().poll(&mut inner_cx)))
                })
            }
            Gate::Detached => CURRENT_COORDINATOR.sync_scope(Arc::clone(&this.coordinator), || {
                catch_unwind(AssertUnwindSafe(|| inner.as_mut().poll(cx)))
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
                match catch_unwind(AssertUnwindSafe(move || drop(inner))) {
                    Ok(()) => TaskOutcome::Value(value),
                    Err(payload) => {
                        // The destructor's panic is the outcome; a panic
                        // while dropping the value has nowhere to go.
                        discard_panic(move || drop(value));
                        TaskOutcome::Panicked(payload)
                    }
                }
            }
            Err(payload) => {
                // The poll's panic is the outcome. A destructor panic after
                // it is a second, caught panic, and is discarded.
                let inner = this.inner.take();
                discard_panic(move || drop(inner));
                TaskOutcome::Panicked(payload)
            }
        };
        drop(this.bless.take());
        // The poll's permit is held while the outcome is handed over, so
        // the consumer's wake takes its own permit first.
        let permit = match &this.gate {
            Gate::Scheduled { schedule, .. } => schedule.end_ready(),
            Gate::Detached => None,
        };
        this.sink.deliver(outcome);
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
        // An aborted task: its outcome has no consumer, so a panic from the
        // inner future's destructor is discarded, as tokio discards the
        // panic of an aborted task whose handle is gone.
        if let Some(inner) = self.inner.take() {
            discard_panic(move || drop(inner));
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

/// Aborts its task on drop. Holds the outcome slot it shares with its
/// task, and the task's `AbortHandle`. A handle without an abort handle
/// belongs to a spawn the coordinator refused: it never resolves.
pub(crate) struct TrackedTask<T> {
    slot: Arc<Slot<T>>,
    abort: Option<AbortHandle>,
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
            slot: Arc::new(Slot::new()),
            abort: None,
        }
    }

    /// Aborts the task. Aborting a finished task does nothing, so an outcome
    /// the task already stored stays in the slot.
    pub(crate) fn abort(&self) {
        if let Some(abort) = &self.abort {
            abort.abort();
        }
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
        self.abort();
    }
}

/// Starts an operation task, counted on the coordinator's tracker and
/// aborted by [`TaskTracker::abort_all`].
///
/// Blesses the task in `ownership` before the first poll of `future`. Takes
/// an activity permit for the task on the calling task. When the phase is
/// not `Running`, starts nothing and returns a handle that never resolves.
pub(crate) fn spawn_tracked<F>(
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
    let slot = Arc::new(Slot::new());
    let handle = raw::spawn(Wrapped::new(
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
    let slot = Arc::new(Slot::new());
    let handle = raw::spawn(Wrapped::new(
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
    };
    Ok((task, handle))
}

/// Not aborted on drop or by `abort_all`, still counted. For the batch
/// flusher and for detached checkpoint writes. Takes no ownership handle
/// and never blesses its task, because this work never creates a durable
/// operation. Takes no activity permit and is polled whatever the phase:
/// the checkpoint permit of the write it carries keeps the invocation open.
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
        future,
        None,
        Arc::clone(coordinator),
        Gate::Detached,
        Sink::Discard,
        Some(permit),
    )));
    Ok(())
}

/// Polls `future` to completion and returns its value, or the payload of a
/// panic from one of its polls or from its destructor.
///
/// A detached task delivers its outcome nowhere. A detached task whose
/// consumer awaits a result therefore runs its work through this function
/// and sends the outcome itself. So a panic in that work resumes on the
/// consumer with its original payload, as for every other SDK task.
///
/// The precedence is the task gateway's (see `Wrapped`). A poll that
/// returns `Ready` drops the future at once; when that destructor panics,
/// its panic is the outcome and the value is dropped. A poll that panics is
/// the outcome, and a destructor panic after it is discarded.
pub(crate) async fn catch_panic<F: Future>(future: F) -> TaskOutcome<F::Output> {
    let mut inner = Some(Box::pin(future));
    std::future::poll_fn(move |cx| {
        #[expect(clippy::expect_used)]
        // reason: the future is taken only when this poll returns `Ready`,
        // and the `.await` below never polls a finished `poll_fn` again.
        let future = inner
            .as_mut()
            .expect("a finished catch_panic is never polled");
        let polled = catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx)));
        let outcome = match polled {
            Ok(Poll::Pending) => return Poll::Pending,
            Ok(Poll::Ready(value)) => {
                let finished = inner.take();
                match catch_unwind(AssertUnwindSafe(move || drop(finished))) {
                    Ok(()) => TaskOutcome::Value(value),
                    Err(payload) => {
                        discard_panic(move || drop(value));
                        TaskOutcome::Panicked(payload)
                    }
                }
            }
            Err(payload) => {
                let finished = inner.take();
                discard_panic(move || drop(finished));
                TaskOutcome::Panicked(payload)
            }
        };
        Poll::Ready(outcome)
    })
    .await
}

/// A set of operation tasks whose outcomes are joined in completion order.
///
/// The outcomes of finished tasks wait in a queue it shares with its tasks;
/// the `JoinSet<()>` is used only to abort. Dropping the set aborts every
/// task in it.
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
/// Takes no ownership handle and never blesses, because blocking serdes
/// work never creates a durable operation.
pub(crate) async fn run_blocking<F, R>(f: F) -> Result<TaskOutcome<R>, NoRuntime>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    ensure_runtime()?;
    let permits = match CURRENT_COORDINATOR.try_with(Arc::clone) {
        Ok(coordinator) => {
            let Some(activity) = coordinator.acquire(PermitKind::Activity) else {
                return std::future::pending().await;
            };
            Some((activity, coordinator.tasks().acquire()))
        }
        Err(_) => None,
    };
    let slot = Arc::new(Slot::new());
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

    impl<T> TestSlot<T> {
        /// Stores `value` as the outcome and wakes the consumer.
        pub(crate) fn deliver_value(&self, value: T) {
            self.0.deliver(TaskOutcome::Value(value));
        }
    }

    /// A slot whose consumer is `waker`.
    pub(crate) fn slot_with_waker<T>(waker: &Waker) -> TestSlot<T> {
        let slot = Arc::new(Slot::new());
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
        NoRuntime, TaskOutcome, TrackedJoinSet, catch_panic, run_blocking, spawn_detached,
        spawn_root, spawn_tracked,
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
        let task = spawn_tracked(&coord, &ownership(), async { 7 }).expect("runtime");
        let outcome = task.await;
        assert_eq!(tasks.live(), 0, "the task released its permit on exit");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
    }

    #[tokio::test]
    async fn tracked_task_catches_panic() {
        let coord = coordinator();
        let task = spawn_tracked(&coord, &ownership(), async {
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
        let task = spawn_tracked(&coord, &ownership(), async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        tokio::task::yield_now().await;
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
        let _task = spawn_tracked(&coord, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        let guard = DropFlag(Arc::clone(&joined_dropped));
        set.spawn(&own, async move {
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
        let parent = spawn_tracked(&coord, &own, std::future::pending::<()>()).expect("runtime");
        tokio::task::yield_now().await;
        tasks.abort_all();

        let tracked_dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&tracked_dropped));
        let late_task = spawn_tracked(&coord, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let joined_dropped = Arc::new(AtomicBool::new(false));
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        let guard = DropFlag(Arc::clone(&joined_dropped));
        set.spawn(&own, async move {
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

        let task = spawn_tracked(&coord, &own, async { 7 }).expect("runtime");
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
        let task = spawn_tracked(&coord, &ownership(), async { 7 }).expect("runtime");
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("the task was not aborted");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
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
                spawn_tracked(&coord.clone(), &own.clone(), async move {
                    loop {
                        let nested = spawn_tracked(&coord, &own, std::future::pending::<()>())
                            .expect("runtime");
                        escaped.lock().expect("lock").push(nested);
                        started.notify_one();
                        tokio::task::yield_now().await;
                    }
                })
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
            let tracked = spawn_tracked(&thread_coord, &own, async {}).map(drop);
            let detached = spawn_detached(&thread_coord, async {});
            let mut set = TrackedJoinSet::<()>::new(Arc::clone(&thread_coord));
            let joined = set.spawn(&own, async {}).map(drop);
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
        let tracked = spawn_tracked(&coord, &own, async move { task_own.check_current_task() })
            .expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        let set_own = Arc::clone(&own);
        set.spawn(&own, async move { set_own.check_current_task() })
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
            set.spawn(&own, async move { task_own.check_current_task() })
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
        let task = spawn_tracked(&coord, &own, async move {
            let _ = tx.send(());
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        rx.await.expect("the task signals");
        assert_eq!(own.blessed_count(), 1);
        drop(task);
        tasks.wait_idle().await;
        assert_eq!(own.blessed_count(), 0);
    }

    #[tokio::test]
    async fn finished_task_with_escaped_handle_does_not_block_idle() {
        let coord = coordinator();
        let tasks = Arc::clone(coord.tasks());
        let task = spawn_tracked(&coord, &ownership(), async { 1 }).expect("runtime");
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
        let task = spawn_tracked(&coord, &own, ReadyThenDropPanics).expect("runtime");
        assert_eq!(panicked_with(task.await), Some("drop boom"));
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(&own, ReadyThenDropPanics).expect("runtime");
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
        let task = spawn_tracked(&coord, &ownership(), ReadyValueThenDropPanics).expect("runtime");
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
        let task = spawn_tracked(&coord, &own, async {}).expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(&own, async {}).expect("runtime");
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
        let task = spawn_tracked(&coord, &own, PollPanicsThenDropPanics).expect("runtime");
        assert_eq!(panicked_with(task.await), Some("poll boom"));
        let mut set = TrackedJoinSet::new(Arc::clone(&coord));
        set.spawn(&own, PollPanicsThenDropPanics).expect("runtime");
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

    /// A value whose destructor panics with `payload`.
    struct PanicOnDrop(&'static str);

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("{}", self.0);
        }
    }

    /// A future that returns `Ready` with a value whose destructor panics,
    /// and whose own destructor panics with `"future drop"`.
    struct ReadyThenDropPanic;

    impl Future for ReadyThenDropPanic {
        type Output = PanicOnDrop;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<PanicOnDrop> {
            Poll::Ready(PanicOnDrop("value drop"))
        }
    }

    impl Drop for ReadyThenDropPanic {
        fn drop(&mut self) {
            panic!("future drop");
        }
    }

    /// A future whose poll panics with `"poll"` and whose destructor
    /// panics with `"future drop"`.
    struct PollPanicThenDropPanic;

    impl Future for PollPanicThenDropPanic {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            panic!("poll");
        }
    }

    impl Drop for PollPanicThenDropPanic {
        fn drop(&mut self) {
            panic!("future drop");
        }
    }

    /// A future that returns `Ready` and then panics in its destructor:
    /// that panic is the outcome, as in the task gateway, and the value's
    /// own destructor panic is discarded.
    #[tokio::test]
    async fn catch_panic_returns_a_destructor_panic_after_ready() {
        let TaskOutcome::Panicked(payload) = catch_panic(ReadyThenDropPanic).await else {
            panic!("expected the destructor's panic");
        };
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| str_payload(payload.as_ref()));
        assert_eq!(message, Some("future drop"));
    }

    /// A poll's panic wins over a destructor panic after it.
    #[tokio::test]
    async fn catch_panic_prefers_the_poll_panic() {
        let TaskOutcome::Panicked(payload) = catch_panic(PollPanicThenDropPanic).await else {
            panic!("expected the poll's panic");
        };
        assert_eq!(str_payload(payload.as_ref()), Some("poll"));
    }

    /// A future that finishes normally gives its value.
    #[tokio::test]
    async fn catch_panic_returns_the_value() {
        let TaskOutcome::Value(value) = catch_panic(async { 7_u32 }).await else {
            panic!("expected the value");
        };
        assert_eq!(value, 7);
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
        let _wait = coord.register_wait("op", Waker::noop()).expect("running");
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
        let _wait = coord.register_wait("op", Waker::noop()).expect("running");
        drop(coord.acquire(PermitKind::Activity));
        assert_eq!(coord.phase(), Phase::Suspending);

        let ran = Arc::new(AtomicBool::new(false));
        let task_ran = Arc::clone(&ran);
        let mut task = spawn_tracked(&coord, &ownership(), async move {
            task_ran.store(true, Ordering::SeqCst);
        })
        .expect("runtime");
        let polled = Pin::new(&mut task).poll(&mut Context::from_waker(Waker::noop()));
        assert!(polled.is_pending());
        tokio::task::yield_now().await;
        assert!(!ran.load(Ordering::SeqCst), "nothing was started");
        assert_eq!(coord.tasks().live(), 0);

        let mut set = TrackedJoinSet::<()>::new(Arc::clone(&coord));
        let id = set.spawn(&ownership(), async {}).expect("runtime");
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
        let _wait = coord.register_wait("op", Waker::noop()).expect("running");
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
        let task = spawn_tracked(&coord, &ownership(), async move { rx.await.unwrap_or(0) })
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
        let _wait = coord.register_wait("op", Waker::noop()).expect("running");
        let (root, join) = spawn_root(&ownership(), &coord, async { 3 }).expect("runtime");
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
        let (root, join) = spawn_root(&ownership(), &coord, async move {
            root_ran.store(true, Ordering::SeqCst);
        })
        .expect("runtime");
        let waited = tokio::time::timeout(std::time::Duration::from_millis(100), root).await;
        assert!(waited.is_err(), "the root never resolves");
        assert!(!ran.load(Ordering::SeqCst));
        join.abort();
        let _ = join.await;
    }
}
