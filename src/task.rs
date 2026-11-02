//! The task gateway: every tokio task the SDK starts goes through here.
//!
//! The SDK starts tasks for `.spawn()`, for each map item and parallel
//! branch, for each combinator input, for the checkpoint batch flusher, for
//! blocking serdes work, and for the handler itself. Each of these tasks
//! goes through one of the functions below:
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

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

use tokio::task::{AbortHandle, Id, JoinSet};

use crate::driver::TaskOwnership;
use crate::error::{OperationError, OperationErrorKind, StepError, StepErrorKind};

/// A captured panic payload.
type Payload = Box<dyn Any + Send + 'static>;

/// The message of the error an operation returns when it needs a task and
/// no Tokio runtime is running on the calling thread.
pub(crate) const NO_RUNTIME_MESSAGE: &str = "durable operation started with `.spawn()` while no \
     Tokio runtime is running; call it from the handler or from an SDK operation";

tokio::task_local! {
    /// The tracker of the invocation whose task is being polled. The
    /// gateway wrapper sets it for each poll of its future, so blocking
    /// work started from any SDK task is counted on that invocation.
    static CURRENT_TASKS: Arc<TaskTracker>;
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

/// The future of every gateway task: polls the inner future under
/// `catch_unwind`, blesses the task before the first poll, and hands the
/// outcome to the consumer before it releases its permit.
///
/// The fields drop in declaration order, so after `Drop::drop` has dropped
/// the inner future, the bless guard drops before the permit.
struct Wrapped<F: Future> {
    /// The inner future. `None` once it finished or was dropped.
    inner: Option<Pin<Box<F>>>,
    /// Present from the first poll of an operation task until it exits.
    bless: Option<BlessGuard>,
    /// The ownership handle to bless the task in, taken at the first poll.
    ownership: Option<Arc<TaskOwnership>>,
    /// The tracker set as `CURRENT_TASKS` during each poll.
    scope: Arc<TaskTracker>,
    /// Where the outcome goes.
    sink: Sink<F::Output>,
    /// The task's count, dropped after the outcome is stored.
    permit: Option<TaskPermit>,
}

impl<F: Future> Wrapped<F> {
    fn new(
        future: F,
        ownership: Option<Arc<TaskOwnership>>,
        scope: Arc<TaskTracker>,
        sink: Sink<F::Output>,
        permit: Option<TaskPermit>,
    ) -> Self {
        Self {
            inner: Some(Box::pin(future)),
            bless: None,
            ownership,
            scope,
            sink,
            permit,
        }
    }
}

impl<F: Future> Future for Wrapped<F> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
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
        let polled = CURRENT_TASKS.sync_scope(Arc::clone(&this.scope), || {
            catch_unwind(AssertUnwindSafe(|| inner.as_mut().poll(cx)))
        });
        let outcome = match polled {
            Ok(Poll::Pending) => return Poll::Pending,
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
        this.sink.deliver(outcome);
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
/// task, and the task's `AbortHandle`.
pub(crate) struct TrackedTask<T> {
    slot: Arc<Slot<T>>,
    abort: AbortHandle,
}

impl<T> std::fmt::Debug for TrackedTask<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackedTask")
            .field("id", &self.abort.id())
            .finish_non_exhaustive()
    }
}

impl<T> TrackedTask<T> {
    /// Aborts the task. Aborting a finished task does nothing, so an outcome
    /// the task already stored stays in the slot.
    pub(crate) fn abort(&self) {
        self.abort.abort();
    }

    /// Takes the outcome the task stored, if it has stored one and no poll
    /// has taken it yet.
    pub(crate) fn take_finished(&self) -> Option<TaskOutcome<T>> {
        self.slot
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .outcome
            .take()
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
        self.abort.abort();
    }
}

/// Starts an operation task, counted on `tasks` and aborted by
/// [`TaskTracker::abort_all`].
///
/// Blesses the task in `ownership` before the first poll of `future`.
pub(crate) fn spawn_tracked<F>(
    tasks: &Arc<TaskTracker>,
    ownership: &Arc<TaskOwnership>,
    future: F,
) -> Result<TrackedTask<F::Output>, NoRuntime>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    ensure_runtime()?;
    let permit = tasks.acquire();
    let slot = Arc::new(Slot::new());
    let handle = raw::spawn(Wrapped::new(
        future,
        Some(Arc::clone(ownership)),
        Arc::clone(tasks),
        Sink::Slot(Arc::clone(&slot)),
        Some(permit),
    ));
    tasks.register_abort(handle.abort_handle());
    Ok(TrackedTask {
        slot,
        abort: handle.abort_handle(),
    })
}

/// Starts the handler of one invocation as an operation task. Blessed in
/// `ownership` before its first poll and polled by the same wrapper as
/// `spawn_tracked`, with `tasks` as the current tracker. Not counted in
/// `tasks` and not reached by `abort_all`: `drive_invocation` aborts it and
/// awaits the returned `JoinHandle` before it aborts the other tasks.
pub(crate) fn spawn_root<F>(
    ownership: &Arc<TaskOwnership>,
    tasks: &Arc<TaskTracker>,
    future: F,
) -> Result<(TrackedTask<F::Output>, tokio::task::JoinHandle<()>), NoRuntime>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    ensure_runtime()?;
    let slot = Arc::new(Slot::new());
    let handle = raw::spawn(Wrapped::new(
        future,
        Some(Arc::clone(ownership)),
        Arc::clone(tasks),
        Sink::Slot(Arc::clone(&slot)),
        None,
    ));
    let task = TrackedTask {
        slot,
        abort: handle.abort_handle(),
    };
    Ok((task, handle))
}

/// Not aborted on drop or by `abort_all`, still counted. For the batch
/// flusher and for detached checkpoint writes. Takes no ownership handle
/// and never blesses its task, because this work never creates a durable
/// operation.
pub(crate) fn spawn_detached<F>(tasks: &Arc<TaskTracker>, future: F) -> Result<(), NoRuntime>
where
    F: Future<Output = ()> + Send + 'static,
{
    ensure_runtime()?;
    let permit = tasks.acquire();
    drop(raw::spawn(Wrapped::new(
        future,
        None,
        Arc::clone(tasks),
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
    tracker: Arc<TaskTracker>,
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
    /// Creates an empty set whose tasks are counted on `tasks`.
    pub(crate) fn new(tasks: Arc<TaskTracker>) -> Self {
        Self {
            outcomes: Arc::new(Mutex::new(JoinState {
                outcomes: VecDeque::new(),
                waker: None,
            })),
            tasks: JoinSet::new(),
            tracker: tasks,
            pending: 0,
        }
    }

    /// Starts an operation task in the set.
    ///
    /// Blesses the task in `ownership` before the first poll of `future`.
    pub(crate) fn spawn<F>(
        &mut self,
        ownership: &Arc<TaskOwnership>,
        future: F,
    ) -> Result<Id, NoRuntime>
    where
        F: Future<Output = T> + Send + 'static,
    {
        ensure_runtime()?;
        let permit = self.tracker.acquire();
        let abort = raw::join_set_spawn(
            &mut self.tasks,
            Wrapped::new(
                future,
                Some(Arc::clone(ownership)),
                Arc::clone(&self.tracker),
                Sink::Queue(Arc::clone(&self.outcomes)),
                Some(permit),
            ),
        );
        let id = abort.id();
        self.tracker.register_abort(abort);
        self.pending += 1;
        Ok(id)
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

/// Runs `f` on the blocking pool, counted on the tracker of the current
/// invocation (the `CURRENT_TASKS` task-local). Without one, the work is not
/// counted. Takes no ownership handle and never blesses, because blocking
/// serdes work never creates a durable operation.
pub(crate) async fn run_blocking<F, R>(f: F) -> Result<TaskOutcome<R>, NoRuntime>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    ensure_runtime()?;
    let permit = CURRENT_TASKS
        .try_with(TaskTracker::acquire)
        .unwrap_or(TaskPermit { tasks: None });
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
        drop(permit);
    });
    Ok(SlotWait(slot).await)
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Helpers for tests in other modules.

    use std::future::Future;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    use super::{CURRENT_TASKS, TaskTracker};

    /// Calls `f` with `tasks` as the current tracker.
    pub(crate) fn with_current_tasks<R>(tasks: &Arc<TaskTracker>, f: impl FnOnce() -> R) -> R {
        CURRENT_TASKS.sync_scope(Arc::clone(tasks), f)
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

    use super::test_support::{poll_once_without_runtime, str_payload, with_current_tasks};
    use super::{
        NoRuntime, TaskOutcome, TaskTracker, TrackedJoinSet, catch_panic, run_blocking,
        spawn_detached, spawn_tracked,
    };
    use crate::driver::TaskOwnership;

    fn tracker() -> Arc<TaskTracker> {
        Arc::new(TaskTracker::new())
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
        let tasks = tracker();
        let task = spawn_tracked(&tasks, &ownership(), async { 7 }).expect("runtime");
        let outcome = task.await;
        assert_eq!(tasks.live(), 0, "the task released its permit on exit");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
    }

    #[tokio::test]
    async fn tracked_task_catches_panic() {
        let tasks = tracker();
        let task = spawn_tracked(&tasks, &ownership(), async {
            panic!("boom");
        })
        .expect("runtime");
        let outcome: TaskOutcome<()> = task.await;
        assert_eq!(panicked_with(outcome), Some("boom"));
    }

    #[tokio::test]
    async fn dropped_tracked_task_counts_until_unwound() {
        let tasks = tracker();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&dropped));
        let task = spawn_tracked(&tasks, &ownership(), async move {
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
        let tasks = tracker();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let work_barrier = Arc::clone(&barrier);
        let mut work = Box::pin(run_blocking(move || {
            work_barrier.wait();
        }));
        let polled = with_current_tasks(&tasks, || {
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
        let tasks = tracker();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        spawn_detached(&tasks, async move {
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
        let tasks = tracker();
        let own = ownership();
        let tracked_dropped = Arc::new(AtomicBool::new(false));
        let joined_dropped = Arc::new(AtomicBool::new(false));
        let detached_done = Arc::new(AtomicBool::new(false));

        let guard = DropFlag(Arc::clone(&tracked_dropped));
        let _task = spawn_tracked(&tasks, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
        let guard = DropFlag(Arc::clone(&joined_dropped));
        set.spawn(&own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let done = Arc::clone(&detached_done);
        spawn_detached(&tasks, async move {
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
        let tasks = tracker();
        let own = ownership();
        // A live task, as during teardown: the parent of the late tasks.
        let parent = spawn_tracked(&tasks, &own, std::future::pending::<()>()).expect("runtime");
        tokio::task::yield_now().await;
        tasks.abort_all();

        let tracked_dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(Arc::clone(&tracked_dropped));
        let late_task = spawn_tracked(&tasks, &own, async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        })
        .expect("runtime");
        let joined_dropped = Arc::new(AtomicBool::new(false));
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
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

        let task = spawn_tracked(&tasks, &own, async { 7 }).expect("runtime");
        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .expect("an idle tracker no longer aborts new tasks");
        assert!(matches!(outcome, TaskOutcome::Value(7)));
    }

    /// `abort_all` on a tracker with no live task leaves it usable: no task
    /// of a teardown can register afterwards, so new tasks are not aborted.
    #[tokio::test]
    async fn abort_all_on_idle_tracker_keeps_it_usable() {
        let tasks = tracker();
        tasks.abort_all();
        let task = spawn_tracked(&tasks, &ownership(), async { 7 }).expect("runtime");
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
            let tasks = tracker();
            let own = ownership();
            let escaped = Arc::new(Mutex::new(Vec::new()));
            let started = Arc::new(tokio::sync::Notify::new());
            let spawner = {
                let tasks = Arc::clone(&tasks);
                let own = Arc::clone(&own);
                let escaped = Arc::clone(&escaped);
                let started = Arc::clone(&started);
                spawn_tracked(&tasks.clone(), &own.clone(), async move {
                    loop {
                        let nested = spawn_tracked(&tasks, &own, std::future::pending::<()>())
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
        let tasks = tracker();
        let ran = Arc::new(AtomicBool::new(false));
        let thread_tasks = Arc::clone(&tasks);
        let thread_ran = Arc::clone(&ran);
        let polled = poll_once_without_runtime(move || {
            let own = ownership();
            let tracked = spawn_tracked(&thread_tasks, &own, async {}).map(drop);
            let detached = spawn_detached(&thread_tasks, async {});
            let mut set = TrackedJoinSet::<()>::new(Arc::clone(&thread_tasks));
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
        let tasks = tracker();
        let own = ownership();
        let task_own = Arc::clone(&own);
        let tracked = spawn_tracked(&tasks, &own, async move { task_own.check_current_task() })
            .expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
        let set_own = Arc::clone(&own);
        set.spawn(&own, async move { set_own.check_current_task() })
            .expect("runtime");

        assert!(matches!(tracked.await, TaskOutcome::Value(Ok(()))));
        let (_, joined) = set.join_next_with_id().await.expect("one task");
        assert!(matches!(joined, TaskOutcome::Value(Ok(()))));
    }

    #[tokio::test]
    async fn detached_task_is_not_blessed() {
        let tasks = tracker();
        let own = ownership();
        let task_own = Arc::clone(&own);
        let (tx, rx) = tokio::sync::oneshot::channel();
        spawn_detached(&tasks, async move {
            let _ = tx.send(task_own.check_current_task());
        })
        .expect("runtime");
        let checked = rx.await.expect("the detached task sends");
        let err = checked.expect_err("a detached task is not blessed");
        assert!(err.contains("Use .spawn()"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn finished_operation_task_is_unblessed_before_it_exits() {
        let tasks = tracker();
        let own = ownership();
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
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
        let tasks = tracker();
        let own = ownership();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let task = spawn_tracked(&tasks, &own, async move {
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
        let tasks = tracker();
        let task = spawn_tracked(&tasks, &ownership(), async { 1 }).expect("runtime");
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
        let tasks = tracker();
        let own = ownership();
        let task = spawn_tracked(&tasks, &own, ReadyThenDropPanics).expect("runtime");
        assert_eq!(panicked_with(task.await), Some("drop boom"));
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
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
        let tasks = tracker();
        let task = spawn_tracked(&tasks, &ownership(), ReadyValueThenDropPanics).expect("runtime");
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
        let tasks = tracker();
        let own = ownership();
        let task = spawn_tracked(&tasks, &own, async {}).expect("runtime");
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
        set.spawn(&own, async {}).expect("runtime");
        assert_eq!(tasks.abortable_count(), 2);
        let _ = task.await;
        let _ = set.join_next_with_id().await;
        tasks.wait_idle().await;
        assert_eq!(tasks.abortable_count(), 0);
    }

    #[tokio::test]
    async fn poll_panic_wins_over_drop_panic() {
        let tasks = tracker();
        let own = ownership();
        let task = spawn_tracked(&tasks, &own, PollPanicsThenDropPanics).expect("runtime");
        assert_eq!(panicked_with(task.await), Some("poll boom"));
        let mut set = TrackedJoinSet::new(Arc::clone(&tasks));
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
        let tasks = tracker();
        let own = ownership();
        let waker_slot = Arc::new(Mutex::new(None));
        let task = spawn_tracked(
            &tasks,
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
}
