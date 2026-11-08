//! The live controls of a run: [`RunHandle`], [`OperationHandle`] and
//! [`WaitingStatus`], and the shared state that the runner futures of a run,
//! its driver and its backend use (see "Reacting to a run in progress" in
//! the [`test_util`](super) module docs).

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use aws_sdk_lambda::types::Event;
use serde::de::DeserializeOwned;

use super::{
    BackendState, CallbackFailure, CallbackReply, CallbackReplyKind, CloudApi, CloudRunner,
    InvocationType, RunnerKind, TestOperation, TestResult, TestRunnerError, aws_error,
    check_heartbeat, fetch_history, fold_history, is_terminal_status, reply_live, snapshot_of,
    terminal_execution_event,
};
use crate::OperationStatus;

#[cfg(doc)]
use super::LocalRunner;

/// The `Unsupported` text of `wait_for_data` under
/// [`InvocationType::RequestResponse`].
pub(super) const WAIT_FOR_DATA_REQUEST_RESPONSE_UNSUPPORTED: &str =
    "wait_for_data with InvocationType::RequestResponse (use InvocationType::Event)";

/// The `Unsupported` text of `send_callback` under
/// [`InvocationType::RequestResponse`].
pub(super) const SEND_CALLBACK_REQUEST_RESPONSE_UNSUPPORTED: &str =
    "send_callback with InvocationType::RequestResponse (use InvocationType::Event)";

/// The `Unsupported` text of `send_callback_heartbeat` under
/// [`InvocationType::RequestResponse`].
pub(super) const SEND_CALLBACK_HEARTBEAT_REQUEST_RESPONSE_UNSUPPORTED: &str =
    "send_callback_heartbeat with InvocationType::RequestResponse (use InvocationType::Event)";

/// The `Unsupported` text of a timeout reply under [`CloudRunner`].
pub(super) const CLOUD_TIMEOUT_REPLY_UNSUPPORTED: &str =
    "CallbackReply::timeout and heartbeat_timeout under CloudRunner";

/// The `Unsupported` text of pause and resume under [`CloudRunner`].
pub(super) const CLOUD_PAUSE_UNSUPPORTED: &str = "pause and resume under CloudRunner";

/// The status that [`OperationHandle::wait_for_data`] waits for, as JS
/// `WaitingOperationStatus`.
///
/// For a `wait_for_callback` context, the status follows its children: the
/// submitter step and the callback.
///
/// | Status | Reached by an operation | Reached by a `wait_for_callback` context |
/// | --- | --- | --- |
/// | `Started` | it exists | it has a step or callback child |
/// | `Submitted` | it has a terminal status | a step or callback child has a terminal status |
/// | `Completed` | it has a terminal status | its callback child and the context have a terminal status |
///
/// The terminal statuses are `Succeeded`, `Failed`, `Cancelled`, `TimedOut`
/// and `Stopped`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum WaitingStatus {
    /// The operation exists.
    #[default]
    Started,
    /// The operation ended, or the submitter of a `wait_for_callback`
    /// finished.
    Submitted,
    /// The operation ended.
    Completed,
}

/// Which operation an [`OperationHandle`] names. It is evaluated on the
/// operations of the run each time the handle is used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Selector {
    /// The `index`-th operation with this name, in operation order.
    Name(String, usize),
    /// The operation at this 0-based position in operation order.
    Index(usize),
    /// The operation with this ID.
    Id(String),
}

impl Selector {
    /// The operation of `ops` this selector names, if any.
    pub(super) fn resolve<'a>(&self, ops: &'a [TestOperation]) -> Option<&'a TestOperation> {
        match self {
            Self::Name(name, index) => ops
                .iter()
                .filter(|op| op.name.as_deref() == Some(name.as_str()))
                .nth(*index),
            Self::Index(index) => ops.get(*index),
            Self::Id(id) => ops.iter().find(|op| op.id == *id),
        }
    }
}

/// Whether `op` is a `wait_for_callback` context.
pub(super) fn is_wait_for_callback(op: &TestOperation) -> bool {
    op.kind == RunnerKind::Context && op.sub_type.as_deref() == Some(crate::callback::WFCB_SUB_TYPE)
}

/// Whether `op` reached `status`, as the table of [`WaitingStatus`] states.
/// `ops` holds every operation of the run, so the children of a
/// `wait_for_callback` context can be read.
pub(super) fn status_reached(
    op: &TestOperation,
    status: WaitingStatus,
    ops: &[TestOperation],
) -> bool {
    if !is_wait_for_callback(op) {
        return status == WaitingStatus::Started || is_terminal_status(op.status);
    }
    let mut children = ops.iter().filter(|child| {
        child.parent_id.as_deref() == Some(op.id.as_str())
            && matches!(child.kind, RunnerKind::Step | RunnerKind::Callback)
    });
    match status {
        WaitingStatus::Started => children.next().is_some(),
        WaitingStatus::Submitted => children.any(|child| is_terminal_status(child.status)),
        // The context's own `SUCCEED` or `FAIL` follows its callback's end,
        // so the snapshot `Completed` returns shows the finished context.
        WaitingStatus::Completed => {
            is_terminal_status(op.status)
                && children.any(|child| {
                    child.kind == RunnerKind::Callback && is_terminal_status(child.status)
                })
        }
    }
}

/// The operation `selector` names in `ops`, when it reached `status`.
pub(super) fn reached_operation(
    selector: &Selector,
    status: WaitingStatus,
    ops: &[TestOperation],
) -> Option<TestOperation> {
    selector
        .resolve(ops)
        .filter(|op| status_reached(op, status, ops))
        .cloned()
}

/// The callback record a callback control targets: the operation
/// `selector` names when it is a callback, or the callback child of a
/// `wait_for_callback` context.
///
/// # Errors
///
/// - [`TestRunnerError::OperationNotFound`] when no operation matches.
/// - [`TestRunnerError::CallbackNotPending`] for a `wait_for_callback`
///   context with no callback child yet.
/// - [`TestRunnerError::NotACallback`] for any other operation.
pub(super) fn callback_target<'a>(
    selector: &Selector,
    ops: &'a [TestOperation],
) -> Result<&'a TestOperation, TestRunnerError> {
    let op = selector
        .resolve(ops)
        .ok_or(TestRunnerError::OperationNotFound)?;
    match op.kind {
        RunnerKind::Callback => Ok(op),
        _ if is_wait_for_callback(op) => ops
            .iter()
            .find(|child| {
                child.kind == RunnerKind::Callback
                    && child.parent_id.as_deref() == Some(op.id.as_str())
            })
            .ok_or(TestRunnerError::CallbackNotPending),
        _ => Err(TestRunnerError::NotACallback),
    }
}

/// Whether the driver finished with a [`TestResult`], as the runner futures
/// see it. A run failure is kept apart, in [`RunState::failure`]: the driver
/// ends with it instead of a `TestResult`, or the run fails before the
/// driver starts. A control's own failed Lambda call is neither.
#[derive(Default)]
pub(super) enum RunEnd {
    /// The driver has not finished.
    #[default]
    Running,
    /// The driver returned a [`TestResult`], boxed as `Any`. The
    /// [`RunHandle`] takes it, so it is `None` after the handle returned it.
    Finished(Option<Box<dyn std::any::Any + Send>>),
}

impl std::fmt::Debug for RunEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => f.write_str("Running"),
            Self::Finished(_) => f.write_str("Finished"),
        }
    }
}

/// What a control of a run is.
#[derive(Debug)]
pub(super) enum ControlKind {
    /// A [`RunHandle`] awaited for the run's end.
    Run,
    /// A `LocalRunner` [`OperationHandle::wait_for_data`].
    Data {
        selector: Selector,
        status: WaitingStatus,
    },
    /// A `LocalRunner` [`RunHandle::pause`] that waits for its invocation
    /// to end.
    Pause,
    /// A `CloudRunner` control: it polls the driver and makes Lambda calls.
    Cloud,
}

impl ControlKind {
    /// Whether a pending control of this kind lets a `LocalRunner` run move
    /// (the hold rule). A `CloudRunner` run has no hold rule.
    fn drives_local(&self) -> bool {
        !matches!(self, Self::Cloud)
    }
}

/// The value a control became ready with.
#[derive(Debug)]
pub(super) enum ControlValue {
    /// The snapshot a `wait_for_data` recheck took when the status was
    /// reached.
    Data(Box<TestOperation>),
    /// The invocation a `pause` waited for ended.
    PauseDone,
}

/// Where a control is. The hold eligibility of a control is part of its
/// phase, so the transition that makes a control ready removes its
/// eligibility in the same step.
#[derive(Debug)]
pub(super) enum ControlPhase {
    /// Created, not polled, or polled and found nothing to wait for.
    Dormant,
    /// Polled and waiting, woken by `waker`. With `drives_local`, it lets
    /// a `LocalRunner` run move.
    Pending {
        waker: std::task::Waker,
        drives_local: bool,
    },
    /// Ready with `ControlValue`; it no longer waits.
    Ready(ControlValue),
    /// A `CloudRunner` Lambda call with the ticket `request` is in flight,
    /// and `waker` wakes the control when the run fails meanwhile. A
    /// pending `wait_for_data` then returns the run failure and drops its
    /// call; a callback control keeps awaiting its call (see
    /// [`await_in_flight`]).
    InFlight {
        request: u64,
        waker: std::task::Waker,
    },
}

/// One control of a run: a runner future that is alive.
#[derive(Debug)]
pub(super) struct Control {
    pub(super) kind: ControlKind,
    pub(super) phase: ControlPhase,
}

/// Where a `LocalRunner` run's invocations are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LocalPhase {
    /// No invocation runs, and the run may start one.
    Idle,
    /// The driver awaits the service function. With
    /// `tokenless_requested`, a `pause` asked the backend to answer the
    /// next checkpoint it applies without a token.
    Invoking { tokenless_requested: bool },
    /// The driver decided that the run ends: after a terminal envelope,
    /// when the invocation budget ran out, or when nothing can advance the
    /// execution. A callback no longer waits for a reply or a heartbeat.
    Closed,
}

/// The `LocalRunner` part of a run's state.
#[derive(Debug)]
pub(super) struct LocalState {
    /// The top-level backend's records: operations, clock, queued replies,
    /// updated IDs and history events.
    pub(super) records: BackendState,
    pub(super) phase: LocalPhase,
    /// Set by [`RunHandle::pause`], cleared by [`RunHandle::resume`].
    pub(super) paused: bool,
    /// Set by a live callback reply, taken by the driver after a `PENDING`
    /// invocation: the next invocation starts without moving the clock.
    pub(super) invocation_due: bool,
}

/// The `CloudRunner` part of a run's state.
#[derive(Debug, Default)]
pub(super) struct CloudState {
    /// The configured poll limit: one budget for the driver's
    /// `GetDurableExecution` calls and the controls' history reads.
    pub(super) limit: Option<usize>,
    /// The polls the budget spent.
    pub(super) used: usize,
    /// The execution ARN, once the invoke returned it.
    pub(super) execution_arn: Option<String>,
    /// The operations of the newest history read that finished.
    pub(super) fold: Option<Vec<TestOperation>>,
    /// The ticket of the history read `fold` came from.
    pub(super) fold_request: u64,
    /// The last ticket given to a Lambda call.
    pub(super) next_request: u64,
}

/// The whole mutable state of one run. Every decision of the run's
/// controls, its driver and its backend is a transition on it, under one
/// mutex.
#[derive(Debug)]
pub(super) struct RunState {
    pub(super) next_id: u64,
    /// The run failure: the first runner error of the run itself, recorded
    /// by [`RunCore::failed`] before the driver starts or by
    /// [`RunCore::poll_driver`] when the driver returns it. A control's own
    /// failed Lambda call is never recorded here. `LocalRunner` controls
    /// and waits that started while the run was active commit their
    /// outcome against it (see [`RunState::commit`]).
    pub(super) failure: Option<TestRunnerError>,
    pub(super) end: RunEnd,
    /// The live controls, by ID.
    pub(super) controls: BTreeMap<u64, Control>,
    pub(super) local: LocalState,
    pub(super) cloud: CloudState,
    /// In-crate test seam: a run failure that the next [`RunState::commit`]
    /// records before it decides, as a concurrent failure of the run would
    /// be recorded just before the commit.
    #[cfg(test)]
    pub(super) fail_at_commit: Option<TestRunnerError>,
}

impl RunState {
    /// The state of a new run whose top-level backend records are
    /// `records`.
    pub(super) fn new(records: BackendState) -> Self {
        Self {
            next_id: 0,
            failure: None,
            end: RunEnd::Running,
            controls: BTreeMap::new(),
            local: LocalState {
                records,
                phase: LocalPhase::Idle,
                paused: false,
                invocation_due: false,
            },
            cloud: CloudState::default(),
            #[cfg(test)]
            fail_at_commit: None,
        }
    }

    /// Adds a dormant control of `kind`, and returns its ID.
    pub(super) fn add_control(&mut self, kind: ControlKind) -> u64 {
        self.next_id += 1;
        self.controls.insert(
            self.next_id,
            Control {
                kind,
                phase: ControlPhase::Dormant,
            },
        );
        self.next_id
    }

    /// Whether a pending control lets a `LocalRunner` run move.
    pub(super) fn hold_open(&self) -> bool {
        self.controls.values().any(|control| {
            matches!(
                control.phase,
                ControlPhase::Pending {
                    drives_local: true,
                    ..
                }
            )
        })
    }

    /// Marks the control `id` as pending with `waker`, unless the run
    /// failed, the control is ready, or it no longer exists. Returns `None`
    /// then, else whether the registration opened the hold.
    ///
    /// Refusing after a run failure leaves no control without an outcome:
    /// a `wait_for_data` first polled after the failure decides from the
    /// stored operations before it registers, and a `CloudRunner` callback
    /// control after the failure needs the driver for nothing, because it
    /// makes its calls only once the target was published.
    pub(super) fn register(&mut self, id: u64, waker: &std::task::Waker) -> Option<bool> {
        if self.failure.is_some() {
            return None;
        }
        let was_open = self.hold_open();
        let control = self.controls.get_mut(&id)?;
        if matches!(control.phase, ControlPhase::Ready(_)) {
            return None;
        }
        control.phase = ControlPhase::Pending {
            waker: waker.clone(),
            drives_local: control.kind.drives_local(),
        };
        Some(!was_open && self.hold_open())
    }

    /// The wakers of every pending or in-flight control.
    pub(super) fn wakers(&self) -> Vec<std::task::Waker> {
        self.controls
            .values()
            .filter_map(|control| match &control.phase {
                ControlPhase::Pending { waker, .. } | ControlPhase::InFlight { waker, .. } => {
                    Some(waker.clone())
                }
                ControlPhase::Dormant | ControlPhase::Ready(_) => None,
            })
            .collect()
    }

    /// Records `err` as the run failure unless one is already recorded.
    /// Only a failure of the run itself is recorded here, never the failed
    /// Lambda call of a control. Returns the recorded failure, and the
    /// wakers of every pending or in-flight control, to wake after the lock
    /// is released.
    pub(super) fn record_failure(
        &mut self,
        err: TestRunnerError,
    ) -> (TestRunnerError, Vec<std::task::Waker>) {
        let winner = self.failure.get_or_insert(err).clone();
        (winner, self.wakers())
    }

    /// Commits the outcome of a control: the run failure when there is
    /// one, else `candidate`.
    ///
    /// `LocalRunner` controls and `wait_for_data` calls that started while
    /// the run was active return through a commit under the state lock, and
    /// any state change their outcome makes happens under the same lock
    /// after the commit. So a run failure recorded before the commit wins
    /// over every other outcome, and one recorded after it finds the
    /// control already decided. A `CloudRunner` callback control and a
    /// `wait_for_data` first polled after the run failed return their own
    /// outcome without a commit.
    pub(super) fn commit<T>(
        &mut self,
        candidate: Result<T, TestRunnerError>,
    ) -> Result<T, TestRunnerError> {
        #[cfg(test)]
        if let Some(err) = self.fail_at_commit.take() {
            self.record_failure(err);
        }
        match &self.failure {
            Some(err) => Err(err.clone()),
            None => candidate,
        }
    }

    /// [`TestRunnerError::RunFinished`] with `name` once the driver
    /// finished with a `TestResult`.
    pub(super) fn run_in_progress(&self, name: &'static str) -> Result<(), TestRunnerError> {
        match self.end {
            RunEnd::Running => Ok(()),
            RunEnd::Finished(_) => Err(TestRunnerError::RunFinished(name)),
        }
    }

    /// Takes the snapshot of the `wait_for_data` control `id`, when a
    /// recheck made it ready.
    pub(super) fn take_ready_data(&mut self, id: u64) -> Option<TestOperation> {
        let control = self.controls.get_mut(&id)?;
        match std::mem::replace(&mut control.phase, ControlPhase::Dormant) {
            ControlPhase::Ready(ControlValue::Data(op)) => Some(*op),
            other => {
                control.phase = other;
                None
            }
        }
    }

    /// Whether the `pause` control `id` saw its invocation end.
    pub(super) fn pause_done(&self, id: u64) -> bool {
        self.controls.get(&id).is_some_and(|control| {
            matches!(control.phase, ControlPhase::Ready(ControlValue::PauseDone))
        })
    }

    /// Makes ready every `wait_for_data` control whose status the stored
    /// operations reach, with the snapshot at this moment. Returns the
    /// wakers of the pending ones, to wake after the lock is released.
    pub(super) fn recheck(&mut self) -> Vec<std::task::Waker> {
        let ops = snapshot_of(&self.local.records);
        let mut wakers = Vec::new();
        for control in self.controls.values_mut() {
            let ControlKind::Data { selector, status } = &control.kind else {
                continue;
            };
            if matches!(control.phase, ControlPhase::Ready(_)) {
                continue;
            }
            let Some(op) = reached_operation(selector, *status, &ops) else {
                continue;
            };
            let phase = std::mem::replace(
                &mut control.phase,
                ControlPhase::Ready(ControlValue::Data(Box::new(op))),
            );
            if let ControlPhase::Pending { waker, .. } = phase {
                wakers.push(waker);
            }
        }
        wakers
    }

    /// The hold rule's admission of an invocation: when a control lets the
    /// run move, the run is not paused and no invocation runs, the
    /// invocation starts. Returns whether it started.
    pub(super) fn admit_invocation(&mut self) -> bool {
        let admitted =
            self.hold_open() && !self.local.paused && self.local.phase == LocalPhase::Idle;
        if admitted {
            self.local.phase = LocalPhase::Invoking {
                tokenless_requested: false,
            };
        }
        admitted
    }

    /// Takes a pause's request to answer the next checkpoint without a
    /// token.
    pub(super) fn take_tokenless(&mut self) -> bool {
        match &mut self.local.phase {
            LocalPhase::Invoking {
                tokenless_requested,
            } => std::mem::take(tokenless_requested),
            LocalPhase::Idle | LocalPhase::Closed => false,
        }
    }

    /// Records the end of an invocation: the run is idle, or closed for a
    /// terminal envelope, and every `pause` control is ready, so it stops
    /// letting the run move before the driver decides what to advance.
    /// Returns the wakers of the pending pauses.
    pub(super) fn end_invocation(&mut self, closing: bool) -> Vec<std::task::Waker> {
        if closing || self.local.phase == LocalPhase::Closed {
            self.local.phase = LocalPhase::Closed;
        } else {
            self.local.phase = LocalPhase::Idle;
        }
        let mut wakers = Vec::new();
        for control in self.controls.values_mut() {
            if !matches!(control.kind, ControlKind::Pause) {
                continue;
            }
            let phase = std::mem::replace(
                &mut control.phase,
                ControlPhase::Ready(ControlValue::PauseDone),
            );
            if let ControlPhase::Pending { waker, .. } = phase {
                wakers.push(waker);
            }
        }
        wakers
    }

    /// Ends the in-flight phase of the control `id` for the ticket
    /// `request`.
    pub(super) fn settle_in_flight(&mut self, id: u64, request: u64) {
        if let Some(control) = self.controls.get_mut(&id)
            && matches!(control.phase, ControlPhase::InFlight { request: r, .. } if r == request)
        {
            control.phase = ControlPhase::Dormant;
        }
    }

    /// Takes one poll from the `CloudRunner` budget. Returns `false` when it
    /// is spent.
    pub(super) fn take_poll(&mut self) -> bool {
        if self.budget_spent() {
            return false;
        }
        self.cloud.used += 1;
        true
    }

    /// Whether a configured `CloudRunner` budget is spent.
    pub(super) fn budget_spent(&self) -> bool {
        self.cloud
            .limit
            .is_some_and(|limit| self.cloud.used >= limit)
    }
}

/// The state cell of one run and the notification of its changes.
#[derive(Debug)]
pub(super) struct LiveState {
    inner: Mutex<RunState>,
    /// Notified when the hold opens and on `resume`, so held checkpoints
    /// and the driver's gates look again.
    pub(super) notify: tokio::sync::Notify,
}

impl LiveState {
    /// The cell of a new run whose top-level backend records are `records`.
    pub(super) fn new(records: BackendState) -> Self {
        Self {
            inner: Mutex::new(RunState::new(records)),
            notify: tokio::sync::Notify::new(),
        }
    }

    pub(super) fn lock(&self) -> std::sync::MutexGuard<'_, RunState> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Returns once `ready` holds for the state. `ready` runs under the
    /// lock, so a `ready` that also changes the state when it returns
    /// `true` makes the check and the change one transition.
    pub(super) async fn wait_until(&self, mut ready: impl FnMut(&mut RunState) -> bool) {
        loop {
            let notified = self.notify.notified();
            let mut notified = std::pin::pin!(notified);
            notified.as_mut().enable();
            if ready(&mut self.lock()) {
                return;
            }
            notified.await;
        }
    }

    /// Wakes every pending or in-flight control.
    pub(super) fn wake_all(&self) {
        let wakers = self.lock().wakers();
        wake(wakers);
    }
}

/// Wakes `wakers`. Callers extract them under the state lock and wake them
/// after it is released.
pub(super) fn wake(wakers: Vec<std::task::Waker>) {
    for waker in wakers {
        waker.wake();
    }
}

/// The driver of a run: the future that invokes and advances it. It ends
/// with the run's [`TestResult`], boxed as `Any`.
pub(super) type DriverFuture =
    Pin<Box<dyn Future<Output = Result<Box<dyn std::any::Any + Send>, TestRunnerError>> + Send>>;

/// Who polls the driver. It decides nothing that a control observes: the
/// run's end is in [`RunState::end`] and [`RunState::failure`].
pub(super) enum DriverSlot {
    /// Not being polled.
    Idle(DriverFuture),
    /// A control polls it. Another control that finds it here does not
    /// poll it; the driver's waker wakes that control when the driver can
    /// make progress.
    Polling,
    /// It finished, or the run failed before it started.
    Done,
}

/// The waker the driver is polled with: it wakes every pending control, so
/// whichever one polls next drives the run on.
pub(super) struct DriverWaker(Arc<LiveState>);

impl std::task::Wake for DriverWaker {
    fn wake(self: Arc<Self>) {
        self.0.wake_all();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.wake_all();
    }
}

/// What a run runs on.
pub(super) enum Engine {
    /// A `LocalRunner` run: its records are in [`RunState::local`].
    Local,
    /// A `CloudRunner` run.
    Cloud(Arc<dyn CloudControl>),
}

/// One run: its state cell, its driver and its engine. The [`RunHandle`]
/// and every [`OperationHandle`] hold it, so the driver is dropped with the
/// last handle. The state mutex and the driver mutex are never held
/// together.
pub(super) struct RunCore {
    pub(super) live: Arc<LiveState>,
    pub(super) driver: Mutex<DriverSlot>,
    pub(super) engine: Engine,
}

impl std::fmt::Debug for RunCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunCore")
            .field("live", &self.live)
            .finish_non_exhaustive()
    }
}

impl RunCore {
    /// A run whose driver is `driver`, not yet polled.
    pub(super) fn new(live: Arc<LiveState>, engine: Engine, driver: DriverFuture) -> Arc<Self> {
        Arc::new(Self {
            live,
            driver: Mutex::new(DriverSlot::Idle(driver)),
            engine,
        })
    }

    /// A run that failed with `err` before its driver started.
    pub(super) fn failed(live: Arc<LiveState>, engine: Engine, err: TestRunnerError) -> Arc<Self> {
        live.lock().failure = Some(err);
        Arc::new(Self {
            live,
            driver: Mutex::new(DriverSlot::Done),
            engine,
        })
    }

    /// Marks the control `id` as pending with `waker`, then polls the
    /// driver once if no other control polls it. A control that
    /// [`RunState::register`] refuses does not poll the driver: the run
    /// moves only while a control waits.
    pub(super) fn drive(&self, id: u64, waker: &std::task::Waker) {
        let Some(opened) = self.live.lock().register(id, waker) else {
            return;
        };
        if opened {
            self.live.notify.notify_waiters();
        }
        self.poll_driver();
    }

    /// Polls the driver once, unless it is being polled or has finished.
    pub(super) fn poll_driver(&self) {
        let mut driver = {
            let mut slot = self.lock_driver();
            match std::mem::replace(&mut *slot, DriverSlot::Polling) {
                DriverSlot::Idle(driver) => driver,
                other => {
                    *slot = other;
                    return;
                }
            }
        };
        let waker = std::task::Waker::from(Arc::new(DriverWaker(Arc::clone(&self.live))));
        let poll = driver
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker));
        let std::task::Poll::Ready(outcome) = poll else {
            *self.lock_driver() = DriverSlot::Idle(driver);
            return;
        };
        // The end is published in the state before the slot changes, so a
        // control never sees the driver done while the run still runs.
        let wakers = {
            let mut state = self.live.lock();
            match outcome {
                Ok(result) => {
                    state.end = RunEnd::Finished(Some(result));
                    state.wakers()
                }
                Err(err) => state.record_failure(err).1,
            }
        };
        *self.lock_driver() = DriverSlot::Done;
        drop(driver);
        wake(wakers);
    }

    pub(super) fn lock_driver(&self) -> std::sync::MutexGuard<'_, DriverSlot> {
        self.driver
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Forgets the control `id`: it no longer waits.
    pub(super) fn leave(&self, id: u64) {
        self.live.lock().controls.remove(&id);
    }

    /// The run failure, if the run has one.
    #[cfg(test)]
    pub(super) fn terminal_error(&self) -> Option<TestRunnerError> {
        self.live.lock().failure.clone()
    }

    /// The decision of a `wait_for_data` call when it is first polled, as
    /// one transition: `None` while the run has no failure, so the call
    /// waits as usual; else the operation `selector` names in the stored
    /// operations when it reached `status`, or else the run failure. It
    /// makes no Lambda call, polls no driver and spends no budget.
    pub(super) fn late_wait(
        &self,
        selector: &Selector,
        status: WaitingStatus,
    ) -> Option<Result<TestOperation, TestRunnerError>> {
        let state = self.live.lock();
        let err = state.failure.clone()?;
        let ops = match &self.engine {
            Engine::Local => snapshot_of(&state.local.records),
            Engine::Cloud(_) => state.cloud.fold.clone().unwrap_or_default(),
        };
        Some(reached_operation(selector, status, &ops).ok_or(err))
    }

    /// Whether the run ended, with a `TestResult` or a runner error.
    pub(super) fn ended(&self) -> bool {
        let state = self.live.lock();
        state.failure.is_some() || !matches!(state.end, RunEnd::Running)
    }

    /// [`RunState::commit`] under the lock.
    pub(super) fn commit<T>(
        &self,
        candidate: Result<T, TestRunnerError>,
    ) -> Result<T, TestRunnerError> {
        self.live.lock().commit(candidate)
    }

    /// The precondition of `pause` and `resume`, decided on `state`:
    /// [`TestRunnerError::Unsupported`] under [`CloudRunner`], whether the
    /// run failed or not, else [`RunState::run_in_progress`] committed
    /// against the run failure.
    pub(super) fn pause_precondition(
        &self,
        state: &mut RunState,
        name: &'static str,
    ) -> Result<(), TestRunnerError> {
        if matches!(self.engine, Engine::Cloud(_)) {
            return Err(TestRunnerError::Unsupported(CLOUD_PAUSE_UNSUPPORTED));
        }
        let precondition = state.run_in_progress(name);
        state.commit(precondition)
    }

    /// The operations of the run now: the backend's records under
    /// `LocalRunner`, the newest history read under `CloudRunner`.
    pub(super) fn operations(&self) -> Option<Vec<TestOperation>> {
        let state = self.live.lock();
        match &self.engine {
            Engine::Local => Some(snapshot_of(&state.local.records)),
            Engine::Cloud(_) => state.cloud.fold.clone(),
        }
    }
}

/// A control's ID. Dropping it forgets the control, so a dropped future no
/// longer lets the run move.
pub(super) struct Registration<'a> {
    pub(super) core: &'a RunCore,
    pub(super) id: u64,
}

impl<'a> Registration<'a> {
    pub(super) fn new(core: &'a RunCore, kind: ControlKind) -> Self {
        Self {
            id: core.live.lock().add_control(kind),
            core,
        }
    }
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        self.core.leave(self.id);
    }
}

/// Polls the run's driver as the control `registration` until `done`
/// returns a value. `done` runs before and after each poll of the driver,
/// with the context of the awaiting task.
pub(super) async fn drive_until<T>(
    registration: &Registration<'_>,
    mut done: impl FnMut(&mut std::task::Context<'_>) -> Option<T>,
) -> T {
    let core = registration.core;
    std::future::poll_fn(|cx| {
        if let Some(value) = done(cx) {
            return std::task::Poll::Ready(value);
        }
        core.drive(registration.id, cx.waker());
        done(cx).map_or(std::task::Poll::Pending, std::task::Poll::Ready)
    })
    .await
}

/// Which `CloudRunner` control makes a Lambda call, which decides how a run
/// failure affects the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ControlRole {
    /// A `wait_for_data` that started while the run was active. A run
    /// failure ends it: it starts no read after the failure, and a read in
    /// flight is dropped.
    Wait,
    /// A callback control with a published target. It is independent of a
    /// run failure: it starts its reads and calls after the failure, and
    /// keeps awaiting a call in flight.
    Callback,
}

/// Awaits `call`, the Lambda call with the ticket `request` of the control
/// `registration`, outside every lock.
///
/// While `call` is pending the control is in flight, and a run failure
/// recorded meanwhile wakes it. For [`ControlRole::Wait`] it then returns
/// the run failure and drops `call`, also when `call` would be ready on
/// that poll: a wait checks the run failure before it polls `call`. For [`ControlRole::Callback`] it keeps
/// awaiting `call`, and stays pending while `call` never settles. Otherwise
/// it returns the call's output, which the caller then finishes against the
/// state (see [`CloudShared::finish_history`] and
/// [`CloudShared::finish_call`]). While it waits, the control polls the
/// driver, as every control does.
pub(super) async fn await_in_flight<T>(
    registration: &Registration<'_>,
    role: ControlRole,
    request: u64,
    call: impl Future<Output = T>,
) -> Result<T, TestRunnerError> {
    let core = registration.core;
    let mut call = std::pin::pin!(call);
    std::future::poll_fn(|cx| {
        // A wait checks the run failure before it polls `call`, so a call
        // that would be ready on this poll is still dropped unconsumed.
        if role == ControlRole::Wait
            && let Some(err) = &core.live.lock().failure
        {
            return std::task::Poll::Ready(Err(err.clone()));
        }
        if let std::task::Poll::Ready(output) = call.as_mut().poll(cx) {
            return std::task::Poll::Ready(Ok(output));
        }
        {
            let mut state = core.live.lock();
            if role == ControlRole::Wait
                && let Some(err) = &state.failure
            {
                return std::task::Poll::Ready(Err(err.clone()));
            }
            if let Some(control) = state.controls.get_mut(&registration.id) {
                control.phase = ControlPhase::InFlight {
                    request,
                    waker: cx.waker().clone(),
                };
            }
        }
        core.poll_driver();
        std::task::Poll::Pending
    })
    .await
}

/// A run started by [`LocalRunner::run`] or [`CloudRunner::run`].
///
/// Awaiting the handle drives the run to its end and returns its
/// [`TestResult`]. Before that, the test can take [`OperationHandle`]s
/// with [`operation`](Self::operation) and the other selectors, wait for an
/// operation, answer its callback, and [`pause`](Self::pause) the run.
///
/// The run starts nothing until the test polls one of its futures, and
/// its driver runs in the test's own task. Dropping the handle and every
/// [`OperationHandle`] stops the run.
///
/// A `RunHandle` is `Send` and `Unpin`, so a test can await `&mut run` under
/// `tokio::time::timeout` and await it again later. It is not `Clone`:
///
/// ```compile_fail
/// use aws_durable_execution_sdk::test_util::RunHandle;
///
/// fn assert_clone<T: Clone>() {}
/// assert_clone::<RunHandle<u32>>();
/// ```
#[must_use = "a RunHandle does nothing unless awaited"]
#[non_exhaustive]
pub struct RunHandle<O> {
    pub(super) shared: Arc<RunCore>,
    pub(super) id: u64,
    pub(super) output: std::marker::PhantomData<fn() -> O>,
}

impl<O> std::fmt::Debug for RunHandle<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunHandle")
            .field("shared", &self.shared)
            .finish_non_exhaustive()
    }
}

impl<O> RunHandle<O> {
    /// The handle of the run `shared`.
    pub(super) fn from_shared(shared: Arc<RunCore>) -> Self {
        let id = shared.live.lock().add_control(ControlKind::Run);
        Self {
            id,
            shared,
            output: std::marker::PhantomData,
        }
    }

    /// The handle of the first operation named `name`, in
    /// [`TestResult::operations`] order.
    #[must_use]
    pub fn operation(&self, name: &str) -> OperationHandle {
        self.handle(Selector::Name(name.to_owned(), 0))
    }

    /// The handle of the operation at the 0-based `index`, in
    /// [`TestResult::operations`] order.
    #[must_use]
    pub fn operation_by_index(&self, index: usize) -> OperationHandle {
        self.handle(Selector::Index(index))
    }

    /// The handle of the `index`-th operation named `name` (0-based), in
    /// [`TestResult::operations`] order.
    #[must_use]
    pub fn operation_by_name_and_index(&self, name: &str, index: usize) -> OperationHandle {
        self.handle(Selector::Name(name.to_owned(), index))
    }

    /// The handle of the operation with the ID `id`.
    #[must_use]
    pub fn operation_by_id(&self, id: &str) -> OperationHandle {
        self.handle(Selector::Id(id.to_owned()))
    }

    pub(super) fn handle(&self, selector: Selector) -> OperationHandle {
        OperationHandle {
            shared: Arc::clone(&self.shared),
            selector,
        }
    }

    /// Pauses a [`LocalRunner`] run, as JS `pauseExecution`.
    ///
    /// When an invocation is in progress, the backend answers the next
    /// checkpoint it applies in that invocation without a token, so the
    /// invocation ends with that checkpoint kept, and `pause` returns once
    /// it ended. Otherwise `pause` returns at once.
    ///
    /// While the run is paused, waits and retry delays keep completing in
    /// due order and callback replies apply, but the next invocation waits
    /// for [`resume`](Self::resume). Calling `pause` again changes nothing.
    ///
    /// # Errors
    ///
    /// - Under [`LocalRunner`], the run's runner error, if the run failed.
    /// - Under [`LocalRunner`], [`TestRunnerError::RunFinished`] once the
    ///   run finished.
    /// - [`TestRunnerError::Unsupported`] under [`CloudRunner`], always,
    ///   also after the run failed.
    pub async fn pause(&self) -> Result<(), TestRunnerError> {
        let shared = &*self.shared;
        let registration = Registration::new(shared, ControlKind::Pause);
        let id = registration.id;
        {
            let mut state = shared.live.lock();
            shared.pause_precondition(&mut state, "pause")?;
            state.local.paused = true;
            // The end of this invocation makes the pause ready, in the
            // transition that ends it, before the driver decides what to
            // advance.
            match &mut state.local.phase {
                LocalPhase::Invoking {
                    tokenless_requested,
                } => *tokenless_requested = true,
                LocalPhase::Idle | LocalPhase::Closed => return Ok(()),
            }
        }
        drive_until(&registration, |_| {
            let mut state = shared.live.lock();
            (state.pause_done(id) || state.failure.is_some()).then(|| state.commit(Ok(())))
        })
        .await
    }

    /// Resumes a paused [`LocalRunner`] run, as JS `resumeExecution`. The
    /// invocation the pause held starts the next time the test awaits a
    /// runner future. Calling `resume` on a run that is not paused changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// - Under [`LocalRunner`], the run's runner error, if the run failed.
    /// - Under [`LocalRunner`], [`TestRunnerError::RunFinished`] once the
    ///   run finished.
    /// - [`TestRunnerError::Unsupported`] under [`CloudRunner`], always,
    ///   also after the run failed.
    // reason: `pause` and `resume` are both async, as in JS
    #[expect(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn resume(&self) -> Result<(), TestRunnerError> {
        let shared = &*self.shared;
        {
            let mut state = shared.live.lock();
            shared.pause_precondition(&mut state, "resume")?;
            state.local.paused = false;
        }
        shared.live.notify.notify_waiters();
        Ok(())
    }

    /// The run's end, once there is one: the run failure, or the
    /// `TestResult` the first time it is taken. A control's own failed call
    /// is never recorded as the run's end, so it replaces neither. After
    /// the `TestResult` was taken, this returns `None`, and polling the
    /// handle again stays pending.
    pub(super) fn take_end(&self) -> Option<Result<TestResult<O>, TestRunnerError>>
    where
        O: Send + 'static,
    {
        let mut state = self.shared.live.lock();
        if let Some(err) = &state.failure {
            return Some(Err(err.clone()));
        }
        match &mut state.end {
            RunEnd::Running => None,
            RunEnd::Finished(result) => result.take().map(|result| {
                // The driver of a `RunHandle<O>` is built by `run` with the
                // same `O`, so it ends with a `TestResult<O>`.
                #[expect(clippy::expect_used)]
                let result = result
                    .downcast::<TestResult<O>>()
                    .expect("the driver of a RunHandle<O> returns a TestResult<O>");
                Ok(*result)
            }),
        }
    }
}

impl<O: DeserializeOwned + Send + 'static> Future for RunHandle<O> {
    type Output = Result<TestResult<O>, TestRunnerError>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(end) = this.take_end() {
            this.shared.leave(this.id);
            return std::task::Poll::Ready(end);
        }
        this.shared.drive(this.id, cx.waker());
        match this.take_end() {
            Some(end) => {
                this.shared.leave(this.id);
                std::task::Poll::Ready(end)
            }
            None => std::task::Poll::Pending,
        }
    }
}

impl<O> Drop for RunHandle<O> {
    fn drop(&mut self) {
        self.shared.leave(self.id);
    }
}

/// One operation of a run, named by a selector of [`RunHandle`].
///
/// The selector is evaluated on the run's operations each time the handle is
/// used, so a handle can be taken before its operation exists. A handle
/// keeps working after the [`RunHandle`] was awaited or dropped.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperationHandle {
    pub(super) shared: Arc<RunCore>,
    pub(super) selector: Selector,
}

impl OperationHandle {
    /// The operation now, or `None` while no operation matches the handle.
    ///
    /// Under [`CloudRunner`] it is the operation in the last history the
    /// runner read successfully. A runner error does not clear the stored
    /// operations, so after a failed run this still returns the
    /// [`LocalRunner`] records or the newest history [`CloudRunner`] read.
    pub fn snapshot(&self) -> Option<TestOperation> {
        self.shared
            .operations()
            .and_then(|ops| self.selector.resolve(&ops).cloned())
    }

    /// Waits until the operation reaches `status` (see [`WaitingStatus`]),
    /// and returns the operation at that moment, as JS `waitForData`.
    ///
    /// It returns at once when the status is already reached. Under
    /// [`LocalRunner`] it checks again after every checkpoint, invocation
    /// end, callback reply and clock advance. Under [`CloudRunner`] it reads
    /// the whole history one `poll_interval` after the run knows the
    /// execution ARN, then one `poll_interval` after each read that does not
    /// reach `status`.
    ///
    /// The call starts when its future is first polled.
    ///
    /// # Errors
    ///
    /// - The run's runner error, when the run fails while the call is
    ///   pending.
    /// - The run's runner error, when the call is first polled after the run
    ///   failed and the stored operation does not reach `status`. When it
    ///   does, the call returns the stored operation instead, without a
    ///   Lambda call.
    /// - [`TestRunnerError::Aws`] when a [`CloudRunner`] history read of
    ///   this call fails. It belongs only to this call: the run goes on, and
    ///   other controls are not affected.
    /// - [`TestRunnerError::OperationNotReached`] when the execution ended,
    ///   or a configured [`CloudRunner::max_poll_attempts`] budget is spent,
    ///   before the operation reached `status`.
    /// - [`TestRunnerError::Unsupported`] under [`CloudRunner`] with
    ///   [`InvocationType::RequestResponse`].
    pub async fn wait_for_data(
        &self,
        status: WaitingStatus,
    ) -> Result<TestOperation, TestRunnerError> {
        let shared = &*self.shared;
        if let Engine::Cloud(cloud) = &shared.engine {
            return cloud
                .wait_for_data(&self.shared, &self.selector, status)
                .await;
        }
        // A call first polled after the run failed decides from the stored
        // records, without a commit, so reached data is not replaced by
        // the failure.
        if let Some(late) = shared.late_wait(&self.selector, status) {
            return late;
        }
        let registration = Registration::new(
            shared,
            ControlKind::Data {
                selector: self.selector.clone(),
                status,
            },
        );
        let id = registration.id;
        // Each outcome is decided and committed in one transition on the
        // run's state, the state a checkpoint is applied to. This call
        // started while the run was active, so a run failure recorded
        // before the commit wins over ready data.
        drive_until(&registration, |_| {
            let mut state = shared.live.lock();
            let ready = state.take_ready_data(id).or_else(|| {
                reached_operation(&self.selector, status, &snapshot_of(&state.local.records))
            });
            let candidate = match ready {
                Some(op) => Ok(op),
                None if state.failure.is_some() || matches!(state.end, RunEnd::Finished(_)) => {
                    Err(TestRunnerError::OperationNotReached(status))
                }
                None => return None,
            };
            Some(state.commit(candidate))
        })
        .await
    }

    /// Sends `reply` to the callback, as JS `sendCallbackSuccess` and
    /// `sendCallbackFailure`.
    ///
    /// The target is the operation when it is a callback, or the callback
    /// child of a `wait_for_callback` context. Under [`LocalRunner`] the
    /// reply applies at once, as a queued reply does (see
    /// [`LocalRunner`]), and needs no runner future to wait. Under
    /// [`CloudRunner`] the runner reads the history again, then calls
    /// `SendDurableExecutionCallbackSuccess` or
    /// `SendDurableExecutionCallbackFailure`.
    ///
    /// # Errors
    ///
    /// - Under [`LocalRunner`], the run's runner error, if the run failed.
    /// - Under [`CloudRunner`], the run's runner error, when the run fails
    ///   before the invoke published the execution ARN. Once it is
    ///   published, the control goes on after a run failure, and returns
    ///   its own outcome: success, or one of the errors below.
    /// - [`TestRunnerError::OperationNotFound`],
    ///   [`TestRunnerError::NotACallback`] and
    ///   [`TestRunnerError::CallbackNotPending`] when there is no pending
    ///   callback to answer. Under [`LocalRunner`], a callback no longer
    ///   waits for a reply once the run decided to end: after a terminal
    ///   envelope, or when nothing can advance the execution.
    /// - [`TestRunnerError::Unsupported`] under [`CloudRunner`] with
    ///   [`InvocationType::RequestResponse`], and for
    ///   [`CallbackReply::timeout`] and [`CallbackReply::heartbeat_timeout`].
    /// - [`TestRunnerError::Aws`] when a [`CloudRunner`] history read or
    ///   callback call of this control fails. It belongs only to this
    ///   control: the run goes on, and other controls are not affected.
    pub async fn send_callback(&self, reply: CallbackReply) -> Result<(), TestRunnerError> {
        let shared = &*self.shared;
        if let Engine::Cloud(cloud) = &shared.engine {
            return cloud
                .send_callback(&self.shared, &self.selector, reply)
                .await;
        }
        let wakers = reply_live(&mut shared.live.lock(), &self.selector, reply)?;
        wake(wakers);
        Ok(())
    }

    /// Sends a heartbeat to the callback, as JS `sendCallbackHeartbeat`.
    ///
    /// The target follows [`send_callback`](Self::send_callback). Under
    /// [`LocalRunner`] a heartbeat changes nothing: the runner runs no
    /// heartbeat timer. Under [`CloudRunner`] the runner reads the history
    /// again, then calls `SendDurableExecutionCallbackHeartbeat`.
    ///
    /// # Errors
    ///
    /// The errors of [`send_callback`](Self::send_callback), and
    /// [`TestRunnerError::NoHeartbeatTimeout`] under [`LocalRunner`] when
    /// the callback has no heartbeat timeout.
    pub async fn send_callback_heartbeat(&self) -> Result<(), TestRunnerError> {
        let shared = &*self.shared;
        if let Engine::Cloud(cloud) = &shared.engine {
            return cloud.send_heartbeat(&self.shared, &self.selector).await;
        }
        check_heartbeat(&mut shared.live.lock(), &self.selector)
    }
}

/// A boxed future of a [`CloudControl`] method.
pub(super) type ControlFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, TestRunnerError>> + Send + 'a>>;

/// The live controls of a [`CloudRunner`] run, with the Lambda client type
/// erased.
pub(super) trait CloudControl: Send + Sync {
    /// [`OperationHandle::wait_for_data`] under [`CloudRunner`].
    fn wait_for_data<'a>(
        &'a self,
        core: &'a RunCore,
        selector: &'a Selector,
        status: WaitingStatus,
    ) -> ControlFuture<'a, TestOperation>;

    /// [`OperationHandle::send_callback`] under [`CloudRunner`].
    fn send_callback<'a>(
        &'a self,
        core: &'a RunCore,
        selector: &'a Selector,
        reply: CallbackReply,
    ) -> ControlFuture<'a, ()>;

    /// [`OperationHandle::send_callback_heartbeat`] under [`CloudRunner`].
    fn send_heartbeat<'a>(
        &'a self,
        core: &'a RunCore,
        selector: &'a Selector,
    ) -> ControlFuture<'a, ()>;
}

/// How a history read starts (see [`CloudShared::begin_history`]).
#[derive(Debug)]
pub(super) enum HistoryStart {
    /// Read the history with this ticket.
    Read(u64),
    /// Read nothing: the run finished, or the budget is spent. The newest
    /// fold, if any.
    Done(Option<Vec<TestOperation>>),
}

/// The configuration of a [`CloudRunner`] run and its Lambda client. Every
/// mutable field of the run is in [`RunState::cloud`].
pub(super) struct CloudShared<A> {
    pub(super) invocation_type: InvocationType,
    pub(super) poll_interval: std::time::Duration,
    /// The client, set once by the driver before it publishes the execution
    /// ARN in [`CloudState::execution_arn`]. It decides nothing: a control
    /// reads it only after it read the ARN under the state lock.
    pub(super) api: std::sync::OnceLock<Arc<A>>,
    pub(super) live: Arc<LiveState>,
}

impl<A> CloudShared<A> {
    /// The configuration of a run of `runner` on the state cell `live`,
    /// which takes the runner's poll limit.
    pub(super) fn new(runner: &CloudRunner, live: Arc<LiveState>) -> Self {
        live.lock().cloud.limit = runner.max_poll_attempts;
        Self {
            invocation_type: runner.invocation_type,
            poll_interval: runner.poll_interval,
            api: std::sync::OnceLock::new(),
            live,
        }
    }

    /// Publishes the client and the execution ARN the invoke returned.
    pub(super) fn set_target(&self, api: Arc<A>, execution_arn: String) {
        // The driver sets the client once per run, before the ARN.
        let _ = self.api.set(api);
        self.live.lock().cloud.execution_arn = Some(execution_arn);
    }

    /// [`RunState::take_poll`].
    pub(super) fn take_poll(&self) -> bool {
        self.live.lock().take_poll()
    }

    /// [`RunState::budget_spent`].
    pub(super) fn budget_spent(&self) -> bool {
        self.live.lock().budget_spent()
    }

    /// The polls the budget spent.
    pub(super) fn used(&self) -> usize {
        self.live.lock().cloud.used
    }

    /// Starts a history read of a control with `role` as one transition.
    ///
    /// For [`ControlRole::Wait`]: the run failure, else
    /// [`HistoryStart::Done`] when the run finished or the budget is spent,
    /// else a new ticket. A pending wait starts no read after the run
    /// failed.
    ///
    /// For [`ControlRole::Callback`]: [`HistoryStart::Done`] when the budget
    /// is spent, else a new ticket, whether the run failed or not.
    ///
    /// A read that starts spends one poll of the budget, however many pages
    /// and attempts it makes, and also when it fails.
    pub(super) fn begin_history(&self, role: ControlRole) -> Result<HistoryStart, TestRunnerError> {
        let mut state = self.live.lock();
        if role == ControlRole::Wait {
            if let Some(err) = &state.failure {
                return Err(err.clone());
            }
            if matches!(state.end, RunEnd::Finished(_)) {
                return Ok(HistoryStart::Done(state.cloud.fold.clone()));
            }
        }
        if !state.take_poll() {
            return Ok(HistoryStart::Done(state.cloud.fold.clone()));
        }
        state.cloud.next_request += 1;
        Ok(HistoryStart::Read(state.cloud.next_request))
    }

    /// Finishes the history read `request` of the control `reader`, an ID
    /// and a role, or of the driver when `reader` is `None`, with
    /// `fetched`, as one transition.
    ///
    /// A wait whose run failed meanwhile returns the run failure and
    /// discards the read, so a pending wait stores no fold after the
    /// failure. Otherwise a failed read returns its error to the reader
    /// without recording it: the driver's error reaches
    /// [`RunCore::poll_driver`], which records it as the run failure, and a
    /// control's error belongs to that control. A failed read leaves the
    /// fold as it is. A successful read becomes the run's fold unless a
    /// newer read already finished, also when a callback control read it
    /// after the run failed, and its own events and operations are
    /// returned.
    pub(super) fn finish_history(
        &self,
        reader: Option<(u64, ControlRole)>,
        request: u64,
        fetched: Result<Vec<Event>, TestRunnerError>,
    ) -> Result<(Vec<Event>, Vec<TestOperation>), TestRunnerError> {
        let mut state = self.live.lock();
        if let Some((id, role)) = reader {
            state.settle_in_flight(id, request);
            if role == ControlRole::Wait
                && let Some(err) = &state.failure
            {
                return Err(err.clone());
            }
        }
        let events = fetched?;
        let ops = fold_history(&events);
        if request > state.cloud.fold_request {
            state.cloud.fold = Some(ops.clone());
            state.cloud.fold_request = request;
        }
        Ok((events, ops))
    }

    /// Starts a Lambda call that needs no budget, as one transition: a new
    /// ticket. The driver's final history read and a callback control's
    /// callback call use it; a callback control with a published target
    /// makes its call also after the run failed.
    pub(super) fn begin_request(&self) -> u64 {
        let mut state = self.live.lock();
        state.cloud.next_request += 1;
        state.cloud.next_request
    }

    /// Finishes the callback call `request` of the control `id` with
    /// `sent`, as one transition: it ends the in-flight phase and returns
    /// the call's own outcome. A failed call is neither recorded as the run
    /// failure nor committed against it.
    pub(super) fn finish_call(
        &self,
        id: u64,
        request: u64,
        sent: Result<(), TestRunnerError>,
    ) -> Result<(), TestRunnerError> {
        self.live.lock().settle_in_flight(id, request);
        sent
    }
}

impl<A: CloudApi + Send + Sync + 'static> CloudShared<A> {
    /// Polls the driver as the control `registration` until the invoke
    /// returned the execution ARN. A published target is returned also
    /// when the run failed after it was published. Without one, the run
    /// failure is returned.
    pub(super) async fn until_target(
        &self,
        registration: &Registration<'_>,
    ) -> Result<(Arc<A>, String), TestRunnerError> {
        drive_until(registration, |_| {
            let state = self.live.lock();
            if let Some(arn) = &state.cloud.execution_arn
                && let Some(api) = self.api.get()
            {
                return Some(Ok((Arc::clone(api), arn.clone())));
            }
            state.failure.clone().map(Err)
        })
        .await
    }

    /// Reads the whole history, as the driver, and finishes the read (see
    /// [`finish_history`](Self::finish_history)).
    pub(super) async fn driver_history(
        &self,
        api: &A,
        execution_arn: &str,
    ) -> Result<(Vec<Event>, Vec<TestOperation>), TestRunnerError> {
        let request = self.begin_request();
        let fetched = fetch_history(api, execution_arn, self.poll_interval).await;
        self.finish_history(None, request, fetched)
    }

    /// The operations a callback control reads: a fresh history read when
    /// the budget allows one, else the newest fold. Once the target is
    /// published, a run failure changes neither.
    pub(super) async fn control_operations(
        &self,
        registration: &Registration<'_>,
    ) -> Result<(Arc<A>, Vec<TestOperation>), TestRunnerError> {
        let (api, execution_arn) = self.until_target(registration).await?;
        match self.begin_history(ControlRole::Callback)? {
            HistoryStart::Read(request) => {
                let fetched = await_in_flight(
                    registration,
                    ControlRole::Callback,
                    request,
                    fetch_history(&*api, &execution_arn, self.poll_interval),
                )
                .await?;
                let (_, ops) = self.finish_history(
                    Some((registration.id, ControlRole::Callback)),
                    request,
                    fetched,
                )?;
                Ok((api, ops))
            }
            HistoryStart::Done(fold) => Ok((api, fold.ok_or(TestRunnerError::OperationNotFound)?)),
        }
    }

    /// Polls the driver as the control `registration` until `duration`
    /// elapsed or the run ended.
    pub(super) async fn drive_for(registration: &Registration<'_>, duration: std::time::Duration) {
        let core = registration.core;
        let mut sleep = std::pin::pin!(tokio::time::sleep(duration));
        drive_until(registration, |cx| {
            (sleep.as_mut().poll(cx).is_ready() || core.ended()).then_some(())
        })
        .await;
    }

    /// The outcome of a `wait_for_data` that started while the run was
    /// active; the wrapper commits it against the run failure.
    async fn wait_for_data_outcome(
        &self,
        core: &RunCore,
        selector: &Selector,
        status: WaitingStatus,
    ) -> Result<TestOperation, TestRunnerError> {
        let registration = Registration::new(core, ControlKind::Cloud);
        let (api, execution_arn) = self.until_target(&registration).await?;
        loop {
            // As the JS history poller, each read starts one poll interval
            // after the ARN is known or after the previous read.
            Self::drive_for(&registration, self.poll_interval).await;
            let request = match self.begin_history(ControlRole::Wait)? {
                HistoryStart::Read(request) => request,
                HistoryStart::Done(fold) => {
                    return reached_operation(selector, status, &fold.unwrap_or_default())
                        .ok_or(TestRunnerError::OperationNotReached(status));
                }
            };
            let fetched = await_in_flight(
                &registration,
                ControlRole::Wait,
                request,
                fetch_history(&*api, &execution_arn, self.poll_interval),
            )
            .await?;
            let (events, ops) =
                self.finish_history(Some((registration.id, ControlRole::Wait)), request, fetched)?;
            if let Some(op) = reached_operation(selector, status, &ops) {
                return Ok(op);
            }
            if terminal_execution_event(&events).is_some() {
                return Err(TestRunnerError::OperationNotReached(status));
            }
        }
    }

    /// The outcome of `send_callback`: the `Unsupported` checks first, then
    /// the target rule of [`until_target`](Self::until_target), then the
    /// control's own validation and calls.
    async fn send_callback_outcome(
        &self,
        core: &RunCore,
        selector: &Selector,
        reply: CallbackReply,
    ) -> Result<(), TestRunnerError> {
        if self.invocation_type == InvocationType::RequestResponse {
            return Err(TestRunnerError::Unsupported(
                SEND_CALLBACK_REQUEST_RESPONSE_UNSUPPORTED,
            ));
        }
        let call = match reply.kind {
            CallbackReplyKind::Success(payload) => CallbackCall::Success(payload),
            CallbackReplyKind::Failure(failure) => CallbackCall::Failure(failure),
            CallbackReplyKind::Timeout | CallbackReplyKind::HeartbeatTimeout => {
                return Err(TestRunnerError::Unsupported(
                    CLOUD_TIMEOUT_REPLY_UNSUPPORTED,
                ));
            }
        };
        let registration = Registration::new(core, ControlKind::Cloud);
        let (api, ops) = self.control_operations(&registration).await?;
        let target = callback_target(selector, &ops)?;
        if target.status != OperationStatus::Started {
            return Err(TestRunnerError::CallbackNotPending);
        }
        let id = callback_id(target)?;
        let request = self.begin_request();
        let sent = match call {
            CallbackCall::Success(payload) => {
                await_in_flight(
                    &registration,
                    ControlRole::Callback,
                    request,
                    api.send_callback_success(&id, payload.into_bytes()),
                )
                .await?
            }
            CallbackCall::Failure(failure) => {
                await_in_flight(
                    &registration,
                    ControlRole::Callback,
                    request,
                    api.send_callback_failure(&id, failure.error_object()),
                )
                .await?
            }
        };
        self.finish_call(registration.id, request, sent.map_err(aws_error))
    }

    /// The outcome of `send_callback_heartbeat`, decided as
    /// [`send_callback_outcome`](Self::send_callback_outcome) decides.
    async fn send_heartbeat_outcome(
        &self,
        core: &RunCore,
        selector: &Selector,
    ) -> Result<(), TestRunnerError> {
        if self.invocation_type == InvocationType::RequestResponse {
            return Err(TestRunnerError::Unsupported(
                SEND_CALLBACK_HEARTBEAT_REQUEST_RESPONSE_UNSUPPORTED,
            ));
        }
        let registration = Registration::new(core, ControlKind::Cloud);
        let (api, ops) = self.control_operations(&registration).await?;
        let id = callback_id(callback_target(selector, &ops)?)?;
        let request = self.begin_request();
        let sent = await_in_flight(
            &registration,
            ControlRole::Callback,
            request,
            api.send_callback_heartbeat(&id),
        )
        .await?;
        self.finish_call(registration.id, request, sent.map_err(aws_error))
    }
}

/// The callback ID of a callback target.
pub(super) fn callback_id(op: &TestOperation) -> Result<String, TestRunnerError> {
    op.callback
        .callback_id
        .clone()
        .ok_or(TestRunnerError::CallbackNotPending)
}

impl<A: CloudApi + Send + Sync + 'static> CloudControl for CloudShared<A> {
    fn wait_for_data<'a>(
        &'a self,
        core: &'a RunCore,
        selector: &'a Selector,
        status: WaitingStatus,
    ) -> ControlFuture<'a, TestOperation> {
        Box::pin(async move {
            if self.invocation_type == InvocationType::RequestResponse {
                return Err(TestRunnerError::Unsupported(
                    WAIT_FOR_DATA_REQUEST_RESPONSE_UNSUPPORTED,
                ));
            }
            // A call first polled after the run failed decides from the
            // newest fold; one that started while the run was active
            // commits its outcome against a run failure.
            if let Some(late) = core.late_wait(selector, status) {
                return late;
            }
            core.commit(self.wait_for_data_outcome(core, selector, status).await)
        })
    }

    fn send_callback<'a>(
        &'a self,
        core: &'a RunCore,
        selector: &'a Selector,
        reply: CallbackReply,
    ) -> ControlFuture<'a, ()> {
        // A callback control returns its own outcome: once the target is
        // published, it is independent of a run failure.
        Box::pin(self.send_callback_outcome(core, selector, reply))
    }

    fn send_heartbeat<'a>(
        &'a self,
        core: &'a RunCore,
        selector: &'a Selector,
    ) -> ControlFuture<'a, ()> {
        Box::pin(self.send_heartbeat_outcome(core, selector))
    }
}

/// The Lambda call a [`CallbackReply`] makes under [`CloudRunner`].
pub(super) enum CallbackCall {
    /// `SendDurableExecutionCallbackSuccess` with this payload.
    Success(String),
    /// `SendDurableExecutionCallbackFailure` with this error.
    Failure(CallbackFailure),
}
