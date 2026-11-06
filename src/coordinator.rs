//! The invocation coordinator: the one place that decides when an
//! invocation suspends.
//!
//! An invocation suspends when no SDK work can make progress and at least
//! one operation waits on the service. The coordinator counts the two
//! parts of that rule.
//!
//! - A **permit** stands for local work that can make progress. An
//!   activity permit is held by a task while the SDK polls it, by a task
//!   that is woken and waits to be polled, by a step body, a
//!   `wait_for_condition` check or a serdes call while it runs, and by
//!   blocking work while it runs. A checkpoint permit is held by an
//!   admitted checkpoint write until its response is merged and the
//!   operations it readied are woken.
//! - A **wait registration** stands for an operation that waits for its
//!   record to change: a wait, a callback result, an invoke, a
//!   `wait_for_condition` between checks, or a step between retries. A
//!   merged checkpoint response wakes the registrations of the operations
//!   it carries.
//!
//! Every release of a permit and every retired registration runs the
//! suspension check under the coordinator's lock. When no permit is held,
//! at least one registration is live, and no fatal error is recorded, the
//! phase becomes [`Phase::Suspending`] and the invocation driver is woken.
//! From then on no permit, registration, task or checkpoint write is
//! granted, so the decision cannot be undone.
//!
//! The lock is never held while a waker, a user future or a destructor of
//! user data runs. Wakers are collected under the lock and woken after it
//! is released.
//!
//! # Refresh scheduler
//!
//! A waiting operation can become ready on the service while other SDK
//! work runs, for example a wait whose end time passes during a long step.
//! A merged checkpoint response wakes it only when the response carries
//! it. So the invocation driver polls one refresh scheduler beside the
//! handler. The scheduler sends a checkpoint with no updates when a
//! registration is due, and the merge of the response wakes every
//! operation it readied, so the operation continues in the same
//! invocation.
//!
//! Each registration keeps its own schedule, as the JS SDK does:
//!
//! 1. The first refresh is due at the operation's end time (a wait's
//!    scheduled end, or a step's next attempt), or 1 s after the
//!    registration when the end time is unknown. No refresh is armed when
//!    it would leave less than 1 s before the Lambda deadline.
//! 2. When a refresh finds the status unchanged, the next one is due
//!    `min(n × 1 s, 10 s)` later, after the `n`-th unchanged refresh, and
//!    60 s later after more than 95 of them. The deadline rule of step 1
//!    applies again.
//! 3. When a refresh finds the status changed, the merge wakes the
//!    operation. When it waits again, the schedule starts again at step 1,
//!    from the record's new end time.
//! 4. When a due refresh would leave less than 1 s before the deadline,
//!    the scheduler sends nothing and does not arm that registration again.
//!
//! Registrations due at the same time share one request. A refresh starts
//! only while an activity or checkpoint permit is held, so a lone wait
//! never sends one: the invocation suspends instead. The refresh holds a
//! scheduler permit until its response is merged and the operations it
//! readied are woken. A wait, a callback result, an invoke or a
//! `wait_for_condition` started on its own task is refreshed only once its
//! handle is first polled. A step retry is refreshed whether or not it is
//! awaited.
//!
//! # Checkpoint token withdrawal
//!
//! A checkpoint response without a `CheckpointToken`, to a request that
//! holds no `EXECUTION` update, means the service accepts no further
//! checkpoints from this invocation. [`InvocationCoordinator::withdraw`]
//! sets the withdrawal latch, which closes admission at once, logs one
//! WARN, drops every queued write without resolving its caller, and then
//! moves the phase to [`Phase::Suspending`]. From then on no checkpoint
//! request is sent: a request that already waited for the
//! checkpoint-token lock sees the latch there and is dropped unresolved
//! too. The invocation answers `PENDING` on every exit that returns a
//! response or a Lambda error.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::Waker;
use std::time::{Duration, SystemTime};

use tokio::time::Instant;

use crate::driver::{FatalError, FatalSeverity};
use crate::engine::CheckpointStatus;
use crate::task::TaskTracker;

/// The delay before the first refresh of a registration whose end time is
/// unknown.
const INITIAL_REFRESH_DELAY: Duration = Duration::from_secs(1);
/// The time that must remain before the Lambda deadline for a refresh to
/// be armed or sent.
const MIN_REMAINING_TO_REFRESH: Duration = Duration::from_secs(1);
/// The longest delay between two unchanged refreshes, until
/// [`REFRESHES_BEFORE_EXTENDED_INTERVAL`] of them ran.
const REFRESH_INTERVAL_CEILING: Duration = Duration::from_secs(10);
/// The delay between two unchanged refreshes after
/// [`REFRESHES_BEFORE_EXTENDED_INTERVAL`] of them ran.
const EXTENDED_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
/// The number of unchanged refreshes after which the delay becomes
/// [`EXTENDED_REFRESH_INTERVAL`].
const REFRESHES_BEFORE_EXTENDED_INTERVAL: u32 = 95;

/// Whether a refresh `delay` from now leaves at least
/// [`MIN_REMAINING_TO_REFRESH`] before the deadline. Without a deadline it
/// always does.
fn fits_before_deadline(delay: Duration, remaining: Option<Duration>) -> bool {
    remaining.is_none_or(|r| delay.saturating_add(MIN_REMAINING_TO_REFRESH) <= r)
}

/// Delay before the first refresh of a new registration, or `None` when
/// no refresh may be armed.
fn first_refresh_delay(
    due_at: Option<SystemTime>,
    now: SystemTime,
    remaining: Option<Duration>,
) -> Option<Duration> {
    let delay = due_at.map_or(INITIAL_REFRESH_DELAY, |due| {
        due.duration_since(now).unwrap_or(Duration::ZERO)
    });
    fits_before_deadline(delay, remaining).then_some(delay)
}

/// Delay after `unchanged` refreshes found no change (`unchanged >= 1`).
fn refresh_delay(unchanged: u32) -> Duration {
    if unchanged > REFRESHES_BEFORE_EXTENDED_INTERVAL {
        EXTENDED_REFRESH_INTERVAL
    } else {
        Duration::from_secs(u64::from(unchanged)).min(REFRESH_INTERVAL_CEILING)
    }
}

/// The message of the WARN that [`InvocationCoordinator::withdraw`] logs.
pub(crate) const WITHDRAWAL_WARNING: &str = "Checkpoint response contained no CheckpointToken: \
     the service will accept no further checkpoints from this invocation. Suspending; the \
     execution continues on the next invocation.";

/// The phase of one invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Work may run, and permits and registrations are granted.
    Running,
    /// No SDK work can make progress and an operation waits. The driver
    /// ends the invocation with `PENDING`.
    Suspending,
    /// The drive ended. Teardown is aborting the remaining tasks.
    Finished,
}

/// The kind of a [`Permit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermitKind {
    /// Local work that can make progress: a polled or woken task, a step
    /// body, a check, a serdes call, blocking work.
    Activity,
    /// An admitted checkpoint write whose response is not merged yet.
    Checkpoint,
    /// A refresh of waiting operations, from its start until its response
    /// is merged and the operations it readied are woken.
    Scheduler,
}

/// One live wait registration.
#[derive(Debug)]
struct Registration {
    /// Woken when a merged response carries the operation.
    waker: Waker,
    /// The record's end time when the schedule last started.
    due_at: Option<SystemTime>,
    /// The record status the registration last saw.
    status: CheckpointStatus,
    /// The first-poll flag of a handle started on its own task. `None`
    /// means awaited.
    awaited: Option<Arc<AtomicBool>>,
    /// Whether the first refresh was ever computed.
    armed: bool,
    /// When the next refresh is due. `None` when none is armed.
    next_refresh: Option<Instant>,
    /// The number of refreshes in a row that found the status unchanged.
    unchanged: u32,
    /// Whether a refresh found the status changed, so the schedule starts
    /// again when the operation waits again.
    restart: bool,
}

impl Registration {
    fn is_awaited(&self) -> bool {
        self.awaited
            .as_ref()
            .is_none_or(|flag| flag.load(Ordering::SeqCst))
    }

    /// Computes the first refresh, as for a new registration.
    fn arm(&mut self, now: Instant, system_now: SystemTime, remaining: Option<Duration>) {
        self.armed = true;
        self.next_refresh =
            first_refresh_delay(self.due_at, system_now, remaining).map(|delay| now + delay);
    }
}

/// The registrations a fired scheduler must refresh.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DueRefresh {
    /// No registration is due.
    Nothing,
    /// Registrations were due, but less than
    /// [`MIN_REMAINING_TO_REFRESH`] remains before the deadline. They are
    /// not armed again, and nothing is sent.
    UnderFloor,
    /// These registrations are due. `first_wire_id` is the wire ID of the
    /// first of them, in key order.
    Due {
        keys: Vec<(String, u64)>,
        first_wire_id: String,
    },
}

/// The counts behind the suspension rule.
#[derive(Debug)]
struct State {
    phase: Phase,
    activity: usize,
    checkpoint: usize,
    scheduler: usize,
    /// Live wait registrations, keyed by the operation wire ID and a
    /// sequence number, so two waits on one operation stay apart.
    waits: BTreeMap<(String, u64), Registration>,
    next_seq: u64,
    fatal: Option<FatalError>,
    /// The first invocation-failing panic, kept by
    /// [`InvocationCoordinator::record_panic`].
    panic: Option<Box<dyn std::any::Any + Send + 'static>>,
    /// Whether [`InvocationCoordinator::record_panic`] can still keep a
    /// panic. Set from creation until [`InvocationCoordinator::take_panic`]
    /// closes it; it never reopens.
    panic_open: bool,
    driver: Option<Waker>,
    /// The Lambda deadline in tokio time, set when the drive starts.
    deadline: Option<Instant>,
    /// The withdrawal latch: set when a checkpoint response had no token
    /// (see [`InvocationCoordinator::withdraw`]). It is never cleared.
    withdrawn: bool,
    /// The write-failure latch: the classification of the first failed
    /// checkpoint write (see [`InvocationCoordinator::fail_write`]). It is
    /// never cleared.
    write_failure: Option<FatalError>,
    /// The wire ID of each registration [`InvocationCoordinator::wake_operations`]
    /// woke, in the order it woke them.
    #[cfg(test)]
    wake_log: Vec<String>,
}

impl State {
    /// Moves to `Suspending` when the rule holds, and returns the driver
    /// waker to wake after the lock is released.
    fn check_suspension(&mut self) -> Option<Waker> {
        let quiescent = self.activity == 0 && self.checkpoint == 0 && self.scheduler == 0;
        // After the withdrawal latch, `InvocationCoordinator::withdraw`
        // moves the phase itself, once the queued writes are dropped.
        if self.phase == Phase::Running
            && !self.withdrawn
            && quiescent
            && !self.waits.is_empty()
            && self.fatal.is_none()
            && self.panic.is_none()
        {
            self.phase = Phase::Suspending;
            return self.driver.clone();
        }
        None
    }

    /// Whether a permit or a wait registration can be taken: the phase is
    /// `Running` and the withdrawal latch is not set. The latch closes
    /// admission in the same lock that sets it, before the phase moves.
    fn admits(&self) -> bool {
        self.phase == Phase::Running && !self.withdrawn
    }

    /// Whether a buffered checkpoint write must stop waiting out its
    /// coalescing window and flush now.
    ///
    /// Only an admitted write holds a checkpoint permit, so with no
    /// activity permit and a live wait the invocation would suspend but for
    /// the buffered writes. No local work can join the batch then, so
    /// waiting out the window only delays the suspension. After the drive
    /// ended, no contributor can join either, and the window would only
    /// delay the end of the invocation.
    fn window_preempted(&self) -> bool {
        self.phase != Phase::Running || (self.activity == 0 && !self.waits.is_empty())
    }

    /// The time left until the deadline at `now`. `None` without one.
    fn remaining(&self, now: Instant) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(now))
    }
}

/// Decides suspension for one invocation. Every context of the invocation
/// shares one coordinator.
#[derive(Debug)]
pub(crate) struct InvocationCoordinator {
    state: Mutex<State>,
    tasks: Arc<TaskTracker>,
    /// Notified when [`State::window_preempted`] may have become true, so a
    /// flush waiting out its coalescing window re-checks it.
    window: tokio::sync::Notify,
    /// Notified when a registration's next refresh may have moved earlier
    /// than the one the scheduler sleeps until, or the earliest one
    /// retired.
    rearm: tokio::sync::Notify,
    /// A tokio instant and the wall-clock time it stands for, taken
    /// together when the coordinator was created. An end time is a
    /// wall-clock time. The scheduler converts it through this pair, so
    /// every refresh time follows tokio time, and two registrations with
    /// the same end time are due at the same instant.
    clock: (Instant, SystemTime),
}

impl InvocationCoordinator {
    /// Creates a coordinator in the `Running` phase with no permit and no
    /// registration.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                phase: Phase::Running,
                activity: 0,
                checkpoint: 0,
                scheduler: 0,
                waits: BTreeMap::new(),
                next_seq: 0,
                fatal: None,
                panic: None,
                panic_open: true,
                driver: None,
                deadline: None,
                withdrawn: false,
                write_failure: None,
                #[cfg(test)]
                wake_log: Vec::new(),
            }),
            tasks: Arc::new(TaskTracker::new()),
            window: tokio::sync::Notify::new(),
            rearm: tokio::sync::Notify::new(),
            clock: (Instant::now(), SystemTime::now()),
        })
    }

    /// The wall-clock time that the tokio instant `now` stands for (see
    /// the `clock` field).
    fn system_time_at(&self, now: Instant) -> SystemTime {
        let (anchor, anchor_system) = self.clock;
        anchor_system + now.saturating_duration_since(anchor)
    }

    /// Stores the Lambda deadline, `remaining` from now in tokio time.
    /// Called when the drive starts. Every later remaining-time check
    /// counts down from it.
    pub(crate) fn start_deadline(&self, remaining: Option<Duration>) {
        let now = Instant::now();
        self.lock().deadline = remaining.map(|d| now + d);
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Takes a permit of `kind`. Returns `None` once the phase is not
    /// `Running`, or once the withdrawal latch is set. A checkpoint permit
    /// is also refused once the write-failure latch is set (see
    /// [`Self::fail_write`]): the coalescer takes it under its buffer
    /// lock, so a join either precedes the latch, and the failure drops
    /// it, or is refused.
    pub(crate) fn acquire(self: &Arc<Self>, kind: PermitKind) -> Option<Permit> {
        let mut state = self.lock();
        if !state.admits() {
            return None;
        }
        match kind {
            PermitKind::Activity => state.activity += 1,
            PermitKind::Checkpoint => {
                if state.write_failure.is_some() {
                    return None;
                }
                state.checkpoint += 1;
            }
            PermitKind::Scheduler => {
                // A refresh starts only beside other local work. A lone
                // wait suspends instead.
                if state.activity == 0 && state.checkpoint == 0 {
                    return None;
                }
                state.scheduler += 1;
            }
        }
        drop(state);
        Some(Permit {
            coordinator: Arc::clone(self),
            kind,
        })
    }

    /// Registers a wait on the record of `wire_id`, woken through `waker`
    /// when a merged response carries that operation. Returns `None` once
    /// the phase is not `Running`, or once the withdrawal latch is set.
    ///
    /// `due_at` and `status` are the record's end time and status when the
    /// wait starts. `awaited` is the first-poll flag of a handle started on
    /// its own task, and `None` for an awaited operation. The first refresh
    /// is armed now when the operation is awaited, and by
    /// [`Self::notify_awaited`] otherwise.
    pub(crate) fn register_wait(
        self: &Arc<Self>,
        wire_id: &str,
        due_at: Option<SystemTime>,
        status: &CheckpointStatus,
        awaited: Option<Arc<AtomicBool>>,
        waker: &Waker,
    ) -> Option<WaitRegistration> {
        let now = Instant::now();
        let mut state = self.lock();
        if !state.admits() {
            return None;
        }
        let mut registration = Registration {
            waker: waker.clone(),
            due_at,
            status: status.clone(),
            awaited,
            armed: false,
            next_refresh: None,
            unchanged: 0,
            restart: false,
        };
        if registration.is_awaited() {
            registration.arm(now, self.system_time_at(now), state.remaining(now));
        }
        let armed = registration.next_refresh.is_some();
        let key = (wire_id.to_owned(), state.next_seq);
        state.next_seq += 1;
        state.waits.insert(key.clone(), registration);
        drop(state);
        if armed {
            self.rearm.notify_one();
        }
        Some(WaitRegistration {
            coordinator: Arc::clone(self),
            key,
        })
    }

    /// Registers an awaited wait with no end time on a `STARTED` record.
    #[cfg(test)]
    pub(crate) fn register_test_wait(
        self: &Arc<Self>,
        wire_id: &str,
        waker: &Waker,
    ) -> Option<WaitRegistration> {
        self.register_wait(wire_id, None, &CheckpointStatus::Started, None, waker)
    }

    /// Arms the first refresh of every registration whose first-poll flag
    /// is set and whose first refresh was never computed. A handle started
    /// on its own task calls it on its first poll.
    pub(crate) fn notify_awaited(&self) {
        let now = Instant::now();
        let armed = {
            let mut state = self.lock();
            let remaining = state.remaining(now);
            let system_now = self.system_time_at(now);
            let mut armed = false;
            for registration in state.waits.values_mut() {
                if !registration.armed && registration.is_awaited() {
                    registration.arm(now, system_now, remaining);
                    armed |= registration.next_refresh.is_some();
                }
            }
            armed
        };
        if armed {
            self.rearm.notify_one();
        }
    }

    /// The earliest next refresh among the live registrations.
    pub(crate) fn next_refresh_at(&self) -> Option<Instant> {
        self.lock()
            .waits
            .values()
            .filter_map(|registration| registration.next_refresh)
            .min()
    }

    /// Resolves when a registration's next refresh may have moved earlier,
    /// or the earliest one retired. A notification sent while nobody waits
    /// is kept for the next call.
    pub(crate) async fn rearmed(&self) {
        self.rearm.notified().await;
    }

    /// Returns the registrations due at `now`.
    ///
    /// When less than [`MIN_REMAINING_TO_REFRESH`] remains before the
    /// deadline, the due registrations are not armed again, and the result
    /// is [`DueRefresh::UnderFloor`].
    pub(crate) fn take_due(&self, now: Instant) -> DueRefresh {
        let mut state = self.lock();
        let under_floor = state
            .remaining(now)
            .is_some_and(|r| r < MIN_REMAINING_TO_REFRESH);
        let mut keys = Vec::new();
        for (key, registration) in &mut state.waits {
            if registration.next_refresh.is_some_and(|at| at <= now) {
                if under_floor {
                    registration.next_refresh = None;
                }
                keys.push(key.clone());
            }
        }
        match keys.first() {
            None => DueRefresh::Nothing,
            Some(_) if under_floor => DueRefresh::UnderFloor,
            Some((first_wire_id, _)) => DueRefresh::Due {
                first_wire_id: first_wire_id.clone(),
                keys,
            },
        }
    }

    /// Schedules the next refresh of each registration in `keys` after a
    /// refresh response was merged. `records` holds the status and end
    /// time of each operation's record after the merge, keyed by wire ID.
    ///
    /// A registration whose status is unchanged backs off by
    /// [`refresh_delay`]. A registration whose status changed keeps the new
    /// status and end time, and its operation is woken. The schedule starts
    /// again when the operation waits again (see
    /// [`WaitRegistration::resume_schedule`]). A registration that retired
    /// meanwhile is skipped.
    pub(crate) fn after_refresh(
        &self,
        keys: &[(String, u64)],
        records: &BTreeMap<String, (CheckpointStatus, Option<SystemTime>)>,
    ) {
        let now = Instant::now();
        let mut changed: Vec<Waker> = Vec::new();
        {
            let mut state = self.lock();
            let remaining = state.remaining(now);
            for key in keys {
                let Some(registration) = state.waits.get_mut(key) else {
                    continue;
                };
                match records.get(&key.0) {
                    Some((status, due_at)) if *status != registration.status => {
                        registration.status = status.clone();
                        registration.due_at = *due_at;
                        registration.unchanged = 0;
                        registration.next_refresh = None;
                        registration.restart = true;
                        changed.push(registration.waker.clone());
                    }
                    _ => {
                        registration.unchanged = registration.unchanged.saturating_add(1);
                        let delay = refresh_delay(registration.unchanged);
                        registration.next_refresh =
                            fits_before_deadline(delay, remaining).then(|| now + delay);
                    }
                }
            }
        }
        // The merge already woke these operations, and one of them can
        // have read its record before the restart was marked. So each is
        // woken again: its next poll either finds it ready, or restarts
        // the schedule. A wake of a task that is already queued takes no
        // second permit.
        for waker in changed {
            waker.wake();
        }
    }

    /// Wakes every registration on one of `wire_ids`, in key order. The
    /// wakers are collected under the lock and woken after it is released.
    pub(crate) fn wake_operations(&self, wire_ids: &[String]) {
        if wire_ids.is_empty() {
            return;
        }
        let wakers: Vec<Waker> = {
            #[cfg_attr(not(test), expect(unused_mut))]
            // reason: only the test build appends to the wake log.
            let mut state = self.lock();
            let woken: Vec<(String, Waker)> = state
                .waits
                .iter()
                .filter(|((id, _), _)| wire_ids.iter().any(|wanted| wanted == id))
                .map(|((id, _), registration)| (id.clone(), registration.waker.clone()))
                .collect();
            #[cfg(test)]
            state
                .wake_log
                .extend(woken.iter().map(|(id, _)| id.clone()));
            woken.into_iter().map(|(_, waker)| waker).collect()
        };
        for waker in wakers {
            waker.wake();
        }
    }

    /// The wire ID of each registration woken so far, in wake order.
    #[cfg(test)]
    pub(crate) fn wake_log(&self) -> Vec<String> {
        self.lock().wake_log.clone()
    }

    /// Returns the current phase.
    pub(crate) fn phase(&self) -> Phase {
        self.lock().phase
    }

    /// Returns the number of live wait registrations.
    #[cfg(test)]
    pub(crate) fn live_waits(&self) -> usize {
        self.lock().waits.len()
    }

    /// Returns the number of held permits of `kind`.
    #[cfg(test)]
    pub(crate) fn permits(&self, kind: PermitKind) -> usize {
        let state = self.lock();
        match kind {
            PermitKind::Activity => state.activity,
            PermitKind::Checkpoint => state.checkpoint,
            PermitKind::Scheduler => state.scheduler,
        }
    }

    /// Returns the invocation-wide tracker of the SDK tasks still running.
    pub(crate) fn tasks(&self) -> &Arc<TaskTracker> {
        &self.tasks
    }

    /// Moves to `Finished`, whatever the phase. Called when the handler
    /// resolves or panics, before the root task releases its permit, and
    /// by the driver when the drive ends.
    pub(crate) fn finish(&self) {
        self.lock().phase = Phase::Finished;
        self.window.notify_waiters();
    }

    /// Waits out a coalescing window of `delay`, or less.
    ///
    /// The window ends early when the invocation would suspend but for the
    /// buffered checkpoint writes, or when the drive ended (see
    /// [`State::window_preempted`]). Otherwise it ends after `delay`.
    pub(crate) async fn coalescing_window(&self, delay: Duration) {
        let mut sleep = std::pin::pin!(tokio::time::sleep(delay));
        loop {
            // `enable()` registers the waiter before the check, so a
            // notification between the check and the await is not lost.
            let mut notified = std::pin::pin!(self.window.notified());
            notified.as_mut().enable();
            if self.lock().window_preempted() {
                return;
            }
            tokio::select! {
                biased;
                () = &mut sleep => return,
                () = notified => {}
            }
        }
    }

    /// Handles a checkpoint response without a token: the service accepts
    /// no further checkpoints from this invocation.
    ///
    /// The caller has emitted the lifecycle events of the accepted request,
    /// discards the response's `NewExecutionState`, and leaves the
    /// request's contributors unresolved. It calls this under the
    /// checkpoint-token lock. This then does, in order:
    ///
    /// 1. It sets the withdrawal latch. The same lock closes admission: no
    ///    permit or wait registration is taken after it (see
    ///    [`State::admits`]).
    /// 2. It logs the WARN with `execution_arn` and `request_id`.
    /// 3. It calls `drop_queued`, which drops every queued write without
    ///    resolving its caller.
    /// 4. It moves the phase to [`Phase::Suspending`] when it is `Running`.
    ///
    /// A later call does nothing, and does not call `drop_queued`.
    pub(crate) fn withdraw(
        &self,
        execution_arn: &str,
        request_id: &str,
        drop_queued: impl FnOnce(),
    ) {
        {
            let mut state = self.lock();
            if state.withdrawn {
                return;
            }
            state.withdrawn = true;
        }
        tracing::warn!(
            executionArn = %execution_arn,
            requestId = %request_id,
            "{WITHDRAWAL_WARNING}"
        );
        drop_queued();
        let driver = {
            let mut state = self.lock();
            if state.phase == Phase::Running {
                state.phase = Phase::Suspending;
            }
            state.driver.clone()
        };
        self.after_change((driver, true));
    }

    /// Records a failed checkpoint write: the SDK sends no further
    /// checkpoint in this invocation.
    ///
    /// The checkpoint-token lock is held by the caller. This does, in
    /// order:
    ///
    /// 1. It sets the write-failure latch to `fatal`, the failure's
    ///    classification. The first failure wins, and the latch is never
    ///    cleared.
    /// 2. It records `fatal` in the fatal slot, where the first record
    ///    wins, so the failure decides the outcome of the drive.
    /// 3. It calls `drop_queued`, which drops every queued write without
    ///    resolving its caller.
    /// 4. It wakes the driver.
    ///
    /// The latch is set before `drop_queued` takes the queue locks, and
    /// every step that adds a write to a queue or hands a queued write a
    /// result without sending it (a writer that cannot be spawned) reads
    /// the latch under the queue's lock. So each such step happens wholly
    /// before the failure, and the failure then drops what it queued, or
    /// wholly after it, and it adds nothing and resolves nothing.
    pub(crate) fn fail_write(&self, fatal: FatalError, drop_queued: impl FnOnce()) {
        let driver = {
            let mut state = self.lock();
            if state.write_failure.is_none() {
                state.write_failure = Some(fatal.clone());
            }
            if state.fatal.is_none() {
                state.fatal = Some(fatal);
            }
            state.driver.clone()
        };
        drop_queued();
        if let Some(driver) = driver {
            driver.wake();
        }
    }

    /// Whether the write-failure latch is set (see [`Self::fail_write`]).
    pub(crate) fn write_failed(&self) -> bool {
        self.lock().write_failure.is_some()
    }

    /// The classification of the first failed checkpoint write, if any
    /// (see [`Self::fail_write`]).
    pub(crate) fn write_failure(&self) -> Option<FatalError> {
        self.lock().write_failure.clone()
    }

    /// Whether the withdrawal latch is set (see [`Self::withdraw`]).
    pub(crate) fn withdrawn(&self) -> bool {
        self.lock().withdrawn
    }

    /// Records an execution-fatal error (first record wins) and wakes the
    /// invocation driver.
    pub(crate) fn record_fatal(&self, error_type: String, error_message: String) {
        self.record(FatalError {
            error_type,
            error_message,
            severity: FatalSeverity::FailExecution,
        });
    }

    /// Records an invocation-fatal error (first record wins): the
    /// invocation dies with a Lambda runtime error, the service invokes the
    /// function again, and the execution stays alive. Used by the retryable
    /// checkpoint-failure path and by the serdes fault path
    /// ([`crate::context::DurableContext::serdes_fault`]).
    pub(crate) fn record_invocation_fault(&self, error_type: &'static str, error_message: String) {
        self.record(FatalError {
            error_type: error_type.to_owned(),
            error_message,
            severity: FatalSeverity::FailInvocation,
        });
    }

    /// Records `fatal` (first record wins) and wakes the invocation driver.
    pub(crate) fn record(&self, fatal: FatalError) {
        let driver = {
            let mut state = self.lock();
            if state.fatal.is_none() {
                state.fatal = Some(fatal);
            }
            state.driver.clone()
        };
        if let Some(driver) = driver {
            driver.wake();
        }
    }

    /// Keeps `payload` as the invocation's panic, and wakes the driver.
    ///
    /// Every invocation-failing panic is passed here, except one that user
    /// code awaits. Panic ownership is open from the coordinator's creation
    /// until [`Self::take_panic`]. While it is open, the first payload is
    /// kept and this returns `None`. A later payload, and every payload
    /// after ownership closed, is returned unchanged; the caller disposes of
    /// it with [`crate::task::discard_panic`].
    ///
    /// Keeping a panic ends the drive, as recording a fatal error does. A
    /// fatal error recorded before or after does not change what is kept;
    /// the driver decides between the two. The driver is woken as
    /// [`Self::record_fatal`] wakes it, after the lock is released. No
    /// payload is dropped here.
    pub(crate) fn record_panic(
        &self,
        payload: Box<dyn std::any::Any + Send + 'static>,
    ) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        let (driver, rejected) = {
            let mut state = self.lock();
            let rejected = if state.panic_open && state.panic.is_none() {
                state.panic = Some(payload);
                None
            } else {
                Some(payload)
            };
            (state.driver.clone(), rejected)
        };
        if let Some(driver) = driver {
            driver.wake();
        }
        rejected
    }

    /// Closes panic ownership and returns the kept panic, if any, in one
    /// step under the lock. Ownership never reopens, so a later
    /// [`Self::record_panic`] returns its payload.
    pub(crate) fn take_panic(&self) -> Option<Box<dyn std::any::Any + Send + 'static>> {
        let mut state = self.lock();
        state.panic_open = false;
        state.panic.take()
    }

    /// Whether a panic is kept.
    pub(crate) fn has_panic(&self) -> bool {
        self.lock().panic.is_some()
    }

    /// Returns the recorded fatal error, if any.
    pub(crate) fn fatal_error(&self) -> Option<FatalError> {
        self.lock().fatal.clone()
    }

    /// Registers (or refreshes) the invocation driver's waker.
    pub(crate) fn register_driver_waker(&self, waker: &Waker) {
        let mut state = self.lock();
        match &state.driver {
            Some(existing) if existing.will_wake(waker) => {}
            _ => state.driver = Some(waker.clone()),
        }
    }

    /// Whether the phase is `Running`, no permit is held and no wait is
    /// live, read under one lock.
    ///
    /// One lock matters. The root task moves the phase to `Finished` and
    /// then releases its permit. Two separate reads could see `Running`
    /// before the move and no permit after the release. That state never
    /// existed, and the driver would log a false untracked-deadline
    /// warning. So the method reads the phase and the counts together.
    pub(crate) fn running_and_idle(&self) -> bool {
        let state = self.lock();
        state.phase == Phase::Running
            && state.activity == 0
            && state.checkpoint == 0
            && state.scheduler == 0
            && state.waits.is_empty()
    }

    fn release(&self, kind: PermitKind) {
        let driver = {
            let mut state = self.lock();
            match kind {
                PermitKind::Activity => state.activity = state.activity.saturating_sub(1),
                PermitKind::Checkpoint => state.checkpoint = state.checkpoint.saturating_sub(1),
                PermitKind::Scheduler => state.scheduler = state.scheduler.saturating_sub(1),
            }
            (state.check_suspension(), state.window_preempted())
        };
        self.after_change(driver);
    }

    /// Wakes the driver when the suspension check moved the phase, and the
    /// flushes waiting out a coalescing window when it may have to end.
    fn after_change(&self, (driver, preempted): (Option<Waker>, bool)) {
        if let Some(driver) = driver {
            driver.wake();
        }
        if preempted {
            self.window.notify_waiters();
        }
    }

    fn retire(&self, key: &(String, u64)) {
        let (removed, driver) = {
            let mut state = self.lock();
            let removed = state.waits.remove(key);
            (removed, state.check_suspension())
        };
        // A retired registration that was armed may have been the earliest,
        // so the scheduler re-arms.
        let was_armed = removed
            .as_ref()
            .is_some_and(|registration| registration.next_refresh.is_some());
        // The waker is dropped outside the lock.
        drop(removed);
        if was_armed {
            self.rearm.notify_one();
        }
        // Retiring a registration removes a wait, so it never makes a
        // coalescing window end early that did not already end.
        self.after_change((driver, false));
    }
}

/// One held permit. Dropping it releases its count and runs the
/// suspension check.
#[derive(Debug)]
pub(crate) struct Permit {
    coordinator: Arc<InvocationCoordinator>,
    kind: PermitKind,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.coordinator.release(self.kind);
    }
}

/// One live wait registration. Dropping it retires the registration and
/// runs the suspension check.
#[derive(Debug)]
pub(crate) struct WaitRegistration {
    coordinator: Arc<InvocationCoordinator>,
    key: (String, u64),
}

impl WaitRegistration {
    /// Replaces the registration's waker when it would wake another task.
    pub(crate) fn update_waker(&self, waker: &Waker) {
        let old = {
            let mut state = self.coordinator.lock();
            match state.waits.get_mut(&self.key) {
                Some(existing) if !existing.waker.will_wake(waker) => {
                    Some(std::mem::replace(&mut existing.waker, waker.clone()))
                }
                _ => None,
            }
        };
        drop(old);
    }
}

impl WaitRegistration {
    /// Starts the refresh schedule again, as for a new registration, when a
    /// refresh found the status changed. The operation calls it when it
    /// re-reads its record and waits again. Does nothing otherwise.
    ///
    /// The schedule starts here rather than in the refresh itself. A
    /// refresh that readied the operation would otherwise arm a refresh at
    /// the record's end time, which has passed, and the scheduler could
    /// send it before the woken operation retires its registration.
    pub(crate) fn resume_schedule(&self) {
        let now = Instant::now();
        let armed = {
            let mut state = self.coordinator.lock();
            let remaining = state.remaining(now);
            let system_now = self.coordinator.system_time_at(now);
            match state.waits.get_mut(&self.key) {
                Some(registration) if registration.restart => {
                    // Only a due registration is refreshed, and only an
                    // awaited one is due, so this one is awaited.
                    registration.restart = false;
                    registration.arm(now, system_now, remaining);
                    registration.next_refresh.is_some()
                }
                _ => false,
            }
        };
        if armed {
            self.coordinator.rearm.notify_one();
        }
    }
}

impl Drop for WaitRegistration {
    fn drop(&mut self) {
        self.coordinator.retire(&self.key);
    }
}

/// Points in the checkpoint write path where a test runs code on the
/// thread that reaches them, so it can order that thread against a
/// failed write without sleeping.
#[cfg(test)]
pub(crate) mod race_points {
    use std::cell::RefCell;

    /// A point in the write path (see [`reach`]).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum RacePoint {
        /// A direct write is about to take the direct queue's lock to
        /// enter the queue.
        DirectQueueLock,
        /// A buffered join is about to take the buffer lock.
        BufferLock,
        /// The direct writer could not be spawned, and the write is about
        /// to take the direct queue's lock to drain the queue.
        DirectNoRuntimeDrain,
        /// A batch flusher could not be spawned, and the write is about to
        /// take the buffer lock to claim the batch.
        BatchNoRuntimeClaim,
    }

    type Hook = Box<dyn FnMut(RacePoint)>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Runs `hook` each time this thread reaches a point.
    pub(crate) fn set(hook: impl FnMut(RacePoint) + 'static) {
        HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs this thread's hook for `point`, if one is set.
    pub(crate) fn reach(point: RacePoint) {
        let hook = HOOK.with(|slot| slot.borrow_mut().take());
        if let Some(mut hook) = hook {
            hook(point);
            HOOK.with(|slot| {
                let mut slot = slot.borrow_mut();
                if slot.is_none() {
                    *slot = Some(hook);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    use super::{
        DueRefresh, InvocationCoordinator, PermitKind, Phase, first_refresh_delay, refresh_delay,
    };
    use crate::driver::{FatalError, FatalSeverity};
    use crate::engine::CheckpointStatus;
    use crate::task::test_support::{Schedule, ScheduleState};
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, SystemTime};

    /// Counts its wakes.
    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counting_waker() -> (Arc<CountingWaker>, Waker) {
        let inner = Arc::new(CountingWaker(AtomicUsize::new(0)));
        (Arc::clone(&inner), Waker::from(inner))
    }

    fn activity(coordinator: &InvocationCoordinator) -> usize {
        coordinator.permits(PermitKind::Activity)
    }

    #[tokio::test(start_paused = true)]
    async fn coalescing_window_runs_full_delay_without_wait() {
        let coordinator = InvocationCoordinator::new();
        let started = tokio::time::Instant::now();
        coordinator.coalescing_window(Duration::from_secs(10)).await;
        assert_eq!(started.elapsed(), Duration::from_secs(10));
    }

    #[tokio::test(start_paused = true)]
    // reason: the test runs the window on a plain Tokio task so it can
    // change the counts between the window's polls. The window starts no
    // task and runs no user code.
    #[expect(clippy::disallowed_methods)]
    async fn coalescing_window_rechecks_after_notification() {
        let coordinator = InvocationCoordinator::new();
        // The buffered write's permit keeps the invocation from suspending.
        let _buffered = coordinator
            .acquire(PermitKind::Checkpoint)
            .expect("running");
        let first = coordinator.acquire(PermitKind::Activity).expect("running");
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let window_coordinator = Arc::clone(&coordinator);
        let ended = Arc::new(AtomicUsize::new(0));
        let window_ended = Arc::clone(&ended);
        let window = tokio::spawn(async move {
            window_coordinator
                .coalescing_window(Duration::from_secs(3600))
                .await;
            window_ended.store(1, Ordering::SeqCst);
        });
        tokio::task::yield_now().await;
        // The release notifies the window. Another permit is taken before
        // the window runs, so its re-check finds local work and it waits on.
        drop(first);
        let second = coordinator.acquire(PermitKind::Activity).expect("running");
        tokio::task::yield_now().await;
        assert_eq!(ended.load(Ordering::SeqCst), 0, "the window waits on");
        let started = tokio::time::Instant::now();
        drop(second);
        window.await.expect("the window task does not panic");
        assert_eq!(ended.load(Ordering::SeqCst), 1);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "the release with a live wait ends the window"
        );
    }

    #[test]
    fn first_wake_takes_one_permit() {
        let coordinator = InvocationCoordinator::new();
        let schedule = Schedule::idle(&coordinator);
        let waker = schedule.waker();
        waker.wake_by_ref();
        assert_eq!(activity(&coordinator), 1);
        assert_eq!(schedule.state(), ScheduleState::Queued);
        waker.wake_by_ref();
        assert_eq!(activity(&coordinator), 1, "a duplicate wake takes nothing");
        assert_eq!(schedule.state(), ScheduleState::Queued);
    }

    #[test]
    fn wake_during_poll_keeps_permit() {
        let coordinator = InvocationCoordinator::new();
        let schedule = Schedule::idle(&coordinator);
        assert!(schedule.begin_poll());
        assert_eq!(schedule.state(), ScheduleState::Polling);
        schedule.waker().wake_by_ref();
        schedule.end_pending(false);
        assert_eq!(schedule.state(), ScheduleState::Queued);
        assert_eq!(activity(&coordinator), 1);

        let other = Schedule::idle(&coordinator);
        drop(schedule);
        assert_eq!(activity(&coordinator), 0);
        assert!(other.begin_poll());
        other.end_pending(false);
        assert_eq!(other.state(), ScheduleState::Idle);
        assert_eq!(activity(&coordinator), 0);
    }

    #[test]
    fn completion_wakes_consumer_before_release() {
        let coordinator = InvocationCoordinator::new();
        let consumer = Schedule::idle(&coordinator);
        let finished = Schedule::idle(&coordinator);
        let slot = crate::task::test_support::slot_with_waker::<u32>(&consumer.waker());
        assert!(finished.begin_poll());
        assert_eq!(activity(&coordinator), 1);
        // The finished task hands its outcome over while it still holds
        // its permit, and only then releases it.
        let permit = finished.end_ready();
        slot.deliver_value(7);
        assert_eq!(consumer.state(), ScheduleState::Queued);
        assert_eq!(activity(&coordinator), 2);
        drop(permit);
        assert_eq!(activity(&coordinator), 1);
        assert_eq!(consumer.state(), ScheduleState::Queued);
    }

    #[test]
    fn suspends_at_zero_with_a_live_wait() {
        let coordinator = InvocationCoordinator::new();
        let (driver_count, driver) = counting_waker();
        coordinator.register_driver_waker(&driver);
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        drop(permit);
        assert_eq!(coordinator.phase(), Phase::Suspending);
        assert_eq!(driver_count.0.load(Ordering::SeqCst), 1);
        assert!(coordinator.acquire(PermitKind::Activity).is_none());
        assert!(coordinator.acquire(PermitKind::Checkpoint).is_none());
        assert!(
            coordinator
                .register_test_wait("op", Waker::noop())
                .is_none()
        );
    }

    #[test]
    fn no_wait_no_suspension() {
        let coordinator = InvocationCoordinator::new();
        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        drop(permit);
        assert_eq!(coordinator.phase(), Phase::Running);
    }

    #[test]
    fn held_permits_block_suspension() {
        let coordinator = InvocationCoordinator::new();
        let wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let activity = coordinator.acquire(PermitKind::Activity).expect("running");
        let checkpoint = coordinator
            .acquire(PermitKind::Checkpoint)
            .expect("running");
        drop(activity);
        assert_eq!(coordinator.phase(), Phase::Running);
        drop(checkpoint);
        assert_eq!(coordinator.phase(), Phase::Suspending);
        drop(wait);
        assert_eq!(coordinator.phase(), Phase::Suspending);
    }

    #[test]
    fn fatal_blocks_suspension() {
        let coordinator = InvocationCoordinator::new();
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        coordinator.record_fatal("Error".to_owned(), "boom".to_owned());
        drop(permit);
        assert_eq!(coordinator.phase(), Phase::Running);
    }

    #[test]
    fn panic_blocks_suspension() {
        let coordinator = InvocationCoordinator::new();
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        assert!(coordinator.record_panic(Box::new("boom")).is_none());
        drop(permit);
        assert_eq!(coordinator.phase(), Phase::Running);
    }

    /// The first payload is kept. The second call returns the second
    /// payload unchanged, and `take_panic` returns the first.
    #[test]
    fn first_panic_is_kept() {
        let coordinator = InvocationCoordinator::new();
        assert!(!coordinator.has_panic());
        assert!(coordinator.record_panic(Box::new("first")).is_none());
        assert!(coordinator.has_panic());
        let second = coordinator
            .record_panic(Box::new("second"))
            .expect("the second payload is returned");
        assert_eq!(second.downcast_ref::<&str>(), Some(&"second"));
        let kept = coordinator.take_panic().expect("kept");
        assert_eq!(kept.downcast_ref::<&str>(), Some(&"first"));
        assert!(!coordinator.has_panic());
    }

    /// `take_panic` closes panic ownership for good: a payload recorded
    /// after it is returned, even though no panic is kept.
    #[test]
    fn record_after_take_returns_the_payload() {
        let coordinator = InvocationCoordinator::new();
        assert!(coordinator.take_panic().is_none());
        let late = coordinator
            .record_panic(Box::new("late"))
            .expect("ownership is closed");
        assert_eq!(late.downcast_ref::<&str>(), Some(&"late"));
        assert!(coordinator.take_panic().is_none());
    }

    /// A driver waker that records whether the coordinator's lock was free
    /// when it was woken.
    struct LockProbe {
        coordinator: std::sync::Weak<InvocationCoordinator>,
        woken_unlocked: AtomicUsize,
        woken_locked: AtomicUsize,
    }

    impl Wake for LockProbe {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            let coordinator = self.coordinator.upgrade().expect("the test holds it");
            if coordinator.state.try_lock().is_ok() {
                self.woken_unlocked.fetch_add(1, Ordering::SeqCst);
            } else {
                self.woken_locked.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Keeping a panic wakes the driver, after the lock was released.
    #[test]
    fn kept_panic_wakes_driver() {
        let coordinator = InvocationCoordinator::new();
        let probe = Arc::new(LockProbe {
            coordinator: Arc::downgrade(&coordinator),
            woken_unlocked: AtomicUsize::new(0),
            woken_locked: AtomicUsize::new(0),
        });
        coordinator.register_driver_waker(&Waker::from(Arc::clone(&probe)));
        assert!(coordinator.record_panic(Box::new("boom")).is_none());
        assert_eq!(probe.woken_unlocked.load(Ordering::SeqCst), 1);
        assert_eq!(probe.woken_locked.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn retired_last_wait_does_not_suspend() {
        let coordinator = InvocationCoordinator::new();
        let wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        drop(wait);
        assert_eq!(coordinator.phase(), Phase::Running);
        assert_eq!(coordinator.live_waits(), 0);
    }

    #[test]
    fn wake_operations_wakes_only_named_registrations() {
        let coordinator = InvocationCoordinator::new();
        let (a_count, a) = counting_waker();
        let (b_count, b) = counting_waker();
        let _a = coordinator.register_test_wait("a", &a).expect("running");
        let _b = coordinator.register_test_wait("b", &b).expect("running");
        coordinator.wake_operations(&["a".to_owned()]);
        assert_eq!(a_count.0.load(Ordering::SeqCst), 1);
        assert_eq!(b_count.0.load(Ordering::SeqCst), 0);
        coordinator.wake_operations(&[]);
        assert_eq!(a_count.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn update_waker_replaces_the_registered_waker() {
        let coordinator = InvocationCoordinator::new();
        let (a_count, a) = counting_waker();
        let (b_count, b) = counting_waker();
        let registration = coordinator.register_test_wait("op", &a).expect("running");
        registration.update_waker(&b);
        coordinator.wake_operations(&["op".to_owned()]);
        assert_eq!(a_count.0.load(Ordering::SeqCst), 0);
        assert_eq!(b_count.0.load(Ordering::SeqCst), 1);
        // The same waker again is kept.
        registration.update_waker(&b);
        coordinator.wake_operations(&["op".to_owned()]);
        assert_eq!(b_count.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn finish_refuses_permits_and_registrations() {
        let coordinator = InvocationCoordinator::new();
        coordinator.finish();
        assert_eq!(coordinator.phase(), Phase::Finished);
        assert!(coordinator.acquire(PermitKind::Activity).is_none());
        assert!(
            coordinator
                .register_test_wait("op", Waker::noop())
                .is_none()
        );
    }

    /// The root task moves the phase to `Finished` and then releases its
    /// permit. The state after the release is idle but `Finished`, so the
    /// untracked-deadline condition must not hold in it.
    #[test]
    fn running_and_idle_reads_phase_and_counts_together() {
        let coordinator = InvocationCoordinator::new();
        assert!(coordinator.running_and_idle());

        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        assert!(!coordinator.running_and_idle());
        let checkpoint = coordinator
            .acquire(PermitKind::Checkpoint)
            .expect("running");
        drop(permit);
        assert!(!coordinator.running_and_idle());
        drop(checkpoint);
        let registration = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        assert!(!coordinator.running_and_idle());
        drop(registration);
        assert!(coordinator.running_and_idle());

        // The root's ending: finish, then release the permit.
        let root = coordinator.acquire(PermitKind::Activity).expect("running");
        coordinator.finish();
        drop(root);
        assert_eq!(coordinator.permits(PermitKind::Activity), 0);
        assert_eq!(coordinator.live_waits(), 0);
        assert!(!coordinator.running_and_idle());
    }

    #[test]
    fn invocation_fault_is_recorded_once() {
        let coordinator = InvocationCoordinator::new();
        coordinator.record_invocation_fault("FirstError", "first".to_owned());
        coordinator.record_fatal("Error".to_owned(), "second".to_owned());
        let fatal = coordinator.fatal_error().expect("recorded");
        assert_eq!(fatal.error_message, "first");
        assert_eq!(fatal.severity, FatalSeverity::FailInvocation);
    }

    /// The first withdrawal sets the latch, moves `Running` to
    /// `Suspending` with a permit still held, and wakes the driver. A
    /// second one does nothing.
    #[test]
    fn withdraw_latches_once_and_suspends() {
        let coordinator = InvocationCoordinator::new();
        let (counter, waker) = counting_waker();
        coordinator.register_driver_waker(&waker);
        let _permit = coordinator.acquire(PermitKind::Activity).expect("running");
        assert!(!coordinator.withdrawn());
        coordinator.withdraw("arn", "req", || {});
        assert!(coordinator.withdrawn());
        assert_eq!(coordinator.phase(), Phase::Suspending);
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        coordinator.withdraw("arn", "req", || {});
        assert_eq!(
            counter.0.load(Ordering::SeqCst),
            1,
            "a second withdrawal does nothing"
        );
    }

    /// The steps of a withdrawal run in order: when the queued writes are
    /// dropped, the latch is set and admission is closed, but the phase is
    /// still `Running`. No suspension happens before the drop, even when
    /// the drop releases the last permit beside a live wait. A second
    /// withdrawal does not drop again.
    #[test]
    fn withdraw_closes_admission_before_dropping_and_suspends_after() {
        let coordinator = InvocationCoordinator::new();
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let permit = std::cell::RefCell::new(Some(
            coordinator
                .acquire(PermitKind::Checkpoint)
                .expect("running"),
        ));
        let drops = std::cell::Cell::new(0);
        coordinator.withdraw("arn", "req", || {
            drops.set(drops.get() + 1);
            assert!(coordinator.withdrawn(), "the latch is set first");
            assert!(
                coordinator.acquire(PermitKind::Activity).is_none(),
                "no activity permit after the latch"
            );
            assert!(
                coordinator.acquire(PermitKind::Checkpoint).is_none(),
                "no checkpoint permit after the latch"
            );
            assert!(
                coordinator.acquire(PermitKind::Scheduler).is_none(),
                "no scheduler permit after the latch"
            );
            assert!(
                coordinator
                    .register_test_wait("other", Waker::noop())
                    .is_none(),
                "no wait registration after the latch"
            );
            // Dropping the last permit beside a live wait does not suspend
            // here: the withdrawal moves the phase after the drop.
            drop(permit.borrow_mut().take());
            assert_eq!(coordinator.phase(), Phase::Running, "the phase moves last");
        });
        assert_eq!(drops.get(), 1);
        assert_eq!(coordinator.phase(), Phase::Suspending);
        coordinator.withdraw("arn", "req", || drops.set(drops.get() + 1));
        assert_eq!(drops.get(), 1, "a second withdrawal drops nothing");
    }

    /// A withdrawal after the drive ended keeps the phase `Finished`.
    #[test]
    fn withdraw_after_finish_keeps_finished() {
        let coordinator = InvocationCoordinator::new();
        coordinator.finish();
        coordinator.withdraw("arn", "req", || {});
        assert!(coordinator.withdrawn());
        assert_eq!(coordinator.phase(), Phase::Finished);
    }

    fn write_fatal(name: &str, severity: FatalSeverity) -> FatalError {
        FatalError {
            error_type: name.to_owned(),
            error_message: name.to_lowercase(),
            severity,
        }
    }

    /// A failed write sets the latch, records its classification in the
    /// fatal slot, drops the queued writes and wakes the driver. The first
    /// failure keeps the latch.
    #[test]
    fn fail_write_latches_records_drops_and_wakes() {
        let coordinator = InvocationCoordinator::new();
        let (counter, waker) = counting_waker();
        coordinator.register_driver_waker(&waker);
        assert!(!coordinator.write_failed());
        let drops = std::cell::Cell::new(0);
        coordinator.fail_write(write_fatal("First", FatalSeverity::FailInvocation), || {
            drops.set(drops.get() + 1);
        });
        assert!(coordinator.write_failed());
        assert_eq!(drops.get(), 1);
        assert_eq!(counter.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            coordinator.fatal_error().expect("recorded").error_type,
            "First"
        );
        coordinator.fail_write(write_fatal("Second", FatalSeverity::FailExecution), || {});
        assert_eq!(
            coordinator.write_failure().expect("latched").error_type,
            "First",
            "the first failure keeps the latch"
        );
        assert_eq!(
            coordinator.fatal_error().expect("recorded").error_type,
            "First"
        );
    }

    /// A failed write after another fatal error keeps that error in the
    /// fatal slot, and still sets the latch with its own classification.
    #[test]
    fn fail_write_after_another_fatal_keeps_the_fatal_slot() {
        let coordinator = InvocationCoordinator::new();
        coordinator.record(write_fatal("Usage", FatalSeverity::FailExecution));
        coordinator.fail_write(write_fatal("Write", FatalSeverity::FailExecution), || {});
        assert_eq!(
            coordinator.fatal_error().expect("recorded").error_type,
            "Usage"
        );
        assert_eq!(
            coordinator.write_failure().expect("latched").error_type,
            "Write"
        );
    }

    #[test]
    fn wake_precedes_checkpoint_release() {
        let coordinator = InvocationCoordinator::new();
        let schedule = Schedule::idle(&coordinator);
        let registration = coordinator
            .register_test_wait("op", &schedule.waker())
            .expect("running");
        let checkpoint = coordinator
            .acquire(PermitKind::Checkpoint)
            .expect("running");
        let log = crate::engine::CheckpointLog::empty();
        log.insert(
            "op".to_owned(),
            crate::engine::CheckpointRecord::with_status("op", CheckpointStatus::Succeeded),
        );
        // The merge path wakes the operations of the response, and only
        // then releases the response's checkpoint permit.
        coordinator.wake_operations(&["op".to_owned()]);
        drop(checkpoint);
        assert_eq!(schedule.state(), ScheduleState::Queued);
        assert_eq!(activity(&coordinator), 1);
        assert_eq!(coordinator.phase(), Phase::Running);
        drop(registration);
    }

    #[tokio::test]
    async fn checkpoint_write_after_suspension_is_rejected() {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = crate::context::DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        let coordinator = Arc::clone(ctx.coordinator());
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        drop(coordinator.acquire(PermitKind::Activity));
        assert_eq!(coordinator.phase(), Phase::Suspending);
        let update = aws_sdk_lambda::types::OperationUpdate::builder()
            .id("u")
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Start)
            .build()
            .expect("all required fields set");
        let written =
            tokio::time::timeout(Duration::from_secs(1), ctx.checkpoint_updates(vec![update]))
                .await;
        assert!(written.is_err(), "the rejected write never resolves");
        assert!(client.recorded_updates().is_empty(), "nothing was sent");
    }

    #[test]
    fn refresh_delay_table() {
        let table = [(1, 1), (2, 2), (10, 10), (11, 10), (95, 10), (96, 60)];
        for (unchanged, expected) in table {
            assert_eq!(
                refresh_delay(unchanged),
                Duration::from_secs(expected),
                "after {unchanged} unchanged refreshes"
            );
        }
    }

    #[test]
    fn first_refresh_rules() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let due = Some(now + Duration::from_secs(5));
        let secs = Duration::from_secs;
        assert_eq!(first_refresh_delay(due, now, None), Some(secs(5)));
        assert_eq!(
            first_refresh_delay(due, now, Some(Duration::from_millis(5_500))),
            None
        );
        assert_eq!(first_refresh_delay(due, now, Some(secs(6))), Some(secs(5)));
        assert_eq!(first_refresh_delay(None, now, None), Some(secs(1)));
        assert_eq!(
            first_refresh_delay(None, now, Some(Duration::from_millis(1_500))),
            None
        );
        assert_eq!(
            first_refresh_delay(Some(now - secs(2)), now, None),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn scheduler_permit_needs_other_local_work() {
        let coordinator = InvocationCoordinator::new();
        assert!(
            coordinator.acquire(PermitKind::Scheduler).is_none(),
            "no other permit is held"
        );
        let activity = coordinator.acquire(PermitKind::Activity).expect("running");
        let with_activity = coordinator.acquire(PermitKind::Scheduler);
        assert!(with_activity.is_some(), "beside an activity permit");
        drop(activity);
        drop(with_activity);
        let checkpoint = coordinator
            .acquire(PermitKind::Checkpoint)
            .expect("running");
        let with_checkpoint = coordinator.acquire(PermitKind::Scheduler);
        assert!(with_checkpoint.is_some(), "beside a checkpoint permit");
        coordinator.finish();
        assert!(
            coordinator.acquire(PermitKind::Scheduler).is_none(),
            "after the drive ended"
        );
        drop((checkpoint, with_checkpoint));
    }

    #[test]
    fn scheduler_permit_blocks_suspension() {
        let coordinator = InvocationCoordinator::new();
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        let activity = coordinator.acquire(PermitKind::Activity).expect("running");
        let scheduler = coordinator
            .acquire(PermitKind::Scheduler)
            .expect("beside work");
        drop(activity);
        assert_eq!(coordinator.phase(), Phase::Running);
        assert!(!coordinator.running_and_idle());
        drop(scheduler);
        assert_eq!(coordinator.phase(), Phase::Suspending);
    }

    #[tokio::test(start_paused = true)]
    async fn unawaited_registration_is_armed_by_notify_awaited() {
        let coordinator = InvocationCoordinator::new();
        let flag = Arc::new(AtomicBool::new(false));
        let _wait = coordinator
            .register_wait(
                "op",
                None,
                &CheckpointStatus::Started,
                Some(Arc::clone(&flag)),
                Waker::noop(),
            )
            .expect("running");
        assert_eq!(coordinator.next_refresh_at(), None, "not awaited yet");
        coordinator.notify_awaited();
        assert_eq!(coordinator.next_refresh_at(), None, "the flag is not set");
        flag.store(true, Ordering::SeqCst);
        let polled = tokio::time::Instant::now();
        coordinator.notify_awaited();
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(polled + Duration::from_secs(1))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn due_registration_under_the_floor_is_not_armed_again() {
        let coordinator = InvocationCoordinator::new();
        coordinator.start_deadline(Some(Duration::from_secs(3)));
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        assert_eq!(
            coordinator.take_due(tokio::time::Instant::now()),
            DueRefresh::Nothing
        );
        tokio::time::advance(Duration::from_millis(2_500)).await;
        assert_eq!(
            coordinator.take_due(tokio::time::Instant::now()),
            DueRefresh::UnderFloor
        );
        assert_eq!(coordinator.next_refresh_at(), None);
    }

    #[tokio::test(start_paused = true)]
    async fn after_refresh_backs_off_restarts_and_skips_retired() {
        let coordinator = InvocationCoordinator::new();
        let (woken, waker) = counting_waker();
        let unchanged = coordinator
            .register_test_wait("same", Waker::noop())
            .expect("running");
        let changed = coordinator
            .register_test_wait("changed", &waker)
            .expect("running");
        let retired = coordinator
            .register_test_wait("retired", Waker::noop())
            .expect("running");
        tokio::time::advance(Duration::from_secs(1)).await;
        let (keys, first_wire_id) = match coordinator.take_due(tokio::time::Instant::now()) {
            DueRefresh::Due {
                keys,
                first_wire_id,
            } => (keys, first_wire_id),
            other => unreachable!("three registrations are due, got {other:?}"),
        };
        assert_eq!(keys.len(), 3);
        assert_eq!(first_wire_id, "changed", "key order");
        drop(retired);
        let records = BTreeMap::from([
            ("same".to_owned(), (CheckpointStatus::Started, None)),
            ("changed".to_owned(), (CheckpointStatus::Pending, None)),
        ]);
        coordinator.after_refresh(&keys, &records);
        let after = tokio::time::Instant::now();
        assert_eq!(
            woken.0.load(Ordering::SeqCst),
            1,
            "the changed operation is woken"
        );
        // Only the unchanged registration is armed: 1 s after its first
        // unchanged refresh.
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(after + Duration::from_secs(1))
        );
        // The changed one restarts when its operation waits again.
        changed.resume_schedule();
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(after + Duration::from_secs(1))
        );
        drop(unchanged);
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(after + Duration::from_secs(1)),
            "the restarted schedule is due 1 s after the restart"
        );
        // A second call, later, finds nothing to restart.
        tokio::time::advance(Duration::from_millis(500)).await;
        changed.resume_schedule();
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(after + Duration::from_secs(1))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn after_refresh_stops_arming_near_the_deadline() {
        let coordinator = InvocationCoordinator::new();
        coordinator.start_deadline(Some(Duration::from_millis(2_500)));
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        tokio::time::advance(Duration::from_secs(1)).await;
        let DueRefresh::Due { keys, .. } = coordinator.take_due(tokio::time::Instant::now()) else {
            unreachable!("the registration is due");
        };
        let records = BTreeMap::from([("op".to_owned(), (CheckpointStatus::Started, None))]);
        coordinator.after_refresh(&keys, &records);
        // 1.5 s remain, and the next refresh would be due in 1 s.
        assert_eq!(coordinator.next_refresh_at(), None);
    }

    #[tokio::test(start_paused = true)]
    async fn notify_awaited_keeps_an_armed_registration() {
        let coordinator = InvocationCoordinator::new();
        let registered = tokio::time::Instant::now();
        let _wait = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        tokio::time::advance(Duration::from_millis(500)).await;
        coordinator.notify_awaited();
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(registered + Duration::from_secs(1))
        );
    }

    #[test]
    fn scheduler_permit_is_not_idle() {
        let coordinator = InvocationCoordinator::new();
        let activity = coordinator.acquire(PermitKind::Activity).expect("running");
        let scheduler = coordinator
            .acquire(PermitKind::Scheduler)
            .expect("beside work");
        drop(activity);
        assert!(!coordinator.running_and_idle());
        assert_eq!(coordinator.permits(PermitKind::Scheduler), 1);
        drop(scheduler);
        assert_eq!(coordinator.permits(PermitKind::Scheduler), 0);
        assert!(coordinator.running_and_idle());
    }

    #[tokio::test(start_paused = true)]
    async fn rearm_is_notified_when_an_armed_registration_changes() {
        let coordinator = InvocationCoordinator::new();
        // No armed registration: no notification.
        let flag = Arc::new(AtomicBool::new(false));
        let unarmed = coordinator
            .register_wait(
                "op",
                None,
                &CheckpointStatus::Started,
                Some(flag),
                Waker::noop(),
            )
            .expect("running");
        drop(unarmed);
        let quiet = tokio::time::timeout(Duration::from_millis(10), coordinator.rearmed()).await;
        assert!(
            quiet.is_err(),
            "neither the registration nor its retirement notifies"
        );
        // An armed registration notifies, and so does its retirement.
        let armed = coordinator
            .register_test_wait("op", Waker::noop())
            .expect("running");
        coordinator.rearmed().await;
        drop(armed);
        coordinator.rearmed().await;
    }

    #[test]
    fn wake_log_records_woken_registrations_in_key_order() {
        let coordinator = InvocationCoordinator::new();
        let _b = coordinator
            .register_test_wait("b", Waker::noop())
            .expect("running");
        let _a = coordinator
            .register_test_wait("a", Waker::noop())
            .expect("running");
        coordinator.wake_operations(&["b".to_owned(), "a".to_owned(), "c".to_owned()]);
        assert_eq!(coordinator.wake_log(), vec!["a".to_owned(), "b".to_owned()]);
    }
}
