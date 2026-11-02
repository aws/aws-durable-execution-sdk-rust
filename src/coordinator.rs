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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::Waker;

use crate::driver::{FatalError, FatalSeverity};
use crate::task::TaskTracker;

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
}

/// The counts behind the suspension rule.
#[derive(Debug)]
struct State {
    phase: Phase,
    activity: usize,
    checkpoint: usize,
    /// Live wait registrations, keyed by the operation wire ID and a
    /// sequence number, so two waits on one operation stay apart.
    waits: BTreeMap<(String, u64), Waker>,
    next_seq: u64,
    fatal: Option<FatalError>,
    driver: Option<Waker>,
}

impl State {
    /// Moves to `Suspending` when the rule holds, and returns the driver
    /// waker to wake after the lock is released.
    fn check_suspension(&mut self) -> Option<Waker> {
        let quiescent = self.activity == 0 && self.checkpoint == 0;
        if self.phase == Phase::Running
            && quiescent
            && !self.waits.is_empty()
            && self.fatal.is_none()
        {
            self.phase = Phase::Suspending;
            return self.driver.clone();
        }
        None
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
                waits: BTreeMap::new(),
                next_seq: 0,
                fatal: None,
                driver: None,
            }),
            tasks: Arc::new(TaskTracker::new()),
            window: tokio::sync::Notify::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Takes a permit of `kind`. Returns `None` once the phase is not
    /// `Running`.
    pub(crate) fn acquire(self: &Arc<Self>, kind: PermitKind) -> Option<Permit> {
        let mut state = self.lock();
        if state.phase != Phase::Running {
            return None;
        }
        match kind {
            PermitKind::Activity => state.activity += 1,
            PermitKind::Checkpoint => state.checkpoint += 1,
        }
        drop(state);
        Some(Permit {
            coordinator: Arc::clone(self),
            kind,
        })
    }

    /// Registers a wait on the record of `wire_id`, woken through `waker`
    /// when a merged response carries that operation. Returns `None` once
    /// the phase is not `Running`.
    pub(crate) fn register_wait(
        self: &Arc<Self>,
        wire_id: &str,
        waker: &Waker,
    ) -> Option<WaitRegistration> {
        let mut state = self.lock();
        if state.phase != Phase::Running {
            return None;
        }
        let key = (wire_id.to_owned(), state.next_seq);
        state.next_seq += 1;
        state.waits.insert(key.clone(), waker.clone());
        drop(state);
        Some(WaitRegistration {
            coordinator: Arc::clone(self),
            key,
        })
    }

    /// Wakes every registration on one of `wire_ids`, in key order. The
    /// wakers are collected under the lock and woken after it is released.
    pub(crate) fn wake_operations(&self, wire_ids: &[String]) {
        if wire_ids.is_empty() {
            return;
        }
        let wakers: Vec<Waker> = {
            let state = self.lock();
            state
                .waits
                .iter()
                .filter(|((id, _), _)| wire_ids.iter().any(|wanted| wanted == id))
                .map(|(_, waker)| waker.clone())
                .collect()
        };
        for waker in wakers {
            waker.wake();
        }
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
    pub(crate) async fn coalescing_window(&self, delay: std::time::Duration) {
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
    pub(crate) fn record_invocation_fault(&self, error_message: String) {
        self.record(FatalError {
            error_type: crate::error::CHECKPOINT_FAILED_ERROR_TYPE.to_owned(),
            error_message,
            severity: FatalSeverity::FailInvocation,
        });
    }

    fn record(&self, fatal: FatalError) {
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
            && state.waits.is_empty()
    }

    fn release(&self, kind: PermitKind) {
        let driver = {
            let mut state = self.lock();
            match kind {
                PermitKind::Activity => state.activity = state.activity.saturating_sub(1),
                PermitKind::Checkpoint => state.checkpoint = state.checkpoint.saturating_sub(1),
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
        // The waker is dropped outside the lock.
        drop(removed);
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
                Some(existing) if !existing.will_wake(waker) => {
                    Some(std::mem::replace(existing, waker.clone()))
                }
                _ => None,
            }
        };
        drop(old);
    }
}

impl Drop for WaitRegistration {
    fn drop(&mut self) {
        self.coordinator.retire(&self.key);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Wake, Waker};

    use super::{InvocationCoordinator, PermitKind, Phase};
    use crate::task::test_support::{Schedule, ScheduleState};

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
        coordinator
            .coalescing_window(std::time::Duration::from_secs(10))
            .await;
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(10));
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
            .register_wait("op", Waker::noop())
            .expect("running");
        let window_coordinator = Arc::clone(&coordinator);
        let ended = Arc::new(AtomicUsize::new(0));
        let window_ended = Arc::clone(&ended);
        let window = tokio::spawn(async move {
            window_coordinator
                .coalescing_window(std::time::Duration::from_secs(3600))
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
            started.elapsed() < std::time::Duration::from_secs(1),
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
            .register_wait("op", Waker::noop())
            .expect("running");
        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        drop(permit);
        assert_eq!(coordinator.phase(), Phase::Suspending);
        assert_eq!(driver_count.0.load(Ordering::SeqCst), 1);
        assert!(coordinator.acquire(PermitKind::Activity).is_none());
        assert!(coordinator.acquire(PermitKind::Checkpoint).is_none());
        assert!(coordinator.register_wait("op", Waker::noop()).is_none());
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
            .register_wait("op", Waker::noop())
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
            .register_wait("op", Waker::noop())
            .expect("running");
        let permit = coordinator.acquire(PermitKind::Activity).expect("running");
        coordinator.record_fatal("Error".to_owned(), "boom".to_owned());
        drop(permit);
        assert_eq!(coordinator.phase(), Phase::Running);
    }

    #[test]
    fn retired_last_wait_does_not_suspend() {
        let coordinator = InvocationCoordinator::new();
        let wait = coordinator
            .register_wait("op", Waker::noop())
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
        let _a = coordinator.register_wait("a", &a).expect("running");
        let _b = coordinator.register_wait("b", &b).expect("running");
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
        let registration = coordinator.register_wait("op", &a).expect("running");
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
        assert!(coordinator.register_wait("op", Waker::noop()).is_none());
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
            .register_wait("op", Waker::noop())
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
        coordinator.record_invocation_fault("first".to_owned());
        coordinator.record_fatal("Error".to_owned(), "second".to_owned());
        let fatal = coordinator.fatal_error().expect("recorded");
        assert_eq!(fatal.error_message, "first");
        assert_eq!(fatal.severity, crate::driver::FatalSeverity::FailInvocation);
    }

    #[test]
    fn wake_precedes_checkpoint_release() {
        let coordinator = InvocationCoordinator::new();
        let schedule = Schedule::idle(&coordinator);
        let registration = coordinator
            .register_wait("op", &schedule.waker())
            .expect("running");
        let checkpoint = coordinator
            .acquire(PermitKind::Checkpoint)
            .expect("running");
        let log = crate::engine::CheckpointLog::empty();
        log.insert(
            "op".to_owned(),
            crate::engine::CheckpointRecord::with_status(
                "op",
                crate::engine::CheckpointStatus::Succeeded,
            ),
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
            .register_wait("op", Waker::noop())
            .expect("running");
        drop(coordinator.acquire(PermitKind::Activity));
        assert_eq!(coordinator.phase(), Phase::Suspending);
        let update = aws_sdk_lambda::types::OperationUpdate::builder()
            .id("u")
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Start)
            .build()
            .expect("all required fields set");
        let written = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            ctx.checkpoint_updates(vec![update]),
        )
        .await;
        assert!(written.is_err(), "the rejected write never resolves");
        assert!(client.recorded_updates().is_empty(), "nothing was sent");
    }
}
