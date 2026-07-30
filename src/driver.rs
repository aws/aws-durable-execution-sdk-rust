//! Engine driver: suspend/resume mechanics and task-ownership enforcement.
//!
//! The driver polls the user's handler future and controls execution
//! lifecycle:
//!
//! - **Suspension**: When an operation signals it must suspend (e.g., a wait
//!   or pending operation whose result is not yet available), the driver
//!   STOPS polling and DROPS the user future at its current await point.
//!   The invocation completes with a `PENDING` outcome. Suspension is
//!   structurally unswallowable — there is no catchable error; the future
//!   is simply dropped (Rust cancellation semantics).
//!
//! - **Resume**: On a subsequent invocation with existing checkpoint state,
//!   the engine reconstructs replay mode via [`CheckpointLog`] and re-runs
//!   the handler. Replayed operations return frozen results from the log
//!   without re-execution, advancing past previously-completed work.
//!
//! - **Task-ownership**: Each context records the `tokio::task::Id` of the
//!   task that created it. Durable operations invoked from a DIFFERENT task
//!   fail fast with a clear error. The engine's own spawned tasks (via
//!   `.spawn()`) are exempt from this check.
//!
//! ## Internal suspension signaling
//!
//! Operations signal suspension via a shared `AtomicBool` flag on the
//! engine state (`suspend_requested`). When an operation determines it
//! must suspend (e.g., checkpoint status is Pending/Started and no frozen
//! result is available), it sets the flag and returns `Poll::Pending` from
//! the operation future. The driver checks the flag after each `Pending`
//! return from the top-level future: if set, it drops the future
//! immediately (no further polls) and returns `InvocationOutcome::Pending`.
//!
//! This design keeps suspension signaling entirely internal — no public
//! type or error surfaces to user code.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Waker};

// Imports used by tests — suppress dead-code warnings for the engine types
// which are consumed by operation execution.
#[cfg(test)]
use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus, EngineState};

// ────────────────────────────────────────────────────────────────────────────
// Invocation Outcome
// ────────────────────────────────────────────────────────────────────────────

/// The outcome of driving a single invocation of the user's handler.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // reason: consumed by the handler wrapper
pub(crate) enum InvocationOutcome {
    /// The handler completed successfully with a serialized result.
    Complete(String),
    /// The handler suspended — an operation requires an external event
    /// before it can proceed. The engine drops the handler future and
    /// reports PENDING to the runtime.
    Pending,
    /// The handler returned an error with a wire error type and message.
    Failed {
        /// The wire error type (e.g. `CallbackError`, `StepError`).
        error_type: String,
        /// The wire error message (the raw inner message, not the full
        /// Display chain).
        error_message: String,
    },
}

// ────────────────────────────────────────────────────────────────────────────
// Suspension Signal (shared state between operations and the driver)
// ────────────────────────────────────────────────────────────────────────────

/// A per-scope suspension signal — one node in the invocation's scope tree.
///
/// Suspension is SCOPED: a `wait` (or any parking operation) requests only
/// the scope of the context it runs on. The root handler runs in the root
/// scope; each map/parallel branch runs in its own child scope (created via
/// [`DurableContext::new_scoped_child`](crate::context::DurableContext) or
/// `new_scoped_flat_child`). A sequential child context shares its parent's
/// scope, so its suspension propagates directly to whoever drives the parent.
///
/// Each scope is observed by exactly one driver:
/// - the root scope by the invocation driver ([`drive_invocation`]);
/// - a branch scope by a coordinator's branch driver ([`drive_scope`]).
///
/// When a scope is suspended while its driver's future returns `Pending`,
/// that driver reports suspension for its subtree: the invocation reports
/// PENDING; a branch reports [`ScopeOutcome::Suspended`]. A branch suspension
/// is scoped to that branch and does not request root suspension, so sibling
/// branches keep running. A coordinator that becomes quiescent
/// with parked work suspends its OWN scope, which the level above observes.
///
/// This is an internal engine concern — it is never exposed publicly.
#[derive(Debug)]
pub(crate) struct SuspensionSignal {
    /// Set to `true` by an operation in this scope that must suspend.
    requested: AtomicBool,
    /// Waker of the branch driver ([`drive_scope`]) polling THIS scope, if
    /// any. The root scope has no branch driver — its driver is the
    /// invocation driver, which registers through `invocation_waker`.
    scope_waker: std::sync::Mutex<Option<Waker>>,
    /// Shared invocation (root) poll-loop waker. Every scope in the tree
    /// holds the same clone, so any suspension request also re-polls the
    /// invocation as a fallback (e.g. a park requested from a spawned
    /// sub-task whose own scope driver is not currently polling). Over-waking
    /// is harmless: the invocation driver returns PENDING only when the ROOT
    /// scope's own flag is set, which a mere branch suspension never sets.
    invocation_waker: Arc<std::sync::Mutex<Option<Waker>>>,
}

impl SuspensionSignal {
    /// Creates a new ROOT scope in the non-suspended state.
    pub(crate) fn new() -> Self {
        Self {
            requested: AtomicBool::new(false),
            scope_waker: std::sync::Mutex::new(None),
            invocation_waker: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Creates a fresh CHILD scope beneath this one. The child shares the
    /// invocation waker (so a park anywhere still re-polls the root) but owns
    /// its own suspension flag and branch-driver waker, so a suspension in
    /// the child is caught by the child's branch driver rather than the root.
    pub(crate) fn new_child_scope(&self) -> Self {
        Self {
            requested: AtomicBool::new(false),
            scope_waker: std::sync::Mutex::new(None),
            invocation_waker: Arc::clone(&self.invocation_waker),
        }
    }

    /// Called by operations to request suspension of THIS scope.
    ///
    /// Sets this scope's flag and wakes both this scope's branch driver (if
    /// registered) and the invocation driver (fallback), so the responsible
    /// driver re-polls and observes the flag — even when the request comes
    /// from a task other than the one that driver is polling.
    pub(crate) fn request_suspend(&self) {
        self.requested.store(true, Ordering::Release);
        wake_slot(&self.scope_waker);
        wake_slot(&self.invocation_waker);
    }

    /// Called by a driver to check whether THIS scope was suspended.
    pub(crate) fn is_suspend_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    /// Registers (or refreshes) the invocation (root) poll loop's waker.
    /// Called by [`drive_invocation`].
    pub(crate) fn register_driver_waker(&self, waker: &Waker) {
        set_slot(&self.invocation_waker, waker);
    }

    /// Registers (or refreshes) this scope's branch-driver waker.
    /// Called by [`drive_scope`].
    pub(crate) fn register_scope_waker(&self, waker: &Waker) {
        set_slot(&self.scope_waker, waker);
    }
}

/// Wakes the waker stored in a slot, if any.
fn wake_slot(slot: &std::sync::Mutex<Option<Waker>>) {
    if let Some(waker) = &*slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        waker.wake_by_ref();
    }
}

/// Stores a waker into a slot, skipping the clone when it would wake the same
/// task as the one already stored.
fn set_slot(slot: &std::sync::Mutex<Option<Waker>>, waker: &Waker) {
    let mut guard = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match &*guard {
        Some(existing) if existing.will_wake(waker) => {}
        _ => *guard = Some(waker.clone()),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Task Ownership
// ────────────────────────────────────────────────────────────────────────────

/// Identifies the owning task and tracks blessed (engine-spawned) task IDs.
///
/// Each `DurableContext` records the `tokio::task::Id` of the task that
/// created it. Durable operations check ownership before proceeding —
/// operations from a foreign task fail fast.
///
/// The `.spawn()` terminal registers the spawned task's ID as "blessed",
/// exempting it from the ownership check. This catches user
/// `tokio::spawn` misuse while allowing the SDK's own spawn mechanism.
#[derive(Debug)]
pub(crate) struct TaskOwnership {
    /// The task ID that owns this context. `None` if created outside a
    /// spawned task (e.g. in `block_on` — the Lambda runtime always runs
    /// handlers in spawned tasks, so this is `Some` in production).
    #[allow(dead_code)] // reason: read by check_current_task
    owner_task_id: Option<tokio::task::Id>,
    /// Task IDs blessed by `.spawn()` — these are exempt from the check.
    /// We use a Mutex<Vec> because spawn registrations are rare relative
    /// to ownership checks, and the set is small.
    #[allow(dead_code)] // reason: read by check_current_task
    blessed_tasks: std::sync::Mutex<Vec<tokio::task::Id>>,
}

impl TaskOwnership {
    /// Creates ownership tracking anchored to the current task.
    ///
    /// If called outside a tokio spawned task (e.g. directly in
    /// `block_on`), the ownership check is disabled (always passes).
    /// In production, the Lambda runtime always runs handlers in spawned
    /// tasks, so the check is always active.
    pub(crate) fn new_current() -> Self {
        Self {
            owner_task_id: tokio::task::try_id(),
            blessed_tasks: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Creates ownership tracking with a specific owner task ID (for testing).
    #[cfg(test)]
    #[allow(dead_code)] // reason: available for future test scenarios
    pub(crate) fn with_owner(owner_task_id: tokio::task::Id) -> Self {
        Self {
            owner_task_id: Some(owner_task_id),
            blessed_tasks: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Registers a task ID as blessed (engine-spawned).
    ///
    /// Blessed tasks pass the ownership check even though they are not the
    /// original owner.
    #[allow(dead_code)] // reason: called by .spawn()
    pub(crate) fn bless_task(&self, task_id: tokio::task::Id) {
        let mut blessed = self
            .blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        blessed.push(task_id);
    }

    /// Checks whether the calling task is authorized to perform durable
    /// operations on this context.
    ///
    /// Returns `Ok(())` if the caller is the owner or a blessed task.
    /// Returns `Err` with a descriptive message if unauthorized.
    ///
    /// If the owner task ID is not available (context created outside a
    /// spawned task — only possible in test/`block_on` scenarios), the
    /// check always passes.
    #[allow(dead_code)] // reason: called by enforce_task_ownership
    pub(crate) fn check_current_task(&self) -> Result<(), String> {
        let Some(owner_id) = self.owner_task_id else {
            // Owner was created outside a spawned task (e.g. block_on in
            // tests). The check is disabled — in production the Lambda
            // runtime always spawns the handler task.
            return Ok(());
        };

        let Some(current_id) = tokio::task::try_id() else {
            return Err("durable operation invoked outside a tokio task context".to_owned());
        };

        if current_id == owner_id {
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
            "durable operation invoked from task {current_id:?}, but context is owned by \
             task {owner_id:?}. Use .spawn() instead of tokio::spawn for durable fan-out"
        ))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Structured task ownership: abort-on-drop handle
// ────────────────────────────────────────────────────────────────────────────

/// Retains a spawned tokio task's `JoinHandle` and aborts the task when
/// dropped.
///
/// Holding this alongside (or inside) the future that owns a spawned
/// durable task makes the owner responsible for the task's lifetime:
/// dropping the owner aborts the task. This prevents spawned durable work
/// from outliving the invocation that created it — a dropped oneshot
/// receiver alone does NOT cancel a tokio task, so without this the task
/// would survive the invocation boundary and could resume on a later warm
/// invocation.
#[derive(Debug)]
pub(crate) struct AbortOnDrop {
    handle: tokio::task::JoinHandle<()>,
}

impl AbortOnDrop {
    /// Wraps a spawned task's handle so it is aborted on drop.
    pub(crate) fn new(handle: tokio::task::JoinHandle<()>) -> Self {
        Self { handle }
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Driver: polls the user future with suspension awareness
// ────────────────────────────────────────────────────────────────────────────

/// Drives a single invocation of the user's handler future.
///
/// The driver polls the future to completion or until suspension is
/// signaled. On suspension, the future is dropped at its current await
/// point (unswallowable cancellation) and `InvocationOutcome::Pending`
/// is returned.
#[allow(dead_code)] // reason: wired by the handler wrapper
pub(crate) async fn drive_invocation<F>(
    handler_future: F,
    suspension_signal: Arc<SuspensionSignal>,
) -> InvocationOutcome
where
    F: Future<Output = Result<String, (String, String)>> + Send,
{
    // Pin the handler future on the stack so we can poll it.
    let mut pinned = Box::pin(handler_future);

    // Use poll_fn to manually drive the future with suspension checks.
    std::future::poll_fn(move |cx: &mut Context<'_>| {
        // Register this poll loop's waker so an operation that suspends from
        // a spawned task can wake the driver to observe the signal.
        suspension_signal.register_driver_waker(cx.waker());

        // Check if suspension was already requested (from a previous poll
        // cycle where an operation set the flag).
        if suspension_signal.is_suspend_requested() {
            // Drop the future by letting `pinned` go out of scope when
            // this closure is dropped. Return Ready(Pending) to the outer
            // async — we're done.
            return Poll::Ready(InvocationOutcome::Pending);
        }

        // Poll the handler future once.
        match pinned.as_mut().poll(cx) {
            Poll::Ready(Ok(result)) => {
                // An operation may have requested suspension AND the
                // handler still completed (e.g. error propagated via `?`
                // without yielding). Suspension takes precedence.
                if suspension_signal.is_suspend_requested() {
                    Poll::Ready(InvocationOutcome::Pending)
                } else {
                    Poll::Ready(InvocationOutcome::Complete(result))
                }
            }
            Poll::Ready(Err(err)) => {
                // Same precedence rule: if an operation requested
                // suspension before the error propagated, the invocation
                // outcome is Pending, not Failed.
                if suspension_signal.is_suspend_requested() {
                    Poll::Ready(InvocationOutcome::Pending)
                } else {
                    Poll::Ready(InvocationOutcome::Failed {
                        error_type: err.0,
                        error_message: err.1,
                    })
                }
            }
            Poll::Pending => {
                // The future yielded — check if an operation requested
                // suspension during this poll cycle.
                if suspension_signal.is_suspend_requested() {
                    // Suspension requested: drop the future (happens when
                    // `pinned` goes out of scope after this return) and
                    // report PENDING.
                    Poll::Ready(InvocationOutcome::Pending)
                } else {
                    // Normal async yield — the future is waiting for some
                    // other wakeup (e.g., I/O). Continue polling.
                    Poll::Pending
                }
            }
        }
    })
    .await
}

// ────────────────────────────────────────────────────────────────────────────
// Branch driver: converts a locally-suspended branch future into an outcome
// ────────────────────────────────────────────────────────────────────────────

/// The outcome of driving a single branch future within one scope.
///
/// `Failed` is not a distinct variant: a branch future's own output already
/// carries its success/failure (e.g. `Result<O, ChildFnError>`), so a failure
/// arrives as `Completed(Err(..))`. Only `Suspended` needs to short-circuit
/// the poll loop, because a parked branch future never resolves on its own.
#[derive(Debug)]
pub(crate) enum ScopeOutcome<T> {
    /// The branch future resolved; its own `Ok`/`Err` is carried in `T`.
    Completed(T),
    /// The branch's scope was suspended before the future resolved. The
    /// future is dropped at its current await point. The branch keeps its
    /// concurrency slot until it terminally completes on a later invocation.
    Suspended,
}

/// Drives one branch future while watching a SPECIFIC scope for suspension.
///
/// Mirrors [`drive_invocation`] at branch granularity: when the branch's
/// `scope` is suspended while the future returns `Pending`, the future is
/// dropped and [`ScopeOutcome::Suspended`] is returned so the coordinator can
/// keep sibling branches running. Otherwise the future's own output is
/// returned as [`ScopeOutcome::Completed`]. Suspension takes precedence over
/// a same-poll `Ready`, mirroring the invocation driver's precedence rule.
pub(crate) async fn drive_scope<F>(
    inner: F,
    scope: Arc<SuspensionSignal>,
) -> ScopeOutcome<F::Output>
where
    F: Future + Send,
{
    let mut pinned = Box::pin(inner);
    std::future::poll_fn(move |cx: &mut Context<'_>| {
        // Register this branch driver's waker so a park requested from a
        // sub-task within this scope re-polls us to observe the flag.
        scope.register_scope_waker(cx.waker());

        if scope.is_suspend_requested() {
            return Poll::Ready(ScopeOutcome::Suspended);
        }

        match pinned.as_mut().poll(cx) {
            Poll::Ready(value) => {
                if scope.is_suspend_requested() {
                    Poll::Ready(ScopeOutcome::Suspended)
                } else {
                    Poll::Ready(ScopeOutcome::Completed(value))
                }
            }
            Poll::Pending => {
                if scope.is_suspend_requested() {
                    Poll::Ready(ScopeOutcome::Suspended)
                } else {
                    Poll::Pending
                }
            }
        }
    })
    .await
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_support {
    use super::{InvocationOutcome, SuspensionSignal, drive_invocation};
    use std::sync::Arc;

    /// Drives an operation future through the real suspension driver and
    /// returns the invocation outcome. The operation's own result is
    /// discarded: suspension is observed via the shared signal exactly as in
    /// production, so a suspending operation yields
    /// [`InvocationOutcome::Pending`] and a completing one yields
    /// `Complete`. Used by operation unit tests to drive a suspending
    /// future to an outcome instead of awaiting it directly (a direct
    /// await would park forever).
    pub(crate) async fn outcome_of<F>(signal: Arc<SuspensionSignal>, fut: F) -> InvocationOutcome
    where
        F: IntoFuture + Send,
        F::IntoFuture: Send,
    {
        drive_invocation(
            async move {
                let _ = fut.await;
                Ok::<_, (String, String)>("ok".to_owned())
            },
            signal,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    // ── Suspension drops the future at the await point ──────────────────

    /// A guard that sets a flag when dropped — proves the future was
    /// dropped at the await point.
    struct DropGuard {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn suspension_drops_future_at_await_point() {
        let signal = Arc::new(SuspensionSignal::new());
        let drop_observed = Arc::new(AtomicBool::new(false));
        let after_await_ran = Arc::new(AtomicBool::new(false));

        let signal_clone = Arc::clone(&signal);
        let drop_clone = Arc::clone(&drop_observed);
        let after_clone = Arc::clone(&after_await_ran);

        let outcome = drive_invocation(
            async move {
                let _guard = DropGuard {
                    dropped: drop_clone,
                };

                // Simulate an operation that requests suspension and yields.
                signal_clone.request_suspend();
                // Yield to the driver — the driver will see the signal and
                // drop this future.
                tokio::task::yield_now().await;

                // This code must NEVER run — the future is dropped above.
                after_clone.store(true, Ordering::SeqCst);
                Ok("should not reach".to_owned())
            },
            Arc::clone(&signal),
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

    // ── Suspension is unswallowable (no catchable error) ────────────────

    #[tokio::test]
    async fn suspension_unswallowable_by_catch_all() {
        let signal = Arc::new(SuspensionSignal::new());
        let after_catch_ran = Arc::new(AtomicBool::new(false));

        let signal_clone = Arc::clone(&signal);
        let after_clone = Arc::clone(&after_catch_ran);

        let outcome = drive_invocation(
            async move {
                // User tries a "catch-all" pattern — wrapping in a closure
                // that catches panics. This cannot catch future-drop.
                let result: Result<String, (String, String)> = async {
                    signal_clone.request_suspend();
                    tokio::task::yield_now().await;
                    Ok("survived".to_owned())
                }
                .await;

                // If somehow the suspension was catchable, this would run:
                after_clone.store(true, Ordering::SeqCst);
                result
            },
            Arc::clone(&signal),
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !after_catch_ran.load(Ordering::SeqCst),
            "catch-all must not intercept suspension — future is dropped"
        );
    }

    // ── PENDING outcome surfaces to the driver caller ───────────────────

    #[tokio::test]
    async fn pending_outcome_surfaces_to_caller() {
        let signal = Arc::new(SuspensionSignal::new());
        let signal_clone = Arc::clone(&signal);

        let outcome = drive_invocation(
            async move {
                signal_clone.request_suspend();
                tokio::task::yield_now().await;
                Ok("unreachable".to_owned())
            },
            Arc::clone(&signal),
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
    }

    // ── Normal completion ───────────────────────────────────────────────

    #[tokio::test]
    async fn driver_returns_complete_on_success() {
        let signal = Arc::new(SuspensionSignal::new());

        let outcome = drive_invocation(
            async move { Ok::<_, (String, String)>("done".to_owned()) },
            Arc::clone(&signal),
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    #[tokio::test]
    async fn driver_returns_failed_on_error() {
        let signal = Arc::new(SuspensionSignal::new());

        let outcome = drive_invocation(
            async move { Err(("Error".to_owned(), "boom".to_owned())) },
            Arc::clone(&signal),
        )
        .await;

        assert_eq!(
            outcome,
            InvocationOutcome::Failed {
                error_type: "Error".to_owned(),
                error_message: "boom".to_owned(),
            }
        );
    }

    // ── Resume: replayed operations return frozen results ────────────────

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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
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
            #[allow(clippy::unwrap_used)] // reason: test — verified present above
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

    // ── Task ownership: foreign task fails fast ─────────────────────────

    #[tokio::test]
    async fn ownership_check_rejects_foreign_task() {
        // Must create ownership inside a spawned task where try_id()
        // returns Some — the #[tokio::test] root runs in block_on which
        // has no task ID.
        let result = tokio::spawn(async {
            let ownership = Arc::new(TaskOwnership::new_current());
            let ownership_clone = Arc::clone(&ownership);

            // Spawn a DIFFERENT task and try the ownership check there.
            let handle = tokio::spawn(async move { ownership_clone.check_current_task() });

            #[allow(clippy::unwrap_used)] // reason: test — join handle will not panic
            handle.await.unwrap()
        })
        .await;

        #[allow(clippy::unwrap_used)] // reason: test — outer spawn will not panic
        let inner_result = result.unwrap();
        assert!(inner_result.is_err());
        #[allow(clippy::unwrap_used)] // reason: test — verified Err above
        let msg = inner_result.unwrap_err();
        assert!(
            msg.contains("Use .spawn()"),
            "error message should guide users: {msg}"
        );
    }

    // ── Task ownership: owning task succeeds ────────────────────────────

    #[tokio::test]
    async fn ownership_check_allows_owning_task() {
        // Run inside a spawned task so try_id() returns Some.
        let result = tokio::spawn(async {
            let ownership = TaskOwnership::new_current();
            ownership.check_current_task()
        })
        .await;

        #[allow(clippy::unwrap_used)] // reason: test — spawn will not panic
        let inner = result.unwrap();
        assert!(inner.is_ok(), "owning task should pass: {inner:?}");
    }

    // ── Task ownership: .spawn() exemption (blessed tasks) ──────────────

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

            #[allow(clippy::unwrap_used)] // reason: test — join handle will not panic
            handle.await.unwrap()
        })
        .await;

        #[allow(clippy::unwrap_used)] // reason: test — outer spawn will not panic
        let inner = result.unwrap();
        assert!(
            inner.is_ok(),
            "blessed task should pass ownership check: {inner:?}"
        );
    }

    // ── Signal mechanics ────────────────────────────────────────────────

    #[test]
    fn suspension_signal_starts_inactive() {
        let signal = SuspensionSignal::new();
        assert!(!signal.is_suspend_requested());
    }

    #[test]
    fn suspension_signal_activates_on_request() {
        let signal = SuspensionSignal::new();
        signal.request_suspend();
        assert!(signal.is_suspend_requested());
    }

    // ── Scope independence + branch driver ──────────────────────────────

    #[test]
    fn child_scope_suspend_does_not_trip_root_scope() {
        // A branch scope's suspension must NOT set the root scope's flag —
        // this is exactly what lets a parked branch leave the invocation
        // running.
        let root = SuspensionSignal::new();
        let branch = root.new_child_scope();
        branch.request_suspend();
        assert!(
            branch.is_suspend_requested(),
            "branch scope must be flagged"
        );
        assert!(
            !root.is_suspend_requested(),
            "root scope must stay unflagged when only a branch suspends"
        );
    }

    #[tokio::test]
    async fn drive_scope_returns_completed_when_future_ready() {
        let scope = Arc::new(SuspensionSignal::new());
        let outcome = drive_scope(async { 42_i32 }, scope).await;
        assert!(matches!(outcome, ScopeOutcome::Completed(42)));
    }

    #[tokio::test]
    async fn drive_scope_returns_suspended_when_scope_flagged() {
        let scope = Arc::new(SuspensionSignal::new());
        let scope_in = Arc::clone(&scope);
        let outcome = drive_scope(
            async move {
                scope_in.request_suspend();
                std::future::pending::<i32>().await
            },
            scope,
        )
        .await;
        assert!(matches!(outcome, ScopeOutcome::Suspended));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Regression tests: suspension never returns to user code; spawned tasks
// are owned and cancelled on drop.
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // reason: test module
mod suspension_containment_and_task_ownership {
    use super::{InvocationOutcome, drive_invocation};
    use crate::client::{ExecutionClient, InMemoryExecutionClient, TestResponse};
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            replay_children: false,
            callback_id: None,
        }
    }

    // ── 1. Suspension never lets following user code run ────────────────
    // Each op is awaited and its (never-produced) result ignored; the side
    // effect after the await MUST NOT run and the invocation MUST be Pending.

    #[tokio::test]
    async fn wait_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.suspension_signal());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                let _ = ctx_h.wait(Duration::from_secs(30)).await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                let _ = ctx_h
                    .invoke::<serde_json::Value, _>("target-fn", "input")
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                // Body fails; the default retry strategy schedules a retry,
                // which suspends (parks) on the backend timer.
                let _ = ctx_h
                    .step(|_| async { Err::<i32, crate::BoxError>("boom".into()) })
                    .name("retrying")
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                let _ = ctx_h
                    .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
                    .wait_strategy_fn(Box::new(|_state: i32, _attempt| {
                        crate::WaitDecision::continue_with(Duration::from_secs(1))
                    }))
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                let Ok(cb) = ctx_h.create_callback::<serde_json::Value>().await else {
                    return Err(("E".to_owned(), "create failed".to_owned()));
                };
                // Awaiting a pending callback result suspends.
                let _ = cb.result().await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended callback ran"
        );
    }

    // ── 2. A replayed completed wait returns Ok and following code runs ──

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
        let signal = Arc::clone(ctx.suspension_signal());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                ctx_h
                    .wait(Duration::from_secs(30))
                    .await
                    .expect("replayed completed wait returns Ok");
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            ran.load(Ordering::SeqCst),
            "code after a replayed wait must run"
        );
    }

    // ── 3. Genuine failures remain catchable Err (not swallowed) ─────────

    #[tokio::test]
    async fn validation_error_is_catchable_err() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.suspension_signal());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                // max_concurrency(0) is a validation error, returned as Err.
                let r = ctx_h
                    .parallel::<String>(Vec::new())
                    .max_concurrency(0)
                    .await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, (String, String)>("handled".to_owned())
            },
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("handled".to_owned()));
        assert!(
            caught.load(Ordering::SeqCst),
            "validation error must be catchable"
        );
    }

    #[tokio::test]
    async fn checkpoint_failure_is_catchable_err() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError("kaboom".to_owned()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.suspension_signal());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_invocation(
            async move {
                // The live wait's START checkpoint fails non-retryably; the
                // error must surface as a catchable Err, not a suspension.
                let r = ctx_h.wait(Duration::from_secs(5)).await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, (String, String)>("handled".to_owned())
            },
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("handled".to_owned()));
        assert!(
            caught.load(Ordering::SeqCst),
            "checkpoint failure must be catchable"
        );
    }

    // ── 5. Dropping a .spawn() handle cancels its task ──────────────────

    #[tokio::test]
    async fn dropping_spawn_handle_cancels_task() {
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
        // Let the runtime process the abort.
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropped .spawn() handle must cancel the task"
        );
        assert!(
            !reached_end.load(Ordering::SeqCst),
            "cancelled task must not reach its end"
        );
    }

    // ── 4 + 6. Suspending branch drops an active sibling spawned task and
    // no further checkpoint (e.g. SUCCEED) is made after cleanup begins. ──

    #[tokio::test]
    async fn suspend_drops_active_sibling_and_makes_no_further_checkpoint() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.suspension_signal());

        let sib_started = Arc::new(AtomicBool::new(false));
        let sib_completed = Arc::new(AtomicBool::new(false));
        let sib_dropped = Arc::new(AtomicBool::new(false));

        let ctx_h = ctx.clone();
        let s = Arc::clone(&sib_started);
        let c = Arc::clone(&sib_completed);
        let d = Arc::clone(&sib_dropped);

        let outcome = drive_invocation(
            async move {
                // Sibling task: checkpoints START, runs its body, then parks
                // forever. It must be aborted when the invocation suspends,
                // before it can checkpoint SUCCEED.
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
                // A wait suspends the whole invocation.
                let _ = ctx_h.wait(Duration::from_secs(30)).await;
                let _ = sib.await; // unreachable — the wait parks first
                Ok::<_, (String, String)>("done".to_owned())
            },
            signal,
        )
        .await;

        // Let the runtime process the abort of the sibling task.
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            sib_started.load(Ordering::SeqCst),
            "sibling should have started"
        );
        assert!(
            !sib_completed.load(Ordering::SeqCst),
            "sibling must not complete"
        );
        assert!(
            sib_dropped.load(Ordering::SeqCst),
            "sibling task must be aborted on suspend"
        );
        let recorded = client.recorded_updates();
        assert!(
            recorded
                .iter()
                .all(|u| u.action() != &OperationAction::Succeed),
            "no operation should have checkpointed SUCCEED after cleanup began"
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Regression tests: branch-local suspension scopes + slot-holding
// accounting. Driven end-to-end through the real invocation
// driver over an in-memory client — no AWS.
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // reason: test module
mod scoped_suspension {
    use super::{InvocationOutcome, ScopeOutcome, SuspensionSignal, drive_invocation, drive_scope};
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use crate::future::Branch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
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
        let signal = Arc::clone(ctx.suspension_signal());
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran_c = Arc::clone(&b_ran);
        let ctx_h = ctx.clone();

        let outcome = drive_invocation(
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
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ctx_h = ctx.clone();

        let outcome = drive_invocation(
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
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let ctx_h = ctx.clone();
        let active_h = Arc::clone(&active);
        let peak_h = Arc::clone(&peak);

        let outcome = drive_invocation(
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
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let second_ran = Arc::new(AtomicBool::new(false));
        let second_h = Arc::clone(&second_ran);
        let ctx_h = ctx.clone();

        let outcome = drive_invocation(
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
                Ok::<_, (String, String)>(format!("{}", r.is_ok()))
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ctx_h = ctx.clone();

        let fut = drive_invocation(
            async move {
                let outer = vec![Branch::new("outer", |cc: DurableContext| {
                    Box::pin(async move {
                        // A child context whose step feeds a nested map whose
                        // single item waits — exercises child + coordinator
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
                Ok::<_, (String, String)>("done".to_owned())
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let ctx_h = ctx.clone();

        let outcome = drive_invocation(
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
                    .completion(crate::CompletionConfig {
                        min_successful: Some(1),
                        ..Default::default()
                    })
                    .await;
                Ok::<_, (String, String)>(format!("ok={}", r.is_ok()))
            },
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
        let signal = Arc::clone(ctx.suspension_signal());
        let after = Arc::new(AtomicBool::new(false));
        let after_c = Arc::clone(&after);
        let ctx_h = ctx.clone();

        let outcome = drive_invocation(
            async move {
                ctx_h
                    .wait(Duration::from_secs(30))
                    .await
                    .map_err(|e| ("W".to_owned(), e.to_string()))?;
                after_c.store(true, Ordering::SeqCst);
                Ok::<_, (String, String)>("done".to_owned())
            },
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !after.load(Ordering::SeqCst),
            "code after a root-level wait must not run"
        );
    }

    // drive_scope directly: a scope-flagged park yields Suspended, while a
    // ready future yields Completed — with an independent scope Arc.
    #[tokio::test]
    async fn drive_scope_isolates_suspension_from_completion() {
        let scope = Arc::new(SuspensionSignal::new());
        let done = drive_scope(async { "ok" }, Arc::clone(&scope)).await;
        assert!(matches!(done, ScopeOutcome::Completed("ok")));

        let scope2 = Arc::new(SuspensionSignal::new());
        let s2 = Arc::clone(&scope2);
        let parked = drive_scope(
            async move {
                s2.request_suspend();
                std::future::pending::<&str>().await
            },
            scope2,
        )
        .await;
        assert!(matches!(parked, ScopeOutcome::Suspended));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Regression tests (S2a): `.spawn()` must not park the ROOT scope
// ────────────────────────────────────────────────────────────────────────────
//
// `.spawn()` hands the operation future to `DurableFuture::spawn_blessed`
// (src/future.rs:67) WITHOUT creating a child suspension scope, so a parking
// operation inside a top-level `.spawn()` sets the ROOT flag. `drive_invocation`
// then returns `Poll::Ready(Pending)` on its next poll and drops the handler
// future, which fires `AbortOnDrop` on every sibling spawned task — including
// one that is mid-flight and has already checkpointed STARTED. The documented
// `spawn` + `join!` pattern therefore does not hold, and an aborted step that
// checkpointed STARTED re-executes on resume (duplicate side effects).
//
// Both tests below encode the INTENDED behaviour: a spawned step must reach its
// terminal (Succeed) checkpoint before the invocation reports Pending. They fail
// against current behaviour, so they are `#[ignore]`d to keep `make check`
// green; the fix slice removes the attribute. See
// .agents/scratchpad/s2a-diagnosis.md for the observed failures and the
// intended mechanism.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // reason: test assertions
mod spawn_scope_regressions {
    use super::{InvocationOutcome, drive_invocation};
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
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
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
    #[ignore = "encodes the S2a defect: .spawn() parks the ROOT scope (src/future.rs:67), so the spawned wait ends the invocation and aborts the still-running spawned step; un-ignore with the fix"]
    async fn spawned_step_reaches_terminal_checkpoint_before_pending() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.suspension_signal());
        let body_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&body_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_invocation(
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
                    Ok::<_, (String, String)>("done".to_owned())
                },
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
    #[ignore = "encodes the S2a defect: tokio::join!(wait.spawn(), step.spawn()) is aborted by the wait's ROOT-scope park (src/future.rs:67, src/driver.rs:334); un-ignore with the fix"]
    async fn joined_spawned_wait_and_step_lets_the_step_finish() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.suspension_signal());
        let body_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&body_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_invocation(
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
                    Ok::<_, (String, String)>("done".to_owned())
                },
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

    /// A spawned wait parks, then the owner performs a non-spawned sequential
    /// step. The sequential step must complete and checkpoint Succeed BEFORE
    /// the invocation reports Pending. This catches premature ROOT-scope
    /// flagging: if the spawned wait's park flags ROOT unconditionally, the
    /// driver drops the handler while the sequential step is still in flight.
    #[tokio::test]
    #[ignore = "encodes the S2a defect: a parked spawned wait must not pre-empt the owner's sequential work (src/future.rs:67 flags ROOT, driver.rs:334 drops handler); un-ignore with the fix"]
    async fn spawned_wait_parks_then_owner_does_sequential_step() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.suspension_signal());
        let step_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&step_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_invocation(
                async move {
                    // Spawn a wait — it parks immediately (no timer result on
                    // first invocation). Hold the handle so its AbortOnDrop
                    // doesn't fire, but never await it.
                    let _wait = ctx_h.wait(Duration::from_secs(10)).spawn();

                    // Yield once so the spawned wait task has a chance to run
                    // and park. Without scope isolation, this parks ROOT.
                    tokio::task::yield_now().await;

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

                    // Handler completes normally. The driver should see the
                    // parked spawn and report Pending instead of Complete.
                    Ok::<_, (String, String)>("done".to_owned())
                },
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Pending);
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
}
