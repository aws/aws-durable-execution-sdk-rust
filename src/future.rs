//! Future types for durable operations.
//!
//! [`DurableFuture`] is the uniform handle for all durable operation
//! results. [`Settled`] represents the outcome of a single future in
//! `join_all`. [`Branch`] defines a named parallel branch. [`Callback`]
//! holds a callback token and its result future.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::BoxError;
use crate::DurableContext;
use crate::error::{ChildFnError, OperationError};

/// The uniform future handle for all durable operations.
///
/// Every builder's [`IntoFuture`](std::future::IntoFuture) target is
/// `DurableFuture<O>`. The operation ID is already claimed at builder
/// creation; the body runs when polled (lazy) unless the builder was
/// terminated with `.spawn()` (eager).
///
/// When produced by `.spawn()` the operation runs eagerly on an owned tokio
/// task. Dropping the `DurableFuture` before it resolves aborts that task,
/// so spawned durable work never outlives its handle.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use aws_durable_execution_sdk_rust::DurableFuture;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     // DurableFuture implements Future
///     let future: DurableFuture<i32> = ctx.step(|_| async { Ok(42) }).future();
///     let value: i32 = future.await?;
///     assert_eq!(value, 42);
///     Ok(())
/// }
/// ```
#[must_use = "futures do nothing unless polled or spawned"]
#[non_exhaustive]
pub struct DurableFuture<O> {
    inner: Pin<Box<dyn Future<Output = Result<O, OperationError>> + Send>>,
    /// Optional shared cell that a spawned handle reads at park-time.
    /// When set to `Some(scope)`, the handle parks that scope instead of the
    /// one captured at `.spawn()` time. Enables spawned combinators to
    /// redirect constituent parking onto the combinator's spawn scope,
    /// preventing deadlock when a constituent parks and the combinator is
    /// outstanding on the same owner scope.
    park_redirect: Option<std::sync::Arc<ParkRedirect>>,
}

/// Shared mutable cell read by a spawned handle at park-time to determine
/// which scope to park. Default is `None` (use the originally captured scope).
#[derive(Debug)]
pub(crate) struct ParkRedirect {
    target: std::sync::Mutex<Option<std::sync::Arc<crate::driver::SuspensionSignal>>>,
}

impl ParkRedirect {
    fn new() -> Self {
        Self {
            target: std::sync::Mutex::new(None),
        }
    }

    fn set(&self, scope: std::sync::Arc<crate::driver::SuspensionSignal>) {
        // A poisoned Mutex means a panic occurred while the lock was held.
        // Since ParkRedirect access is trivial (clone-in / clone-out with no
        // user code under the lock), poisoning is an irrecoverable internal
        // state; silently falling back to the default park target is safe.
        if let Ok(mut guard) = self.target.lock() {
            *guard = Some(scope);
        }
    }

    fn get(&self) -> Option<std::sync::Arc<crate::driver::SuspensionSignal>> {
        self.target.lock().ok().and_then(|guard| guard.clone())
    }
}

impl<O> std::fmt::Debug for DurableFuture<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableFuture").finish_non_exhaustive()
    }
}

impl<O: Send + 'static> DurableFuture<O> {
    /// Creates a `DurableFuture` from an async computation.
    pub(crate) fn from_async<F>(fut: F) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        Self {
            inner: Box::pin(fut),
            park_redirect: None,
        }
    }

    /// Redirects this future's park behaviour to `scope`.
    ///
    /// When the underlying operation parks, the future parks `scope`
    /// instead of the scope captured when the future was created. Both
    /// terminals participate: a spawned future (produced by `.spawn()`)
    /// reads the redirect when its task reports `Parked`, and a lazy
    /// future (produced by [`Self::lazy_scoped`] via `.future()` or
    /// `.await`) reads it when its own scope suspends. This is what lets
    /// a combinator isolate a losing input's suspension onto a scope it
    /// controls (issue #49). No-op on futures without a redirect cell
    /// (immediate-error preflight futures, raw test futures).
    pub(crate) fn set_park_scope(&self, scope: std::sync::Arc<crate::driver::SuspensionSignal>) {
        if let Some(redirect) = &self.park_redirect {
            redirect.set(scope);
        }
    }

    /// Creates a lazy `DurableFuture` that runs `fut` inside its OWN
    /// suspension scope, mirroring [`Self::spawn_blessed`] without the
    /// eager tokio task.
    ///
    /// The builder's context was rebound onto `op_scope` (a fresh child
    /// scope) before `fut` was constructed, so every park inside the
    /// operation lands on `op_scope` rather than on the scope the builder
    /// was created from. [`drive_scope`](crate::driver::drive_scope)
    /// observes that park and this wrapper forwards it: to the redirect
    /// target when [`Self::set_park_scope`] installed one (the future is a
    /// combinator constituent), otherwise to `owner_scope` (the scope the
    /// operation was created on — a direct `.await` must still suspend the
    /// caller). After forwarding, the future pends forever: a parked
    /// operation resumes only on a later invocation, exactly like the
    /// spawned path.
    pub(crate) fn lazy_scoped<F>(
        fut: F,
        owner_scope: std::sync::Arc<crate::driver::SuspensionSignal>,
        op_scope: std::sync::Arc<crate::driver::SuspensionSignal>,
    ) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        use crate::driver::{ScopeOutcome, drive_scope};

        let redirect = std::sync::Arc::new(ParkRedirect::new());
        let redirect_handle = std::sync::Arc::clone(&redirect);

        Self {
            inner: Box::pin(async move {
                match drive_scope(fut, op_scope).await {
                    ScopeOutcome::Completed(result) => result,
                    ScopeOutcome::Suspended => {
                        let park_target = redirect_handle.get().unwrap_or(owner_scope);
                        park_target.park_owner();
                        std::future::pending().await
                    }
                }
            }),
            park_redirect: Some(redirect),
        }
    }

    /// Spawns the future on a tokio task with its OWN suspension scope, and
    /// registers it as blessed.
    ///
    /// `owner_scope` is the scope the `.spawn()` call was made from;
    /// `spawn_scope` is the fresh child scope the operation runs in (the
    /// builder's context was rebound onto it by
    /// [`DurableContext::spawn_scope`](crate::context::DurableContext)).
    ///
    /// Scope isolation means a parking operation inside the spawned task
    /// suspends only `spawn_scope`, which this task's
    /// [`drive_scope`](crate::driver::drive_scope) observes — the owner's scope
    /// is untouched, so runnable siblings keep running. The task then reports
    /// how it settled to the owner's scope-quiescence accounting, which is what
    /// lets the owner suspend exactly once everything runnable has finished or
    /// parked.
    ///
    /// The accounting transitions happen ON THIS TASK, never in the returned
    /// handle: the runtime always polls a live task, but nothing guarantees the
    /// handle is ever polled. A task cancelled before it settles reports
    /// `Aborted` from its RAII guard.
    pub(crate) fn spawn_blessed(
        future: Self,
        task_ownership: std::sync::Arc<crate::driver::TaskOwnership>,
        owner_scope: std::sync::Arc<crate::driver::SuspensionSignal>,
        spawn_scope: std::sync::Arc<crate::driver::SuspensionSignal>,
    ) -> Self
    where
        O: Send + 'static,
    {
        use crate::driver::{ScopeOutcome, SpawnSettlement, drive_scope};
        use tokio::sync::oneshot;

        /// Reports `Aborted` if the task is dropped before it settles, so a
        /// cancelled spawn can never leave the owner's counter stuck above
        /// zero (which would park the owner forever).
        struct SpawnAccounting {
            scope: std::sync::Arc<crate::driver::SuspensionSignal>,
            settled: bool,
        }

        impl SpawnAccounting {
            fn settle(&mut self, settlement: SpawnSettlement) {
                if !self.settled {
                    self.settled = true;
                    self.scope.settle_spawn(settlement);
                }
            }
        }

        impl Drop for SpawnAccounting {
            fn drop(&mut self) {
                self.settle(SpawnSettlement::Aborted);
            }
        }

        let (tx, rx) = oneshot::channel();

        // Park-redirect cell: the handle reads this at park-time. A
        // combinator's `.spawn()` may set it to the combinator's spawn scope
        // before polling the handle, redirecting parking away from the
        // captured `owner_scope`.
        let redirect = std::sync::Arc::new(ParkRedirect::new());
        let redirect_handle = std::sync::Arc::clone(&redirect);

        // Register BEFORE spawning so the count is already correct even if the
        // task settles before the owner is polled again.
        owner_scope.register_spawn();
        // Construct the settling guard BEFORE spawning and move it into the
        // task. The guard must exist from the moment `register_spawn` takes
        // effect: dropping the returned future aborts the task, and an abort
        // that lands before the task's first poll drops the task body — and
        // this guard with it — settling `Aborted`. Constructing the guard
        // inside the task body instead would leave that window unguarded,
        // permanently counting a phantom outstanding spawn that parks the
        // owner forever (issue #48).
        let mut accounting = SpawnAccounting {
            scope: std::sync::Arc::clone(&owner_scope),
            settled: false,
        };

        let handle = tokio::spawn(async move {
            // Register this task as blessed AFTER spawn (we need the task ID).
            if let Some(task_id) = tokio::task::try_id() {
                task_ownership.bless_task(task_id);
            }
            // Catch a panic in the operation body INSIDE the task so the
            // payload can be shipped through the channel instead of being
            // lost to a `JoinError` nobody observes (the JoinHandle is held
            // only for abort-on-drop, never awaited).
            let mut future = future;
            let body = std::future::poll_fn(move |cx| {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Pin::new(&mut future).poll(cx)
                })) {
                    Ok(Poll::Ready(outcome)) => Poll::Ready(Ok(outcome)),
                    Ok(Poll::Pending) => Poll::Pending,
                    Err(payload) => Poll::Ready(Err(payload)),
                }
            });
            // Drive the operation in its OWN scope: a park inside it suspends
            // that scope only, and surfaces here as `Suspended`.
            let settled = match drive_scope(body, spawn_scope).await {
                ScopeOutcome::Completed(result) => {
                    accounting.settle(SpawnSettlement::Completed);
                    SpawnMessage::Settled(result)
                }
                ScopeOutcome::Suspended => {
                    accounting.settle(SpawnSettlement::Parked);
                    SpawnMessage::Parked
                }
            };
            // Ignore send error — receiver was dropped.
            let _ = tx.send(settled);
        });

        // Own the spawned task: hold its abort-on-drop guard inside the
        // returned future so that dropping this `DurableFuture` cancels the
        // task. Without this the task would be detached and could outlive
        // the invocation.
        let guard = crate::driver::AbortOnDrop::new(handle);

        Self {
            inner: Box::pin(async move {
                let _guard = guard;
                match rx.await {
                    Ok(SpawnMessage::Settled(Ok(result))) => result,
                    // The operation body panicked: re-raise the ORIGINAL panic
                    // payload on the awaiting task, exactly as the lazy
                    // (non-spawned) path would have, instead of masking it as a
                    // fabricated step error.
                    Ok(SpawnMessage::Settled(Err(panic_payload))) => {
                        std::panic::resume_unwind(panic_payload)
                    }
                    // The operation parked durably: it resumes on a later
                    // invocation, so this handle can never resolve now. Park
                    // the effective scope — either the redirect target (when
                    // this future is a constituent of a spawned combinator) or
                    // the original owner scope — then never return.
                    Ok(SpawnMessage::Parked) => {
                        let park_target = redirect_handle
                            .get()
                            .unwrap_or_else(|| std::sync::Arc::clone(&owner_scope));
                        park_target.park_owner();
                        std::future::pending().await
                    }
                    // The sender was dropped without a value: the task was
                    // cancelled (aborted) before completing.
                    Err(_) => Err(OperationError::from_kind(
                        crate::error::OperationErrorKind::Step(crate::error::StepError::new(
                            crate::error::StepErrorKind::ExecutionFailed,
                            Some("spawned task was cancelled".into()),
                        )),
                    )),
                }
            }),
            park_redirect: Some(redirect),
        }
    }
}

/// What a spawned task reports to its handle.
enum SpawnMessage<O> {
    /// The operation resolved (`Ok`) or its body panicked (`Err(payload)`).
    Settled(Result<Result<O, OperationError>, Box<dyn std::any::Any + Send>>),
    /// The operation suspended durably; it resumes on a later invocation.
    Parked,
}

impl<O: Send + 'static> Future for DurableFuture<O> {
    type Output = Result<O, OperationError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

// DurableFuture<O> is automatically Send when O: Send, since it only
// contains a pinned boxed future that is Send when O: Send.

/// The outcome of a single future within [`DurableContext::join_all`].
///
/// Each item records either success or failure without short-circuiting
/// the collection.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{Settled, OperationError};
///
/// let ok: Settled<i32> = Settled::Fulfilled(42);
/// let err: Settled<i32> = Settled::Rejected(OperationError::__test_error());
/// match &ok {
///     Settled::Fulfilled(v) => assert_eq!(*v, 42),
///     _ => panic!("unexpected"),
/// }
/// # drop(err);
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Settled<O> {
    /// The operation completed successfully.
    Fulfilled(O),
    /// The operation failed with an error.
    Rejected(OperationError),
}

/// A named branch for [`DurableContext::parallel`].
///
/// Each branch defines a name and an async closure that receives a child
/// [`DurableContext`].
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust::{Branch, DurableContext};
///
/// let branch = Branch::new("process-a", |ctx: DurableContext| async move {
///     let v = ctx.step(|_| async { Ok(1) }).await?;
///     Ok(v)
/// });
/// # drop(branch);
/// ```
#[non_exhaustive]
pub struct Branch<O> {
    name: String,
    body: BranchBody<O>,
}

/// The erased body of one parallel branch (crate-internal).
///
/// Branch bodies are inherently heterogeneous — every `Branch::new` call
/// site captures a different closure and future type, and they are all
/// collected into one `Vec` — so exactly ONE erasure per branch is
/// unavoidable. This type keeps it to exactly one: the user's closure and
/// its future live UNERASED inside a single boxed future built at
/// [`Branch::new`]. The child [`DurableContext`] is not known until the
/// batch coordinator dispatches the branch, so it is delivered through a
/// oneshot channel the body awaits on first poll. There is no nested
/// closure-box-returning-future-box layering: polling a running branch goes
/// through one vtable hop.
pub(crate) struct BranchBody<O> {
    /// Delivers the child context to the body. The coordinator sends the
    /// context BEFORE first polling `future`, so the body's receive
    /// resolves immediately on first poll.
    ctx_tx: tokio::sync::oneshot::Sender<DurableContext>,
    /// The single erased branch future: awaits the child context, then runs
    /// the user's factory closure and its future in place.
    future: Pin<Box<dyn Future<Output = Result<O, ChildFnError>> + Send>>,
}

impl<O> BranchBody<O> {
    /// Injects the child context and returns the branch future, ready to
    /// poll. Called exactly once by the batch coordinator at dispatch time.
    pub(crate) fn start(
        self,
        ctx: DurableContext,
    ) -> Pin<Box<dyn Future<Output = Result<O, ChildFnError>> + Send>> {
        // Send cannot fail: the receiver lives inside `self.future`, which
        // we hold. Ignore the impossible error rather than panicking.
        let _ = self.ctx_tx.send(ctx);
        self.future
    }
}

impl<O> std::fmt::Debug for Branch<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Branch")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> Branch<O> {
    /// Creates a new named branch.
    ///
    /// The factory function receives a child [`DurableContext`] and returns
    /// a future producing the branch result. The factory is not invoked
    /// here — it runs when the parallel operation dispatches the branch.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::{Branch, DurableContext};
    ///
    /// let branch: Branch<i32> = Branch::new("my-branch", |_ctx: DurableContext| async move {
    ///     Ok(42)
    /// });
    /// # drop(branch);
    /// ```
    pub fn new<F, Fut>(name: impl Into<String>, factory: F) -> Self
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    {
        let (ctx_tx, ctx_rx) = tokio::sync::oneshot::channel::<DurableContext>();
        Self {
            name: name.into(),
            body: BranchBody {
                ctx_tx,
                // The SDK does the pinning and BoxError -> internal-carrier
                // type erasure so callers write a plain `async move` body
                // with `?`. This is the branch's ONE erasure: the factory
                // closure and its concrete future both live inside this box
                // — there is no separate closure box producing a second
                // future box.
                future: Box::pin(async move {
                    let ctx = ctx_rx.await.map_err(|_| {
                        ChildFnError::new("branch context was never delivered (coordinator gone)")
                    })?;
                    factory(ctx).await.map_err(ChildFnError::from)
                }),
            },
        }
    }

    /// Returns the branch name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Consumes the branch and returns its name and erased body (internal).
    pub(crate) fn into_parts(self) -> (String, BranchBody<O>) {
        (self.name, self.body)
    }
}

/// A callback token and its result future.
///
/// Returned by [`DurableContext::create_callback`]. Call [`id()`](Self::id)
/// to obtain the token string that external systems use to complete the
/// callback, and [`result()`](Self::result) to get the future that
/// resolves when the callback arrives.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Approval { approved: bool }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<bool, durable::BoxError> {
///     let cb = ctx.create_callback::<Approval>()
///         .name("wait-approval")
///         .await?;
///     let _id = cb.id(); // send to external system
///     let approval = cb.result().await?;
///     Ok(approval.approved)
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct Callback<O> {
    id: String,
    state: CallbackState<O>,
}

/// Internal state for a callback — either settled (replay) or pending (live).
#[derive(Debug)]
enum CallbackState<O> {
    /// The callback has a known outcome from the checkpoint log (replay).
    Settled(Option<Result<O, OperationError>>),
    /// The callback is in flight — requesting the result triggers suspension.
    Pending(DurableContext),
}

impl<O: Send + 'static> Callback<O> {
    /// Creates a settled callback (replay: outcome already known).
    pub(crate) fn new_settled(id: String, result: Result<O, OperationError>) -> Self {
        Self {
            id,
            state: CallbackState::Settled(Some(result)),
        }
    }

    /// Creates a pending callback (live: outcome not yet known).
    pub(crate) fn new_pending(id: String, ctx: DurableContext) -> Self {
        Self {
            id,
            state: CallbackState::Pending(ctx),
        }
    }

    /// Returns the callback ID token.
    ///
    /// This string is passed to an external system, which uses it to
    /// complete the callback via the durable execution API.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct Data { value: i32 }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let cb = ctx.create_callback::<Data>()
    ///         .await?;
    ///     println!("callback id: {}", cb.id());
    ///     let _data = cb.result().await?;
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the callback result as a [`DurableFuture`].
    ///
    /// For settled callbacks (replay), the future resolves with the stored
    /// outcome immediately. For pending callbacks (live), polling the
    /// future triggers suspension so the orchestrator can wait for the
    /// external completion signal.
    ///
    /// Because the return type is a [`DurableFuture`], a callback result
    /// participates in the durable combinators
    /// ([`try_join_all`](DurableContext::try_join_all),
    /// [`join_all`](DurableContext::join_all),
    /// [`select_ok`](DurableContext::select_ok),
    /// [`race`](DurableContext::race)) exactly like any other durable
    /// operation.
    ///
    /// Awaiting the future yields `Result<O, OperationError>`: an
    /// [`OperationError`] is returned if the callback failed (timeout,
    /// heartbeat timeout, or external failure).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct Payload { msg: String }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let cb = ctx.create_callback::<Payload>()
    ///         .await?;
    ///     let payload = cb.result().await?;
    ///     Ok(payload.msg)
    /// }
    /// ```
    pub fn result(self) -> DurableFuture<O> {
        match self.state {
            CallbackState::Settled(outcome) => DurableFuture::from_async(async move {
                // Replay path: return the stored outcome directly.
                outcome.unwrap_or_else(|| {
                    Err(OperationError::from_kind(
                        crate::error::OperationErrorKind::Callback(
                            crate::error::CallbackError::new(
                                crate::error::CallbackErrorKind::Internal,
                                Some("settled callback had no outcome".into()),
                            ),
                        ),
                    ))
                })
            }),
            CallbackState::Pending(ctx) => {
                // Live path: request suspension and park. The stored context
                // carries the CALLER's scope (create_callback deliberately
                // does not rebind — see `CreateCallbackBuilder::into_future`),
                // so rebind onto a fresh scope here: a direct `.await` still
                // suspends the caller, while a combinator can redirect this
                // future's park onto a scope it controls instead of parking
                // the caller after a sibling already settled (issue #49).
                // Either way control never returns here; the callback result
                // surfaces only when a later invocation replays with a
                // settled record.
                let owner_scope = std::sync::Arc::clone(ctx.suspension_signal());
                let (scoped_ctx, op_scope) = ctx.spawn_scope();
                DurableFuture::lazy_scoped(
                    async move { scoped_ctx.suspend_now().await },
                    owner_scope,
                    op_scope,
                )
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // reason: test assertions
mod tests {
    use super::*;
    use crate::engine::CheckpointLog;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn test_ctx_with_client() -> DurableContext {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token0".to_owned(),
        )
    }

    /// A panic inside a `.spawn()`ed operation body must surface to the
    /// awaiter as the ORIGINAL panic (payload intact), exactly like the lazy
    /// path — not as a fabricated "spawned task was cancelled" step error.
    #[tokio::test]
    async fn spawned_panic_propagates_original_payload() {
        let ctx = test_ctx_with_client();
        let fut: DurableFuture<i32> = ctx.step(|_| async { panic!("kaboom-spawn") }).spawn();

        // Await on a dedicated task so the re-raised panic is observable as
        // a JoinError instead of unwinding the test itself.
        let join_err = tokio::spawn(fut).await.unwrap_err();
        assert!(join_err.is_panic(), "panic must stay a panic: {join_err}");
        let payload = join_err.into_panic();
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(
            msg.contains("kaboom-spawn"),
            "original panic payload must be preserved, got: {msg}"
        );
    }

    /// Dropping the handle of a `.spawn()`ed operation cancels the spawned
    /// task (abort-on-drop) — cancellation, distinct from a panic, tears the
    /// body down without unwinding anything into the caller.
    #[tokio::test]
    async fn dropped_spawned_future_cancels_task_without_panic() {
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let ctx = test_ctx_with_client();
        let dropped = Arc::new(AtomicBool::new(false));
        // The guard lives in the closure environment: whether the spawned
        // task is aborted before or after its first poll, cancelling the
        // task drops the body and the guard with it.
        let guard = DropGuard(Arc::clone(&dropped));

        let fut: DurableFuture<i32> = ctx
            .step(move |_| async move {
                let _guard = guard;
                std::future::pending::<()>().await;
                Ok(0)
            })
            .spawn();
        drop(fut);

        // Give the runtime a moment to process the abort.
        for _ in 0..50 {
            if dropped.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "spawned body must be dropped (cancelled) when the handle is dropped"
        );
    }
}
