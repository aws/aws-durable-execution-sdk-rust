//! Future types for durable operations.
//!
//! [`DurableFuture`] is the uniform handle for all durable operation
//! results. [`Branch`] defines a named parallel branch, and [`BranchSet`]
//! a set of branches with different result types whose results each
//! [`BranchHandle`] returns. [`Callback`] holds a callback token and its
//! result future.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use crate::coordinator::InvocationCoordinator;

use crate::BoxError;
use crate::DurableContext;
use crate::error::{ChildFnError, OperationError};

/// The uniform future handle for all durable operations.
///
/// Every builder's [`IntoFuture`](std::future::IntoFuture) target is
/// `DurableFuture<O>`. Every operation except a combinator has already
/// claimed its operation ID when its builder was created. A combinator
/// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
/// [`DurableContext::select_ok`] and [`DurableContext::race`]) claims its ID
/// on the first poll of this future, or at `.spawn()`. A combinator future
/// that is dropped before its first poll claims no ID and records nothing.
/// A combinator future passed to another combinator must already have
/// claimed its ID: start it with `.spawn()` and pass the returned handle. A
/// still-unclaimed one fails the execution with a context usage error. The
/// body runs when polled (lazy) unless the builder was terminated with
/// `.spawn()` (eager).
///
/// When produced by `.spawn()`, `.spawn_batch()` or
/// [`Callback::spawn_result`], the operation runs eagerly on its own SDK
/// task. Dropping the `DurableFuture` loses only the result: the operation
/// runs until it settles, records its outcome and fires its plugin hooks.
/// Its error goes into its record and is returned to nobody. A panic in it
/// fails the invocation.
///
/// The SDK stops work in three cases:
///
/// - a combinator stops the `.future()` inputs it started that are still
///   running when it resolves, and lets an input started with `.spawn()`
///   run;
/// - a closed scope stops the work still running inside it. A scope closes
///   when a child context's body returns, when a `with_retry` block or one
///   of its attempts ends, and when a `map`, `parallel` or `parallel_mixed`
///   item, or the whole batch, completes;
/// - the end of the invocation stops all remaining operation work.
///
/// A stopped task stops at its next `.await`, and its record can stay
/// `STARTED`. The SDK cannot stop code between two `.await`s,
/// `spawn_blocking` work, tasks or threads the handler starts itself, or a
/// request already sent to another service.
///
/// An operation that waits on the service (a wait, a callback result, an
/// invoke) stays pending while other SDK work runs, and continues in the
/// same invocation when a checkpoint response completes it. The invocation
/// suspends only when no SDK work can make progress and an operation
/// waits.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use aws_durable_execution_sdk::DurableFuture;
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
    /// The first-poll flag of an operation started on its own task, with
    /// the coordinator to notify when the handle is first polled. `None`
    /// for every other handle, and after the first poll.
    awaited: Option<(Arc<AtomicBool>, Arc<InvocationCoordinator>)>,
    /// The pending claim of a combinator that has not claimed its operation
    /// ID yet. `None` for every other future, and once the claim ran. While
    /// it is `Some`, `inner` is a placeholder that is never polled.
    claim: Option<PendingClaim<O>>,
}

/// The claim a combinator future runs on its first poll or at `.spawn()`.
///
/// `run` claims the combinator's operation ID on the context the
/// combinator was created on, checks the claimed ID against the recorded
/// history, and returns the future that runs the combinator.
pub(crate) struct PendingClaim<O> {
    /// The combinator's public method, such as `"race"`.
    method: &'static str,
    /// The display name of the context the combinator was created on:
    /// `"root"` or the positional ID of the operation that created it.
    context: String,
    /// Claims the ID and builds the combinator's future.
    run: Box<dyn FnOnce() -> DurableFuture<O> + Send>,
}

impl<O> PendingClaim<O> {
    /// Creates the pending claim of combinator `method` created on the
    /// context whose display name is `context`.
    pub(crate) fn new(
        method: &'static str,
        context: String,
        run: impl FnOnce() -> DurableFuture<O> + Send + 'static,
    ) -> Self {
        Self {
            method,
            context,
            run: Box::new(run),
        }
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
            awaited: None,
            claim: None,
        }
    }

    /// Creates the future of a combinator that claims its operation ID on
    /// its first poll or at `.spawn()`, through [`Self::claim_now`].
    pub(crate) fn deferred(claim: PendingClaim<O>) -> Self {
        Self {
            // A placeholder: `claim_now` replaces it before the first poll.
            inner: Box::pin(std::future::pending()),
            awaited: None,
            claim: Some(claim),
        }
    }

    /// Runs the pending claim of a combinator future: claims its operation
    /// ID and checks it against the recorded history.
    ///
    /// The first poll and `.spawn()` call this. It runs the claim once. It
    /// does nothing for a future that is not a combinator, and for a
    /// combinator that already claimed.
    pub(crate) fn claim_now(&mut self) {
        if let Some(claim) = self.claim.take() {
            let claimed = (claim.run)();
            self.inner = claimed.inner;
        }
    }

    /// Reports whether this future is a combinator that has not claimed its
    /// operation ID yet.
    pub(crate) fn is_unclaimed_combinator(&self) -> bool {
        self.claim.is_some()
    }

    /// The message of the context usage error an outer combinator records
    /// when it would start this still-unclaimed combinator as an input.
    /// `None` when this future is not an unclaimed combinator.
    ///
    /// `outer` is the positional ID of the outer combinator.
    pub(crate) fn unclaimed_input_message(&self, outer: &str) -> Option<String> {
        self.claim.as_ref().map(|claim| {
            format!(
                "context usage error in \"{method}\": the combinator was created on context \
                 \"{context}\", but combinator \"{outer}\" started it as an input. Start the \
                 inner combinator with .spawn() and pass the returned handle to the outer \
                 combinator.",
                method = claim.method,
                context = claim.context,
            )
        })
    }

    /// Creates the future of one operation created on `ctx`.
    ///
    /// When one of the context's owners is abandoned before the first poll,
    /// `fut` is never polled and the future never resolves (see
    /// [`crate::checkpoint_coalescer::Admission`]). Every builder terminal,
    /// awaited or spawned, passes here before its operation runs.
    pub(crate) fn for_operation<F>(ctx: &DurableContext, fut: F) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        let gate = ctx.clone();
        Self::from_async(async move {
            gate.admit_operation().await;
            drop(gate);
            fut.await
        })
    }

    /// Creates the future of one operation whose execution begins with the
    /// start step (see [`crate::start`]): a step, a wait or a
    /// `create_callback`.
    ///
    /// The start step runs on the first poll, and checks admission itself
    /// before it queues anything. So this adds no admission gate of its
    /// own, unlike [`Self::for_operation`].
    pub(crate) fn for_started_operation<F>(fut: F) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        Self::from_async(fut)
    }

    /// Spawns the operation future on its own task, blessed in the task
    /// ownership of `ctx` before its first poll.
    ///
    /// The task starts through the task gateway
    /// ([`crate::task::spawn_tracked`]) and holds a permit of the
    /// invocation's coordinator whenever it can make progress, so the
    /// invocation does not suspend while it runs. A wait inside it registers
    /// like any other wait. The task is registered with the scopes of `ctx`
    /// (see [`crate::checkpoint_coalescer::Admission`]), so it stops when
    /// one of them closes. Dropping the returned future loses only the
    /// result: the task keeps running. A panic in the operation resumes on
    /// the task that awaits the handle, with its original payload, and
    /// fails the invocation when nothing awaits it. When no Tokio runtime is
    /// running, the returned future resolves to an error and the operation
    /// records nothing.
    ///
    /// `awaited` is the first-poll flag the operation's wait registrations
    /// carry. The returned handle sets it on its first poll and asks the
    /// coordinator to arm those registrations' refreshes (see
    /// [`crate::coordinator`]).
    pub(crate) fn spawn_blessed(
        future: Self,
        ctx: &DurableContext,
        awaited: Option<Arc<AtomicBool>>,
    ) -> Self {
        let coordinator = ctx.coordinator();
        match crate::task::spawn_tracked(coordinator, ctx.task_ownership(), future) {
            Ok(task) => {
                if let Some(abort) = task.abort_handle() {
                    ctx.admission().register_task(ctx.owners(), abort);
                }
                let mut handle = Self::from_async(async move { task.await.into_value() });
                handle.awaited = awaited.map(|flag| (flag, Arc::clone(coordinator)));
                handle
            }
            // No Tokio runtime is running on this thread: the operation
            // records nothing and resolves to an error.
            Err(no_runtime) => {
                Self::from_async(
                    async move { Err(crate::task::NoRuntime::operation_error(no_runtime)) },
                )
            }
        }
    }

    /// Starts an operation whose JS form begins inside the call: runs its
    /// start step on the caller's task, then spawns the rest of it.
    ///
    /// The runtime check runs first: when no Tokio runtime is running, the
    /// returned future resolves to an error, `start` is not called, and
    /// nothing is queued. Otherwise `start` builds the operation future
    /// after it ran the start step (see [`crate::start`]) on this task: the
    /// admission check, the task-ownership check, and the `START` the
    /// operation's record requires, with the start hook its JS order fires
    /// at the call. Then the future runs on its own task, as
    /// [`Self::spawn_blessed`] describes.
    pub(crate) fn spawn_started(
        ctx: &DurableContext,
        awaited: Option<Arc<AtomicBool>>,
        start: impl FnOnce() -> Self,
    ) -> Self {
        if let Err(no_runtime) = crate::task::ensure_runtime() {
            return Self::from_async(async move {
                Err(crate::task::NoRuntime::operation_error(no_runtime))
            });
        }
        // The task-ownership check runs before the builder's replay
        // identity preflight, as in the start step: an unblessed caller
        // gets the task-ownership error even when the record also
        // contradicts the operation. An abandoned owner skips it, so its
        // handle stays refused.
        if ctx.admission().admits(ctx.owners())
            && let Err(err) = ctx.enforce_task_ownership()
        {
            return Self::from_async(async move { Err(err) });
        }
        Self::spawn_blessed(start(), ctx, awaited)
    }
}

impl<O: Send + 'static> Future for DurableFuture<O> {
    type Output = Result<O, OperationError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // The first poll of a combinator claims its operation ID.
        self.claim_now();
        // The first poll of the handle of an operation started on its own
        // task: from now on the scheduler refreshes its registrations.
        if let Some((flag, coordinator)) = self.awaited.take() {
            flag.store(true, Ordering::SeqCst);
            coordinator.notify_awaited();
        }
        self.inner.as_mut().poll(cx)
    }
}

// DurableFuture<O> is automatically Send when O: Send, since it only
// contains a pinned boxed future that is Send when O: Send.

/// A branch for [`DurableContext::parallel`].
///
/// Each branch defines an optional name and an async closure that receives
/// a child [`DurableContext`]. History records the name, or
/// `parallel-branch-{index}` when the branch has no name or an empty one.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk::{Branch, DurableContext};
///
/// let branch = Branch::new("process-a", |ctx: DurableContext| async move {
///     let v = ctx.step(|_| async { Ok(1) }).await?;
///     Ok(v)
/// });
/// # drop(branch);
/// ```
#[non_exhaustive]
pub struct Branch<O> {
    name: Option<String>,
    body: BranchBody<O>,
}

/// The erased body of one parallel branch (crate-internal).
///
/// Branch bodies are inherently heterogeneous: every `Branch::new` call
/// site captures a different closure and future type, and they are all
/// collected into one `Vec`, so exactly ONE erasure per branch is
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
    /// here: it runs when the parallel operation dispatches the branch.
    ///
    /// History records `name`. An empty name records
    /// `parallel-branch-{index}`, and a completion predicate sees
    /// `Some("")`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::{Branch, DurableContext};
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
        Self::with_name(Some(name.into()), factory)
    }

    /// A branch the caller does not name. A completion predicate sees
    /// `None`; history records `parallel-branch-{index}`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::{Branch, DurableContext};
    ///
    /// let branch: Branch<i32> = Branch::unnamed(|_ctx: DurableContext| async move { Ok(42) });
    /// assert_eq!(branch.name(), None);
    /// ```
    pub fn unnamed<F, Fut>(factory: F) -> Self
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    {
        Self::with_name(None, factory)
    }

    /// Builds a branch with the caller's name, if any (internal).
    fn with_name<F, Fut>(name: Option<String>, factory: F) -> Self
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    {
        let (ctx_tx, ctx_rx) = tokio::sync::oneshot::channel::<DurableContext>();
        Self {
            name,
            body: BranchBody {
                ctx_tx,
                // The SDK does the pinning and BoxError -> internal-carrier
                // type erasure so callers write a plain `async move` body
                // with `?`. This is the branch's ONE erasure: the factory
                // closure and its concrete future both live inside this box;
                // there is no separate closure box producing a second
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

    /// Returns the name the caller gave, or `None` for a branch built with
    /// [`Branch::unnamed`].
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Consumes the branch and returns its name and erased body (internal).
    pub(crate) fn into_parts(self) -> (Option<String>, BranchBody<O>) {
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
/// use aws_durable_execution_sdk as durable;
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

/// Resolves a pending callback from its record once the record is
/// terminal.
pub(crate) type CallbackResolver<O> =
    Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<O, OperationError>> + Send>> + Send>;

/// What a pending callback's `result()` needs, besides its context and
/// wire ID, to build the plugin info of the callback.
#[derive(Debug)]
pub(crate) struct CallbackIdentity {
    /// The callback's name.
    pub(crate) name: Option<String>,
    /// The name the plugin hooks report instead of `name`, if any.
    pub(crate) plugin_name: Option<String>,
    /// The callback's resolved subtype.
    pub(crate) sub_type: String,
}

/// Internal state for a callback: either settled (replay) or pending (live).
enum CallbackState<O> {
    /// The callback has a known outcome from the checkpoint log (replay).
    Settled(Option<Result<O, OperationError>>),
    /// The callback is in flight: the result waits on its record.
    Pending {
        /// The context the callback was created on.
        ctx: DurableContext,
        /// The callback operation's wire ID.
        wire_id: String,
        /// The name, plugin name and subtype of the callback.
        identity: CallbackIdentity,
        /// Builds the outcome from the terminal record.
        resolver: CallbackResolver<O>,
    },
}

impl<O: std::fmt::Debug> std::fmt::Debug for CallbackState<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Settled(outcome) => f.debug_tuple("Settled").field(outcome).finish(),
            Self::Pending {
                ctx,
                wire_id,
                identity,
                ..
            } => f
                .debug_struct("Pending")
                .field("ctx", ctx)
                .field("wire_id", wire_id)
                .field("identity", identity)
                .finish_non_exhaustive(),
        }
    }
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
    /// `resolver` builds the outcome from the record once it is terminal.
    pub(crate) fn new_pending(
        id: String,
        ctx: DurableContext,
        wire_id: String,
        identity: CallbackIdentity,
        resolver: CallbackResolver<O>,
    ) -> Self {
        Self {
            id,
            state: CallbackState::Pending {
                ctx,
                wire_id,
                identity,
                resolver,
            },
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
    /// use aws_durable_execution_sdk as durable;
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
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the callback result as a [`DurableFuture`].
    ///
    /// For settled callbacks (replay), the future resolves with the stored
    /// outcome immediately. For pending callbacks (live), the future waits
    /// for the external completion. It resolves in the same invocation when
    /// a checkpoint response carries the completed callback; otherwise the
    /// invocation suspends once no other SDK work can make progress.
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
    /// use aws_durable_execution_sdk as durable;
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
            CallbackState::Pending {
                ctx,
                wire_id,
                identity,
                resolver,
            } => Self::pending_result(ctx, wire_id, identity, resolver),
        }
    }

    /// Waits for the callback result on its own SDK task, as `.spawn()` does
    /// for other operations, and returns its handle. The SDK refreshes the
    /// callback's status only after the handle is first polled. Dropping the
    /// handle loses only the result: the wait keeps running until the
    /// callback settles, until its scope closes, or until the invocation
    /// ends, and dropping it does not count as the first poll. A stopped
    /// wait stops at its next `.await`.
    ///
    /// For a settled callback (replay) it returns the same future as
    /// [`result()`](Self::result), which resolves with the stored outcome
    /// and starts no task.
    ///
    /// # Errors
    ///
    /// For a pending callback, when no Tokio runtime is running, the
    /// returned future resolves to an error. A settled callback needs no
    /// runtime.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let cb = ctx.create_callback::<String>().name("approval").await?;
    ///     let approval = cb.spawn_result();
    ///     ctx.step(|_| async { Ok(()) }).name("other-work").await?;
    ///     Ok(approval.await?)
    /// }
    /// ```
    pub fn spawn_result(self) -> DurableFuture<O> {
        match self.state {
            CallbackState::Settled(_) => self.result(),
            CallbackState::Pending {
                ctx,
                wire_id,
                identity,
                resolver,
            } => {
                let flag = Arc::new(AtomicBool::new(false));
                let waiting = ctx.with_awaited(Arc::clone(&flag));
                let future = Self::pending_result(waiting, wire_id, identity, resolver);
                DurableFuture::spawn_blessed(future, &ctx, Some(flag))
            }
        }
    }

    /// The future of a pending callback's result: it waits until the record
    /// is terminal, then builds the outcome from it, as the create-time
    /// replay does.
    ///
    /// The callback was created in this invocation, so a `SUCCEEDED`,
    /// `FAILED` or `TIMED_OUT` record it reads changed in this invocation.
    /// So it fires `on_operation_end` with `is_replay` `false` before the
    /// outcome is built, as JS does after its status wait.
    fn pending_result(
        ctx: DurableContext,
        wire_id: String,
        identity: CallbackIdentity,
        resolver: CallbackResolver<O>,
    ) -> DurableFuture<O> {
        let gate = ctx.clone();
        DurableFuture::for_operation(&gate, async move {
            ctx.wait_for_record(&wire_id, |status| {
                !matches!(
                    status,
                    crate::engine::CheckpointStatus::Started
                        | crate::engine::CheckpointStatus::Pending
                )
            })
            .await;
            if let Some(plugins) = ctx.plugins() {
                let terminal = ctx
                    .with_checkpoint_record_by_wire(&wire_id, |record| {
                        crate::callback::is_terminal_callback_status(&record.status)
                    })
                    .unwrap_or(false);
                if terminal {
                    let info = crate::callback::callback_info(
                        &ctx,
                        &wire_id,
                        &identity.sub_type,
                        identity.name.as_deref(),
                        identity.plugin_name.as_deref(),
                    )
                    .with_replay(false);
                    crate::plugin::operation_ended(plugins, &info);
                }
            }
            resolver().await
        })
    }
}

/// A set of parallel branches whose results have different types, for
/// [`DurableContext::parallel_mixed`].
///
/// Each [`add`](Self::add) or [`add_with_serdes`](Self::add_with_serdes)
/// call adds one branch and returns a [`BranchHandle`] typed with that
/// branch's result. Branches start, and are numbered, in the order they
/// were added. The number of branches is unlimited.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use durable::BranchSet;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     let mut branches = BranchSet::new();
///     let count = branches.add("count", |_ctx| async { Ok(7_u32) });
///     let label = branches.add("label", |_ctx| async { Ok("ready".to_owned()) });
///     ctx.parallel_mixed(branches).name("fan-out").await?;
///     Ok(format!("{}: {}", count.into_result()?, label.into_result()?))
/// }
/// ```
#[non_exhaustive]
pub struct BranchSet {
    /// Each branch's caller name and erased body, in the order added.
    branches: Vec<(Option<String>, BranchBody<MixedValue>)>,
    /// Each branch's decoder, at the branch's index.
    decoders: Vec<Arc<dyn BranchDecoder>>,
    /// Whether each branch uses the default JSON serdes, at its index.
    json: Vec<bool>,
}

impl Default for BranchSet {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BranchSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self
            .branches
            .iter()
            .map(|(name, _)| name.as_deref().unwrap_or_default())
            .collect();
        f.debug_struct("BranchSet")
            .field("names", &names)
            .finish_non_exhaustive()
    }
}

impl BranchSet {
    /// Creates an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            decoders: Vec::new(),
            json: Vec::new(),
        }
    }

    /// Adds a branch whose result is encoded and decoded as JSON.
    ///
    /// This is [`add_with_serdes`](Self::add_with_serdes) with
    /// [`JsonSerdes`](crate::JsonSerdes). The batch record holds the
    /// branch's value as JSON, as [`DurableContext::parallel`] does with
    /// its default serdes.
    #[must_use]
    pub fn add<T, F, Fut>(&mut self, name: impl Into<String>, factory: F) -> BranchHandle<T>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, BoxError>> + Send + 'static,
    {
        self.insert(name.into(), crate::JsonSerdes, factory, true)
    }

    /// Adds a branch whose result is encoded and decoded by `serdes`.
    ///
    /// `name` and `factory` have the same types as in [`Branch::new`].
    /// History records `name`, or `parallel-branch-{index}` when it is
    /// empty, and a completion predicate sees `Some(name)`. `serdes` must
    /// implement [`Serdes<T>`](crate::Serdes), so a serdes written for
    /// another type fails to compile. It is called wherever the item serdes
    /// of [`DurableContext::parallel`] is called for this branch, with the
    /// same [`SerdesContext`](crate::serdes::SerdesContext). To share one
    /// serdes across branches, pass a clone of one `Arc` to each call.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    ///
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::{FileSystemSerdes, SerdesContext};
    /// use durable::{BoxError, BranchSet, Serdes};
    ///
    /// struct Order {
    ///     id: u64,
    /// }
    ///
    /// // A serdes for `Order` only: attaching it to a branch of another
    /// // type fails to compile.
    /// struct OrderSerdes;
    ///
    /// impl Serdes<Order> for OrderSerdes {
    ///     async fn serialize(&self, value: Order, _ctx: SerdesContext) -> Result<String, BoxError> {
    ///         Ok(value.id.to_string())
    ///     }
    ///     async fn deserialize(&self, wire: String, _ctx: SerdesContext) -> Result<Order, BoxError> {
    ///         Ok(Order { id: wire.parse()? })
    ///     }
    /// }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u64, durable::BoxError> {
    ///     let files = Arc::new(FileSystemSerdes::new("/mnt/efs"));
    ///     let mut branches = BranchSet::new();
    ///     let order = branches.add_with_serdes("order", OrderSerdes, |_ctx| async {
    ///         Ok(Order { id: 42 })
    ///     });
    ///     let report = branches.add_with_serdes("report", Arc::clone(&files), |_ctx| async {
    ///         Ok("a large report".to_owned())
    ///     });
    ///     ctx.parallel_mixed(branches).name("process-order").await?;
    ///     let order = order.into_result()?;
    ///     let _report: String = report.into_result()?;
    ///     Ok(order.id)
    /// }
    /// ```
    #[must_use]
    pub fn add_with_serdes<T, S, F, Fut>(
        &mut self,
        name: impl Into<String>,
        serdes: S,
        factory: F,
    ) -> BranchHandle<T>
    where
        T: Send + 'static,
        S: crate::Serdes<T> + 'static,
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, BoxError>> + Send + 'static,
    {
        self.insert(name.into(), serdes, factory, false)
    }

    /// Adds one branch (internal). `json` is `true` when the batch record
    /// holds the branch's value as JSON.
    fn insert<T, S, F, Fut>(
        &mut self,
        name: String,
        serdes: S,
        factory: F,
        json: bool,
    ) -> BranchHandle<T>
    where
        T: Send + 'static,
        S: crate::Serdes<T> + 'static,
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, BoxError>> + Send + 'static,
    {
        let index = self.branches.len();
        let record_name = if name.is_empty() {
            format!("parallel-branch-{index}")
        } else {
            name.clone()
        };
        let slot: BranchSlot<T> = Arc::new(std::sync::Mutex::new(None));
        let serdes = Arc::new(serdes);
        let decoder = BranchShared {
            serdes: Arc::clone(&serdes),
            slot: Arc::clone(&slot),
        };
        let body_slot = Arc::clone(&slot);
        let branch = Branch::new(name, move |ctx: DurableContext| async move {
            let value = factory(ctx).await?;
            Ok(MixedValue(Box::new(TypedBranchValue {
                value,
                serdes,
                slot: body_slot,
            })))
        });
        self.branches.push(branch.into_parts());
        self.decoders.push(Arc::new(decoder));
        self.json.push(json);
        BranchHandle {
            name: record_name,
            slot,
        }
    }

    /// Consumes the set and returns each branch's caller name and body,
    /// the decoders and the JSON flags, all in branch order (internal).
    #[expect(clippy::type_complexity)]
    // reason: the three parts go to three different places, and a named type for each would add nothing
    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<(Option<String>, BranchBody<MixedValue>)>,
        Arc<Vec<Arc<dyn BranchDecoder>>>,
        Arc<[bool]>,
    ) {
        (self.branches, Arc::new(self.decoders), Arc::from(self.json))
    }
}

/// The result of one branch of a [`BranchSet`].
///
/// [`into_result`](Self::into_result) returns the branch's result once the
/// `.await` of the [`DurableContext::parallel_mixed`] that ran the set has
/// returned, or the `.await` of the future its `.spawn()` returned. Before
/// that, the handle has no result, even when its branch
/// has finished, so what a handle returns never depends on timing. Dropping
/// a handle does not cancel or skip its branch.
#[non_exhaustive]
pub struct BranchHandle<T> {
    /// The name history records for the branch.
    name: String,
    /// Shared with the branch: filled when the parallel's `.await` returns.
    slot: BranchSlot<T>,
}

impl<T> std::fmt::Debug for BranchHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_result = lock_slot(&self.slot).is_some();
        f.debug_struct("BranchHandle")
            .field("name", &self.name)
            .field("has_result", &has_result)
            .finish_non_exhaustive()
    }
}

impl<T> BranchHandle<T> {
    /// Returns the branch's result, without waiting.
    ///
    /// - `Ok(value)` when the branch succeeded. `value` is what the
    ///   branch's serdes decoded, so the first run and replay return the
    ///   same value.
    /// - The branch's own error when the branch failed: a
    ///   [`ChildContextError`](crate::ChildContextError) of kind
    ///   [`ChildFailed`](crate::ChildContextErrorKind::ChildFailed) that
    ///   holds the recorded failure, the error `parallel(...).await`
    ///   returns when a failed item ends the batch.
    ///
    /// # Errors
    ///
    /// Besides the branch's own error, returns an error of kind
    /// [`NotCompleted`](crate::ChildContextErrorKind::NotCompleted), whose
    /// message names the branch, when the handle has no result: the
    /// parallel has not returned, it returned an error of its own, the
    /// branch never started, or the batch completed while the branch still
    /// ran.
    pub fn into_result(self) -> Result<T, OperationError> {
        let taken = lock_slot(&self.slot).take();
        taken.unwrap_or_else(|| Err(not_completed_error(&self.name)))
    }
}

/// The slot a [`BranchHandle`] shares with its branch.
type BranchSlot<T> = Arc<std::sync::Mutex<Option<Result<T, OperationError>>>>;

/// Locks a branch slot. A poisoned lock still holds a valid slot value.
fn lock_slot<T>(
    slot: &std::sync::Mutex<Option<Result<T, OperationError>>>,
) -> std::sync::MutexGuard<'_, Option<Result<T, OperationError>>> {
    slot.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The error of a [`BranchHandle`] that has no result.
fn not_completed_error(name: &str) -> OperationError {
    OperationError::from_kind(crate::OperationErrorKind::ChildContext(
        crate::error::ChildContextError::new(
            crate::ChildContextErrorKind::NotCompleted,
            Some(format!("parallel branch \"{name}\" has no result").into()),
        ),
    ))
}

/// The item type of `parallel_mixed`: one branch's value, erased, carrying
/// its own encoder and its handle's slot (crate-internal).
pub(crate) struct MixedValue(pub(crate) Box<dyn ErasedBranchValue>);

/// A branch value whose type is erased (crate-internal).
pub(crate) trait ErasedBranchValue: Send {
    /// Encodes the value with the branch's serdes.
    fn encode(
        self: Box<Self>,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>>;
    /// Stores `Ok(value)` in the branch handle's slot.
    fn publish(self: Box<Self>);
}

/// Decodes the values of one branch and fails its handle (crate-internal).
pub(crate) trait BranchDecoder: Send + Sync {
    /// Decodes a wire string with the branch's serdes.
    fn decode(
        &self,
        wire: String,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<MixedValue, BoxError>> + Send + '_>>;
    /// Stores `Err(err)` in the branch handle's slot.
    fn publish_error(&self, err: OperationError);
}

/// A branch's value with its serdes and its handle's slot.
struct TypedBranchValue<T, S> {
    value: T,
    serdes: Arc<S>,
    slot: BranchSlot<T>,
}

impl<T, S> ErasedBranchValue for TypedBranchValue<T, S>
where
    T: Send + 'static,
    S: crate::Serdes<T>,
{
    fn encode(
        self: Box<Self>,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>> {
        let Self { value, serdes, .. } = *self;
        Box::pin(async move { serdes.serialize(value, ctx).await })
    }

    fn publish(self: Box<Self>) {
        let Self { value, slot, .. } = *self;
        *lock_slot(&slot) = Some(Ok(value));
    }
}

/// One branch's serdes and handle slot: the branch's [`BranchDecoder`].
struct BranchShared<T, S> {
    serdes: Arc<S>,
    slot: BranchSlot<T>,
}

impl<T, S> BranchDecoder for BranchShared<T, S>
where
    T: Send + 'static,
    S: crate::Serdes<T>,
{
    fn decode(
        &self,
        wire: String,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<MixedValue, BoxError>> + Send + '_>> {
        Box::pin(async move {
            let value = self.serdes.deserialize(wire, ctx).await?;
            Ok(MixedValue(Box::new(TypedBranchValue {
                value,
                serdes: Arc::clone(&self.serdes),
                slot: Arc::clone(&self.slot),
            })))
        })
    }

    fn publish_error(&self, err: OperationError) {
        *lock_slot(&self.slot) = Some(Err(err));
    }
}

#[cfg(test)]
#[expect(clippy::panic)]
// reason: test assertions
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
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

    /// Test 16: a combinator future claims one ID however often
    /// `claim_now` runs, and its first poll after a claim claims nothing.
    #[tokio::test]
    async fn claim_now_claims_once() {
        let ctx = test_ctx_with_client();
        let mut all = ctx.try_join_all(Vec::<DurableFuture<i32>>::new()).future();
        assert!(all.is_unclaimed_combinator());
        all.claim_now();
        assert!(!all.is_unclaimed_combinator());
        all.claim_now();
        assert_eq!(
            all.await.expect("an empty join succeeds"),
            Vec::<i32>::new()
        );
        assert_eq!(
            ctx.mint_id().positional(),
            "2",
            "the combinator claimed 1 only"
        );
    }

    /// Test 17: a step's future carries no pending claim.
    #[tokio::test]
    async fn claim_now_is_noop_for_non_combinator() {
        let ctx = test_ctx_with_client();
        let mut step = ctx.step(|_| async { Ok(1_i32) }).future();
        assert!(!step.is_unclaimed_combinator());
        assert_eq!(step.unclaimed_input_message("9"), None);
        step.claim_now();
        assert_eq!(ctx.mint_id().positional(), "2", "only the step claimed");
        assert_eq!(step.await.expect("the step succeeds"), 1);
    }

    /// A panic inside a `.spawn()`ed operation body must surface to the
    /// awaiter as the ORIGINAL panic (payload intact), exactly like the lazy
    /// path, not as a fabricated "spawned task was cancelled" step error.
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

    /// `.spawn()` on a thread with no Tokio runtime resolves to the error
    /// the task-ownership check would return, records nothing, and never
    /// runs the body.
    #[tokio::test]
    async fn spawn_outside_runtime_returns_error() {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        let ran = Arc::new(AtomicBool::new(false));
        let thread_ctx = ctx.clone();
        let thread_ran = Arc::clone(&ran);
        let polled = crate::task::test_support::poll_once_without_runtime(move || {
            thread_ctx
                .step(move |_| async move {
                    thread_ran.store(true, Ordering::SeqCst);
                    Ok(1_i32)
                })
                .spawn()
        });
        let Poll::Ready(Err(err)) = polled else {
            panic!("expected an immediate error, got {polled:?}");
        };
        let crate::error::OperationErrorKind::Step(step_err) = err.kind() else {
            panic!(
                "expected the task-ownership error kind, got {:?}",
                err.kind()
            );
        };
        assert!(matches!(
            step_err.kind(),
            crate::error::StepErrorKind::ExecutionFailed
        ));
        assert!(
            format!("{err:#}").contains(crate::task::NO_RUNTIME_MESSAGE),
            "unexpected error: {err:#}"
        );
        assert!(!ran.load(Ordering::SeqCst), "the body must not run");
        assert!(client.recorded_updates().is_empty(), "nothing is recorded");
    }

    /// Dropping the handle of a `.spawn()`ed operation does not cancel the
    /// spawned task: the body keeps running. The task stops, without
    /// unwinding anything into the caller, when the tracker aborts it.
    #[tokio::test]
    async fn dropped_spawned_future_keeps_task_running_without_panic() {
        #[derive(Clone)]
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let ctx = test_ctx_with_client();
        let dropped = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        // The guard lives in the closure environment, so the body and the
        // guard drop together.
        let guard = DropGuard(Arc::clone(&dropped));
        let body_started = Arc::clone(&started);

        let fut: DurableFuture<i32> = ctx
            .step(move |_| async move {
                let _guard = guard;
                body_started.store(true, Ordering::SeqCst);
                std::future::pending::<()>().await;
                Ok(0)
            })
            .spawn();
        drop(fut);

        for _ in 0..50 {
            if started.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            started.load(Ordering::SeqCst),
            "the body runs after the drop"
        );
        assert!(
            !dropped.load(Ordering::SeqCst),
            "the dropped handle does not cancel the body"
        );
        let tasks = Arc::clone(ctx.coordinator().tasks());
        tasks.abort_all();
        tasks.wait_idle().await;
        assert!(dropped.load(Ordering::SeqCst), "abort_all cancels the body");
    }

    #[test]
    fn handle_of_a_set_that_never_ran() {
        let mut set = BranchSet::new();
        let handle = set.add("inventory", |_ctx| async { Ok(1_u32) });
        drop(set);
        let Err(err) = handle.into_result() else {
            panic!("a handle whose set never ran has no result");
        };
        let crate::OperationErrorKind::ChildContext(child) = err.kind() else {
            panic!("the error is a child context error: {err:?}");
        };
        assert!(matches!(
            child.kind(),
            crate::ChildContextErrorKind::NotCompleted
        ));
        assert_eq!(child.kind().to_string(), "branch has no result");
        let source = std::error::Error::source(child).expect("the error has a source");
        assert_eq!(
            source.to_string(),
            r#"parallel branch "inventory" has no result"#
        );
    }

    #[test]
    fn empty_branch_name_records_the_default_name() {
        let mut set = BranchSet::default();
        let handle = set.add("", |_ctx| async { Ok(1_u32) });
        drop(set);
        let err = handle.into_result().expect_err("no result");
        let source = std::error::Error::source(&err)
            .and_then(std::error::Error::source)
            .map(ToString::to_string);
        assert_eq!(
            source.as_deref(),
            Some(r#"parallel branch "parallel-branch-0" has no result"#)
        );
    }

    #[test]
    fn branch_set_and_handle_debug() {
        let mut set = BranchSet::new();
        assert_eq!(format!("{set:?}"), "BranchSet { names: [], .. }");
        let handle = set.add("a", |_ctx| async { Ok(1_u32) });
        let _other = set.add_with_serdes("b", crate::JsonSerdes, |_ctx| async { Ok(2_u32) });
        assert_eq!(format!("{set:?}"), r#"BranchSet { names: ["a", "b"], .. }"#);
        assert_eq!(
            format!("{handle:?}"),
            r#"BranchHandle { name: "a", has_result: false, .. }"#
        );
        *lock_slot(&handle.slot) = Some(Ok(1));
        assert_eq!(
            format!("{handle:?}"),
            r#"BranchHandle { name: "a", has_result: true, .. }"#
        );
    }
}
