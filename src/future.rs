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
        }
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

    /// Spawns the operation future on its own task, blessed in
    /// `task_ownership` before its first poll.
    ///
    /// The task starts through the task gateway
    /// ([`crate::task::spawn_tracked`]) and holds a permit of `coordinator`
    /// whenever it can make progress, so the invocation does not suspend
    /// while it runs. A wait inside it registers like any other wait.
    /// Dropping the returned future aborts the task. A panic in the
    /// operation resumes on the task that awaits the handle, with its
    /// original payload. When no Tokio runtime is running, the returned
    /// future resolves to an error and the operation records nothing.
    pub(crate) fn spawn_blessed(
        future: Self,
        task_ownership: &std::sync::Arc<crate::driver::TaskOwnership>,
        coordinator: &std::sync::Arc<crate::coordinator::InvocationCoordinator>,
    ) -> Self {
        match crate::task::spawn_tracked(coordinator, task_ownership, future) {
            Ok(task) => Self::from_async(async move { task.await.into_value() }),
            // No Tokio runtime is running on this thread: the operation
            // records nothing and resolves to an error.
            Err(no_runtime) => {
                Self::from_async(
                    async move { Err(crate::task::NoRuntime::operation_error(no_runtime)) },
                )
            }
        }
    }
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
/// use aws_durable_execution_sdk::{Settled, OperationError};
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
    name: String,
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
        let (ctx_tx, ctx_rx) = tokio::sync::oneshot::channel::<DurableContext>();
        Self {
            name: name.into(),
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

    /// Returns the branch name.
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
        /// Builds the outcome from the terminal record.
        resolver: CallbackResolver<O>,
    },
}

impl<O: std::fmt::Debug> std::fmt::Debug for CallbackState<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Settled(outcome) => f.debug_tuple("Settled").field(outcome).finish(),
            Self::Pending { ctx, wire_id, .. } => f
                .debug_struct("Pending")
                .field("ctx", ctx)
                .field("wire_id", wire_id)
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
        resolver: CallbackResolver<O>,
    ) -> Self {
        Self {
            id,
            state: CallbackState::Pending {
                ctx,
                wire_id,
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
                resolver,
            } => {
                // Live path: wait until the record is terminal, then build
                // the outcome from it, as the create-time replay does.
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
                    resolver().await
                })
            }
        }
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

    /// Dropping the handle of a `.spawn()`ed operation cancels the spawned
    /// task (abort-on-drop): cancellation, distinct from a panic, tears the
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
