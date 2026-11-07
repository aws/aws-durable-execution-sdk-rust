//! Future types for durable operations.
//!
//! [`DurableFuture`] is the uniform handle for all durable operation
//! results. [`Branch`] defines a named parallel branch, and [`BranchSet`]
//! a set of branches with different result types whose results each
//! [`BranchHandle`] returns. [`Callback`] holds a callback token and its
//! result future.

use std::future::Future;
use std::marker::PhantomData;
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
/// `DurableFuture<O>`. Every operation except a combinator and a
/// [`DurableContext::with_retry`] block has already claimed its operation
/// ID when its builder was created. A `with_retry` block with its own
/// record claims its ID when the builder becomes this future; an unwrapped
/// block claims none, and its first attempt claims the caller's next ID at
/// the block's start step (its first poll or `.spawn()`). A combinator
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
/// keeps running until it settles, its scope closes or the invocation ends.
/// An operation that settles records its outcome and fires its plugin
/// hooks. Its error goes into its record and is returned to nobody, and a
/// panic in it fails the invocation. A running detached operation keeps the
/// invocation from suspending, and one that waits only on the service lets
/// it suspend. When the handler returns, the invocation ends and stops each
/// detached operation that still runs. A stopped operation fires no end
/// hook. Its record, if it has one, keeps the last state written, which can
/// be `STARTED`, and the execution can end with the handler's result.
///
/// The SDK stops work in three cases:
///
/// - a combinator stops the `.future()` inputs it started that are still
///   running when it resolves, and lets an input started with `.spawn()`
///   run;
/// - a closed scope stops the work still running inside it. A scope closes
///   when a child context's body returns, when a `with_retry` block with
///   its own record ends, when an attempt of any `with_retry` block ends
///   (a block with `.without_child_context()` has no scope of its own,
///   only its attempts do), and when a `map`, `parallel` or
///   `parallel_mixed` item, or the whole batch, completes;
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
    /// The task context captured when the operation was created: the
    /// frames of its context and the span current at the call. A task that
    /// runs this future runs inside it, whoever starts the task.
    task_cx: crate::task::TaskContext,
    /// The start step of a lazy operation whose execution begins with one
    /// (see [`crate::start`]): it runs the start step and returns the rest
    /// of the operation. `None` for every other future, and once it ran.
    /// While it is `Some`, `inner` is a placeholder that is never polled.
    begin: Option<BeginFn<O>>,
    /// Whether a combinator begins this future by polling it once on its
    /// own task: a lazy invoke or `wait_for_condition`, whose first poll
    /// serializes its input before it can queue its `START`.
    first_poll_start: bool,
    /// The ownership gate of an operation future: the context whose owner
    /// may poll it. `None` for a future that executes nothing of its own:
    /// a spawned handle, a settled callback result, an error future, and a
    /// combinator until it claims its ID. Boxed, so the gate adds one
    /// pointer to every future.
    gate: Option<Box<PollGate>>,
}

/// The ownership gate of one operation future (see [`DurableFuture`]).
///
/// Every poll checks ownership, before the operation checks admission: an
/// abandoned scope makes the operation pend, so a later check would never
/// run for a future its owner sent out unpolled. The gate keeps no
/// identity of an earlier poller, because Tokio may give a new task the ID
/// of a task that exited, and a cached ID would then admit a task the
/// invocation does not own. So a future that its owner polled until it
/// suspended and then moved to such a task fails the execution when that
/// task resumes it. The check costs the owner comparison and, for a task
/// other than the owner, one lookup in the set of tasks the SDK started.
struct PollGate {
    ctx: DurableContext,
}

impl PollGate {
    fn new(ctx: &DurableContext) -> Self {
        Self { ctx: ctx.clone() }
    }

    /// Admits the current poll when the calling task or thread owns the
    /// context. A refused poll has recorded the fatal context usage error.
    fn admit(&mut self) -> Result<(), OperationError> {
        self.ctx.enforce_task_ownership()
    }
}

/// The start step of a lazy operation, and the builder of its remainder.
type BeginFn<O> =
    Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<O, OperationError>> + Send>> + Send>;

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
            task_cx: crate::task::TaskContext::none(),
            begin: None,
            first_poll_start: false,
            gate: None,
        }
    }

    /// Maps the output and keeps the task context, so a
    /// task that later runs the mapped future runs it inside the frames
    /// and span of the operation's creation, as for the unmapped future.
    pub(crate) fn map_output<T, M>(self, map: M) -> DurableFuture<T>
    where
        T: Send + 'static,
        M: FnOnce(Result<O, OperationError>) -> Result<T, OperationError> + Send + 'static,
    {
        let Self {
            inner,
            awaited,
            claim,
            task_cx,
            begin,
            first_poll_start,
            gate,
        } = self;
        match claim {
            // An unclaimed combinator stays unclaimed: the mapped future
            // keeps a pending claim that claims the ID and maps the claimed
            // future, so `claim_now` takes the claimed future's context.
            Some(claim) => {
                let PendingClaim {
                    method,
                    context,
                    run,
                } = claim;
                DurableFuture::deferred(PendingClaim::new(method, context, move || {
                    run().map_output(map)
                }))
            }
            // A lazy operation stays lazy: the mapped future keeps the
            // start step, and maps the remainder the start step returns.
            None => match begin {
                Some(begin) => DurableFuture {
                    inner: Box::pin(std::future::pending()),
                    awaited,
                    claim: None,
                    task_cx,
                    begin: Some(Box::new(move || {
                        let rest = begin();
                        Box::pin(async move { map(rest.await) })
                    })),
                    first_poll_start,
                    gate,
                },
                None => DurableFuture {
                    inner: Box::pin(async move { map(inner.await) }),
                    awaited,
                    claim: None,
                    task_cx,
                    begin: None,
                    first_poll_start,
                    gate,
                },
            },
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
            task_cx: crate::task::TaskContext::none(),
            begin: None,
            first_poll_start: false,
            gate: None,
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
            self.task_cx = claimed.task_cx;
            self.gate = claimed.gate;
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
        let mut future = Self::from_async(async move {
            gate.admit_operation().await;
            drop(gate);
            fut.await
        });
        future.task_cx = crate::task::TaskContext::current();
        future.gate = Some(Box::new(PollGate::new(ctx)));
        future
    }

    /// The task context captured when the operation was created.
    pub(crate) fn task_context(&self) -> crate::task::TaskContext {
        self.task_cx.clone()
    }

    /// Creates the future of one operation whose start step already ran on
    /// the caller's task (see [`crate::start`]): `fut` is the rest of the
    /// operation.
    ///
    /// It captures the task context current at the call, as
    /// [`Self::for_operation`] does.
    pub(crate) fn for_started_operation<F>(ctx: &DurableContext, fut: F) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        let mut future = Self::from_async(fut);
        future.task_cx = crate::task::TaskContext::current();
        future.gate = Some(Box::new(PollGate::new(ctx)));
        future
    }

    /// Creates the lazy future of one operation whose execution begins with
    /// the start step (see [`crate::start`]): a step, a wait, a child
    /// context, `with_retry`, a map, a parallel, a `parallel_mixed`, a
    /// `create_callback` or a `wait_for_callback`.
    ///
    /// `begin` runs the start step and returns the rest of the operation.
    /// It runs on the first poll, in the poller's context, or earlier when a
    /// combinator begins the future as an input, in the creator's context
    /// ([`Self::begin_in_creator_context`]). The start step checks
    /// admission itself before it queues anything, so this adds no
    /// admission gate of its own, unlike [`Self::for_operation`].
    pub(crate) fn for_start_step<F>(
        ctx: &DurableContext,
        begin: impl FnOnce() -> F + Send + 'static,
    ) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        Self {
            // A placeholder: `begin_now` replaces it before the first poll.
            inner: Box::pin(std::future::pending()),
            awaited: None,
            claim: None,
            begin: Some(Box::new(move || Box::pin(begin()))),
            first_poll_start: false,
            task_cx: crate::task::TaskContext::current(),
            gate: Some(Box::new(PollGate::new(ctx))),
        }
    }

    /// Runs the start step of a lazy operation now, on the calling task,
    /// in the caller's context: the first poll calls this, so an operation
    /// awaited directly runs both its start step and its body in its
    /// awaiter's frames and span (#126 rule 5.2).
    ///
    /// It runs the start step once, and does nothing for a future without
    /// one.
    fn begin_now(&mut self) {
        if let Some(begin) = self.begin.take() {
            self.inner = begin();
        }
    }

    /// Runs the start step of a lazy combinator input now, on the calling
    /// task, inside the task context captured at the operation's creation:
    /// its frames and its span. So `on_operation_start` and the start step
    /// see the creator's context when a combinator created elsewhere begins
    /// the input, as the input's remainder does on the task the combinator
    /// spawns with that context (#126 rule 5.2).
    ///
    /// Only a combinator that begins its inputs calls this. It runs the
    /// start step once, and does nothing for a future without one.
    pub(crate) fn begin_in_creator_context(&mut self) {
        if let Some(begin) = self.begin.take() {
            self.inner = self.task_cx.enter(begin);
        }
    }

    /// Creates the future of a lazy invoke or `wait_for_condition` on
    /// `ctx`, as [`Self::for_operation`] does. A combinator begins such an
    /// input by polling it once on its own task, because its first poll
    /// serializes its input before it can queue its `START`.
    pub(crate) fn for_first_poll_operation<F>(ctx: &DurableContext, fut: F) -> Self
    where
        F: Future<Output = Result<O, OperationError>> + Send + 'static,
    {
        let mut future = Self::for_operation(ctx, fut);
        future.first_poll_start = true;
        future
    }

    /// Reports whether a combinator begins this future by polling it once
    /// on its own task (see [`Self::for_first_poll_operation`]).
    pub(crate) fn starts_on_first_poll(&self) -> bool {
        self.first_poll_start
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
    /// result: the task keeps running. A caller that does not own the
    /// context fails the execution with a context usage error, no task
    /// starts, and a combinator claims no ID. Otherwise a combinator claims
    /// its ID here, before its task starts. A panic in the operation resumes on
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
        mut future: Self,
        ctx: &DurableContext,
        awaited: Option<Arc<AtomicBool>>,
    ) -> Self {
        // The ownership check runs before a combinator's pending claim, so
        // a refused `.spawn()` records the error of method `spawn` rather
        // than the claim's, and the combinator claims no ID.
        if let Some(refused) = Self::refuse_foreign_spawn(ctx) {
            return refused;
        }
        future.claim_now();
        let coordinator = ctx.coordinator();
        let task_cx = future.task_context();
        match crate::task::spawn_tracked(task_cx.clone(), coordinator, ctx.task_ownership(), future)
        {
            Ok(task) => {
                if let Some(abort) = task.abort_handle() {
                    ctx.admission().register_task(ctx.owners(), abort);
                }
                let mut handle = Self::from_async(async move { task.await.into_value() });
                handle.awaited = awaited.map(|flag| (flag, Arc::clone(coordinator)));
                // The handle keeps the operation's context. A combinator
                // that takes it spawns its input task with that context, so
                // the input runs inside the creator's frames and span.
                handle.task_cx = task_cx;
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

    /// Builds the operation future with `build` and spawns it as
    /// [`Self::spawn_blessed`] describes, for an operation whose `.spawn()`
    /// runs no start step on the caller's task.
    ///
    /// The task-ownership check runs before `build`: building an invoke or
    /// a `wait_for_condition` future runs its replay identity preflight,
    /// which records a fatal error on a mismatch, and the first fatal error
    /// wins. So a caller that does not own the context fails the execution
    /// with the error of method `spawn`, and `build` is not called.
    pub(crate) fn spawn_built(
        ctx: &DurableContext,
        awaited: Option<Arc<AtomicBool>>,
        build: impl FnOnce() -> Self,
    ) -> Self {
        if let Some(refused) = Self::refuse_foreign_spawn(ctx) {
            return refused;
        }
        Self::spawn_blessed(build(), ctx, awaited)
    }

    /// Starts an operation whose JS form begins inside the call: runs its
    /// start step on the caller's task, then spawns the rest of it.
    ///
    /// The task-ownership check runs first: a caller that does not own the
    /// context fails the execution, `start` is not called, and nothing is
    /// queued. The runtime check runs next: when no Tokio runtime is
    /// running, the returned future resolves to an error, `start` is not
    /// called, and nothing is queued. Otherwise `start` builds the
    /// operation future after it ran the start step (see [`crate::start`])
    /// on this task: the admission check, the task-ownership check, and the
    /// `START` the
    /// operation's record requires, with the start hook its JS order fires
    /// at the call. Then the future runs on its own task, as
    /// [`Self::spawn_blessed`] describes.
    pub(crate) fn spawn_started(
        ctx: &DurableContext,
        awaited: Option<Arc<AtomicBool>>,
        start: impl FnOnce() -> Self,
    ) -> Self {
        // The task-ownership check runs first: before the runtime check, so
        // a thread with no runtime is refused rather than told it has none,
        // and before the admission check and the builder's replay identity
        // preflight, so a caller that does not own the context fails the
        // execution in an abandoned owner and when the record also
        // contradicts the operation.
        if let Some(refused) = Self::refuse_foreign_spawn(ctx) {
            return refused;
        }
        if let Err(no_runtime) = crate::task::ensure_runtime() {
            return Self::from_async(async move {
                Err(crate::task::NoRuntime::operation_error(no_runtime))
            });
        }
        Self::spawn_blessed(start(), ctx, awaited)
    }

    /// Refuses a `.spawn()` from a task or thread that does not own `ctx`:
    /// records the fatal context usage error for method `spawn` and returns
    /// a handle that resolves to an error. No task starts, so the operation
    /// takes no permit and its body never runs. Returns `None` for a caller
    /// that owns the context.
    fn refuse_foreign_spawn(ctx: &DurableContext) -> Option<Self> {
        let reason = ctx.task_ownership().check_current_task().err()?;
        ctx.coordinator().record_fatal(
            "Error".to_owned(),
            crate::context::foreign_caller_message("spawn"),
        );
        Some(Self::from_async(async move {
            Err(OperationError::from_kind(
                crate::error::OperationErrorKind::Step(crate::error::StepError::new(
                    crate::error::StepErrorKind::ExecutionFailed,
                    Some(reason.into()),
                )),
            ))
        }))
    }
}

impl<O: Send + 'static> Future for DurableFuture<O> {
    type Output = Result<O, OperationError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // The first poll of a combinator claims its operation ID.
        self.claim_now();
        // Every poll checks ownership before the operation checks
        // admission.
        if let Some(gate) = self.gate.as_mut()
            && let Err(refused) = gate.admit()
        {
            return Poll::Ready(Err(refused));
        }
        // The first poll of a lazy operation runs its start step, unless a
        // combinator ran it already.
        self.begin_now();
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
    /// heartbeat timeout, or external failure). Awaiting the result of a
    /// pending callback on a task or thread that does not own the context
    /// fails the execution with a context usage error.
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
    /// runtime. For a pending callback, a call on a task or thread that
    /// does not own the context fails the execution with a context usage
    /// error.
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
                // `pending_result` builds the future through `for_operation`,
                // so the task gets the creating context's frames and the span
                // current here. The capture point is this call, as for
                // `.spawn()`: it is where the user starts the waiting task.
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
            // Execution entry: a task or thread that does not own the
            // context fails the execution. A settled result executes
            // nothing, so it is not checked.
            ctx.enforce_task_ownership()?;
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
    branches: Vec<(Option<String>, BranchBody<BranchValue>)>,
    /// Each branch's decoder, at the branch's index.
    decoders: Vec<Arc<dyn BranchDecoder>>,
    /// Whether each branch uses the default JSON serdes, at its index.
    json: Vec<bool>,
    /// The set's identity, shared by every key and value of its branches.
    identity: Arc<SetIdentity>,
}

/// The identity of one [`BranchSet`]: each set allocates its own, and two
/// identities are the same exactly when they are the same allocation.
struct SetIdentity;

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
            identity: Arc::new(SetIdentity),
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
        let key = BranchKey {
            identity: Arc::clone(&self.identity),
            index,
            _marker: PhantomData,
        };
        let slot: BranchSlot<T> = Arc::new(std::sync::Mutex::new(None));
        let serdes = Arc::new(serdes);
        let decoder = BranchShared {
            serdes: Arc::clone(&serdes),
            slot: Arc::clone(&slot),
            identity: Arc::clone(&self.identity),
            index,
        };
        let body_slot = Arc::clone(&slot);
        let body_identity = Arc::clone(&self.identity);
        let branch = Branch::new(name, move |ctx: DurableContext| async move {
            let value = factory(ctx).await?;
            Ok(BranchValue {
                value: Box::new(TypedBranchValue {
                    value,
                    serdes,
                    slot: body_slot,
                }),
                identity: body_identity,
                index,
            })
        });
        self.branches.push(branch.into_parts());
        self.decoders.push(Arc::new(decoder));
        self.json.push(json);
        BranchHandle {
            name: record_name,
            slot,
            key,
        }
    }

    /// Consumes the set and returns each branch's caller name and body,
    /// the decoders and the JSON flags, all in branch order (internal).
    #[expect(clippy::type_complexity)]
    // reason: the three parts go to three different places, and a named type for each would add nothing
    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<(Option<String>, BranchBody<BranchValue>)>,
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
    /// The branch's key.
    key: BranchKey<T>,
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
    /// Returns the key that reads this branch's value from the
    /// [`BatchResult`](crate::builders::map_parallel::BatchResult) a
    /// [`ParallelMixedBuilder::summary`](crate::builders::ParallelMixedBuilder::summary)
    /// receives, with
    /// [`BatchResult::value`](crate::builders::map_parallel::BatchResult::value).
    ///
    /// Every call returns a key of the same branch.
    #[must_use]
    pub fn key(&self) -> BranchKey<T> {
        self.key.clone()
    }

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

/// The value of one branch of a [`BranchSet`], with its type erased: the
/// item type of the [`BatchResult`](crate::builders::map_parallel::BatchResult) that a
/// [`ParallelMixedBuilder::summary`](crate::builders::ParallelMixedBuilder::summary)
/// receives.
///
/// A `BranchValue` exposes nothing of its value. Read the value with
/// [`BatchResult::value`](crate::builders::map_parallel::BatchResult::value) and the key that
/// [`BranchHandle::key`] returns. Its `Debug` text is `BranchValue { .. }`.
pub struct BranchValue {
    /// The value, with its serdes and its handle's slot.
    value: Box<dyn ErasedBranchValue>,
    /// The identity of the set the branch belongs to.
    identity: Arc<SetIdentity>,
    /// The branch's index in its set.
    index: usize,
}

impl std::fmt::Debug for BranchValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BranchValue").finish_non_exhaustive()
    }
}

impl BranchValue {
    /// Encodes the value with its branch's serdes (internal).
    pub(crate) fn encode(
        self,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>> {
        self.value.encode(ctx)
    }

    /// Stores `Ok(value)` in the branch handle's slot (internal).
    pub(crate) fn publish(self) {
        self.value.publish();
    }

    /// Returns the value when it belongs to `key`'s branch (internal).
    pub(crate) fn of_branch<T: 'static>(&self, key: &BranchKey<T>) -> Option<&T> {
        if !Arc::ptr_eq(&self.identity, &key.identity) || self.index != key.index {
            return None;
        }
        #[expect(clippy::expect_used)]
        // reason: the set and the index name one branch, and that branch's
        // body and decoder produce only values of the type of its key
        let value = self
            .value
            .value()
            .downcast_ref::<T>()
            .expect("a branch's key has the type of the branch's values");
        Some(value)
    }
}

/// The key of one branch of a [`BranchSet`], which reads the branch's value
/// from a [`BatchResult<BranchValue>`](crate::builders::map_parallel::BatchResult) with
/// [`BatchResult::value`](crate::builders::map_parallel::BatchResult::value).
///
/// [`BranchHandle::key`] returns it. A key reads only values of its own
/// branch: a key from another set returns `None`. It is `Clone`, `Send`
/// and `Sync` for every branch type, and holds no value of type `T`.
///
/// A key's type is exactly its branch's type. Subtyping cannot turn a key
/// of one type into a key of another, so a key never reads a value as a
/// type the branch did not produce:
///
/// ```compile_fail,E0308
/// use aws_durable_execution_sdk::BranchKey;
///
/// fn coerce(key: BranchKey<for<'a> fn(&'a ())>) -> BranchKey<fn(&'static ())> {
///     key
/// }
/// ```
pub struct BranchKey<T> {
    /// The identity of the set the branch belongs to.
    identity: Arc<SetIdentity>,
    /// The branch's index in its set.
    index: usize,
    /// The branch's type, without owning a `T`. `fn(T) -> T` makes the key
    /// invariant in `T`, so subtyping cannot change the type it reads.
    _marker: PhantomData<fn(T) -> T>,
}

impl<T> BranchKey<T> {
    /// The branch's index in its set (internal).
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

impl<T> Clone for BranchKey<T> {
    fn clone(&self) -> Self {
        Self {
            identity: Arc::clone(&self.identity),
            index: self.index,
            _marker: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for BranchKey<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BranchKey")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

/// A branch value whose type is erased (crate-internal).
pub(crate) trait ErasedBranchValue: Send {
    /// Encodes the value with the branch's serdes.
    fn encode(
        self: Box<Self>,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>>;
    /// Stores `Ok(value)` in the branch handle's slot.
    fn publish(self: Box<Self>);
    /// Returns the value.
    fn value(&self) -> &dyn std::any::Any;
}

/// Decodes the values of one branch and fails its handle (crate-internal).
pub(crate) trait BranchDecoder: Send + Sync {
    /// Decodes a wire string with the branch's serdes.
    fn decode(
        &self,
        wire: String,
        ctx: crate::serdes::SerdesContext,
    ) -> Pin<Box<dyn Future<Output = Result<BranchValue, BoxError>> + Send + '_>>;
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

    fn value(&self) -> &dyn std::any::Any {
        &self.value
    }
}

/// One branch's serdes, handle slot, set identity and index: the branch's
/// [`BranchDecoder`].
struct BranchShared<T, S> {
    serdes: Arc<S>,
    slot: BranchSlot<T>,
    identity: Arc<SetIdentity>,
    index: usize,
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
    ) -> Pin<Box<dyn Future<Output = Result<BranchValue, BoxError>> + Send + '_>> {
        Box::pin(async move {
            let value = self.serdes.deserialize(wire, ctx).await?;
            Ok(BranchValue {
                value: Box::new(TypedBranchValue {
                    value,
                    serdes: Arc::clone(&self.serdes),
                    slot: Arc::clone(&self.slot),
                }),
                identity: Arc::clone(&self.identity),
                index: self.index,
            })
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
            "test-function-arn".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token0".to_owned(),
        )
    }

    use std::future::IntoFuture as _;

    thread_local! {
        /// The wraps of [`Tag`] plugins entered on this thread, outermost
        /// first.
        static TAGS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
        /// Each user-function wrap entered on this thread: its tag and the
        /// tags entered around it.
        static ENTERED: std::cell::RefCell<Vec<(String, Vec<String>)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    fn current_tags() -> Vec<String> {
        TAGS.with(|tags| tags.borrow().clone())
    }

    /// A plugin whose wraps record `inv`, or the user function's name (its
    /// ID without one), while they are entered.
    struct Tag;

    impl Tag {
        fn wrap(site: &str, next: &mut dyn FnMut()) {
            TAGS.with(|tags| tags.borrow_mut().push(site.to_owned()));
            next();
            TAGS.with(|tags| tags.borrow_mut().pop());
        }
    }

    impl crate::plugin::Plugin for Tag {
        fn wrap_invocation(&self, _info: &crate::plugin::InvocationInfo, next: &mut dyn FnMut()) {
            Self::wrap("inv", next);
        }

        fn wrap_user_function(&self, info: &crate::plugin::OperationInfo, next: &mut dyn FnMut()) {
            let site = info.name().unwrap_or(info.id());
            ENTERED.with(|entered| {
                entered.borrow_mut().push((site.to_owned(), current_tags()));
            });
            Self::wrap(site, next);
        }
    }

    impl crate::plugin::PluginFactory for Tag {
        fn create_plugin(
            &self,
            _info: &crate::plugin::InvocationInfo,
        ) -> Result<Option<Box<dyn crate::plugin::Plugin>>, BoxError> {
            Ok(Some(Box::new(Tag)))
        }
    }

    /// A root context over the in-memory transport whose one plugin is a
    /// [`Tag`], and its invocation frames.
    fn tagged_ctx() -> (DurableContext, crate::plugin::Frames) {
        let factory: Arc<dyn crate::plugin::PluginFactory> = Arc::new(Tag);
        let plugins = crate::plugin::create_plugins(
            &[("Tag", factory)],
            None,
            crate::plugin::tests_support_info,
        )
        .expect("a factory was given");
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "test-function-arn".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new())),
            "token0".to_owned(),
            None,
            0,
            Some(Arc::clone(&plugins)),
        );
        (ctx, crate::plugin::Frames::invocation(Some(plugins)))
    }

    /// What a step body saw: its name and the wraps entered at its call.
    type Seen = Arc<std::sync::Mutex<Vec<(&'static str, Vec<String>)>>>;

    /// A step body that records what it saw in `seen`, then returns
    /// `value`.
    fn recording(
        seen: &Seen,
        name: &'static str,
        value: u32,
    ) -> impl FnOnce(crate::StepContext) -> std::future::Ready<Result<u32, BoxError>>
    + Clone
    + Send
    + 'static {
        let seen = Arc::clone(seen);
        move |_| {
            seen.lock().unwrap().push((name, current_tags()));
            std::future::ready(Ok(value))
        }
    }

    /// Rule 5.4: in A (child `a`, span `A`) a step future is made and
    /// mapped, and so is an unclaimed `try_join_all` of two steps. The
    /// mapped step keeps A. Mapping claims no ID; the combinator claims the
    /// next positional ID in B (child `b`, span `B`), by its first poll, then
    /// by what `.spawn()` does; and its task context is B's. Each path
    /// yields its inputs' mapped result; the join runs in `[I, b, join]`,
    /// and each input's body in the frames it captured in A.
    #[test]
    fn map_output_keeps_task_context_and_pending_claim() {
        let subscriber = lambda_runtime::tracing::subscriber::registry();
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime");
            let _guard = runtime.enter();
            let span_a = tracing::info_span!("A");
            let span_b = tracing::info_span!("B");
            assert!(span_a.id().is_some() && span_b.id().is_some());
            for spawned in [false, true] {
                let (ctx, inv) = tagged_ctx();
                let frames_a =
                    inv.with_user_function(Arc::new(crate::plugin::OperationInfo::for_test("a")));
                let frames_b =
                    inv.with_user_function(Arc::new(crate::plugin::OperationInfo::for_test("b")));
                let seen: Seen = Arc::default();
                let (step, mut join) = span_a.in_scope(|| {
                    frames_a.enter(|| {
                        let step = ctx
                            .step(recording(&seen, "solo", 1))
                            .name("solo")
                            .future()
                            .map_output(|r| r.map(|v| v + 1));
                        let s1 = ctx.step(recording(&seen, "s1", 1)).name("s1").future();
                        let s2 = ctx.step(recording(&seen, "s2", 2)).name("s2").future();
                        let join = ctx
                            .try_join_all([s1, s2])
                            .name("join")
                            .into_future()
                            .map_output(|r| r.map(|v| v.iter().sum::<u32>() * 10));
                        (step, join)
                    })
                });
                // The mapped step keeps A.
                assert_eq!(
                    step.task_context().frames().enter(current_tags),
                    ["inv", "a"],
                    "spawned: {spawned}"
                );
                assert_eq!(step.task_context().span().id(), span_a.id());
                // The claim is kept, and mapping claimed nothing: the steps
                // took 1 to 3, so the probe takes 4.
                assert!(join.is_unclaimed_combinator(), "spawned: {spawned}");
                assert_eq!(ctx.mint_id().positional(), "4", "spawned: {spawned}");
                span_b.in_scope(|| {
                    frames_b.enter(|| {
                        if spawned {
                            // What `.spawn()` does: claim, then start the task.
                            join.claim_now();
                        } else {
                            // The first poll claims.
                            let waker = std::task::Waker::noop();
                            let mut cx = Context::from_waker(waker);
                            drop(Pin::new(&mut join).poll(&mut cx));
                        }
                    });
                });
                assert!(!join.is_unclaimed_combinator(), "spawned: {spawned}");
                // The combinator claimed 5, the ID after the probe's 4.
                assert_eq!(ctx.mint_id().positional(), "6", "spawned: {spawned}");
                assert_eq!(
                    join.task_context().frames().enter(current_tags),
                    ["inv", "b"],
                    "the claim captured B's frames; spawned: {spawned}"
                );
                assert_eq!(join.task_context().span().id(), span_b.id());
                let step = DurableFuture::spawn_blessed(step, &ctx, None);
                assert_eq!(runtime.block_on(step).expect("the step succeeds"), 2);
                let join = DurableFuture::spawn_blessed(join, &ctx, None);
                assert_eq!(runtime.block_on(join).expect("the join succeeds"), 30);
                let mut seen = std::mem::take(&mut *seen.lock().unwrap());
                seen.sort_by_key(|(name, _)| *name);
                let tags = |list: &[&str]| -> Vec<String> {
                    list.iter().map(|tag| (*tag).to_owned()).collect()
                };
                // Each input's body runs on its own task, in the frames it
                // captured in A.
                assert_eq!(
                    seen,
                    [
                        ("s1", tags(&["inv", "a", "s1"])),
                        ("s2", tags(&["inv", "a", "s2"])),
                        ("solo", tags(&["inv", "a", "solo"])),
                    ],
                    "spawned: {spawned}"
                );
                // The join's node sits above B: its wrap is entered inside
                // B's frames, so the join runs in `[I, b, join]`.
                let entered = ENTERED.with(|entered| std::mem::take(&mut *entered.borrow_mut()));
                let joins: Vec<_> = entered.iter().filter(|(site, _)| site == "join").collect();
                assert!(!joins.is_empty(), "the join ran as a user function");
                assert!(
                    joins
                        .iter()
                        .all(|(_, around)| *around == tags(&["inv", "b"])),
                    "{entered:?}; spawned: {spawned}"
                );
            }
        };
    }

    /// r7 item 5: a `parallel_mixed` future keeps the task context of its
    /// batch, so a task that runs it runs inside the span and frames of its
    /// creation.
    #[test]
    fn parallel_mixed_keeps_the_task_context_of_its_creation() {
        let subscriber = lambda_runtime::tracing::subscriber::registry();
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime");
            let _guard = runtime.enter();
            let ctx = test_ctx_with_client();
            let span = tracing::info_span!("creation");
            let mixed = span.in_scope(|| {
                let mut set = BranchSet::new();
                let _handle = set.add("b", |_c| async { Ok(1_u32) });
                ctx.parallel_mixed(set).into_future()
            });
            assert!(span.id().is_some());
            assert_eq!(mixed.task_context().span().id(), span.id());
            let plain = span.in_scope(|| ctx.step(|_| async { Ok(1_u32) }).future());
            let mapped = plain.map_output(|r| r.map(|v| v + 1));
            assert_eq!(mapped.task_context().span().id(), span.id());
        };
    }

    /// Tokio may give a new task the ID of a task that exited, and the
    /// exited task's blessing is gone. So a raw task whose ID matches a
    /// poller that passed earlier must still be refused: the gate checks
    /// ownership on every poll. The test gives one task the identity the
    /// gate admitted: it blesses the task, passes the gate, then removes
    /// the blessing, as the task gateway does when a blessed task exits,
    /// and polls the same gate again with that task ID. A gate that cached
    /// the ID of its last passing poller would admit the second poll.
    #[test]
    fn poll_gate_refuses_raw_task_with_reused_identity() {
        let ctx = test_ctx_with_client();
        let mut gate = PollGate::new(&ctx);
        let ownership = Arc::clone(ctx.task_ownership());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let (first, second) = runtime
            .block_on(async move {
                tokio::spawn(async move {
                    let id = tokio::task::id();
                    ownership.bless_task(id);
                    let first = gate.admit().is_ok();
                    ownership.unbless_task(id);
                    let second = gate.admit().is_ok();
                    (first, second)
                })
                .await
            })
            .unwrap();
        assert!(first, "the blessed task was refused");
        assert!(!second, "a raw task with a reused task ID was admitted");
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert_eq!(
            fatal.error_message,
            crate::context::foreign_caller_message("execute")
        );
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
        let ran = Arc::new(AtomicBool::new(false));
        let thread_client = Arc::clone(&client);
        let thread_ran = Arc::clone(&ran);
        // The context is built on the polling thread, so that thread owns it.
        let polled = crate::task::test_support::poll_once_without_runtime(move || {
            DurableContext::new_root_with_client(
                "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::empty()),
                thread_client as Arc<dyn crate::client::ExecutionClient>,
                "token0".to_owned(),
            )
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

    /// Whether two keys name the same branch of the same set.
    fn same_branch<T>(a: &BranchKey<T>, b: &BranchKey<T>) -> bool {
        Arc::ptr_eq(&a.identity, &b.identity) && a.index == b.index
    }

    #[test]
    fn branch_keys_name_their_branch_and_set() {
        fn assert_clone_send_sync<K: Clone + Send + Sync>() {}
        assert_clone_send_sync::<BranchKey<std::cell::Cell<String>>>();

        let mut first = BranchSet::new();
        let a = first.add("a", |_ctx| async { Ok(1_u32) });
        let b = first.add("b", |_ctx| async { Ok(2_u32) });
        let mut second = BranchSet::new();
        let other_a = second.add("a", |_ctx| async { Ok(1_u32) });

        assert!(same_branch(&a.key(), &a.key()));
        assert!(same_branch(&a.key(), &a.key().clone()));
        assert!(!same_branch(&a.key(), &b.key()));
        assert!(!same_branch(&a.key(), &other_a.key()));
        assert_eq!(format!("{:?}", b.key()), "BranchKey { index: 1, .. }");
    }
}
