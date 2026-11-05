//! Context types for durable execution.
//!
//! [`DurableContext`] provides access to all durable operations.
//! [`StepContext`] is passed to step bodies and deliberately omits durable
//! operations: the type system enforces the "no nesting" rule at compile
//! time.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use serde::{Serialize, de::DeserializeOwned};

use crate::BoxError;
use crate::builders::{
    ChildBuilder, CreateCallbackBuilder, InvokeBuilder, JoinAllBuilder, MapBuilder,
    ParallelBuilder, ParallelMixedBuilder, RaceBuilder, SelectOkBuilder, StepBuilder,
    TryJoinAllBuilder, WaitBuilder, WaitForCallbackBuilder, WaitForConditionBuilder,
    WithRetryBuilder,
};
use crate::checkpoint_coalescer::{
    Admission, BatchLimits, CheckpointBatch, CheckpointCoalescer, TrackedUpdate,
    split_into_requests,
};
use crate::client::{
    CheckpointOutput, ExecutionClient, ExecutionClientError, ExecutionClientErrorScope,
};
use crate::coordinator::{InvocationCoordinator, PermitKind};
use crate::driver::TaskOwnership;
use crate::engine::{
    CheckpointLog, CheckpointRecord, CheckpointStatus, CheckpointStatusView, EngineState,
    OperationId, TerminalReplaySnapshot,
};
use crate::error::{
    NonDeterministicExecutionError, NonDeterministicExecutionErrorKind, OperationError,
    OperationErrorKind, StepError, StepErrorKind,
};
use crate::future::{Branch, DurableFuture};

use aws_sdk_lambda::types::{OperationAction, OperationUpdate};
use tokio::sync::Mutex;

/// The receiving half of a detached write, which drops a result that
/// arrived but was never read inside a catch when it is dropped.
///
/// The channel carries only the write's result, never a panic payload: a
/// panic of the checkpoint client is recorded by the task gateway. The
/// result can hold an error of the user's checkpoint client, whose
/// destructor can panic, and such a panic is invocation-failing. The write
/// task can send its result after the caller's last poll and before the
/// caller is dropped. Closing the channel first makes the result go to
/// exactly one place: either this guard takes it, or the sender's `send`
/// fails and the sender disposes of it.
struct UnreadResultGuard<T> {
    receiver: tokio::sync::oneshot::Receiver<T>,
    coordinator: Arc<InvocationCoordinator>,
}

impl<T> Drop for UnreadResultGuard<T> {
    fn drop(&mut self) {
        self.receiver.close();
        if let Ok(unread) = self.receiver.try_recv() {
            crate::task::drop_recorded(&self.coordinator, unread);
        }
    }
}

/// The changes one checkpoint call merged, for `on_operation_change`.
#[derive(Default)]
struct ChangeSet {
    /// The wire IDs of the log records whose status changed, each once, in
    /// merge order.
    ids: Vec<String>,
    /// The `EXECUTION` operations that changed the plugins' slot.
    executions: Vec<crate::plugin::OperationInfo>,
}

/// A checkpoint write that has already started, and the future of its
/// result.
///
/// [`DurableContext::submit_checkpoint_updates`] starts the write before it
/// returns: the updates are admitted to the buffer, or enter the
/// [`DirectWriteQueue`]. Awaiting the value only waits for the result. Dropping it
/// does not cancel the write, because the batch flusher and the direct
/// write task do not depend on the caller. An at-least-once step uses this
/// to run its body while its `START` write is in flight.
pub(crate) struct SubmittedWrite {
    result: std::pin::Pin<
        Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send>,
    >,
    /// Set on the first poll (see [`SubmissionRecorder`]).
    #[cfg(test)]
    first_poll: Arc<std::sync::atomic::AtomicBool>,
    /// The plugins and the log whose queued change sets the write delivers
    /// when its result is ready (see [`Self::reported_by`]). `None` without
    /// plugins, and once delivered.
    report: Option<Box<WriteReport>>,
}

/// What a [`SubmittedWrite`] needs to deliver the change sets through its
/// own before it returns its result.
struct WriteReport {
    plugins: Arc<crate::plugin::InvocationPlugins>,
    log: Arc<CheckpointLog>,
    /// Records a panic from the destructor of a retained result that no
    /// reader receives.
    coordinator: Arc<InvocationCoordinator>,
    /// The result, kept while the write waits for the deliveries.
    ready: Option<Result<CheckpointOutput, ExecutionClientError>>,
    /// The write's wait for the deliveries (see
    /// [`InvocationPlugins::poll_changes_delivered`](crate::plugin::InvocationPlugins::poll_changes_delivered)).
    wait: crate::plugin::DeliveryWait,
}

impl Drop for WriteReport {
    /// A write dropped while it waits removes its waker registration, and
    /// drops the retained result, which no reader receives, inside a catch:
    /// the result can hold an error of the user's checkpoint client whose
    /// destructor panics, and such a panic is invocation-failing.
    fn drop(&mut self) {
        self.plugins.cancel_wait(&mut self.wait);
        if let Some(unread) = self.ready.take() {
            crate::task::drop_recorded(&self.coordinator, unread);
        }
    }
}

impl SubmittedWrite {
    fn new(
        result: impl Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + 'static,
    ) -> Self {
        Self {
            result: Box::pin(result),
            #[cfg(test)]
            first_poll: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            report: None,
        }
    }

    /// Makes the write deliver the queued change sets of `ctx`'s plugins
    /// to `on_operation_change` once its result is ready, `Ok` or `Err`, on
    /// the task that polls it to completion. A write dropped before that
    /// delivers nothing: another drain point delivers its change set. Does
    /// nothing without plugins.
    fn reported_by(mut self, ctx: &DurableContext) -> Self {
        self.report = ctx.shared_plugins().map(|plugins| {
            Box::new(WriteReport {
                plugins,
                log: Arc::clone(&ctx.inner.engine.checkpoint_log),
                coordinator: Arc::clone(ctx.coordinator()),
                ready: None,
                wait: crate::plugin::DeliveryWait::default(),
            })
        });
        self
    }

    /// A write whose result `receiver` delivers. A test builds the retained
    /// `START` of an operation with it.
    #[cfg(test)]
    pub(crate) fn from_receiver(
        receiver: tokio::sync::oneshot::Receiver<Result<CheckpointOutput, ExecutionClientError>>,
    ) -> Self {
        Self::new(async move {
            match receiver.await {
                Ok(result) => result,
                Err(_closed) => std::future::pending().await,
            }
        })
    }

    /// A rejected write: nothing is sent, and the result never resolves.
    /// The rejected caller's task is being cancelled (see [`Admission`]).
    fn rejected() -> Self {
        Self::new(std::future::pending())
    }
}

impl Future for SubmittedWrite {
    type Output = Result<CheckpointOutput, ExecutionClientError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        #[cfg(test)]
        this.first_poll
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let Some(report) = this.report.as_mut() else {
            return this.result.as_mut().poll(cx);
        };
        let result = match report.ready.take() {
            Some(result) => result,
            None => std::task::ready!(this.result.as_mut().poll(cx)),
        };
        if report
            .plugins
            .poll_changes_delivered(&report.log, &mut report.wait, cx)
            .is_pending()
        {
            report.ready = Some(result);
            return std::task::Poll::Pending;
        }
        this.report = None;
        std::task::Poll::Ready(result)
    }
}

/// A checkpoint write whose position in the write order is fixed, but
/// that is not sent yet.
///
/// [`DurableContext::reserve_checkpoint_updates`] puts the updates into the
/// coalescer's open batch, or into the [`DirectWriteQueue`], before it
/// returns. A batch that holds a reserved write is not flushed, and later
/// entries of the queue wait behind it. So a later submission cannot reach
/// the service first. [`Self::release`] makes the write eligible to be
/// sent and returns its result future. Dropping the value releases it too.
///
/// The start step of an operation whose JS order is "queue START, then
/// fire `on_operation_start`" reserves START, fires the hook, then
/// releases START. So the hook returns before the START request is sent,
/// on any runtime, as in JS.
pub(crate) struct ReservedWrite {
    write: SubmittedWrite,
    /// What keeps the write from being sent. `None` once released, and for
    /// a rejected write.
    hold: Option<WriteHold>,
}

/// What [`ReservedWrite`] holds back.
enum WriteHold {
    /// A hold on the coalescer's open batch.
    Batch(Arc<CheckpointCoalescer>),
    /// A held entry of the direct write queue.
    Direct {
        queue: Arc<DirectWriteQueue>,
        id: u64,
    },
}

impl WriteHold {
    fn release(self) {
        match self {
            Self::Batch(coalescer) => coalescer.release_hold(),
            Self::Direct { queue, id } => queue.release(id),
        }
    }
}

impl ReservedWrite {
    /// A rejected reservation: nothing is held, and the result never
    /// resolves (see [`SubmittedWrite::rejected`]).
    fn rejected() -> Self {
        Self {
            write: SubmittedWrite::rejected(),
            hold: None,
        }
    }

    /// Makes the write eligible to be sent, and returns its result future.
    /// It does not wait for the write.
    pub(crate) fn release(mut self) -> SubmittedWrite {
        // `self` is dropped when this returns, and the drop releases the
        // hold.
        std::mem::replace(&mut self.write, SubmittedWrite::rejected())
    }
}

impl Drop for ReservedWrite {
    /// A dropped reservation is released, so the writes queued behind it
    /// are not blocked forever.
    fn drop(&mut self) {
        if let Some(hold) = self.hold.take() {
            hold.release();
        }
    }
}

/// The test hook of the direct writer: for an entry's updates it returns
/// the `Notify` the writer awaits before it takes the checkpoint token, or
/// `None` to write at once.
#[cfg(test)]
pub(crate) type PreLockGate =
    Arc<dyn Fn(&[OperationUpdate]) -> Option<Arc<tokio::sync::Notify>> + Send + Sync>;

/// The test hook that sees every admitted submission: its updates and the
/// first-poll flag of its [`SubmittedWrite`]. It runs before the write can
/// be sent: an ordinary write calls it before it joins the coalescer or
/// enters the [`DirectWriteQueue`], and a reservation calls it while its
/// entry is held.
#[cfg(test)]
pub(crate) type SubmissionRecorder =
    Arc<dyn Fn(&[OperationUpdate], &Arc<std::sync::atomic::AtomicBool>) + Send + Sync>;

/// The test hook that each child-backed outcome path calls with the
/// operation's wire ID and the action it is about to write, just before it
/// awaits the operation's retained `START` write.
#[cfg(test)]
pub(crate) type SettlerEntryRecorder = Arc<dyn Fn(&str, OperationAction) + Send + Sync>;

/// The test hooks a [`DirectWriteQueue`] carries for every context of the
/// invocation.
#[cfg(test)]
#[derive(Default)]
struct TestHooks {
    gate: Option<PreLockGate>,
    submissions: Option<SubmissionRecorder>,
    settler_entries: Option<SettlerEntryRecorder>,
}

/// The FIFO of the invocation's unbuffered checkpoint writes.
///
/// Without a coalescer, every write enters this queue under a synchronous
/// lock before the call that submits it returns. One detached task, the
/// direct writer, takes the entries in order, writes each one under the
/// `checkpoint_token` lock, and sends its result to the entry's
/// [`SubmittedWrite`]. So writes reach the service in submission order,
/// whichever task Tokio runs first. The root context and every child share
/// one queue, as they share `checkpoint_token`.
///
/// The writer runs while the queue has entries and ends when it is empty.
/// The next write starts a new one. A held entry (see [`ReservedWrite`])
/// stops the writer until it is released.
#[derive(Default)]
pub(crate) struct DirectWriteQueue {
    state: std::sync::Mutex<DirectQueueState>,
    /// Wakes the writer when a held entry is released.
    released: tokio::sync::Notify,
    /// The hooks tests install (see [`PreLockGate`],
    /// [`SubmissionRecorder`] and [`SettlerEntryRecorder`]).
    #[cfg(test)]
    hooks: std::sync::Mutex<TestHooks>,
}

#[derive(Default)]
struct DirectQueueState {
    entries: std::collections::VecDeque<DirectEntry>,
    /// Whether a writer task is running. Set by the push that starts one,
    /// cleared by the writer when it finds the queue empty.
    writer_running: bool,
    next_id: u64,
}

/// The result channel of a direct write. `None` means the admission check
/// rejected the write.
type DirectResult = Option<Result<CheckpointOutput, ExecutionClientError>>;

/// One write of the [`DirectWriteQueue`].
struct DirectEntry {
    id: u64,
    /// Whether a [`ReservedWrite`] still holds the entry.
    held: bool,
    /// The context that submitted the write: its owners are admitted, and
    /// its checkpoint log takes the response.
    ctx: DurableContext,
    updates: Vec<OperationUpdate>,
    events: Vec<crate::tracing_layer::PendingTransitionEvent>,
    /// The write's checkpoint permit, released after the result is sent.
    permit: Option<crate::coordinator::Permit>,
    /// The operation a failed write's classification is recorded for.
    failure_owner: Option<String>,
    sender: tokio::sync::oneshot::Sender<DirectResult>,
}

impl DirectWriteQueue {
    fn lock(&self) -> std::sync::MutexGuard<'_, DirectQueueState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Releases the held entry `id`, and wakes the writer.
    fn release(&self, id: u64) {
        if let Some(entry) = self.lock().entries.iter_mut().find(|entry| entry.id == id) {
            entry.held = false;
        }
        self.released.notify_waiters();
    }

    /// Takes the next entry once it is not held. Returns `None`, and marks
    /// the writer stopped, when the queue is empty.
    async fn next(&self) -> Option<DirectEntry> {
        loop {
            // `enable()` registers the waiter before the check, so a
            // release between the check and the await still wakes it.
            let mut released = std::pin::pin!(self.released.notified());
            released.as_mut().enable();
            {
                let mut state = self.lock();
                if state.entries.is_empty() {
                    state.writer_running = false;
                    return None;
                }
                if let Some(entry) = state.entries.pop_front_if(|entry| !entry.held) {
                    return Some(entry);
                }
            }
            released.await;
        }
    }

    /// Waits on the pre-lock gate for `updates`, when one is installed and
    /// returns a `Notify` for them.
    #[cfg(test)]
    fn hooks(&self) -> std::sync::MutexGuard<'_, TestHooks> {
        self.hooks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    async fn pass_gate(&self, updates: &[OperationUpdate]) {
        let gate = self.hooks().gate.clone();
        if let Some(notify) = gate.and_then(|gate| gate(updates)) {
            notify.notified().await;
        }
    }
}

/// The result future of a write that joined `batch`: it resolves when the
/// batch's flusher publishes the result.
fn batch_result(batch: Arc<CheckpointBatch>) -> SubmittedWrite {
    SubmittedWrite::new(async move {
        loop {
            // `enable()` registers the waiter BEFORE the result check, so a
            // publish landing between the check and the await still wakes
            // us.
            let mut notified = std::pin::pin!(batch.notified());
            notified.as_mut().enable();
            if let Some(result) = batch.result_clone() {
                return result;
            }
            notified.await;
        }
    })
}

/// Marks the direct writer stopped when its task unwinds, so the next
/// write starts a new writer, which writes the entries still queued.
struct WriterUnwind(Arc<DirectWriteQueue>);

impl Drop for WriterUnwind {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.lock().writer_running = false;
        }
    }
}

/// The direct writer: writes the queue's entries in order until it is
/// empty (see [`DirectWriteQueue`]).
async fn run_direct_writer(queue: Arc<DirectWriteQueue>) {
    let _unwind = WriterUnwind(Arc::clone(&queue));
    while let Some(entry) = queue.next().await {
        #[cfg(test)]
        queue.pass_gate(&entry.updates).await;
        let DirectEntry {
            ctx,
            updates,
            events,
            permit,
            failure_owner,
            sender,
            ..
        } = entry;
        let written = ctx
            .checkpoint_direct_with_events(updates, &events, Some(ctx.owners()))
            .await;
        if let (Some(owner), Some(Err(err))) = (&failure_owner, &written) {
            ctx.record_checkpoint_failure(owner, err);
        }
        // A dropped receiver means the caller was cancelled, or the driver
        // dropped the refresh scheduler when the drive ended: the write
        // already happened, and its result has no reader. The result is
        // then dropped inside a catch. The caller's wake takes its own
        // permit before the write's permit is released.
        if let Err(unread) = sender.send(written) {
            crate::task::drop_recorded(ctx.coordinator(), unread);
        }
        drop(permit);
    }
}

/// The user-code body the SDK is polling on the current task.
///
/// [`DurableContext::claim_operation_id`] compares it with the context an
/// operation is called on.
#[derive(Debug, Clone)]
pub(crate) enum ActiveBody {
    /// A body that runs on a `DurableContext`: the handler or a child body.
    /// Holds that context's positional prefix ("" for the root).
    Context(String),
    /// A body that may not call durable operations: a step body or a
    /// `wait_for_condition` check. Holds the operation's positional ID.
    Step(String),
    /// Plugin code: a wrap, or a start or end hook and its future. A
    /// plugin observes operations, so a durable operation it called would
    /// claim an ID that the handler's code did not create, and replay
    /// would diverge.
    Plugin,
    /// The SDK derives the recorded error of a user function's outcome:
    /// the error's `Display` and `source` run, which are user code. Holds
    /// the operation's positional ID. An operation called there would
    /// claim an ID on a live run only, since replay returns the recorded
    /// error without deriving it again.
    Outcome(String),
    /// No body is active: the check does nothing. A plugin wrap's `next`
    /// sets it when the code around the wrap had no value, so the code
    /// inside sees exactly what it saw without the wrap.
    Unset,
}

/// The body active now, or [`ActiveBody::Unset`] when none is.
pub(crate) fn current_body() -> ActiveBody {
    ACTIVE_BODY
        .try_with(ActiveBody::clone)
        .unwrap_or(ActiveBody::Unset)
}

tokio::task_local! {
    /// The body the SDK is polling. `LocalKey::scope` holds the value only
    /// while the wrapped future is polled. So a parent future joined with a
    /// child body on one task does not see the child's value, and a task
    /// the SDK spawns starts without a value.
    static ACTIVE_BODY: ActiveBody;
}

/// Polls `fut` with `body` as the active body.
///
/// Callers pass a future that calls the user closure inside it, so code
/// the closure runs before it returns its future is checked too.
pub(crate) fn run_body<F: Future>(body: ActiveBody, fut: F) -> impl Future<Output = F::Output> {
    ACTIVE_BODY.scope(body, fut)
}

/// Calls `f` with `body` as the active body.
///
/// A step clones the user closure before each attempt. The clone runs the
/// captures' `Clone` implementations, which are user code. A terminal
/// record replays without cloning. So an operation claimed inside a clone
/// would claim an ID on a live attempt only. This scope makes such a claim
/// fail the execution with a context usage error instead.
pub(crate) fn run_body_sync<R>(body: ActiveBody, f: impl FnOnce() -> R) -> R {
    ACTIVE_BODY.sync_scope(body, f)
}

/// The time left from `now` until `deadline_ms` (milliseconds since the
/// Unix epoch): `None` when the deadline is 0, `Duration::ZERO` once it
/// passed.
fn remaining_until(deadline_ms: u64, now: std::time::SystemTime) -> Option<Duration> {
    if deadline_ms == 0 {
        return None;
    }
    let deadline = std::time::UNIX_EPOCH + Duration::from_millis(deadline_ms);
    Some(deadline.duration_since(now).unwrap_or(Duration::ZERO))
}

/// Shows a context prefix in a context usage error: `root` for the root.
pub(crate) fn display_prefix(prefix: &str) -> &str {
    if prefix.is_empty() { "root" } else { prefix }
}

/// Returns the wire ID and payload of every `wait_for_condition` `RETRY`
/// among `updates`, in order.
fn wfc_retries(updates: &[OperationUpdate]) -> Vec<(String, Option<String>)> {
    updates
        .iter()
        .filter(|update| {
            *update.action() == OperationAction::Retry
                && update.sub_type() == Some(crate::wait_for_condition::WFC_SUB_TYPE)
        })
        .map(|update| (update.id().to_owned(), update.payload().map(str::to_owned)))
        .collect()
}

/// Marks the record of `wire_id` `PENDING` with `payload` as its result,
/// after a `wait_for_condition` `RETRY` write whose response did not carry
/// the operation.
///
/// The service moves a retried operation to `PENDING`. A response that
/// does not carry the operation leaves the log at the status from before
/// the write. A waiting operation that read that status would take it for
/// the next status and run its next check at once. So the log takes the
/// status the service recorded. A record already `PENDING` is kept, and a
/// missing record stays missing. A response that carries the operation is
/// merged instead, so its status is kept.
fn mark_retry_pending(log: &CheckpointLog, wire_id: &str, payload: Option<&str>) {
    log.update(wire_id, |record| {
        if record.status != CheckpointStatus::Pending {
            record.status = CheckpointStatus::Pending;
            record.result = payload.map(str::to_owned);
        }
    });
}

/// Shared inner state for a durable execution context.
struct Inner {
    execution_arn: String,
    lambda_context: lambda_runtime::Context,
    /// Engine state (ID counter + checkpoint log) for this context's
    /// namespace.
    engine: Arc<EngineState>,
    /// The invocation coordinator, shared by every context of the
    /// invocation and by the driver.
    coordinator: Arc<InvocationCoordinator>,
    /// Task-ownership detector: catches user `tokio::spawn` misuse.
    task_ownership: Arc<TaskOwnership>,
    /// Execution client for checkpointing (None in test contexts without a client).
    execution_client: Option<Arc<dyn ExecutionClient>>,
    /// Mutable checkpoint token rotated on each checkpoint call.
    /// Uses `tokio::sync::Mutex` (not `std::sync::Mutex`) because the lock
    /// must be held across `await` points in [`DurableContext::checkpoint_updates`],
    /// serializing all concurrent checkpoint callers through one critical section.
    checkpoint_token: Arc<Mutex<String>>,
    /// Checkpoint-write coalescer, present only when
    /// [`Options`](crate::Options) configured a `checkpoint_delay` and/or
    /// `checkpoint_batching`. Shared with every child context so updates
    /// from all namespaces coalesce into the same batches. A zero-window
    /// batching coalescer is the default. `None` means every checkpoint
    /// writes immediately, which needs `checkpoint_batching(false)` without
    /// a `checkpoint_delay`.
    coalescer: Option<Arc<CheckpointCoalescer>>,
    /// The FIFO of unbuffered writes and its writer, shared by the root and
    /// every child context, as `checkpoint_token` is (see
    /// [`DirectWriteQueue`]).
    direct: Arc<DirectWriteQueue>,
    /// Cached parent wire ID: the SHA-256 hash of this context's prefix
    /// (positional ID of the parent operation). `None` for root contexts.
    parent_wire_id: Option<String>,
    /// Abandoned owners of this invocation, shared by the root and every
    /// context derived from it (see [`Admission`]).
    admission: Arc<Admission>,
    /// The owners this context runs in, outermost first: the wire IDs of
    /// the non-virtual child contexts (batch items included) that enclose
    /// it. Empty at the root. A checkpoint update from this context is
    /// admitted only while none of them is abandoned.
    owners: Arc<[String]>,
    /// This namespace's `durable_execution` span. For a root context it is
    /// the handler-level span wrapping the invocation; for a child context
    /// (sequential child, map/parallel branch, callback body) it is a
    /// detached span wrapping that child's body. [`DurableContext::mint_id`]
    /// re-records its `isReplay` field after every operation claim minted
    /// through this context, so replay-aware filters (see
    /// `tracing_layer::ReplayFilterLayer`) can suppress log events emitted
    /// while THIS namespace is replaying. Per-namespace spans are what keep
    /// concurrent branches, each with its own replay high-water mark, from
    /// clobbering each other's flag.
    replay_span: tracing::Span,
    /// The plugins of the invocation. The root context of a wrapped
    /// invocation holds them, and every context derived from it shares
    /// the same `Arc`. `None` without a registered plugin factory.
    plugins: Option<Arc<crate::plugin::InvocationPlugins>>,
    /// How many context levels this context sits below the root: 0 at the
    /// root, plus 1 for each [`DurableContext::new_child`]. A Flat batch
    /// item's context keeps the batch's level.
    nesting_level: u32,
    /// The invocation's
    /// [`plugin_child_operations_depth`](crate::OptionsBuilder::plugin_child_operations_depth),
    /// copied into every derived context.
    preserve_child_depth: u32,
}

impl std::fmt::Debug for Inner {
    /// Hand-written to keep `checkpoint_token`, a credential-like value,
    /// out of log output, and to skip the engine/client/signal internals
    /// that make a derived impl unreadable. See issue #30.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("execution_arn", &self.execution_arn)
            .field("parent_wire_id", &self.parent_wire_id)
            .field("is_replaying", &self.engine.is_replaying())
            .field("checkpoint_token", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// The durable execution context: a cheap-to-clone handle providing access
/// to all durable operations.
///
/// `DurableContext` is `Clone + Send + Sync` (backed by an `Arc`). Clone it
/// freely to share across async boundaries. Every durable operation method
/// except [`with_retry`](Self::with_retry) and the four combinators claims
/// its operation ID synchronously at the call site: polling order never
/// affects identity. A `with_retry` block with its own record claims its ID
/// when its builder becomes its future (`.await`, `.future()` or
/// `.spawn()`); a block with
/// [`without_child_context`](crate::builders::WithRetryBuilder::without_child_context)
/// claims none, and its first attempt claims the caller's next ID when it
/// starts. A combinator
/// ([`try_join_all`](Self::try_join_all), [`join_all`](Self::join_all),
/// [`select_ok`](Self::select_ok) and [`race`](Self::race)) claims its ID
/// when it is first awaited or polled, or when it is spawned. A combinator
/// that is dropped before that claims no ID and records nothing. A
/// combinator passed to another combinator must already have claimed its
/// ID: start it with `.spawn()` and pass the returned handle. A
/// still-unclaimed combinator input fails the execution with a context
/// usage error.
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
///     // Access execution metadata
///     let arn = ctx.execution_arn();
///     let replaying = ctx.is_replaying();
///
///     // Perform a durable step
///     let result = ctx.step(|_| async { Ok("done".to_owned()) })
///         .name("example")
///         .await?;
///     Ok(result)
/// }
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct DurableContext {
    inner: Arc<Inner>,
    /// The first-poll flag of the handle an operation started on its own
    /// task returns. `None` on every context except the clone that starts
    /// such an operation (see [`Self::with_awaited`]). Every wait
    /// registration of an operation run on that clone carries the flag, so
    /// the refresh scheduler skips it until the handle is first polled.
    awaited: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl std::fmt::Debug for DurableContext {
    /// Hand-written so `tracing::debug!(?ctx)` in a handler cannot leak the
    /// checkpoint token into `CloudWatch` Logs: the token prints as
    /// `"<redacted>"` and engine/client internals are skipped. See issue #30.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableContext")
            .field("execution_arn", &self.inner.execution_arn)
            .field("parent_wire_id", &self.inner.parent_wire_id)
            .field("is_replaying", &self.inner.engine.is_replaying())
            .field("checkpoint_token", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Every checkpoint write failure suffered by an invocation's buffered
/// (coalesced) checkpoints, reported by
/// [`DurableContext::flush_pending_checkpoints`] for the issue #43
/// classification at the end-of-invocation flush point.
#[derive(Debug)]
pub(crate) struct FlushFailure {
    /// The write failures, in occurrence order, each carrying the updates
    /// it did not persist.
    pub(crate) failures: Vec<crate::checkpoint_coalescer::FailedFlush>,
}

impl FlushFailure {
    /// Whether any of the failures is non-retryable. Non-retryable wins
    /// the classification: re-invoking on a deterministic rejection would
    /// loop until the execution timeout (issue #43 defect 1), so a single
    /// non-retryable failure routes the whole flush through the
    /// terminal-FAIL-then-fail-the-execution path.
    pub(crate) fn any_non_retryable(&self) -> bool {
        self.failures.iter().any(|f| !f.error.is_retryable())
    }

    /// The error to report: the first non-retryable failure when one
    /// exists (it decides the classification), otherwise the first
    /// failure.
    ///
    /// # Panics
    ///
    /// Never in practice: `flush_pending_checkpoints` only constructs a
    /// `FlushFailure` with at least one failure.
    pub(crate) fn primary_error(&self) -> &ExecutionClientError {
        self.failures
            .iter()
            .find(|f| !f.error.is_retryable())
            .or_else(|| self.failures.first())
            .map_or_else(
                || unreachable!("FlushFailure is never constructed empty"),
                |f| &f.error,
            )
    }
}

impl DurableContext {
    /// Builds a context for unit tests: a root context with an empty
    /// checkpoint log and no execution client.
    #[cfg(test)]
    pub(crate) fn test_context() -> Self {
        let execution_arn = String::from("arn:aws:lambda:us-east-1:123456789012:function:test");
        let lambda_context = lambda_runtime::Context::default();
        let engine = Arc::new(EngineState::new_root(Arc::new(CheckpointLog::empty())));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            lambda_context.tenant_id.as_deref(),
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                coordinator: InvocationCoordinator::new(),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: None,
                checkpoint_token: Arc::new(Mutex::new(String::new())),
                coalescer: None,
                direct: Arc::new(DirectWriteQueue::default()),
                parent_wire_id: None,
                admission: Arc::new(Admission::new()),
                owners: Arc::from(Vec::new()),
                replay_span,
                plugins: None,
                nesting_level: 0,
                preserve_child_depth: 0,
            }),
            awaited: None,
        }
    }

    /// Creates a root context with the given execution state (test-only:
    /// production roots are built by the handler wrapper with a client and
    /// token via [`Self::new_root_with_client`]-equivalent wiring in
    /// `lib.rs`).
    #[cfg(test)]
    pub(crate) fn new_root(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            lambda_context.tenant_id.as_deref(),
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                coordinator: InvocationCoordinator::new(),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: None,
                checkpoint_token: Arc::new(Mutex::new(String::new())),
                coalescer: None,
                direct: Arc::new(DirectWriteQueue::default()),
                parent_wire_id: None,
                admission: Arc::new(Admission::new()),
                owners: Arc::from(Vec::new()),
                replay_span,
                plugins: None,
                nesting_level: 0,
                preserve_child_depth: 0,
            }),
            awaited: None,
        }
    }

    /// Creates a root context with a client and token (test-only harness
    /// for exercising live-path operation execution against a test double).
    #[cfg(test)]
    pub(crate) fn new_root_with_client(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            lambda_context.tenant_id.as_deref(),
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                coordinator: InvocationCoordinator::new(),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                coalescer: None,
                direct: Arc::new(DirectWriteQueue::default()),
                parent_wire_id: None,
                admission: Arc::new(Admission::new()),
                owners: Arc::from(Vec::new()),
                replay_span,
                plugins: None,
                nesting_level: 0,
                preserve_child_depth: 0,
            }),
            awaited: None,
        }
    }

    /// Creates a root context with a client, token, and checkpoint
    /// buffering window threaded in from
    /// [`Options`](crate::Options) (internal). `checkpoint_buffer_window`
    /// is `Some(Duration::ZERO)` for pure `checkpoint_batching` (the
    /// default: no added delay; writes batch behind the single-writer
    /// lock), `Some(window)` for a `checkpoint_delay` coalescing window,
    /// and `None` for immediate writes, which need
    /// `checkpoint_batching(false)` without a delay.
    /// `preserve_child_depth` is the
    /// [`plugin_child_operations_depth`](crate::OptionsBuilder::plugin_child_operations_depth)
    /// of the invocation.
    #[expect(clippy::too_many_arguments, reason = "one argument per root setting")]
    pub(crate) fn new_root_with_client_and_defaults(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
        checkpoint_buffer_window: Option<Duration>,
        preserve_child_depth: u32,
        plugins: Option<Arc<crate::plugin::InvocationPlugins>>,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            lambda_context.tenant_id.as_deref(),
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                coordinator: InvocationCoordinator::new(),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                coalescer: checkpoint_buffer_window.map(|d| Arc::new(CheckpointCoalescer::new(d))),
                direct: Arc::new(DirectWriteQueue::default()),
                parent_wire_id: None,
                admission: Arc::new(Admission::new()),
                owners: Arc::from(Vec::new()),
                replay_span,
                plugins,
                nesting_level: 0,
                preserve_child_depth,
            }),
            awaited: None,
        }
    }

    /// Test-only root with an empty log, an in-memory client and the
    /// plugin child operations depth `depth`, built through
    /// [`Self::new_root_with_client_and_defaults`]. Returns the client, whose
    /// recorded updates a test reads.
    #[cfg(test)]
    pub(crate) fn test_context_with_depth(
        depth: u32,
    ) -> (Self, Arc<crate::client::InMemoryExecutionClient>) {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = Self::test_context_with_depth_and_client(
            depth,
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
        );
        (ctx, client)
    }

    /// [`Self::test_context_with_depth`] with a caller-supplied client.
    #[cfg(test)]
    pub(crate) fn test_context_with_depth_and_client(
        depth: u32,
        client: Arc<dyn ExecutionClient>,
    ) -> Self {
        Self::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token0".to_owned(),
            None,
            depth,
            None,
        )
    }

    /// Test-only root constructor that accepts an explicit
    /// [`CheckpointCoalescer`], letting tests force small [`BatchLimits`]
    /// so batch splitting is observable with a handful of updates.
    #[cfg(test)]
    pub(crate) fn new_root_with_client_and_coalescer(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
        coalescer: CheckpointCoalescer,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            lambda_context.tenant_id.as_deref(),
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                coordinator: InvocationCoordinator::new(),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                coalescer: Some(Arc::new(coalescer)),
                direct: Arc::new(DirectWriteQueue::default()),
                parent_wire_id: None,
                admission: Arc::new(Admission::new()),
                owners: Arc::from(Vec::new()),
                replay_span,
                plugins: None,
                nesting_level: 0,
                preserve_child_depth: 0,
            }),
            awaited: None,
        }
    }
    /// Creates the child context of the child operation at
    /// `parent_positional_id`. `name` is that operation's name; the child's
    /// span carries it as `operationName`, with the operation's wire ID as
    /// `operationId`.
    pub(crate) fn new_child(&self, parent_positional_id: &str, name: Option<&str>) -> Self {
        let wire_id = crate::engine::compute_wire_id_public(parent_positional_id);
        let engine = Arc::new(EngineState::new_child(
            parent_positional_id,
            Arc::clone(&self.inner.engine.checkpoint_log),
        ));
        // The child namespace has its own replay high-water mark: nested
        // operations can still be replaying while the parent is already
        // live. Give it its own detached span, initialized from the CHILD
        // engine's replay status; the child's mints keep it current, and the
        // parent span keeps the value the parent's own mints gave it.
        let replay_span = crate::tracing_layer::scoped_execution_span(
            &self.inner.execution_arn,
            &self.inner.lambda_context.request_id,
            self.inner.lambda_context.tenant_id.as_deref(),
            &wire_id,
            name,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine,
                coordinator: Arc::clone(&self.inner.coordinator),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                coalescer: self.inner.coalescer.clone(),
                direct: Arc::clone(&self.inner.direct),
                parent_wire_id: Some(wire_id),
                admission: Arc::clone(&self.inner.admission),
                owners: self.owners_with(parent_positional_id),
                replay_span,
                plugins: self.inner.plugins.clone(),
                nesting_level: self.inner.nesting_level.saturating_add(1),
                preserve_child_depth: self.inner.preserve_child_depth,
            }),
            awaited: None,
        }
    }

    /// Whether a context for the child namespace `child_positional_id`
    /// would replay before its first claim: the first operation of that
    /// namespace has a checkpoint record. It is the
    /// [`Self::is_replaying`] a child context created now starts with.
    pub(crate) fn namespace_is_replaying(&self, child_positional_id: &str) -> bool {
        EngineState::new_child(
            child_positional_id,
            Arc::clone(&self.inner.engine.checkpoint_log),
        )
        .is_replaying()
    }

    /// Whether a context-backed record created on this context at `levels_below`
    /// levels under it records `ReplayChildren` on its terminal update.
    pub(crate) fn preserves_children(&self, levels_below: u32) -> bool {
        let level = self.inner.nesting_level.saturating_add(levels_below);
        level <= self.inner.preserve_child_depth
    }

    /// Mints the next operation ID (internal engine concern).
    ///
    /// Also re-records THIS namespace's span `isReplay` flag: after each
    /// claim, the namespace is replaying iff the NEXT operation to be
    /// claimed in it has a checkpoint record. This keeps replay-aware log
    /// filters (see `tracing_layer::ReplayFilterLayer`) in step with the
    /// live replay status as each namespace crosses its own replay
    /// high-water mark.
    pub(crate) fn mint_id(&self) -> OperationId {
        let id = self.inner.engine.mint_id();
        self.inner.replay_span.record(
            crate::tracing_layer::fields::IS_REPLAY,
            self.inner.engine.is_replaying(),
        );
        id
    }

    /// Returns this context's positional prefix: `""` for the root context,
    /// the positional ID of the operation that created it for a child.
    pub(crate) fn prefix(&self) -> &str {
        self.inner.engine.id_counter.prefix()
    }

    /// Claims the next operation ID for the public operation `method`, and
    /// checks that the caller may claim it on this context.
    ///
    /// The SDK records which body it is polling in [`ACTIVE_BODY`]. The
    /// check compares that body with this context. The outcomes are:
    ///
    /// - No body is active, or the body of this context is active: the
    ///   caller must own the context (see [`TaskOwnership`]). A task or
    ///   thread that does not own it fails the execution with a context
    ///   usage error.
    /// - The body of another context is active: the execution fails with a
    ///   context usage error that names both contexts.
    /// - A step body or a `wait_for_condition` check is active: the
    ///   execution fails with a context usage error that names the
    ///   operation.
    /// - Plugin code is active: the SDK records a context usage error.
    ///
    /// The failure is recorded as an execution-fatal error. When it is
    /// recorded before the driver's arbitration, the invocation answers
    /// `FAILED` on the first invocation. Only plugin code can record it
    /// after the arbitration, either before `drive_invocation` returns or
    /// after; such a record changes neither the response nor the
    /// late-panic policy. Without the
    /// check, the misuse surfaces one invocation later as a
    /// non-determinism error that does not name the cause. The ID is
    /// claimed in every case, so the caller builds its builder as usual.
    ///
    /// Only the public operation methods call this. The SDK's own internal
    /// claims (inside `wait_for_callback`) call [`Self::mint_id`], because
    /// they run on the composite's context while a child body may be
    /// active.
    ///
    /// [`TaskOwnership`]: crate::driver::TaskOwnership
    pub(crate) fn claim_operation_id(&self, method: &'static str) -> OperationId {
        let message = ACTIVE_BODY
            .try_with(|active| match active {
                ActiveBody::Context(prefix) if prefix == self.prefix() => None,
                ActiveBody::Context(prefix) => Some(format!(
                    "context usage error in \"{method}\": the operation was called on context \
                     \"{calling}\", but the SDK is running the body of context \"{active}\". \
                     Inside run_in_child_context, with_retry, map and parallel, call operations \
                     on the context passed to the closure.",
                    calling = display_prefix(self.prefix()),
                    active = display_prefix(prefix),
                )),
                ActiveBody::Step(id) => Some(format!(
                    "context usage error in \"{method}\": the operation was called inside the \
                     body of operation \"{id}\". A step body or a wait_for_condition check \
                     cannot call durable operations; use run_in_child_context to group \
                     operations."
                )),
                ActiveBody::Plugin => Some(format!(
                    "context usage error in \"{method}\": the operation was called inside \
                     plugin code. Plugin code cannot call durable operations."
                )),
                ActiveBody::Outcome(id) => Some(format!(
                    "context usage error in \"{method}\": the operation was called while the \
                     SDK records the outcome of operation \"{id}\". An error's Display and \
                     source cannot call durable operations."
                )),
                ActiveBody::Unset => None,
            })
            .ok()
            .flatten();
        let message = message.or_else(|| {
            self.inner
                .task_ownership
                .check_current_task()
                .err()
                .map(|_| foreign_caller_message(method))
        });
        if let Some(message) = message {
            self.coordinator().record_fatal("Error".to_owned(), message);
        }
        self.mint_id()
    }

    /// Returns this namespace's `durable_execution` span. The handler future
    /// (for a root context) or the child body future (for a child context)
    /// is instrumented with it so log events inherit the execution ARN,
    /// request ID, and the namespace's live `isReplay` flag.
    pub(crate) fn replay_span(&self) -> tracing::Span {
        self.inner.replay_span.clone()
    }

    /// Emits the
    /// [`operation_replayed`](crate::observability::event_names::OPERATION_REPLAYED)
    /// lifecycle event: the operation's recorded terminal outcome is being
    /// returned without re-running it (see [`crate::observability`]).
    ///
    /// `recorded_attempt` is the checkpoint record's retry count; the event
    /// reports the 1-based attempt that produced the recorded outcome.
    pub(crate) fn emit_operation_replayed(
        &self,
        wire_id: &str,
        operation_name: Option<&str>,
        operation_type: &str,
        operation_sub_type: Option<&str>,
        recorded_attempt: u32,
    ) {
        crate::tracing_layer::operation_replayed_event(&crate::tracing_layer::OperationIdentity {
            execution_arn: self.execution_arn(),
            request_id: &self.lambda_context().request_id,
            operation_id: wire_id,
            operation_name,
            operation_type,
            operation_sub_type,
            attempt: recorded_attempt.saturating_add(1),
        });
    }

    /// FLAT-nesting counterpart of [`Self::new_child`]: a child that
    /// reports `parent_wire_id_override` as its operations' parent (the
    /// batch parent, not the virtual child). The child's span carries the
    /// virtual child's wire ID as `operationId` and `name` as
    /// `operationName`.
    pub(crate) fn new_flat_child(
        &self,
        child_positional_id: &str,
        parent_wire_id_override: &str,
        name: Option<&str>,
    ) -> Self {
        let engine = Arc::new(EngineState::new_child(
            child_positional_id,
            Arc::clone(&self.inner.engine.checkpoint_log),
        ));
        // A flat branch has its own namespace and replay high-water mark,
        // so it gets its own span, as a child from `new_child` does.
        let replay_span = crate::tracing_layer::scoped_execution_span(
            &self.inner.execution_arn,
            &self.inner.lambda_context.request_id,
            self.inner.lambda_context.tenant_id.as_deref(),
            &crate::engine::compute_wire_id_public(child_positional_id),
            name,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine,
                coordinator: Arc::clone(&self.inner.coordinator),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                coalescer: self.inner.coalescer.clone(),
                direct: Arc::clone(&self.inner.direct),
                parent_wire_id: Some(parent_wire_id_override.to_owned()),
                admission: Arc::clone(&self.inner.admission),
                owners: self.owners_with(child_positional_id),
                replay_span,
                plugins: self.inner.plugins.clone(),
                nesting_level: self.inner.nesting_level,
                preserve_child_depth: self.inner.preserve_child_depth,
            }),
            awaited: None,
        }
    }

    /// The scopes of this invocation: which are closed, and the tasks
    /// started in each (see [`Admission`]).
    pub(crate) fn admission(&self) -> &Arc<Admission> {
        &self.inner.admission
    }

    /// The owners this context runs in, outermost first. Empty at the root.
    pub(crate) fn owners(&self) -> &[String] {
        &self.inner.owners
    }

    /// Returns at once when no owner of this context is abandoned, and
    /// never returns otherwise (see [`Admission`]). Every operation
    /// terminal awaits it before it runs the operation, so an abandoned
    /// owner starts no operation and replays no recorded one.
    pub(crate) async fn admit_operation(&self) {
        if !self.admission().admits(self.owners()) {
            std::future::pending::<()>().await;
        }
    }

    /// This context's owners followed by the wire ID of `positional_id`:
    /// the owners of a child context created for that operation.
    fn owners_with(&self, positional_id: &str) -> Arc<[String]> {
        self.inner
            .owners
            .iter()
            .cloned()
            .chain(std::iter::once(crate::engine::compute_wire_id_public(
                positional_id,
            )))
            .collect()
    }

    /// Installs the direct writer's pre-lock gate (see [`PreLockGate`]).
    /// The root context and every child share it.
    #[cfg(test)]
    pub(crate) fn set_pre_lock_gate(&self, gate: PreLockGate) {
        self.inner.direct.hooks().gate = Some(gate);
    }

    /// Installs the submission recorder (see [`SubmissionRecorder`]). The
    /// root context and every child share it.
    #[cfg(test)]
    pub(crate) fn set_submission_recorder(&self, recorder: SubmissionRecorder) {
        self.inner.direct.hooks().submissions = Some(recorder);
    }

    /// Installs the settler-entry recorder (see [`SettlerEntryRecorder`]).
    /// The root context and every child share it.
    #[cfg(test)]
    pub(crate) fn set_settler_entry_recorder(&self, recorder: SettlerEntryRecorder) {
        self.inner.direct.hooks().settler_entries = Some(recorder);
    }

    /// Passes `updates` and `write`'s first-poll flag to the submission
    /// recorder, when one is installed.
    #[cfg(test)]
    fn record_submission(&self, updates: &[OperationUpdate], write: &SubmittedWrite) {
        let recorder = self.inner.direct.hooks().submissions.clone();
        if let Some(recorder) = recorder
            && !updates.is_empty()
        {
            recorder(updates, &write.first_poll);
        }
    }

    /// Waits for `start`, the retained `START` write of the child-backed
    /// operation `wire_id`, before the operation writes its outcome
    /// `action`. A failed `START` ends the invocation, as
    /// [`Self::checkpoint_failure_unrecoverable`] describes, and the
    /// outcome is never written. `None` means the operation's `START` was
    /// recorded before this cycle.
    pub(crate) async fn await_retained_start(
        &self,
        wire_id: &str,
        start: Option<SubmittedWrite>,
        action: OperationAction,
    ) -> Result<(), OperationError> {
        #[cfg(test)]
        {
            let recorder = self.inner.direct.hooks().settler_entries.clone();
            if let Some(recorder) = recorder {
                recorder(wire_id, action.clone());
            }
        }
        #[cfg(not(test))]
        let _ = action;
        if let Some(start) = start
            && let Err(err) = start.await
        {
            return self
                .checkpoint_failure_unrecoverable(wire_id, err, None)
                .await;
        }
        Ok(())
    }

    /// The IDs of the updates that wait to be sent: those of the direct
    /// write queue, then those of the coalescer's open batch.
    #[cfg(test)]
    pub(crate) fn queued_update_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .inner
            .direct
            .lock()
            .entries
            .iter()
            .flat_map(|entry| entry.updates.iter().map(|u| u.id().to_owned()))
            .collect();
        if let Some(coalescer) = &self.inner.coalescer {
            ids.extend(coalescer.pending_ids());
        }
        ids
    }

    /// The checkpoint token mutex, which a checkpoint call holds.
    #[cfg(test)]
    pub(crate) fn checkpoint_token_mutex(&self) -> Arc<Mutex<String>> {
        Arc::clone(&self.inner.checkpoint_token)
    }

    /// The plugins of the invocation, shared by every context derived from
    /// the root. `None` without a registered plugin factory.
    pub(crate) fn plugins(&self) -> Option<&crate::plugin::InvocationPlugins> {
        self.inner.plugins.as_deref()
    }

    /// Delivers the queued change sets of the plugins to
    /// `on_operation_change`, and waits until every change set queued
    /// before the call was delivered, also when another task delivers it
    /// (see
    /// [`InvocationPlugins::changes_delivered`](crate::plugin::InvocationPlugins::changes_delivered)).
    /// So an operation that awaits this before it fires
    /// `on_operation_start` or `on_operation_end` fires that hook after its
    /// change. Does nothing without plugins. The caller must hold no
    /// checkpoint lock.
    pub(crate) async fn report_operation_changes(&self) {
        if let Some(plugins) = self.plugins() {
            plugins
                .changes_delivered(&self.inner.engine.checkpoint_log)
                .await;
        }
    }

    /// [`Self::plugins`], shared: a user-function runner's own frame node
    /// holds them.
    pub(crate) fn shared_plugins(&self) -> Option<Arc<crate::plugin::InvocationPlugins>> {
        self.inner.plugins.clone()
    }

    /// Returns the invocation coordinator, shared by every context of the
    /// invocation.
    pub(crate) fn coordinator(&self) -> &Arc<InvocationCoordinator> {
        &self.inner.coordinator
    }

    /// Awaits `fut` while holding an activity permit, so the invocation
    /// does not suspend while `fut` runs. When the coordinator refuses the
    /// permit (the phase left `Running`), never resolves.
    ///
    /// Wraps the step body, the `wait_for_condition` check and every serdes
    /// call: work that can make progress without the SDK polling it.
    pub(crate) async fn hold_activity<F: Future>(&self, fut: F) -> F::Output {
        let Some(permit) = self.inner.coordinator.acquire(PermitKind::Activity) else {
            return std::future::pending().await;
        };
        let output = fut.await;
        drop(permit);
        output
    }

    /// Resolves when the record of `wire_id` satisfies `ready`.
    ///
    /// Registers a wait on its first poll, then reads the record. A missing
    /// record is not ready. When the record is ready, the registration is
    /// dropped and the future resolves. Otherwise it returns `Pending`; a
    /// merged checkpoint response that carries the operation wakes it, and
    /// each later poll updates the waker and reads the record again.
    ///
    /// Registering before the read means a merge between the read and the
    /// end of the poll still wakes the waiting task. When the coordinator
    /// refuses the registration (the phase left `Running`), never resolves.
    pub(crate) async fn wait_for_record(
        &self,
        wire_id: &str,
        ready: fn(&CheckpointStatus) -> bool,
    ) {
        let mut registration: Option<crate::coordinator::WaitRegistration> = None;
        std::future::poll_fn(|cx| {
            if let Some(live) = &registration {
                live.update_waker(cx.waker());
            } else {
                // The refresh schedule starts from the record's end time
                // and status. A missing record is taken as `STARTED` with
                // no end time.
                let (status, due_at) = self
                    .refresh_view(wire_id)
                    .unwrap_or((CheckpointStatus::Started, None));
                match self.inner.coordinator.register_wait(
                    wire_id,
                    due_at,
                    &status,
                    self.awaited.clone(),
                    cx.waker(),
                ) {
                    Some(live) => registration = Some(live),
                    None => return std::task::Poll::Pending,
                }
            }
            let is_ready = self
                .inner
                .engine
                .checkpoint_log
                .status_view(wire_id)
                .is_some_and(|view| ready(&view.status));
            if is_ready {
                registration = None;
                std::task::Poll::Ready(())
            } else {
                if let Some(live) = &registration {
                    live.resume_schedule();
                }
                std::task::Poll::Pending
            }
        })
        .await;
        // The response that readied the record queued its change set. It is
        // delivered before the operation reads the record again, so it
        // comes before the operation's end hook.
        self.report_operation_changes().await;
    }

    /// The status and end time of the record of `wire_id`, as the refresh
    /// scheduler reads them.
    pub(crate) fn refresh_view(
        &self,
        wire_id: &str,
    ) -> Option<(CheckpointStatus, Option<std::time::SystemTime>)> {
        self.inner
            .engine
            .checkpoint_log
            .with_record(wire_id, |record| (record.status.clone(), record.due_at))
    }

    /// A clone of this context whose wait registrations carry `flag`, the
    /// first-poll flag of the handle of an operation started on its own
    /// task. The refresh scheduler skips those registrations until the
    /// handle is first polled and sets the flag.
    pub(crate) fn with_awaited(&self, flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            awaited: Some(flag),
        }
    }

    /// Waits until the record of the operation at `positional_id` satisfies
    /// `ready` (see [`Self::wait_for_record`]), then returns its view.
    pub(crate) async fn wait_for_record_view(
        &self,
        positional_id: &str,
        ready: fn(&CheckpointStatus) -> bool,
    ) -> CheckpointStatusView {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.wait_for_record(&wire_id, ready).await;
        #[expect(clippy::expect_used)]
        // reason: `wait_for_record` resolves only once the record exists,
        // and the checkpoint log never removes a record during an
        // invocation.
        self.checkpoint_status_view(positional_id)
            .expect("a record that satisfied the wait is still in the log")
    }

    /// Returns the time left until the Lambda deadline: `None` when the
    /// deadline is 0 (unknown), otherwise the deadline (milliseconds since
    /// the Unix epoch) minus now, and `Duration::ZERO` once it passed.
    pub(crate) fn remaining_time(&self) -> Option<Duration> {
        remaining_until(
            self.inner.lambda_context.deadline,
            std::time::SystemTime::now(),
        )
    }

    /// Returns a reference to the task-ownership tracker.
    pub(crate) fn task_ownership(&self) -> &Arc<TaskOwnership> {
        &self.inner.task_ownership
    }

    /// Checks task ownership and returns an `OperationError` if the caller
    /// is not authorized. Used by every durable operation entry point.
    pub(crate) fn enforce_task_ownership(&self) -> Result<(), OperationError> {
        self.inner
            .task_ownership
            .check_current_task()
            .map_err(|msg| {
                self.coordinator()
                    .record_fatal("Error".to_owned(), foreign_caller_message("execute"));
                OperationError::from_kind(OperationErrorKind::Step(StepError::new(
                    StepErrorKind::ExecutionFailed,
                    Some(msg.into()),
                )))
            })
    }

    /// Reads the checkpoint record for the given positional ID in place,
    /// applying `f` under the log's read guard.
    ///
    /// Returns `None` when no record exists. This (and the targeted
    /// accessors below) replaces the removed full-record clone: `f` borrows
    /// the stored record, so only what `f` returns is cloned. `f` must not
    /// touch the checkpoint log (the read guard is held while it runs).
    ///
    /// NOTE: The checkpoint log is keyed by wire ID (the hash), which is
    /// what the backend returns in Operations[].Id. We look up by wire ID
    /// computed from the positional ID.
    pub(crate) fn with_checkpoint_record<R>(
        &self,
        positional_id: &str,
        f: impl FnOnce(&CheckpointRecord) -> R,
    ) -> Option<R> {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.with_record(&wire_id, f)
    }

    /// [`Self::with_checkpoint_record`], for the record stored under the
    /// wire ID `wire_id`.
    pub(crate) fn with_checkpoint_record_by_wire<R>(
        &self,
        wire_id: &str,
        f: impl FnOnce(&CheckpointRecord) -> R,
    ) -> Option<R> {
        self.inner.engine.checkpoint_log.with_record(wire_id, f)
    }

    /// Returns the compact view (status, attempt,
    /// `replay_children`) of the checkpoint record for the given positional
    /// ID, without cloning any of the record's owned strings.
    pub(crate) fn checkpoint_status_view(
        &self,
        positional_id: &str,
    ) -> Option<CheckpointStatusView> {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.status_view(&wire_id)
    }

    /// Returns whether a checkpoint record exists for the given positional
    /// ID, without cloning anything.
    ///
    /// Production operation paths read record presence through
    /// [`Self::checkpoint_view_validated`] (which also validates identity);
    /// this direct form is retained for unit tests.
    #[cfg(test)]
    pub(crate) fn has_checkpoint_record(&self, positional_id: &str) -> bool {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.contains(&wire_id)
    }

    /// Returns the recorded result payload for the given positional ID,
    /// cloning only that one string.
    ///
    /// `None` means either no record exists or the record carries no
    /// result: callers that need to distinguish should use
    /// [`Self::with_checkpoint_record`].
    pub(crate) fn checkpoint_result_payload(&self, positional_id: &str) -> Option<String> {
        self.with_checkpoint_record(positional_id, |record| record.result.clone())
            .flatten()
    }

    /// Returns the backend-assigned callback ID for the given positional
    /// ID, cloning only that one string.
    pub(crate) fn checkpoint_callback_id(&self, positional_id: &str) -> Option<String> {
        self.with_checkpoint_record(positional_id, |record| record.callback_id.clone())
            .flatten()
    }

    /// Returns the recorded wire failure record for the given positional
    /// ID, cloning only the failure strings. `None` means no record
    /// exists.
    pub(crate) fn checkpoint_wire_error(
        &self,
        positional_id: &str,
    ) -> Option<crate::error::WireError> {
        self.with_checkpoint_record(positional_id, |record| {
            crate::error::WireError::new(record.error_type.clone(), record.error_message.clone())
                .with_error_data(record.error_data.clone())
                .with_stack_trace(record.stack_trace.clone().unwrap_or_default())
        })
    }

    /// Returns the terminal-replay projection (status, `replay_children`,
    /// and the result/error strings) of the checkpoint record for the
    /// given positional ID, cloning only those payload strings.
    ///
    /// This is what the map/parallel replay helpers consume when
    /// reconstructing a terminal batch or child item; the record's invoke,
    /// callback, attempt, ID, and identity fields are never cloned. `None`
    /// means no record exists.
    pub(crate) fn checkpoint_terminal_replay(
        &self,
        positional_id: &str,
    ) -> Option<TerminalReplaySnapshot> {
        self.with_checkpoint_record(positional_id, |record| TerminalReplaySnapshot {
            status: record.status.clone(),
            replay_children: record.replay_children,
            result: record.result.clone(),
            error_message: record.error_message.clone(),
            error_type: record.error_type.clone(),
            error_data: record.error_data.clone(),
            stack_trace: record.stack_trace.clone(),
        })
    }

    /// Validates the claimed operation identity against the checkpoint
    /// record and returns the compact status view, in a single read-guard
    /// pass: the common preamble of every operation's replay check.
    ///
    /// Returns `Ok(None)` when no record exists (live position, nothing to
    /// validate), `Ok(Some(view))` when the identity matches, and the
    /// `NonDeterministicExecution` error (also recorded on the fatal slot,
    /// exactly as [`Self::validate_replay_identity`] does) on mismatch,
    /// or when the record carries a [`CheckpointStatus::Unknown`] status
    /// this SDK version cannot interpret, in which case the error names
    /// the raw status. Nothing is cloned on the match path beyond the
    /// status itself.
    pub(crate) fn checkpoint_view_validated(
        &self,
        positional_id: &str,
        wire_id: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<Option<CheckpointStatusView>, OperationError> {
        let Some((mismatch, view)) = self.with_checkpoint_record(positional_id, |record| {
            (
                replay_identity_mismatch(record, claimed_type, claimed_sub_type, claimed_name),
                CheckpointStatusView {
                    status: record.status.clone(),
                    attempt: record.attempt,
                    replay_children: record.replay_children,
                },
            )
        }) else {
            return Ok(None);
        };
        if let Some(expected) = mismatch {
            // Built (and the fatal slot written) OUTSIDE the read guard.
            return Err(self.replay_mismatch_error(
                wire_id,
                &expected,
                claimed_type,
                claimed_sub_type,
                claimed_name,
            ));
        }
        if let CheckpointStatus::Unknown(raw) = &view.status {
            // A status this SDK cannot interpret must never be acted on:
            // guessing non-terminal re-runs completed work, and guessing
            // terminal fabricates an outcome that was never recorded. Fail
            // the execution naming the raw status (issue #45).
            return Err(self.unrecognized_status_error(wire_id, raw));
        }
        Ok(Some(view))
    }

    /// Eagerly validates a claimed operation's replay identity against the
    /// checkpoint log, BEFORE the operation future first runs.
    ///
    /// Builders call this when they are finalized into a
    /// [`crate::DurableFuture`] (`.future()`, `.spawn()`, or `.await` via
    /// `IntoFuture`). Validating at finalization instead of first poll makes
    /// fatal propagation scheduler-independent: a short-circuiting combinator
    /// (`select_ok`, `race`, `try_join_all`) aborts losers as soon as a
    /// winner settles, so a mismatching constituent might otherwise never be
    /// polled and the mismatch never recorded. The fatal slot is written
    /// synchronously here (inside [`Self::validate_replay_identity`]), so the
    /// invocation driver fails the execution with the dedicated error no
    /// matter which sibling settles first.
    ///
    /// A position with no checkpoint record is live (nothing to validate),
    /// and a matching identity is a no-op: an unchanged handler behaves
    /// identically.
    pub(crate) fn preflight_replay_identity(
        &self,
        op_id: &OperationId,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<(), OperationError> {
        // An abandoned owner replays no operation, so its claim is not
        // checked against history. A mismatch here would record a fatal
        // error and resolve the future. Skipping the check leaves the
        // future to the admission check in `DurableFuture::for_operation`,
        // which never resolves it. An owner abandoned after this check is
        // stopped there too.
        if !self.admission().admits(self.owners()) {
            return Ok(());
        }
        self.checkpoint_view_validated(
            op_id.positional(),
            op_id.wire(),
            claimed_type,
            claimed_sub_type,
            claimed_name,
        )?;
        Ok(())
    }

    /// Validates that the claimed operation's identity matches the
    /// checkpoint record. On mismatch, returns a
    /// `NonDeterministicExecution` error.
    ///
    /// `claimed_type` is the operation type string the SDK sends to the
    /// backend (e.g. `Step`, `Wait`, `Context`, `ChainedInvoke`,
    /// `Callback`).
    ///
    /// `claimed_sub_type` is the sub-type (e.g. `Step`, `Wait`, `Map`,
    /// `RunInChildContext`) or `None` for operations without a sub-type.
    ///
    /// `claimed_name` is the user-supplied `.name("...")` or `None`.
    ///
    /// Production paths validate through [`Self::checkpoint_view_validated`]
    /// (one read-guard pass, no record clone); this record-taking form is
    /// retained as the direct unit-test harness for the identity-matching
    /// rules both share.
    #[cfg(test)]
    pub(crate) fn validate_replay_identity(
        &self,
        record: &CheckpointRecord,
        wire_id: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<(), OperationError> {
        match replay_identity_mismatch(record, claimed_type, claimed_sub_type, claimed_name) {
            None => Ok(()),
            Some(expected) => Err(self.replay_mismatch_error(
                wire_id,
                &expected,
                claimed_type,
                claimed_sub_type,
                claimed_name,
            )),
        }
    }

    /// Builds the `NonDeterministicExecution` mismatch error and records it
    /// on the shared fatal slot.
    ///
    /// A replay identity mismatch is execution-fatal: recording it on the
    /// shared slot makes the invocation driver fail the execution with the
    /// dedicated error even if the returned `Err` is swallowed on its way
    /// up: stored as a rejected outcome by `join_all`, out-raced by a
    /// sibling's success in `select_ok`, stringified through a
    /// child-context boundary, or tolerated by a map/parallel completion
    /// config.
    fn replay_mismatch_error(
        &self,
        wire_id: &str,
        expected: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> OperationError {
        let err = OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
            NonDeterministicExecutionError::from_kind(
                NonDeterministicExecutionErrorKind::OperationMismatch(
                    crate::error::OperationMismatch::new(
                        wire_id,
                        expected,
                        format_op_identity(claimed_type, claimed_sub_type, claimed_name),
                    ),
                ),
            ),
        ));
        self.inner.coordinator.record_fatal(
            "NonDeterministicExecutionError".to_owned(),
            crate::error::chain_string(&err),
        );
        err
    }

    /// Builds the execution-fatal error for a checkpoint record whose
    /// status this SDK version does not recognize, recording it on the
    /// shared fatal slot (issue #45).
    ///
    /// Exactly like [`Self::replay_mismatch_error`], the fatal-slot write
    /// makes the invocation driver fail the execution with the dedicated
    /// error even if the returned `Err` is swallowed on its way up. The
    /// error message carries the raw status verbatim, so the operator can
    /// see which service-side status the SDK build predates.
    pub(crate) fn unrecognized_status_error(&self, wire_id: &str, raw: &str) -> OperationError {
        let err = OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
            NonDeterministicExecutionError::from_kind(
                NonDeterministicExecutionErrorKind::UnrecognizedStatus(
                    crate::error::UnrecognizedStatus::new(wire_id, raw),
                ),
            ),
        ));
        self.inner.coordinator.record_fatal(
            "NonDeterministicExecutionError".to_owned(),
            crate::error::chain_string(&err),
        );
        err
    }

    /// Routes a checkpoint write failure through the unrecoverable path
    /// and never returns control to the caller (issue #43).
    ///
    /// A failed outcome write must never become a catchable operation
    /// error: a handler that catches it branches on a decision no
    /// checkpoint records, which replay cannot reproduce. Instead the
    /// failure's classification (see
    /// [`ExecutionClientError::is_retryable`](crate::client::ExecutionClientError)) picks
    /// the recovery scope:
    ///
    /// - **Retryable** (exhausted transient failure, stale token): the
    ///   write channel is down, so a follow-up write would fail the same
    ///   way. Nothing more is written; the invocation fails with a Lambda
    ///   runtime error and the service re-invokes: the same recovery as
    ///   an interruption.
    /// - **Non-retryable** (permanent rejection, e.g. oversized payload):
    ///   re-invoking would re-run the body into the same rejection, an
    ///   infinite loop with side effects firing once per lap. When the
    ///   caller supplies `terminal_fail`, required at every site where
    ///   user code already ran, so its side effects get a recorded
    ///   outcome, that small terminal `FAIL` is written first (it goes
    ///   through on a channel that rejected only the payload; a failure
    ///   here is logged, not propagated, because the execution dies
    ///   either way). Then the execution fails with the record of
    ///   [`checkpoint_execution_failure_wire`](crate::error::checkpoint_execution_failure_wire):
    ///   `CheckpointUnrecoverableExecutionError`, as in JS.
    ///
    /// The returned future never resolves: the fatal slot is recorded and
    /// the invocation driver, which checks it with priority over completion
    /// and suspension, drops the handler at its current await point, so
    /// user code can neither catch nor ignore the failure.
    ///
    /// One error is not a write failure: a direct write that found no Tokio
    /// runtime sent nothing (see [`Self::write_direct_detached`]). The
    /// operation then returns the error `.spawn()` returns without a
    /// runtime, at once, and nothing is recorded.
    pub(crate) async fn checkpoint_failure_unrecoverable<T: NoRuntimeOutcome>(
        &self,
        op_wire_id: &str,
        err: ExecutionClientError,
        terminal_fail: Option<OperationUpdate>,
    ) -> T {
        if let Some(no_runtime) = std::error::Error::source(&err)
            .and_then(|source| source.downcast_ref::<crate::task::NoRuntime>())
        {
            return T::no_runtime(*no_runtime);
        }
        if !err.is_retryable()
            && let Some(update) = terminal_fail
        {
            // Direct write: the coalescing buffer may hold the very
            // updates the service just rejected, and this record must
            // not queue behind them.
            if let Err(fail_err) = self.checkpoint_updates_direct(vec![update]).await {
                tracing::error!(
                    operation_id = %op_wire_id,
                    error = %fail_err,
                    "terminal FAIL write also failed after non-retryable checkpoint \
                     failure; failing the execution without a recorded outcome"
                );
            }
        }
        self.record_checkpoint_failure(op_wire_id, &err);
        std::future::pending::<T>().await
    }

    /// Records the classification of a failed checkpoint write of
    /// `op_wire_id` in the fatal slot, and returns at once.
    ///
    /// A retryable failure fails the invocation. A non-retryable failure
    /// fails the execution with the record of
    /// [`checkpoint_execution_failure_wire`](crate::error::checkpoint_execution_failure_wire).
    /// [`Self::checkpoint_failure_unrecoverable`] calls it after its
    /// terminal `FAIL` write. A refresh write calls it on the task that
    /// receives the response, so the classification is recorded even when
    /// the refresh's caller was dropped.
    pub(crate) fn record_checkpoint_failure(&self, op_wire_id: &str, err: &ExecutionClientError) {
        let message = format!("checkpoint write failed for operation {op_wire_id}: {err}");
        if err.is_retryable() {
            tracing::error!(
                operation_id = %op_wire_id,
                error = %err,
                "retryable checkpoint failure exhausted retries; failing the invocation \
                 so the service re-invokes"
            );
            self.inner.coordinator.record_invocation_fault(
                crate::error::CHECKPOINT_INVOCATION_FAILED_ERROR_TYPE,
                message,
            );
        } else {
            tracing::error!(
                operation_id = %op_wire_id,
                error = %err,
                "non-retryable checkpoint failure; failing the execution"
            );
            let wire = crate::error::checkpoint_execution_failure_wire(err);
            self.inner.coordinator.record_fatal(
                wire.error_type().unwrap_or_default().to_owned(),
                wire.error_message().unwrap_or_default().to_owned(),
            );
        }
    }

    /// Ends the invocation because a serdes call failed with a marked
    /// error. Never resolves.
    ///
    /// A serdes marks a transient failure with
    /// [`RetryableSerdesError`](crate::serdes::RetryableSerdesError). The
    /// failure records nothing: the invocation fails with a Lambda runtime
    /// error, and the service invokes the function again. The next
    /// invocation then repeats the failed serdes call. Like
    /// [`Self::checkpoint_failure_unrecoverable`], the fault goes into the
    /// fatal slot that every scope shares, so user code can neither catch
    /// nor ignore it.
    pub(crate) async fn serdes_fault<T>(
        &self,
        call: &'static str,
        operation_id: &str,
        name: Option<&str>,
        err: BoxError,
    ) -> T {
        let chain = crate::error::chain_string(&*err);
        let message = match name {
            Some(name) => {
                format!("serdes {call} failed for operation \"{name}\" ({operation_id}): {chain}")
            }
            None => format!("serdes {call} failed for operation {operation_id}: {chain}"),
        };
        tracing::error!(
            operation_id = %operation_id,
            error = %chain,
            "serdes {call} failed with a retryable error; failing the invocation so the \
             service re-invokes"
        );
        self.inner
            .coordinator
            .record_invocation_fault(crate::error::SERDES_FAILED_ERROR_TYPE, message);
        std::future::pending::<T>().await
    }

    /// Ends the invocation when `err` is marked with
    /// `RetryableSerdesError`. Returns `err` unchanged otherwise.
    ///
    /// Every call to a user [`Serdes`](crate::serdes::Serdes) passes its
    /// error through this method before any other handling. So an
    /// unmarked error takes the caller's permanent-failure path, and a
    /// marked error never reaches it.
    pub(crate) async fn route_serdes_error(
        &self,
        call: &'static str,
        operation_id: &str,
        name: Option<&str>,
        err: BoxError,
    ) -> BoxError {
        if crate::serdes::is_retryable_serdes_error(&*err) {
            return self.serdes_fault(call, operation_id, name, err).await;
        }
        err
    }

    /// Checkpoints operation updates via the execution client.
    ///
    /// When [`Options`](crate::Options) configured a `checkpoint_delay`
    /// and/or `checkpoint_batching`, the updates join the shared coalescing
    /// batch and this call resolves with the batch's result once it flushes
    /// (after at most the configured delay, or earlier if a flush point
    /// drains the buffer). Without either knob, the write happens
    /// immediately via [`Self::checkpoint_updates_direct`].
    pub(crate) async fn checkpoint_updates(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        self.submit_checkpoint_updates(updates).await
    }

    /// Starts the checkpoint write of [`Self::checkpoint_updates`] before
    /// it returns, and returns the future of its result.
    ///
    /// With buffering, the updates join the batch and take their checkpoint
    /// permit under the buffer lock. Without it, they enter the
    /// [`DirectWriteQueue`], whose writer sends them in submission order.
    /// So the write keeps its place in the write order, and continues when
    /// the returned value is dropped.
    pub(crate) fn submit_checkpoint_updates(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> SubmittedWrite {
        self.checkpoint_with_urgency(updates, false)
    }

    /// Checkpoints operation updates, flushing the coalescing buffer
    /// immediately instead of waiting out the delay window.
    ///
    /// This is the "callback creation" flush point of the
    /// [`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay)
    /// contract: the caller needs the backend's response right away (the
    /// service assigns the callback ID in it), so the batch, including any
    /// previously buffered updates, which keeps write order intact, is
    /// written now. Identical to [`Self::submit_checkpoint_updates`] when
    /// no delay is configured.
    ///
    /// The updates join the batch, or the direct write queue, before this
    /// returns, as with [`Self::submit_checkpoint_updates`].
    pub(crate) fn submit_checkpoint_updates_urgent(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> SubmittedWrite {
        self.checkpoint_with_urgency(updates, true)
    }

    /// Fixes the position of `updates` in the write order, as
    /// [`Self::submit_checkpoint_updates`] does, but holds them back until
    /// the returned [`ReservedWrite`] is released or dropped.
    ///
    /// With buffering, the updates join the open batch, and the batch is
    /// not flushed while the reservation holds it. Without buffering, they
    /// enter the [`DirectWriteQueue`], and the writer waits at them. So no
    /// request carries them, and no response is merged for them, before the
    /// release.
    pub(crate) fn reserve_checkpoint_updates(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> ReservedWrite {
        #[cfg(test)]
        let recorded = updates.clone();
        let tracked = self.track_updates(updates);
        let Some(coalescer) = self.inner.coalescer.clone() else {
            let (updates, events): (Vec<_>, Vec<_>) =
                tracked.into_iter().map(|t| (t.update, t.event)).unzip();
            let Some(permit) = self.inner.coordinator.acquire(PermitKind::Checkpoint) else {
                return ReservedWrite::rejected();
            };
            let (write, id) = self.enqueue_direct(updates, events, Some(permit), None, true);
            let write = write.reported_by(self);
            #[cfg(test)]
            if id.is_some() {
                self.record_submission(&recorded, &write);
            }
            return ReservedWrite {
                write,
                hold: id.map(|id| WriteHold::Direct {
                    queue: Arc::clone(&self.inner.direct),
                    id,
                }),
            };
        };
        let Some(batch) = self.join_batch(&coalescer, tracked, false, None, true) else {
            return ReservedWrite::rejected();
        };
        let write = batch_result(batch).reported_by(self);
        #[cfg(test)]
        self.record_submission(&recorded, &write);
        ReservedWrite {
            write,
            hold: Some(WriteHold::Batch(coalescer)),
        }
    }

    /// Sends one checkpoint with no updates for the refresh scheduler, and
    /// records the classification of a failure for `first_wire_id` before
    /// it returns the error.
    ///
    /// The driver drops the scheduler, and so this future, when the
    /// handler returns. A failure must still decide the invocation's
    /// outcome, so it is never left to this future alone:
    ///
    /// - Without buffering, the detached write task records the failure
    ///   itself (see [`Self::write_direct_detached`]).
    /// - With buffering, the refresh joins the batch with `first_wire_id`
    ///   as a failure owner, and the detached flusher that publishes the
    ///   batch's failure records it (see [`Self::publish_batch`]).
    pub(crate) async fn checkpoint_refresh(
        &self,
        first_wire_id: &str,
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        let write = if self.inner.coalescer.is_some() {
            self.write_with_urgency(Vec::new(), true, Some(first_wire_id))
        } else {
            // No checkpoint permit: the scheduler's permit keeps the
            // invocation open until the response is merged and the readied
            // operations are woken. The write task is counted, so the
            // drive's cleanup waits for it after the scheduler is dropped.
            self.write_direct_detached(Vec::new(), Vec::new(), None, Some(first_wire_id.to_owned()))
        };
        // The task that sends the refresh delivers the change sets queued
        // so far once the call returns.
        write.reported_by(self).await
    }

    /// Shared body of the buffered checkpoint paths: pairs each update
    /// with its captured lifecycle-event metadata, then hands the pairs to
    /// the write path, which emits each event as soon as the write that
    /// persists its update succeeds. The write starts before this returns.
    fn checkpoint_with_urgency(
        &self,
        updates: Vec<OperationUpdate>,
        urgent: bool,
    ) -> SubmittedWrite {
        // Lifecycle events (see `crate::observability`): every operation
        // transition the SDK records passes through here, so this one site
        // covers `operation_started` / `operation_succeeded` /
        // `operation_failed` / `operation_retry_scheduled` for every
        // operation type. The metadata is captured BEFORE the write (the
        // checkpoint response mutates the log the attempt is derived from,
        // and the updates themselves are consumed by it; the capture also
        // snapshots this call's span so the event keeps its originating
        // context). Ownership then travels WITH the update into the write
        // path, the buffered flush task, not this possibly-cancelled
        // future, emits the event, so a contributor dropped after joining
        // a batch (a lost `race`/`select_ok` branch) cannot suppress
        // telemetry for a transition the flush still persists, and events
        // are emitted per persisted chunk rather than per batch. A rejected
        // write records nothing, so it emits nothing.
        let tracked = self.track_updates(updates);
        self.write_with_urgency(tracked, urgent, None)
            .reported_by(self)
    }

    /// Pairs each update with its lifecycle-event metadata, captured now
    /// (see [`Self::checkpoint_with_urgency`]).
    fn track_updates(&self, updates: Vec<OperationUpdate>) -> Vec<TrackedUpdate> {
        updates
            .into_iter()
            .map(|update| {
                // The current attempt is one past the recorded retry count,
                // exactly as the step live path derives it.
                let attempt = self
                    .inner
                    .engine
                    .checkpoint_log
                    .status_view(update.id())
                    .map_or(0, |view| view.attempt)
                    .saturating_add(1);
                TrackedUpdate::capture(update, attempt)
            })
            .collect()
    }

    /// Dispatches a checkpoint write: joins the coalescing batch when one
    /// is configured, otherwise writes directly. Under a non-zero delay, the
    /// non-urgent join that opens a batch starts a detached flush that
    /// waits out the window; an urgent contributor, or any contributor
    /// under a zero-delay coalescer (pure `checkpoint_batching` mode),
    /// requests the flush right away. Either way the flush task claims and
    /// writes the batch under the coalescer's writer lock, so batching still
    /// emerges while an earlier write is in flight.
    ///
    /// Joining the batch transfers ownership of each update's lifecycle
    /// event to the coalescer: the flush task that persists the update
    /// emits it, so this future can be dropped mid-await without losing
    /// telemetry for a transition the flush still records.
    ///
    /// `failure_owner` applies only with buffering: the joined batch
    /// records a failed write's classification for it when the flusher
    /// publishes the failure (see [`Self::publish_batch`]).
    ///
    /// The join or the direct write happens before this returns, so the
    /// returned [`SubmittedWrite`] only waits for the result.
    fn write_with_urgency(
        &self,
        updates: Vec<TrackedUpdate>,
        urgent: bool,
        failure_owner: Option<&str>,
    ) -> SubmittedWrite {
        // The recorder sees the submission before it joins the coalescer
        // or enters the queue, so before the writer can send it. The write
        // built below takes the flag the recorder saw.
        #[cfg(test)]
        let flag = {
            let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let recorder = self.inner.direct.hooks().submissions.clone();
            if let Some(recorder) = recorder
                && !updates.is_empty()
                && self.admission().admits(self.owners())
            {
                let submitted: Vec<OperationUpdate> =
                    updates.iter().map(|t| t.update.clone()).collect();
                recorder(&submitted, &flag);
            }
            flag
        };
        #[cfg_attr(not(test), expect(unused_mut))] // reason: only the test build sets the flag
        let mut write = self.write_with_urgency_unrecorded(updates, urgent, failure_owner);
        #[cfg(test)]
        {
            write.first_poll = flag;
        }
        write
    }

    /// [`Self::write_with_urgency`] without the test submission recorder.
    fn write_with_urgency_unrecorded(
        &self,
        updates: Vec<TrackedUpdate>,
        urgent: bool,
        failure_owner: Option<&str>,
    ) -> SubmittedWrite {
        let Some(coalescer) = self.inner.coalescer.clone() else {
            let (updates, events): (Vec<_>, Vec<_>) =
                updates.into_iter().map(|t| (t.update, t.event)).unzip();
            let Some(permit) = self.inner.coordinator.acquire(PermitKind::Checkpoint) else {
                // The phase left `Running`: the write is rejected, as a
                // write from an abandoned owner is.
                return SubmittedWrite::rejected();
            };
            return self.write_direct_detached(updates, events, Some(permit), None);
        };

        match self.join_batch(&coalescer, updates, urgent, failure_owner, false) {
            Some(batch) => batch_result(batch),
            None => SubmittedWrite::rejected(),
        }
    }

    /// Joins `updates` to the coalescer's open batch and schedules its
    /// flush. With `reserve`, the join holds the batch until the hold is
    /// released (see [`ReservedWrite`]). Returns `None` when admission
    /// rejects the join.
    fn join_batch(
        &self,
        coalescer: &Arc<CheckpointCoalescer>,
        updates: Vec<TrackedUpdate>,
        urgent: bool,
        failure_owner: Option<&str>,
        reserve: bool,
    ) -> Option<Arc<CheckpointBatch>> {
        // Admission (see `Admission`): an update from an abandoned owner is
        // rejected, and a rejected write never resolves. Its task is being
        // cancelled. Returning `Ok` instead would let an operation continue
        // as if its update were recorded. The admitted join holds a
        // checkpoint permit until the batch's response is merged.
        let (batch, opened) = coalescer.join_owned(
            self.admission(),
            self.coordinator(),
            self.owners(),
            updates,
            failure_owner,
            reserve,
        )?;
        if urgent || coalescer.delay().is_zero() {
            self.spawn_batch_flush(coalescer, &batch, Duration::ZERO);
        } else if opened {
            // The join that opened the batch arms its window on a detached
            // task, so the batch is written even when every contributor is
            // dropped before the window elapses.
            self.spawn_batch_flush(coalescer, &batch, coalescer.delay());
        }
        Some(batch)
    }

    /// Writes one request's worth of tracked updates immediately and, if,
    /// and only if, the checkpoint call that persists them succeeds, emits
    /// each update's lifecycle event (inside the span captured with it).
    /// This is the single site that pairs "the transition was persisted"
    /// with "its telemetry was emitted": both the unbuffered path and every
    /// chunk of a batched write funnel through it. The emission happens
    /// inside [`Self::checkpoint_direct_with_events`], immediately after
    /// the service accepts the write, before the fallible pagination
    /// hydration that follows it, so a transition the service recorded
    /// always emits its event even when hydrating the paginated state
    /// fails afterwards.
    ///
    /// Only the batch flusher calls this. Its updates were admitted when
    /// they joined the buffer, so it skips the admission check.
    async fn write_tracked_direct(
        &self,
        tracked: Vec<TrackedUpdate>,
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        let (updates, events): (Vec<_>, Vec<_>) =
            tracked.into_iter().map(|t| (t.update, t.event)).unzip();
        self.write_unchecked(updates, &events).await
    }

    /// Writes updates that skip the admission check: the batch flusher's.
    async fn write_unchecked(
        &self,
        updates: Vec<OperationUpdate>,
        events: &[crate::tracing_layer::PendingTransitionEvent],
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        #[expect(clippy::expect_used)]
        // reason: with no owners to check, `checkpoint_direct_with_events`
        // never rejects the write, so it always returns a result.
        self.checkpoint_direct_with_events(updates, events, None)
            .await
            .expect("a write without an admission check is never rejected")
    }

    /// Writes updates directly, through the [`DirectWriteQueue`] and its
    /// detached writer, after the admission check, and returns the result
    /// to the caller.
    ///
    /// `permit` is the write's checkpoint permit, held until the response is
    /// merged, the operations it readied are woken, and the result is
    /// handed to the caller. `None` is passed in two cases: after the drive
    /// ended, and for an unbatched refresh, whose scheduler permit holds the
    /// invocation open instead (see [`Self::checkpoint_refresh`]). In both
    /// cases the cleanup's `wait_idle` waits for the counted writer.
    ///
    /// The write runs on the writer's task because a cancelled caller must not
    /// cancel an admitted request. A batch abandons its running items at
    /// early completion and aborts their tasks. If an aborted item
    /// cancelled a request the service had accepted, the checkpoint token
    /// would not rotate, and the parent's next write would use a stale
    /// token. The task is counted, so the invocation waits for it.
    ///
    /// The outcomes are:
    ///
    /// - No Tokio runtime is running: nothing is sent, and the caller gets
    ///   an invocation-scope error whose source is
    ///   [`NoRuntime`](crate::task::NoRuntime). An operation that receives
    ///   it returns the error `.spawn()` returns without a runtime (see
    ///   [`Self::checkpoint_failure_unrecoverable`]).
    /// - The admission check rejects the write: nothing is sent, and the
    ///   returned future never resolves (see [`Admission`]).
    /// - The checkpoint client panics: the task gateway records the panic on
    ///   the coordinator, which ends the drive. The channel carries no
    ///   payload, so the caller stays pending until the drive's cleanup
    ///   cancels it.
    /// - Otherwise: the caller gets the write's result.
    ///
    /// With `failure_owner` set, the task itself records a failed write's
    /// classification for that operation (see
    /// [`Self::record_checkpoint_failure`]) before it releases the permit.
    /// A refresh sets it, because the driver drops the refresh's caller
    /// when the handler returns, and the failure must still decide the
    /// invocation's outcome.
    ///
    /// The write enters the queue before this returns, so its position in
    /// the write order is fixed, and the returned [`SubmittedWrite`] only
    /// waits for the result.
    fn write_direct_detached(
        &self,
        updates: Vec<OperationUpdate>,
        events: Vec<crate::tracing_layer::PendingTransitionEvent>,
        permit: Option<crate::coordinator::Permit>,
        failure_owner: Option<String>,
    ) -> SubmittedWrite {
        self.enqueue_direct(updates, events, permit, failure_owner, false)
            .0
    }

    /// Appends a write to the [`DirectWriteQueue`] and starts the direct
    /// writer when none runs. Returns the write's result future and, while
    /// the entry is queued, its ID. A `held` entry is not written until
    /// [`DirectWriteQueue::release`] releases it.
    ///
    /// When no Tokio runtime is running, the writer cannot start: every
    /// queued entry gets an invocation-scope error whose source is
    /// [`NoRuntime`](crate::task::NoRuntime), and no ID is returned.
    fn enqueue_direct(
        &self,
        updates: Vec<OperationUpdate>,
        events: Vec<crate::tracing_layer::PendingTransitionEvent>,
        permit: Option<crate::coordinator::Permit>,
        failure_owner: Option<String>,
        held: bool,
    ) -> (SubmittedWrite, Option<u64>) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let queue = &self.inner.direct;
        let (id, start_writer) = {
            let mut state = queue.lock();
            let id = state.next_id;
            state.next_id = state.next_id.wrapping_add(1);
            state.entries.push_back(DirectEntry {
                id,
                held,
                ctx: self.clone(),
                updates,
                events,
                permit,
                failure_owner,
                sender,
            });
            let start_writer = !state.writer_running;
            state.writer_running = true;
            (id, start_writer)
        };
        let mut queued = Some(id);
        if start_writer
            && let Err(no_runtime) = crate::task::spawn_detached(
                self.coordinator(),
                run_direct_writer(Arc::clone(queue)),
            )
        {
            // The source marks the error, so that
            // `checkpoint_failure_unrecoverable` returns the operation error
            // `.spawn()` returns instead of failing the invocation.
            let err = ExecutionClientError::with_source(
                ExecutionClientErrorScope::Invocation,
                no_runtime.to_string(),
                no_runtime,
            );
            let drained = {
                let mut state = queue.lock();
                state.writer_running = false;
                std::mem::take(&mut state.entries)
            };
            for entry in drained {
                // The receivers are this call's and those of callers that
                // queued behind it on a thread without a runtime.
                let _ = entry.sender.send(Some(Err(err.clone())));
            }
            queued = None;
        }
        let mut receiver = UnreadResultGuard {
            receiver,
            coordinator: Arc::clone(self.coordinator()),
        };
        // The writer is never aborted, so the sender is dropped unsent only
        // when the checkpoint client panicked. The gateway recorded that
        // panic, so the caller stays pending until the cleanup cancels it.
        let write = SubmittedWrite::new(async move {
            let Ok(written) = (&mut receiver.receiver).await else {
                return std::future::pending().await;
            };
            match written {
                Some(result) => result,
                None => std::future::pending().await,
            }
        });
        (write, queued)
    }

    /// Spawns a task that claims `batch` (if it is still the open batch)
    /// and writes its buffered updates, publishing the result to every
    /// contributor. The claim and the write both happen while holding the
    /// coalescer's writer lock, so buffered writes are totally ordered and
    /// a flush point that acquires the lock waits for this write to finish.
    /// If an earlier buffered write already failed (the coalescer's
    /// failure latch, checked under the same lock), nothing is written:
    /// the claimed updates are retained as unwritten and the prior error
    /// is published to this batch's contributors (issue #43, "no further
    /// writes" after a checkpoint channel failure).
    ///
    /// The write runs on its own task deliberately: a contributor dropped
    /// mid-await (a lost `race`, a dropped `DurableFuture`) must not cancel
    /// an in-flight batch write that other contributors are waiting on.
    fn spawn_batch_flush(
        &self,
        coalescer: &Arc<CheckpointCoalescer>,
        batch: &Arc<CheckpointBatch>,
        delay: Duration,
    ) {
        let ctx = self.clone();
        let coalescer = Arc::clone(coalescer);
        let publish_to = Arc::clone(batch);
        let batch = Arc::clone(batch);
        // Detached mode of the task gateway (src/task.rs): this internal
        // task runs no handler code and creates no durable operations, and it
        // must survive a cancelled contributor. It is still counted, so the
        // invocation waits for an in-flight write before it returns. The
        // batch's checkpoint permits keep the invocation from suspending
        // until the response is merged and published.
        // A panic of the checkpoint client unwinds the flush, and nothing
        // publishes the batch. The task gateway records the panic on the
        // coordinator, which ends the drive. The contributors stay pending
        // until the drive's cleanup cancels them.
        let flush = async move {
            if !delay.is_zero() {
                // The window ends early when the invocation would suspend
                // but for buffered writes, or when the drive ended. A
                // counted task that slept out the whole window would keep
                // a resolved handler from returning for up to the full
                // `checkpoint_delay`, which has no upper bound.
                ctx.coordinator().coalescing_window(delay).await;
            }
            let _writer = coalescer.writer_lock().lock().await;
            if let Some((updates, permits)) = coalescer.take_batch_unheld(&batch).await {
                // Failure latch (issue #43): once any buffered write has
                // failed, the channel is down for the rest of the
                // invocation: "no further writes". A flusher that queued
                // on the writer lock behind the failing write must not
                // perform another checkpoint call: doing so could persist
                // replay-visible transitions after the invocation is
                // already doomed. Instead it retains its updates for the
                // flush-point classification and publishes the prior
                // error, so its contributors route through the same
                // unrecoverable path the failing batch's contributors
                // took. The latch is set by the failing flusher while it
                // holds this same lock, so the check cannot race it.
                if let Some(prior) = coalescer.latched_failure() {
                    coalescer.record_failed_flush(
                        prior.clone(),
                        updates.into_iter().map(|t| t.update).collect(),
                    );
                    ctx.publish_batch(&batch, Err(prior));
                    drop(permits);
                    return;
                }
                let result = match ctx.write_batched_updates(updates, coalescer.limits()).await {
                    Ok(output) => Ok(output),
                    Err((error, unwritten)) => {
                        // Retain the failure for the end-of-invocation
                        // flush point (issue #43): every contributor of
                        // this batch may already be dropped (a lost
                        // `race`/`select_ok` branch, a dropped
                        // `DurableFuture`), and a failure published to
                        // nobody would otherwise be fully discarded,
                        // leaving the affected operations' records
                        // claiming less than what executed.
                        coalescer.record_failed_flush(error.clone(), unwritten);
                        Err(error)
                    }
                };
                // The merge already woke the operations the response
                // readied. The contributors' wakes take their own permits
                // before the batch's permits are released.
                ctx.publish_batch(&batch, result);
                drop(permits);
            }
        };
        let spawned = crate::task::spawn_detached(self.coordinator(), flush);
        // No Tokio runtime is running on this thread, so nothing can write
        // the batch: every contributor receives the error instead of waiting
        // forever. Nothing was written, so the invocation fails and the
        // service invokes the function again.
        if let Err(no_runtime) = spawned {
            self.publish_batch(
                &publish_to,
                Err(ExecutionClientError::new(
                    ExecutionClientErrorScope::Invocation,
                    no_runtime.to_string(),
                )),
            );
        }
    }

    /// Publishes a spawned flush's `result` to `batch`'s contributors.
    ///
    /// A failure is first recorded for each of the batch's failure owners
    /// (see [`Self::record_checkpoint_failure`]). The publisher runs on the
    /// detached flush task, which the drive's cleanup waits for, so the
    /// classification is in the fatal slot before the driver reads the
    /// outcome, even when every contributor was dropped. The
    /// end-of-invocation flush point publishes without this record,
    /// because it classifies the failures it drains itself.
    fn publish_batch(
        &self,
        batch: &CheckpointBatch,
        result: Result<CheckpointOutput, ExecutionClientError>,
    ) {
        if let Err(err) = &result {
            for owner in batch.take_failure_owners() {
                self.record_checkpoint_failure(&owner, err);
            }
        }
        batch.publish(result);
    }

    /// Writes a sealed batch, splitting it into request-sized chunks that
    /// respect the coalescer's [`BatchLimits`] (operation count and
    /// estimated payload bytes) while preserving join order. Returns the
    /// last chunk's output on success: backend-assigned fields from every
    /// chunk are already merged into the checkpoint log by
    /// [`Self::checkpoint_updates_direct`], or the first chunk error,
    /// which aborts the remaining chunks so contributors observe it. The
    /// error carries the updates the write did NOT persist, the rejected
    /// chunk plus every chunk after it, never a chunk that already
    /// succeeded, so the #43 flush point can write terminal `FAIL`
    /// records for exactly the operations whose outcomes were lost.
    ///
    /// Each chunk's lifecycle events are emitted immediately after that
    /// chunk's write succeeds (see [`Self::write_tracked_direct`]): a chunk
    /// persisted before a later chunk fails still emits its events, even
    /// though the batch as a whole publishes the error, so telemetry stays
    /// faithful to what was actually recorded.
    ///
    /// Callers must hold the coalescer's writer lock (see the invariants in
    /// [`crate::checkpoint_coalescer`]).
    async fn write_batched_updates(
        &self,
        updates: Vec<TrackedUpdate>,
        limits: BatchLimits,
    ) -> Result<CheckpointOutput, (ExecutionClientError, Vec<OperationUpdate>)> {
        if updates.is_empty() {
            // An empty seal (possible only defensively) still performs one
            // call so a waiting contributor receives a published result.
            return self
                .write_unchecked(Vec::new(), &[])
                .await
                .map_err(|e| (e, Vec::new()));
        }
        let mut chunks = split_into_requests(updates, &limits).into_iter();
        let mut last = None;
        while let Some(chunk) = chunks.next() {
            // Snapshot the chunk before the write consumes it: on
            // rejection this chunk is the unwritten head, and the
            // remaining chunks the unwritten tail.
            let snapshot: Vec<OperationUpdate> = chunk.iter().map(|t| t.update.clone()).collect();
            match self.write_tracked_direct(chunk).await {
                Ok(output) => last = Some(output),
                Err(err) => {
                    let mut unwritten = snapshot;
                    unwritten.extend(chunks.flatten().map(|t| t.update));
                    return Err((err, unwritten));
                }
            }
        }
        last.ok_or_else(|| {
            (
                ExecutionClientError::new_non_retryable(
                    "internal: batched checkpoint produced no requests",
                ),
                Vec::new(),
            )
        })
    }

    /// Unconditionally drains the checkpoint coalescing buffer, writing any
    /// pending batches now, and waits for any in-flight batch write to
    /// finish before returning.
    ///
    /// This is the "suspension" and "execution completion" flush point of
    /// the [`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay) /
    /// [`checkpoint_batching`](crate::OptionsBuilder::checkpoint_batching)
    /// contract: the invocation wrapper calls it after the driver settles,
    /// before reporting `PENDING`, `SUCCEEDED`, or `FAILED` to the service,
    /// so a checkpoint that must land before the invocation ends is never
    /// held back by the buffer. Because every claimed batch is written while
    /// holding the coalescer's writer lock, acquiring that lock here makes
    /// this a true barrier: a batch a delay timer already claimed cannot
    /// still be in flight when this returns. A no-op when no buffering is
    /// configured or the buffer is idle.
    ///
    /// A failure is returned as a [`FlushFailure`] carrying every write
    /// failure this invocation's buffered checkpoints suffered, the one
    /// hit here directly, the remaining drained-but-unwritten batches, and
    /// any failure a spawned batch flush retained because its contributors
    /// were all dropped, together with the updates that were not
    /// persisted, so the caller can apply the issue #43 classification
    /// (retryable fails the invocation with no further writes;
    /// non-retryable persists terminal `FAIL` records for the affected
    /// operations, then fails the execution).
    pub(crate) async fn flush_pending_checkpoints(&self) -> Result<(), FlushFailure> {
        let Some(coalescer) = self.inner.coalescer.clone() else {
            return Ok(());
        };
        // Acquiring the writer lock waits out any in-flight batch write
        // (batches are only claimed and written under it), then holding it
        // across the drain keeps this flush ordered after those writes.
        let _writer = coalescer.writer_lock().lock().await;
        // Failure latch (issue #43): a buffered write that already failed
        // (in a spawned flush this drain never contributed to) poisons
        // the channel for the rest of the invocation. Seed the drain's
        // "already failed" state from it, so pending batches are published
        // the prior error and retained as unwritten instead of written.
        let mut prior: Option<ExecutionClientError> = coalescer.latched_failure();
        let mut failures: Vec<crate::checkpoint_coalescer::FailedFlush> = Vec::new();
        while let Some((batch, updates, permits)) = coalescer.take_any_unheld().await {
            if let Some(error) = prior.clone() {
                // The channel already failed: do not attempt further
                // outcome writes ("no further writes" for the retryable
                // case; the non-retryable case writes only the small
                // terminal FAILs, and those go through the caller, not
                // here). Publish the error and collect the updates as
                // unwritten.
                batch.publish(Err(error.clone()));
                failures.push(crate::checkpoint_coalescer::FailedFlush {
                    error,
                    unwritten: updates.into_iter().map(|t| t.update).collect(),
                });
                drop(permits);
                continue;
            }
            match self
                .write_batched_updates(updates, coalescer.limits())
                .await
            {
                Ok(output) => batch.publish(Ok(output)),
                Err((error, unwritten)) => {
                    batch.publish(Err(error.clone()));
                    prior = Some(error.clone());
                    failures.push(crate::checkpoint_coalescer::FailedFlush { error, unwritten });
                }
            }
            drop(permits);
        }
        // Failures a spawned flush retained because its contributors were
        // all dropped happened BEFORE this drain: report them first.
        let mut all = coalescer.take_failed_flushes();
        all.extend(failures);
        if all.is_empty() {
            Ok(())
        } else {
            Err(FlushFailure { failures: all })
        }
    }

    /// Drains ONLY the write failures retained by detached (spawned) batch
    /// flushes, attempting no writes and leaving any still-pending batches
    /// in the buffer untouched.
    ///
    /// This is the `Fault`-outcome counterpart of
    /// [`Self::flush_pending_checkpoints`] (issue #43): when a retryable
    /// checkpoint failure is already failing the invocation, the "no
    /// further writes" contract forbids flushing the buffer: a follow-up
    /// write would fail the same way, and re-invocation re-runs the
    /// buffered operations under the interruption contract. But a failure
    /// a spawned flush retained on behalf of dropped contributors (a lost
    /// `race`/`select_ok` branch, a dropped [`crate::DurableFuture`]) has
    /// already happened and must still be classified: silently discarding
    /// a NON-retryable one would leave its operations' records claiming
    /// less than what executed on every future invocation: the very loop
    /// the #43 terminalization exists to end.
    ///
    /// Acquiring the coalescer's writer lock first waits out any batch
    /// write still in flight (batches are only claimed and written under
    /// it), so a failure that write is about to retain is observed rather
    /// than raced past. Waiting is not writing: nothing is sent here.
    pub(crate) async fn take_retained_flush_failures(&self) -> Option<FlushFailure> {
        let coalescer = self.inner.coalescer.clone()?;
        let _writer = coalescer.writer_lock().lock().await;
        let failures = coalescer.take_failed_flushes();
        if failures.is_empty() {
            None
        } else {
            Some(FlushFailure { failures })
        }
    }

    /// Writes a small terminal `FAIL` record for every operation whose
    /// buffered OUTCOME write was lost to a non-retryable flush failure
    /// (issue #43). Called by the invocation wrapper before it fails the
    /// execution, so each affected operation's record claims exactly what
    /// executed instead of a dangling `Started`.
    ///
    /// Per operation (first-appearance order across the failures):
    /// - An operation with an unwritten outcome update (`Succeed`, `Fail`,
    ///   or `Retry`) gets a terminal `FAIL` derived from the flush failure:
    ///   user code ran, so its side effects need a recorded outcome. An
    ///   unwritten `Start` for the same operation is written first so the
    ///   `FAIL` has a record to terminate.
    /// - An operation whose only unwritten update is its `Start` gets
    ///   nothing: no user-visible outcome was discarded, and the execution
    ///   is failing anyway.
    /// - An operation the checkpoint log already shows terminal is
    ///   skipped: a live contributor's unrecoverable routing may already
    ///   have persisted its terminal `FAIL`, and a `FAIL` must never be
    ///   written over a recorded outcome.
    ///
    /// Writes go one operation per request so one rejected record cannot
    /// take its siblings' terminal `FAIL`s down with it; individual
    /// failures are logged, not propagated: the execution dies either way.
    pub(crate) async fn terminalize_unwritten_outcomes(&self, flush: &FlushFailure) {
        use aws_sdk_lambda::types::OperationAction;

        struct PerOp {
            start: Option<OperationUpdate>,
            outcome: Option<OperationUpdate>,
            error: ExecutionClientError,
        }
        let mut order: Vec<String> = Vec::new();
        let mut per_op: std::collections::HashMap<String, PerOp> = std::collections::HashMap::new();
        for failure in &flush.failures {
            for update in &failure.unwritten {
                let entry = per_op.entry(update.id.clone()).or_insert_with(|| {
                    order.push(update.id.clone());
                    PerOp {
                        start: None,
                        outcome: None,
                        error: failure.error.clone(),
                    }
                });
                match update.action {
                    OperationAction::Start => {
                        entry.start.get_or_insert_with(|| update.clone());
                    }
                    _ => {
                        entry.outcome.get_or_insert_with(|| update.clone());
                    }
                }
            }
        }

        for wire_id in order {
            let Some(op) = per_op.remove(&wire_id) else {
                continue;
            };
            let Some(outcome) = op.outcome else {
                // Start-only: no outcome was discarded (see doc above).
                continue;
            };
            if self
                .inner
                .engine
                .checkpoint_log
                .status_view(&wire_id)
                .is_some_and(|view| view.status.is_terminal())
            {
                continue;
            }
            let wire = crate::error::checkpoint_failure_wire(&op.error);
            let mut updates = Vec::new();
            if let Some(start) = op.start {
                updates.push(start);
            }
            let mut builder = OperationUpdate::builder()
                .id(outcome.id.clone())
                .r#type(outcome.r#type.clone())
                .action(OperationAction::Fail)
                .error(wire.to_error_object());
            if let Some(sub_type) = &outcome.sub_type {
                builder = builder.sub_type(sub_type.clone());
            }
            if let Some(name) = &outcome.name {
                builder = builder.name(name.clone());
            }
            if let Some(parent_id) = &outcome.parent_id {
                builder = builder.parent_id(parent_id.clone());
            }
            match builder.build() {
                Ok(fail_update) => {
                    updates.push(fail_update);
                    if let Err(err) = self.checkpoint_updates_after_drive(updates).await {
                        tracing::error!(
                            operation_id = %wire_id,
                            error = %err,
                            "terminal FAIL write failed after non-retryable flush \
                             failure; the execution fails without a recorded outcome \
                             for this operation"
                        );
                    }
                }
                Err(err) => {
                    tracing::error!(
                        operation_id = %wire_id,
                        error = %err,
                        "could not build terminal FAIL update after non-retryable \
                         flush failure"
                    );
                }
            }
        }
    }

    /// Writes operation updates through the execution client immediately.
    ///
    /// Serializes all concurrent callers through a single critical section:
    /// the lock is held across the full read-token → API-call →
    /// rotate-token sequence, and, when the response carries a pagination
    /// marker, through the follow-up `get_state` fetch as well, so no
    /// concurrent branch can rotate the token out from under the paginated
    /// state read.
    ///
    /// The write is admitted (see [`Admission`]) and runs on a detached
    /// task (see [`Self::write_direct_detached`]).
    pub(crate) async fn checkpoint_updates_direct(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        let Some(permit) = self.inner.coordinator.acquire(PermitKind::Checkpoint) else {
            // The phase left `Running`: the write is rejected, as a write
            // from an abandoned owner is.
            return std::future::pending().await;
        };
        let result = self
            .write_direct_detached(updates, Vec::new(), Some(permit), None)
            .await;
        // The detached write task merged the response and dropped the
        // token guard, so the caller's task delivers the change sets.
        self.report_operation_changes().await;
        result
    }

    /// A direct write after the drive ended, when no permit can be taken:
    /// the terminal `FAIL` records of the end-of-invocation flush point.
    async fn checkpoint_updates_after_drive(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        self.write_direct_detached(updates, Vec::new(), None, None)
            .await
    }

    /// Body of [`Self::checkpoint_updates_direct`] that additionally owns
    /// the lifecycle events paired with the updates. Each event is emitted
    /// the moment `client.checkpoint` returns success, the point at which
    /// the service has durably recorded the transitions, and **before**
    /// the pagination hydration below, which can fail after the write
    /// already persisted. Emitting first keeps telemetry faithful to what
    /// was recorded: a rejected write emits nothing, and a persisted write
    /// emits everything even when the follow-up `get_state` fetch fails.
    ///
    /// With `Some(owners)`, the write is admitted while this call holds the
    /// checkpoint-token lock. A rejected write sends nothing, emits no
    /// event, does not rotate the token, and returns `None`. `None` skips
    /// the check: the batch flusher passes it, because its updates were
    /// admitted when they joined the buffer.
    async fn checkpoint_direct_with_events(
        &self,
        updates: Vec<OperationUpdate>,
        events: &[crate::tracing_layer::PendingTransitionEvent],
        owners: Option<&[String]>,
    ) -> Option<Result<CheckpointOutput, ExecutionClientError>> {
        let Some(client) = self.inner.execution_client.as_ref() else {
            return Some(Err(ExecutionClientError::new_non_retryable(
                "no execution client configured",
            )));
        };

        // Hold the async mutex across the entire checkpoint call to
        // serialize concurrent branch checkpoints. This is the Rust
        // equivalent of Go's sync.Mutex held across the blocking API call
        // in checkpointer.checkpoint().
        let token_guard = self.inner.checkpoint_token.lock().await;
        if owners.is_some_and(|owners| !self.admission().admits(owners)) {
            return None;
        }
        Some(
            self.write_locked(client, token_guard, updates, events)
                .await,
        )
    }

    /// The write of [`Self::checkpoint_direct_with_events`], with the
    /// checkpoint-token lock held.
    async fn write_locked(
        &self,
        client: &Arc<dyn ExecutionClient>,
        mut token_guard: tokio::sync::MutexGuard<'_, String>,
        updates: Vec<OperationUpdate>,
        events: &[crate::tracing_layer::PendingTransitionEvent],
    ) -> Result<CheckpointOutput, ExecutionClientError> {
        let retried = wfc_retries(&updates);
        let output = client
            .checkpoint(&self.inner.execution_arn, &token_guard, updates)
            .await?;
        // A transport reports a tokenless response as `None`. Until the
        // SDK gives that response its own meaning, it is the non-retryable
        // error the Lambda transport returned before the token check moved
        // here, so the behavior does not change.
        let Some(next_token) = output.checkpoint_token() else {
            return Err(ExecutionClientError::new_non_retryable(
                "backend returned no checkpoint token",
            ));
        };
        let next_token = next_token.to_owned();

        // The service has recorded the transitions: emit their lifecycle
        // events now, before the fallible pagination hydration below can
        // turn an already-persisted write into a caller-visible error.
        for event in events {
            event.emit(self.execution_arn(), &self.lambda_context().request_id);
        }

        // Rotate the token while still holding the lock.
        *token_guard = next_token;

        // Merge updated operations into the checkpoint log so that
        // subsequent reads (e.g. reading callback_id after START) see
        // backend-assigned fields.
        let mut merged: Vec<String> = Vec::new();
        // The change set of this call, for `on_operation_change`: the wire
        // IDs whose status changed, each once, and the `EXECUTION`
        // operations that changed the plugins' slot. Only collected with
        // plugins.
        let mut change_set = ChangeSet::default();
        if !output.updated_operations.is_empty() {
            let changed_ids = crate::client::merge_operations_into_log(
                &self.inner.engine.checkpoint_log,
                &output.updated_operations,
            );
            self.collect_changes(&mut change_set, changed_ids, &output.updated_operations);
            merged.extend(
                output
                    .updated_operations
                    .iter()
                    .map(|op| op.id().to_owned()),
            );
        }

        // If the checkpoint response is paginated, fetch remaining pages
        // via get_state and merge them into the checkpoint log. The token
        // lock stays held through this fetch: releasing it first would let
        // a concurrent branch checkpoint and rotate the token, leaving this
        // get_state call with a stale token.
        if output.next_marker.is_some() {
            let full_state = match crate::client::get_full_state(
                client.as_ref(),
                &self.inner.execution_arn,
                &token_guard,
            )
            .await
            {
                Ok(full_state) => full_state,
                Err(err) => {
                    // The response is merged already, so its changes are
                    // reported even though the pages failed.
                    self.queue_changes(change_set);
                    return Err(err);
                }
            };
            if !full_state.is_empty() {
                let changed_ids = crate::client::merge_operations_into_log(
                    &self.inner.engine.checkpoint_log,
                    &full_state,
                );
                self.collect_changes(&mut change_set, changed_ids, &full_state);
                merged.extend(full_state.iter().map(|op| op.id().to_owned()));
            }
        }
        // Still under the token lock, so change sets queue in response
        // order. No hook runs here: a drain point delivers them later.
        self.queue_changes(change_set);

        // A `wait_for_condition` RETRY the response did not carry: the log
        // takes the status the service recorded. This runs under the token
        // lock, so a later response, which carries a newer status, merges
        // after it.
        for (wire_id, payload) in &retried {
            if !merged.iter().any(|id| id == wire_id) {
                mark_retry_pending(
                    &self.inner.engine.checkpoint_log,
                    wire_id,
                    payload.as_deref(),
                );
            }
        }

        // The token is rotated and the response merged: release the lock,
        // then wake every operation the response carries. The write's
        // checkpoint permit is still held, so each woken operation holds
        // its own permit before this write's permit is released.
        drop(token_guard);
        self.inner.coordinator.wake_operations(&merged);

        Ok(output)
    }

    /// Adds the changes of one merge to `change_set`: the wire IDs in
    /// `changed_ids` not listed yet, and each `EXECUTION` operation of
    /// `operations`, which replaces the plugins' slot and is a change when
    /// the slot held no operation or one with another status. Does nothing
    /// without plugins.
    fn collect_changes(
        &self,
        change_set: &mut ChangeSet,
        changed_ids: Vec<String>,
        operations: &[aws_sdk_lambda::types::Operation],
    ) {
        let Some(plugins) = self.plugins() else {
            return;
        };
        for id in changed_ids {
            if !change_set.ids.contains(&id) {
                change_set.ids.push(id);
            }
        }
        for info in operations
            .iter()
            .filter_map(crate::client::execution_info_from_operation)
        {
            if plugins.replace_execution(info.clone()) {
                change_set.executions.retain(|held| held.id() != info.id());
                change_set.executions.push(info);
            }
        }
    }

    /// Queues `change_set` for `on_operation_change` when it lists anything.
    /// Each listed operation's info is built from its borrowed record, so
    /// no record is cloned and the full log is not projected.
    fn queue_changes(&self, change_set: ChangeSet) {
        let Some(plugins) = self.plugins() else {
            return;
        };
        let log = &self.inner.engine.checkpoint_log;
        let mut updated: Vec<crate::plugin::OperationInfo> = change_set
            .ids
            .iter()
            .filter_map(|id| {
                log.with_record(id, |record| {
                    crate::plugin::OperationInfo::from_record(id, record)
                })
            })
            .collect();
        updated.extend(change_set.executions);
        if !updated.is_empty() {
            plugins.push_operation_change(updated);
        }
    }

    /// Returns the parent wire ID if this is a child context.
    pub(crate) fn parent_wire_id(&self) -> Option<&str> {
        self.inner.parent_wire_id.as_deref()
    }

    /// Returns the attempt number for the given operation from the checkpoint
    /// log (0 if no prior attempt recorded).
    pub(crate) fn get_attempt(&self, op_id: &OperationId) -> u32 {
        // On re-invocation after a RETRY, the backend returns the operation
        // with step_details.attempt set to the last completed attempt number.
        self.checkpoint_status_view(op_id.positional())
            .map_or(0, |view| view.attempt)
    }

    /// Returns the execution ARN identifying this durable execution.
    ///
    /// The ARN uniquely identifies the execution across invocations and is
    /// stable for the lifetime of the orchestration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     tracing::info!(arn = ctx.execution_arn(), "starting execution");
    ///     Ok(())
    /// }
    /// ```
    pub fn execution_arn(&self) -> &str {
        &self.inner.execution_arn
    }

    /// Returns a reference to the Lambda invocation context.
    ///
    /// Provides access to the request ID, deadline, function ARN, and other
    /// Lambda metadata for the current invocation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let request_id = &ctx.lambda_context().request_id;
    ///     tracing::info!(%request_id, "invoked");
    ///     Ok(())
    /// }
    /// ```
    pub fn lambda_context(&self) -> &lambda_runtime::Context {
        &self.inner.lambda_context
    }

    /// Returns whether the current invocation is in replay mode.
    ///
    /// During replay, previously-checkpointed operation results are returned
    /// without re-execution. Replay mode lasts while the next operation to
    /// be claimed was already claimed by a prior invocation, including a
    /// started-but-unfinished child context, map, or parallel parent that
    /// the resumed invocation re-enters to replay its nested operations.
    /// User code can use this flag to suppress duplicate side effects
    /// (e.g., logging) during replay.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     if !ctx.is_replaying() {
    ///         tracing::info!("first execution, not replay");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub fn is_replaying(&self) -> bool {
        self.inner.engine.is_replaying()
    }

    /// Creates a durable step operation.
    ///
    /// The step body receives a [`StepContext`] and returns a result. The
    /// result is checkpointed on success; on failure the retry strategy
    /// determines whether to retry.
    ///
    /// The step body is `FnOnce + Clone`. The SDK keeps the original closure
    /// and calls a fresh clone of it for each attempt. Each capture follows
    /// its type's `Clone` implementation: a cloned `Vec` or `String` has its
    /// own contents, and a cloned `Arc` shares its allocation.
    ///
    /// The operation ID is claimed synchronously at this call site.
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
    ///     let result = ctx.step(|_step_ctx| async {
    ///         Ok("computed value".to_owned())
    ///     }).name("compute")
    ///       .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn step<O, F, Fut>(&self, f: F) -> StepBuilder<O, F, Fut>
    where
        F: FnOnce(StepContext) -> Fut + Clone + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("step");
        // The closure is stored unerased; the single erasure point is the
        // builder's `.future()` / `.await`, producing one DurableFuture box.
        StepBuilder::new_internal(self.clone(), op_id, f)
    }

    /// Creates a durable wait (timer) operation.
    ///
    /// The execution pauses for the specified duration. The wait is
    /// checkpointed, so replay does not re-wait.
    ///
    /// The duration is rounded up to whole seconds, so the wait never ends
    /// early. A duration under one second, including [`Duration::ZERO`],
    /// waits one second, because the service rejects a zero-second wait.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     ctx.wait(Duration::from_secs(60))
    ///         .name("cooldown")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn wait(&self, duration: Duration) -> WaitBuilder {
        // Round up to whole seconds. The service rejects `WaitSeconds: 0`,
        // so the clamp raises a sub-second duration to one second.
        #[expect(clippy::cast_possible_truncation)]
        // reason: the clamp bounds the value to `1..=i32::MAX` before the cast to i32
        let secs = (duration.as_secs_f64().ceil() as i64).clamp(1, i64::from(i32::MAX)) as i32;
        let op_id = self.claim_operation_id("wait");
        WaitBuilder::new_internal(self.clone(), op_id, secs)
    }

    /// Creates a durable invoke operation to call another durable function.
    ///
    /// The output type parameter `O` comes first so callers can turbofish:
    /// `ctx.invoke::<Receipt, _>(...)`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct ChargeInput { amount: u64 }
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct Receipt { id: String }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let receipt = ctx.invoke::<Receipt, _>("payment-fn", ChargeInput { amount: 100 })
    ///         .name("charge")
    ///         .await?;
    ///     Ok(receipt.id)
    /// }
    /// ```
    pub fn invoke<O, I>(&self, function_id: impl Into<String>, input: I) -> InvokeBuilder<O, I>
    where
        I: Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("invoke");
        // The input is carried TYPED into the builder: the payload serdes
        // receives the owned value directly at execution time (a write-only
        // transfer), so no intermediate representation is constructed here.
        InvokeBuilder::new_internal(self.clone(), op_id, function_id.into(), input)
    }

    /// Creates a child context for fan-out / sub-orchestration.
    ///
    /// The child closure receives its own [`DurableContext`] with an
    /// independent operation-ID namespace. Use `.spawn()` for eager
    /// (background) execution.
    ///
    /// The body should run at least one durable operation. The child's
    /// result is saved, so a replay returns it without running the body,
    /// with three exceptions. A body that returns an error is not retried,
    /// as a step is. A result over 256 KiB makes the body run again on
    /// every replay, to rebuild the result; its record holds the payload of
    /// [`ChildBuilder::summary`] instead. A context that
    /// [`plugin_child_operations_depth`](crate::OptionsBuilder::plugin_child_operations_depth)
    /// preserves also runs its body again on every replay. So put plain
    /// computation in a step.
    ///
    /// To change the error a child returns, call
    /// [`map_err`](Result::map_err) on the awaited result.
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
    ///     let result = ctx.run_in_child_context(|child_ctx| async move {
    ///         let v = child_ctx.step(|_| async { Ok(42) }).await?;
    ///         Ok(v.to_string())
    ///     }).name("branch")
    ///       .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn run_in_child_context<O, F, Fut>(&self, f: F) -> ChildBuilder<O, F, Fut>
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("run_in_child_context");
        // The closure is stored unerased; the child execution converts the
        // BoxError into the internal child-error carrier at the boundary,
        // and the single erasure point is the builder's `.future()` /
        // `.await`, producing one DurableFuture box.
        ChildBuilder::new_internal(self.clone(), op_id, f)
    }

    /// Runs a closure against a child context and retries the closure's
    /// **overall** outcome as a unit.
    ///
    /// Where a step's retry strategy re-runs one operation, `with_retry`
    /// re-runs a whole block: if any operation in the block fails (or the
    /// closure returns an error), the retry strategy decides whether to run
    /// the entire block again. Each attempt receives a **fresh child
    /// operation namespace**, so operations recorded by a failed attempt
    /// are never replayed into the next one: every operation in the block
    /// re-runs on retry. The delay between attempts suspends the execution
    /// (the backend owns the timer, exactly as step retries do), and the
    /// retry progress is derived from checkpointed results, so it survives
    /// suspension and replays deterministically.
    ///
    /// The closure is `Fn` rather than `FnOnce` because the SDK calls it
    /// once per attempt. It receives the attempt's context and the 1-based
    /// attempt number, which is `N` in the attempt's name `attempt-N`. The
    /// number is metadata for names and logs: a resumed attempt receives
    /// the same number, and operation IDs do not depend on it. Configure
    /// the policy with
    /// [`retry_strategy`](WithRetryBuilder::retry_strategy) or
    /// [`retry_strategy_config`](WithRetryBuilder::retry_strategy_config);
    /// without one, the step default applies (6 total attempts with
    /// exponential backoff). When retries exhaust, a block with its own
    /// record fails with a [`ChildContextError`](crate::ChildContextError)
    /// of kind `ChildFailed` whose source is the last attempt's
    /// [`OperationError`](crate::OperationError).
    /// [`without_child_context`](WithRetryBuilder::without_child_context)
    /// runs the attempts in this context, without a record for the block;
    /// on exhaustion such a block returns the last attempt's
    /// `OperationError` unchanged, a `ChildContextError` of kind
    /// `ChildFailed` whose source is the attempt body's error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use aws_durable_execution_sdk::RetryDecision;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     // If either step fails, BOTH re-run on the next attempt.
    ///     let result = ctx.with_retry(|child, attempt| async move {
    ///         tracing::info!(attempt, "reserving and booking");
    ///         let quote = child
    ///             .step(|_| async { Ok("quote-17".to_owned()) })
    ///             .name("reserve-quote")
    ///             .await?;
    ///         let receipt = child
    ///             .step(move |_| async move { Ok(format!("booked:{quote}")) })
    ///             .name("book")
    ///             .await?;
    ///         Ok(receipt)
    ///     })
    ///     .name("reserve-and-book")
    ///     .retry_strategy(|_err, attempt| {
    ///         if attempt >= 3 {
    ///             RetryDecision::Stop
    ///         } else {
    ///             RetryDecision::Retry { delay: Duration::from_secs(5) }
    ///         }
    ///     })
    ///     .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn with_retry<O, F, Fut>(&self, f: F) -> WithRetryBuilder<O, F, Fut>
    where
        F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        // The block claims its operation ID when the builder becomes its
        // future, because only then is its mode known.
        WithRetryBuilder::new_internal(self.clone(), f)
    }

    /// Creates a wait-for-condition operation that polls until a predicate
    /// is satisfied.
    ///
    /// The check function is called repeatedly with the current state, and
    /// the configured strategy decides after each check whether to
    /// complete, keep polling, or fail. Set one with
    /// [`wait_strategy`](crate::builders::WaitForConditionBuilder::wait_strategy)
    /// (a bounded [`WaitStrategy`](crate::builders::wait_for_condition::WaitStrategy)
    /// configuration) or
    /// [`wait_strategy_fn`](crate::builders::WaitForConditionBuilder::wait_strategy_fn)
    /// (a custom closure). With no strategy set, the check runs exactly
    /// once and the operation completes with that state.
    ///
    /// The first check receives the initial state after a round trip
    /// through the configured serdes, so it sees the same shape as a check
    /// that resumes from a recorded state. The round trip runs before the
    /// operation's first checkpoint: a serdes error marked with
    /// [`RetryableSerdesError`](crate::serdes::RetryableSerdesError) ends
    /// the invocation, and any other serdes error returns
    /// `WaitForCondition(SerializationFailed)` without calling the check.
    ///
    /// A check error fails the operation without a retry. It records the
    /// type of a [`TypedError`](crate::TypedError) in its source chain, or
    /// `Error` when there is none. The error kind follows the recorded
    /// type, live and on replay: `WaitForCondition.MaxChecksExceeded` is
    /// reported as `MaxChecksExceeded`, `SerializationError` as
    /// `SerializationFailed`, and every other type as `CheckFailed`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Serialize, Deserialize)]
    /// struct PollState { count: u32 }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     ctx.wait_for_condition(
    ///         |_step_ctx, state: PollState| async move {
    ///             Ok(PollState { count: state.count + 1 })
    ///         },
    ///         PollState { count: 0 },
    ///     ).name("poll-ready")
    ///      .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn wait_for_condition<S, F, Fut>(
        &self,
        check: F,
        initial_state: S,
    ) -> WaitForConditionBuilder<S, F, Fut>
    where
        F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
        S: Clone + Send + Sync + 'static,
    {
        let op_id = self.claim_operation_id("wait_for_condition");
        WaitForConditionBuilder::new_internal(self.clone(), op_id, initial_state, check)
    }

    /// Creates a callback token for external completion.
    ///
    /// The returned [`Callback`](crate::builders::callback::Callback) provides an ID that
    /// external systems use to complete the operation, plus a
    /// [`DurableFuture`] that resolves when the callback arrives.
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
    ///         .name("approval")
    ///         .await?;
    ///     // Send cb.id() to an external system...
    ///     let approval = cb.result().await?;
    ///     Ok(approval.approved)
    /// }
    /// ```
    pub fn create_callback<O>(&self) -> CreateCallbackBuilder<O>
    where
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("create_callback");
        CreateCallbackBuilder::new_internal(self.clone(), op_id)
    }

    /// Creates a wait-for-callback operation that registers and waits for
    /// an external callback in one step.
    ///
    /// The submitter closure receives the callback ID as an owned
    /// [`String`] and is responsible for delivering it to the external
    /// system that will complete it.
    ///
    /// The submitter is `FnOnce + Clone`. The SDK keeps the original closure
    /// and calls a fresh clone of it for each attempt. Each capture follows
    /// its type's `Clone` implementation: a cloned `Vec` or `String` has its
    /// own contents, and a cloned `Arc` shares its allocation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Deserialize, Serialize)]
    /// struct Approval { ok: bool }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<bool, durable::BoxError> {
    ///     let approval = ctx.wait_for_callback::<Approval, _, _>(
    ///         |_step_ctx, callback_id| async move {
    ///             // e.g., send callback_id to an approval queue
    ///             Ok(())
    ///         }
    ///     ).name("await-approval")
    ///      .await?;
    ///     Ok(approval.ok)
    /// }
    /// ```
    pub fn wait_for_callback<O, F, Fut>(&self, submitter: F) -> WaitForCallbackBuilder<O, F, Fut>
    where
        F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
        Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("wait_for_callback");
        WaitForCallbackBuilder::new_internal(self.clone(), op_id, submitter)
    }

    /// Creates a map operation that applies a function to each item in a
    /// collection, with configurable concurrency.
    ///
    /// Each item gets its own child context with an independent operation
    /// namespace.
    ///
    /// The closure should run at least one durable operation. Each item's
    /// result is saved, so a replay returns it without running the
    /// closure, with three exceptions. An item that returns an error is not
    /// retried, as a step is. An item whose result is over 256 KiB runs
    /// again on every replay, to rebuild the result. An item context that
    /// [`plugin_child_operations_depth`](crate::OptionsBuilder::plugin_child_operations_depth)
    /// preserves also runs again on every replay. So put plain computation
    /// in a step.
    ///
    /// When the whole batch result is over 256 KiB, the map records a
    /// summary of its outcome instead. With
    /// [`NestingMode::Normal`](crate::builders::map_parallel::NestingMode::Normal) replay reads each
    /// item from its own record. A
    /// [`NestingMode::Flat`](crate::builders::map_parallel::NestingMode::Flat) item has no record of
    /// its own, so replay runs the items again to rebuild the result, even
    /// when each item's result is small. [`MapBuilder::summary`] adds the
    /// caller's own summary to that record.
    ///
    /// The map claims one operation ID from this context when this method
    /// is called. Its items are numbered inside the map: item `i` has the
    /// ID of item `i` in the map's own namespace, whatever the item count,
    /// the concurrency or the order in which items finish. So this
    /// context's next operation gets this context's next ID.
    ///
    /// A panic in an item propagates to the caller with its original
    /// payload, and the SDK records nothing for it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Serialize, Deserialize)]
    /// struct Image { url: String }
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct Thumbnail { url: String }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let images = vec![Image { url: "a.png".into() }, Image { url: "b.png".into() }];
    ///     let _thumbnails = ctx.map(images, |child_ctx, _img, _idx| async move {
    ///         let thumb = child_ctx
    ///             .step(|_| async { Ok(Thumbnail { url: "thumb.png".into() }) })
    ///             .await?;
    ///         Ok(thumb)
    ///     }).name("resize")
    ///       .max_concurrency(4)
    ///       .await?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// The closure gets one item. JS passes the whole input array as a
    /// fourth argument. To read other inputs, clone an `Arc<[I]>` of the
    /// inputs into the closure:
    ///
    /// ```no_run
    /// use std::sync::Arc;
    ///
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     event: Vec<u32>,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<Vec<u32>, durable::BoxError> {
    ///     let all: Arc<[u32]> = event.clone().into();
    ///     let shares = ctx
    ///         .map(event, move |child, item, _idx| {
    ///             let all = Arc::clone(&all);
    ///             async move {
    ///                 let share = child
    ///                     .step(move |_| async move {
    ///                         let total: u32 = all.iter().sum();
    ///                         Ok(item * 100 / total.max(1))
    ///                     })
    ///                     .name("share")
    ///                     .await?;
    ///                 Ok(share)
    ///             }
    ///         })
    ///         .name("shares")
    ///         .await?;
    ///     Ok(shares)
    /// }
    /// ```
    pub fn map<Items, I, O, F, Fut>(&self, items: Items, f: F) -> MapBuilder<I, O, F, Fut>
    where
        Items: IntoIterator<Item = I>,
        F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        I: Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("map");
        // The closure is stored unerased; at execution it is shared as
        // `Arc<F>` and every item produces a concrete future, so the
        // internal JoinSet holds unboxed futures. The single erasure point
        // is the builder's `.future()` / `.await`, producing one
        // DurableFuture box.
        let items: Vec<I> = items.into_iter().collect();
        MapBuilder::new_internal(self.clone(), op_id, items, f)
    }

    /// Creates a parallel operation that executes named branches
    /// concurrently.
    ///
    /// The parallel claims one operation ID from this context when this
    /// method is called. Its branches are numbered inside the parallel:
    /// branch `i` has the ID of item `i` in the parallel's own namespace,
    /// as a [`map`](Self::map) item does.
    ///
    /// A panic in a branch propagates to the caller with its original
    /// payload, and the SDK records nothing for it.
    ///
    /// Each branch should run at least one durable operation. Each branch's
    /// result is saved, so a replay returns it without running the branch,
    /// with three exceptions. A branch that returns an error is not
    /// retried, as a step is. A branch whose result is over 256 KiB runs
    /// again on every replay, to rebuild the result. A branch context that
    /// [`plugin_child_operations_depth`](crate::OptionsBuilder::plugin_child_operations_depth)
    /// preserves also runs again on every replay. So put plain computation
    /// in a step.
    ///
    /// When the whole batch result is over 256 KiB, the parallel records a
    /// summary of its outcome instead. With
    /// [`NestingMode::Normal`](crate::builders::map_parallel::NestingMode::Normal) replay reads each
    /// branch from its own record. A
    /// [`NestingMode::Flat`](crate::builders::map_parallel::NestingMode::Flat) branch has no record of
    /// its own, so replay runs the branches again to rebuild the result, even
    /// when each branch's result is small. [`ParallelBuilder::summary`] adds the
    /// caller's own summary to that record.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let branches = vec![
    ///         durable::Branch::new("a", |child_ctx: durable::DurableContext| async move {
    ///             let v = child_ctx.step(|_| async { Ok(1) }).await?;
    ///             Ok(v)
    ///         }),
    ///     ];
    ///     let _results: Vec<i32> = ctx.parallel(branches)
    ///         .name("fan-out")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn parallel<Branches, O>(&self, branches: Branches) -> ParallelBuilder<O>
    where
        Branches: IntoIterator<Item = Branch<O>>,
        O: Send + 'static,
    {
        let op_id = self.claim_operation_id("parallel");
        let branch_tuples: Vec<_> = branches.into_iter().map(Branch::into_parts).collect();
        ParallelBuilder::new_internal(self.clone(), op_id, branch_tuples)
    }

    /// Creates a parallel operation over branches whose results have
    /// different types.
    ///
    /// Each branch of `branches` returns its own type, and the
    /// [`BranchHandle`](crate::BranchHandle) that
    /// [`BranchSet::add`](crate::BranchSet::add) or
    /// [`BranchSet::add_with_serdes`](crate::BranchSet::add_with_serdes)
    /// returned gives that type back. The operation records the same
    /// history as [`parallel`](Self::parallel) over branches with the same
    /// names and outcomes: one parallel parent and one child context per
    /// branch. Only the encoded branch values differ, because each branch
    /// encodes its value with its own serdes.
    ///
    /// `.await` returns the [`CompletionReason`](crate::builders::map_parallel::CompletionReason)
    /// of the batch. It returns an error in exactly the cases where
    /// `parallel(...).await` does, with the same error. When it returns,
    /// each handle whose branch settled holds the branch's value or error.
    /// A handle whose branch never started, or still ran when the batch
    /// completed, holds no result. This holds whatever the parallel's own
    /// outcome, including an error from a completion rule that fails the
    /// batch. When the batch does not complete, for example because a
    /// checkpoint write fails, no handle holds a result. With
    /// [`spawn`](crate::builders::ParallelMixedBuilder::spawn), the handles
    /// are filled when the returned future is awaited.
    ///
    /// The operation ID is claimed when this method is called, as
    /// [`parallel`](Self::parallel) claims it, and branch `i` has the ID of
    /// item `i` in the batch's own namespace, as in `parallel`.
    ///
    /// # Which form to use
    ///
    /// - [`parallel`](Self::parallel), when every branch returns the same
    ///   type.
    /// - [`parallel`](Self::parallel) over an enum, when the results are
    ///   cases of one type that the handler processes together. Use serde's
    ///   default (externally tagged) enum representation.
    ///   `#[serde(untagged)]` can decode a result into the wrong variant
    ///   when two variants accept the same JSON.
    /// - `parallel_mixed`, when each branch returns its own type.
    ///
    /// A panic in a branch propagates to the caller with its original
    /// payload, the SDK records nothing for it, and no handle receives a
    /// result.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::SerdesContext;
    /// use durable::{BoxError, BranchSet, Serdes};
    ///
    /// #[derive(serde::Serialize, serde::Deserialize)]
    /// struct Reservation {
    ///     sku: String,
    /// }
    ///
    /// struct PaymentAuth {
    ///     code: u32,
    /// }
    ///
    /// // A serdes for `PaymentAuth` only.
    /// struct PaymentSerdes;
    ///
    /// impl Serdes<PaymentAuth> for PaymentSerdes {
    ///     async fn serialize(&self, value: PaymentAuth, _ctx: SerdesContext) -> Result<String, BoxError> {
    ///         Ok(value.code.to_string())
    ///     }
    ///     async fn deserialize(&self, wire: String, _ctx: SerdesContext) -> Result<PaymentAuth, BoxError> {
    ///         Ok(PaymentAuth { code: wire.parse()? })
    ///     }
    /// }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, BoxError> {
    ///     let mut branches = BranchSet::new();
    ///     let inventory = branches.add("inventory", |_ctx| async {
    ///         Ok(Reservation { sku: "sku-1".to_owned() })
    ///     });
    ///     let payment = branches.add_with_serdes("payment", PaymentSerdes, |_ctx| async {
    ///         Ok(PaymentAuth { code: 7 })
    ///     });
    ///
    ///     ctx.parallel_mixed(branches)
    ///         .name("process-order")
    ///         .max_concurrency(2)
    ///         .await?;
    ///
    ///     let inventory: Reservation = inventory.into_result()?;
    ///     let payment: PaymentAuth = payment.into_result()?;
    ///     Ok(format!("{}: {}", payment.code, inventory.sku))
    /// }
    /// ```
    pub fn parallel_mixed(&self, branches: crate::BranchSet) -> ParallelMixedBuilder {
        let op_id = self.claim_operation_id("parallel_mixed");
        ParallelMixedBuilder::new_internal(self.clone(), op_id, branches)
    }

    /// Joins all futures, failing fast on the first error.
    ///
    /// Returns `Vec<O>` on success, or the first `OperationError`
    /// encountered.
    ///
    /// The returned value has passed through the combinator's JSON
    /// encoding: the SDK decodes the payload it recorded and returns that.
    /// So the first run returns the value a replay returns, and data that
    /// the value's serde form omits, such as a `#[serde(skip)]` field, is
    /// absent on both paths.
    ///
    /// When the combinator resolves, it cancels every input that is still
    /// running. A cancelled input stops at its next `.await`.
    ///
    /// # Oversized results
    ///
    /// A combined result larger than 256 KiB is not stored in the
    /// combinator's record. Replay then drives the inputs again. An input
    /// whose own terminal record is inline replays without running again. An
    /// input whose own terminal record has `ReplayChildren` runs its body
    /// again to rebuild its value, and its recorded descendants replay.
    ///
    /// # Empty input
    ///
    /// Called with no futures, resolves to `Ok` with an empty `Vec`,
    /// matching `futures::future::try_join_all`.
    ///
    /// A panic in an input propagates to the caller with its original
    /// payload, and the SDK records nothing for it.
    ///
    /// # Operation ID
    ///
    /// This method collects `futures` and claims no operation ID. An
    /// operation that the iterator creates claims its ID during this
    /// collection, so the inputs take their IDs before the combinator. The
    /// combinator claims its own ID when it is first awaited or polled, or
    /// when it is spawned, on this context. A combinator that is dropped
    /// before that claims no ID and records nothing. An input that is itself
    /// a combinator must already have claimed its ID: start it with
    /// `.spawn()` and pass the returned handle. A still-unclaimed combinator
    /// input fails the execution with a context usage error. Starting the
    /// combinator inside the body of another context (a child context, a
    /// `with_retry` attempt, a map item or a parallel branch) fails the
    /// execution with a context usage error too.
    ///
    /// # Start order
    ///
    /// When the combinator runs, it begins the start phase of each
    /// `.future()` input, in input order, before its own start step, as a
    /// JS combinator awaits inputs that already run. An input whose JS form
    /// queues its `START` at the call (a step, a wait, a child context,
    /// `with_retry`, a map, a parallel or a `wait_for_callback`) queues it
    /// then, so its first checkpoint comes before the combinator's, and a
    /// start hook it fires when it queues `START` runs before the
    /// combinator's. A start hook that waits for the `START` response (a
    /// wait, a step under `AtMostOncePerRetry`) keeps that timing. A lazy
    /// invoke or `wait_for_condition` is polled once, in input order,
    /// before the combinator's start step, but the combinator does not wait
    /// for its `START`. The combinator then queues its own `START`, fires
    /// its start hook, and runs the inputs without waiting for the `START`
    /// response; its `SUCCEED` or `FAIL` waits for it. On a `STARTED`,
    /// `PENDING` or `READY` record it fires its replay start hook after the
    /// inputs' start phases begin. A replay of a final record starts no
    /// `.future()` input. A handle that `.spawn()` returned already runs
    /// independently, and the combinator does not start it again.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let a = ctx.step(|_| async { Ok(1) }).future();
    ///     let b = ctx.step(|_| async { Ok(2) }).future();
    ///     let _results: Vec<i32> = ctx.try_join_all([a, b])
    ///         .name("gather")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn try_join_all<O>(
        &self,
        futures: impl IntoIterator<Item = DurableFuture<O>>,
    ) -> TryJoinAllBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        TryJoinAllBuilder::new_internal(self.clone(), futures.into_iter().collect())
    }

    /// Joins all futures, collecting every outcome as a `Result`.
    ///
    /// Never fails fast: all futures run to completion regardless of
    /// individual errors. The output holds one `Result<O, OperationError>`
    /// per input, in input order.
    ///
    /// # Outer and inner errors
    ///
    /// Awaiting returns `Result<Vec<Result<O, OperationError>>, OperationError>`,
    /// and the two levels report different failures:
    ///
    /// 1. The outer `Err` means the combinator itself failed, for example
    ///    replay nondeterminism or a checkpoint write failure.
    /// 2. An input that returns `Err` appears only as that input's inner
    ///    `Err`. The outer result is then `Ok`.
    /// 3. A panicking input ends the invocation, as a panic does everywhere
    ///    else in the SDK. The SDK resumes the original panic payload. The
    ///    combinator writes no record and returns no result.
    /// 4. Nothing cancels an input that `join_all` awaits. A combinator stops
    ///    an input only by dropping it, and `join_all` drops no input before
    ///    it settles.
    ///
    /// An inner `Err` is a
    /// [`OperationErrorKind::ChildContext`](crate::OperationErrorKind::ChildContext)
    /// error of kind
    /// [`ChildContextErrorKind::ChildFailed`](crate::ChildContextErrorKind::ChildFailed),
    /// whose `wire()` carries the input's recorded failure (`error_type`,
    /// message, `error_data`), on the first run and on replay. Match on the
    /// `err.wire()` fields, not on the kind of the input that failed.
    ///
    /// # Oversized results
    ///
    /// A combined result larger than 256 KiB is not stored in the
    /// combinator's record. Replay then drives the inputs again. An input
    /// whose own terminal record is inline replays without running again. An
    /// input whose own terminal record has `ReplayChildren` runs its body
    /// again to rebuild its value, and its recorded descendants replay.
    ///
    /// # Empty input
    ///
    /// Called with no futures, resolves to `Ok` with an empty `Vec`,
    /// matching `futures::future::join_all`.
    ///
    /// # Operation ID
    ///
    /// This method collects `futures` and claims no operation ID. An
    /// operation that the iterator creates claims its ID during this
    /// collection, so the inputs take their IDs before the combinator. The
    /// combinator claims its own ID when it is first awaited or polled, or
    /// when it is spawned, on this context. A combinator that is dropped
    /// before that claims no ID and records nothing. An input that is itself
    /// a combinator must already have claimed its ID: start it with
    /// `.spawn()` and pass the returned handle. A still-unclaimed combinator
    /// input fails the execution with a context usage error. Starting the
    /// combinator inside the body of another context (a child context, a
    /// `with_retry` attempt, a map item or a parallel branch) fails the
    /// execution with a context usage error too.
    ///
    /// # Start order
    ///
    /// When the combinator runs, it begins the start phase of each
    /// `.future()` input, in input order, before its own start step, as a
    /// JS combinator awaits inputs that already run. An input whose JS form
    /// queues its `START` at the call (a step, a wait, a child context,
    /// `with_retry`, a map, a parallel or a `wait_for_callback`) queues it
    /// then, so its first checkpoint comes before the combinator's, and a
    /// start hook it fires when it queues `START` runs before the
    /// combinator's. A start hook that waits for the `START` response (a
    /// wait, a step under `AtMostOncePerRetry`) keeps that timing. A lazy
    /// invoke or `wait_for_condition` is polled once, in input order,
    /// before the combinator's start step, but the combinator does not wait
    /// for its `START`. The combinator then queues its own `START`, fires
    /// its start hook, and runs the inputs without waiting for the `START`
    /// response; its `SUCCEED` or `FAIL` waits for it. On a `STARTED`,
    /// `PENDING` or `READY` record it fires its replay start hook after the
    /// inputs' start phases begin. A replay of a final record starts no
    /// `.future()` input. A handle that `.spawn()` returned already runs
    /// independently, and the combinator does not start it again.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let a = ctx.step(|_| async { Ok(1) }).future();
    ///     let b = ctx.step(|_| async { Ok(2) }).future();
    ///     let outcomes: Vec<Result<i32, durable::OperationError>> = ctx
    ///         .join_all([a, b])
    ///         .name("collect")
    ///         .await?;
    ///     let succeeded = outcomes.iter().filter(|outcome| outcome.is_ok()).count();
    ///     assert_eq!(succeeded, 2);
    ///     Ok(())
    /// }
    /// ```
    pub fn join_all<O>(
        &self,
        futures: impl IntoIterator<Item = DurableFuture<O>>,
    ) -> JoinAllBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        JoinAllBuilder::new_internal(self.clone(), futures.into_iter().collect())
    }

    /// Returns the first successful result.
    ///
    /// When the combinator resolves, it cancels every input that is still
    /// running. A cancelled input stops at its next `.await`.
    ///
    /// The returned value has passed through the combinator's JSON
    /// encoding: the SDK decodes the payload it recorded and returns that.
    /// So the first run returns the value a replay returns, and data that
    /// the value's serde form omits, such as a `#[serde(skip)]` field, is
    /// absent on both paths.
    ///
    /// If every future fails, the operation fails with
    /// [`CombinatorErrorKind::AllFailed`](crate::CombinatorErrorKind::AllFailed)
    /// preserving each future's error (see [`CombinatorError::failures`](crate::CombinatorError::failures)).
    ///
    /// # Oversized results
    ///
    /// A winning value larger than 256 KiB is not stored in the
    /// combinator's record. The record stores the winner's input index
    /// instead. Replay then drives only the recorded winner again, and drops
    /// every other input without running it. The winner replays without
    /// running again when its own terminal record is inline. A winner whose
    /// own terminal record has `ReplayChildren` runs its body again to
    /// rebuild its value, and its recorded descendants replay.
    ///
    /// # Empty input
    ///
    /// Called with no futures, fails with
    /// [`CombinatorErrorKind::EmptyInput`](crate::CombinatorErrorKind::EmptyInput):
    /// there is no future that could succeed.
    ///
    /// A panic in an input propagates to the caller with its original
    /// payload, and the SDK records nothing for it.
    ///
    /// # Operation ID
    ///
    /// This method collects `futures` and claims no operation ID. An
    /// operation that the iterator creates claims its ID during this
    /// collection, so the inputs take their IDs before the combinator. The
    /// combinator claims its own ID when it is first awaited or polled, or
    /// when it is spawned, on this context. A combinator that is dropped
    /// before that claims no ID and records nothing. An input that is itself
    /// a combinator must already have claimed its ID: start it with
    /// `.spawn()` and pass the returned handle. A still-unclaimed combinator
    /// input fails the execution with a context usage error. Starting the
    /// combinator inside the body of another context (a child context, a
    /// `with_retry` attempt, a map item or a parallel branch) fails the
    /// execution with a context usage error too.
    ///
    /// # Start order
    ///
    /// When the combinator runs, it begins the start phase of each
    /// `.future()` input, in input order, before its own start step, as a
    /// JS combinator awaits inputs that already run. An input whose JS form
    /// queues its `START` at the call (a step, a wait, a child context,
    /// `with_retry`, a map, a parallel or a `wait_for_callback`) queues it
    /// then, so its first checkpoint comes before the combinator's, and a
    /// start hook it fires when it queues `START` runs before the
    /// combinator's. A start hook that waits for the `START` response (a
    /// wait, a step under `AtMostOncePerRetry`) keeps that timing. A lazy
    /// invoke or `wait_for_condition` is polled once, in input order,
    /// before the combinator's start step, but the combinator does not wait
    /// for its `START`. The combinator then queues its own `START`, fires
    /// its start hook, and runs the inputs without waiting for the `START`
    /// response; its `SUCCEED` or `FAIL` waits for it. On a `STARTED`,
    /// `PENDING` or `READY` record it fires its replay start hook after the
    /// inputs' start phases begin. A replay of a final record starts no
    /// `.future()` input. A handle that `.spawn()` returned already runs
    /// independently, and the combinator does not start it again.
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
    ///     let a = ctx.step(|_| async { Ok("fast".to_owned()) }).future();
    ///     let b = ctx.step(|_| async { Ok("slow".to_owned()) }).future();
    ///     let winner = ctx.select_ok([a, b])
    ///         .name("race-ok")
    ///         .await?;
    ///     Ok(winner)
    /// }
    /// ```
    pub fn select_ok<O>(
        &self,
        futures: impl IntoIterator<Item = DurableFuture<O>>,
    ) -> SelectOkBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        SelectOkBuilder::new_internal(self.clone(), futures.into_iter().collect())
    }

    /// Returns the first settled result, whether success or failure.
    ///
    /// When the combinator resolves, it cancels every input that is still
    /// running. A cancelled input stops at its next `.await`.
    ///
    /// The returned value has passed through the combinator's JSON
    /// encoding: the SDK decodes the payload it recorded and returns that.
    /// So the first run returns the value a replay returns, and data that
    /// the value's serde form omits, such as a `#[serde(skip)]` field, is
    /// absent on both paths.
    ///
    /// When the first settled outcome is a failure, the operation fails
    /// with
    /// [`CombinatorErrorKind::FirstSettledFailed`](crate::CombinatorErrorKind::FirstSettledFailed)
    /// carrying the losing future's error as its source; the same variant
    /// is produced live and on replay.
    ///
    /// # Oversized results
    ///
    /// A winning value larger than 256 KiB is not stored in the
    /// combinator's record. The record stores the winner's input index
    /// instead. Replay then drives only the recorded winner again, and drops
    /// every other input without running it. The winner replays without
    /// running again when its own terminal record is inline. A winner whose
    /// own terminal record has `ReplayChildren` runs its body again to
    /// rebuild its value, and its recorded descendants replay.
    ///
    /// # Empty input
    ///
    /// Called with no futures, fails with
    /// [`CombinatorErrorKind::EmptyInput`](crate::CombinatorErrorKind::EmptyInput):
    /// there is no future that could settle.
    ///
    /// A panic in an input propagates to the caller with its original
    /// payload, and the SDK records nothing for it.
    ///
    /// # Operation ID
    ///
    /// This method collects `futures` and claims no operation ID. An
    /// operation that the iterator creates claims its ID during this
    /// collection, so the inputs take their IDs before the combinator. The
    /// combinator claims its own ID when it is first awaited or polled, or
    /// when it is spawned, on this context. A combinator that is dropped
    /// before that claims no ID and records nothing. An input that is itself
    /// a combinator must already have claimed its ID: start it with
    /// `.spawn()` and pass the returned handle. A still-unclaimed combinator
    /// input fails the execution with a context usage error. Starting the
    /// combinator inside the body of another context (a child context, a
    /// `with_retry` attempt, a map item or a parallel branch) fails the
    /// execution with a context usage error too.
    ///
    /// # Start order
    ///
    /// When the combinator runs, it begins the start phase of each
    /// `.future()` input, in input order, before its own start step, as a
    /// JS combinator awaits inputs that already run. An input whose JS form
    /// queues its `START` at the call (a step, a wait, a child context,
    /// `with_retry`, a map, a parallel or a `wait_for_callback`) queues it
    /// then, so its first checkpoint comes before the combinator's, and a
    /// start hook it fires when it queues `START` runs before the
    /// combinator's. A start hook that waits for the `START` response (a
    /// wait, a step under `AtMostOncePerRetry`) keeps that timing. A lazy
    /// invoke or `wait_for_condition` is polled once, in input order,
    /// before the combinator's start step, but the combinator does not wait
    /// for its `START`. The combinator then queues its own `START`, fires
    /// its start hook, and runs the inputs without waiting for the `START`
    /// response; its `SUCCEED` or `FAIL` waits for it. On a `STARTED`,
    /// `PENDING` or `READY` record it fires its replay start hook after the
    /// inputs' start phases begin. A replay of a final record starts no
    /// `.future()` input. A handle that `.spawn()` returned already runs
    /// independently, and the combinator does not start it again.
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
    ///     let a = ctx.step(|_| async { Ok("first".to_owned()) }).future();
    ///     let b = ctx.step(|_| async { Ok("second".to_owned()) }).future();
    ///     let winner = ctx.race([a, b])
    ///         .name("fastest")
    ///         .await?;
    ///     Ok(winner)
    /// }
    /// ```
    pub fn race<O>(&self, futures: impl IntoIterator<Item = DurableFuture<O>>) -> RaceBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        RaceBuilder::new_internal(self.clone(), futures.into_iter().collect())
    }
}

/// Context passed to step bodies.
///
/// `StepContext` deliberately does **not** expose any durable operations.
/// This provides compile-time enforcement of the rule that durable
/// operations cannot be nested inside step bodies: attempting to call
/// a durable operation method inside a step body is a type error.
///
/// Use `tracing` macros for logging inside steps: the SDK-created
/// operation span automatically attaches execution context fields.
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
///     let result = ctx.step(|step_ctx: durable::StepContext| async move {
///         // step_ctx has no durable operations: type-system enforced
///         tracing::info!("inside step");
///         Ok("done".to_owned())
///     }).await?;
///     Ok(result)
/// }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StepContext {
    attempt: u32,
    _private: (),
}

impl StepContext {
    /// Creates a new step context (internal).
    pub(crate) fn new(attempt: u32) -> Self {
        Self {
            attempt,
            _private: (),
        }
    }

    /// Returns the current attempt number, starting at 1 for the first
    /// attempt.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// Canonicalizes an operation type string to the service wire format via
/// [`aws_sdk_lambda::types::OperationType`].
///
/// Both envelope parsers store the wire token (`CHAINED_INVOKE`, the
/// `OperationType::as_str()` form). The SDK's own claimed types and older
/// fixtures use `PascalCase` (`ChainedInvoke`). A plain case-insensitive
/// comparison misses the underscore, so both spellings are converted to
/// `SCREAMING_SNAKE` and round-tripped through `OperationType` before
/// comparison. Unknown types canonicalize to their `SCREAMING_SNAKE` form,
/// which keeps genuine mismatches detectable.
pub(crate) fn canonical_op_type(raw: &str) -> String {
    // PascalCase → SCREAMING_SNAKE: insert `_` at lower→upper boundaries,
    // then uppercase. Already-wire values (all caps + underscores) pass
    // through unchanged because they contain no lower→upper boundary.
    let mut screaming = String::with_capacity(raw.len() + 4);
    let mut prev_is_lower = false;
    for ch in raw.chars() {
        if ch.is_ascii_uppercase() && prev_is_lower {
            screaming.push('_');
        }
        prev_is_lower = ch.is_ascii_lowercase();
        screaming.push(ch.to_ascii_uppercase());
    }
    // Round-trip through the SDK enum so every known type lands on the
    // exact canonical wire constant (`OperationType::as_str()`).
    aws_sdk_lambda::types::OperationType::from(screaming.as_str())
        .as_str()
        .to_owned()
}

/// Compares a checkpoint record's stored identity against a claimed one.
///
/// Returns `None` when the identities match (or the record carries no
/// operation type), and `Some(expected)`, the stored identity
/// formatted for the mismatch error, when they differ. The comparisons
/// borrow the record, so the match path clones nothing; only the cold
/// mismatch path allocates. This is the shared core behind
/// `DurableContext::validate_replay_identity` and
/// `DurableContext::checkpoint_view_validated`.
fn replay_identity_mismatch(
    record: &CheckpointRecord,
    claimed_type: &str,
    claimed_sub_type: Option<&str>,
    claimed_name: Option<&str>,
) -> Option<String> {
    // A record without a stored operation type has nothing to validate
    // against, so the check skips it, as JS does. This is the ONLY
    // lenient path; once a record carries identity, every field is
    // compared in full.
    let expected_type = record.op_type.as_deref()?;

    let mismatch = || {
        Some(format_op_identity(
            expected_type,
            record.sub_type.as_deref(),
            record.op_name.as_deref(),
        ))
    };

    // Compare canonicalized types: the checkpoint log stores the wire
    // token (e.g. `CHAINED_INVOKE`), while the claimed type can be
    // PascalCase, so both sides canonicalize through `OperationType`
    // before comparison.
    if canonical_op_type(expected_type) != canonical_op_type(claimed_type) {
        return mismatch();
    }

    // Sub-type is compared as a complete Option in both directions: a
    // stored sub-type with no claimed sub-type (or the reverse) is a
    // mismatch, not a skip. Otherwise a reordered same-type operation
    // could consume the wrong checkpoint silently.
    let sub_type_matches = match (record.sub_type.as_deref(), claimed_sub_type) {
        (Some(expected), Some(claimed)) => expected.eq_ignore_ascii_case(claimed),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    if !sub_type_matches {
        return mismatch();
    }

    // Name is likewise compared as a complete Option in both directions.
    // Adding or removing `.name(...)` between runs changes the operation's
    // replay identity and must be flagged, for the same reason as above.
    //
    // Empty names are normalized to `None` on BOTH sides before the
    // comparison: an empty name and no name are the same identity, so an
    // operation named `""` whose record holds no `Name` still matches.
    // Without normalization an UNCHANGED handler would be rejected on
    // resume. Map items and parallel branches never claim an empty name:
    // they claim the recorded name, which defaults to `map-item-{index}`
    // or `parallel-branch-{index}`.
    let claimed_name = claimed_name.filter(|n| !n.is_empty());
    let expected_name = record.op_name.as_deref().filter(|n| !n.is_empty());
    let name_matches = match (expected_name, claimed_name) {
        (Some(expected), Some(claimed)) => expected == claimed,
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    if !name_matches {
        return mismatch();
    }

    None
}

/// Formats an operation's identity as a human-readable string for error
/// messages (e.g. `"Step/Step named \"fetch-name\""` or `"Wait/Wait"`).
fn format_op_identity(op_type: &str, sub_type: Option<&str>, name: Option<&str>) -> String {
    let mut s = String::with_capacity(64);
    s.push_str(op_type);
    if let Some(sub) = sub_type {
        s.push('/');
        s.push_str(sub);
    }
    if let Some(n) = name {
        s.push_str(" named \"");
        s.push_str(n);
        s.push('"');
    }
    s
}

/// The value an operation returns when its direct checkpoint write found
/// no Tokio runtime (see `DurableContext::checkpoint_failure_unrecoverable`).
pub(crate) trait NoRuntimeOutcome {
    /// The outcome for `no_runtime`.
    fn no_runtime(no_runtime: crate::task::NoRuntime) -> Self;
}

impl<T> NoRuntimeOutcome for Result<T, OperationError> {
    fn no_runtime(no_runtime: crate::task::NoRuntime) -> Self {
        Err(crate::task::NoRuntime::operation_error(no_runtime))
    }
}

/// For a call site whose function returns the operation error itself.
impl NoRuntimeOutcome for OperationError {
    fn no_runtime(no_runtime: crate::task::NoRuntime) -> Self {
        crate::task::NoRuntime::operation_error(no_runtime)
    }
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::client::{
        GetStateOutput, InMemoryExecutionClient, TestResponse, operations_to_checkpoint_log,
    };
    use crate::engine::{CheckpointLog, CheckpointStatus};
    use aws_sdk_lambda::types::Operation;
    use std::sync::Arc;

    /// The batch context, its Normal item context and its Flat item
    /// context take their levels from the constructors: `new_child` adds 1
    /// and `new_flat_child` keeps the level.
    #[test]
    fn nesting_levels_follow_constructors() {
        let (root, _client) = DurableContext::test_context_with_depth(5);
        let level = |ctx: &DurableContext| ctx.inner.nesting_level;
        assert_eq!(level(&root), 0);
        assert_eq!(root.inner.preserve_child_depth, 5);

        let batch = root.new_child("1", None);
        let normal = batch.new_child("1-1", None);
        let flat = batch.new_flat_child("1-2", "parent", None);
        assert_eq!((level(&batch), level(&normal), level(&flat)), (1, 2, 1));

        let child = root.new_child("2", None);
        let batch = child.new_child("2-1", None);
        let normal = batch.new_child("2-1-1", None);
        let flat = batch.new_flat_child("2-1-2", "parent", None);
        assert_eq!((level(&batch), level(&normal), level(&flat)), (2, 3, 2));
        for ctx in [&child, &batch, &normal, &flat] {
            assert_eq!(
                ctx.inner.preserve_child_depth, 5,
                "every constructor copies the depth"
            );
        }

        for ctx in [
            DurableContext::test_context(),
            DurableContext::new_root(
                "arn".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::empty()),
            ),
        ] {
            assert_eq!((level(&ctx), ctx.inner.preserve_child_depth), (0, 0));
        }
    }

    /// Each row of the issue's table: the context that creates the record,
    /// asked with `levels_below` 1, against depths 0, 1, 2 and `u32::MAX`.
    #[test]
    fn preserves_children_table() {
        let depths = [0, 1, 2, u32::MAX];
        let row = |make: &dyn Fn(&DurableContext) -> DurableContext| -> Vec<bool> {
            depths
                .iter()
                .map(|&depth| {
                    let (root, _client) = DurableContext::test_context_with_depth(depth);
                    make(&root).preserves_children(1)
                })
                .collect()
        };
        // A top-level child, map or parallel container: level 1.
        assert_eq!(row(&|root| root.clone()), [false, true, true, true]);
        // An item of a top-level map, Normal nesting: level 2.
        assert_eq!(
            row(&|root| root.new_child("1", None)),
            [false, false, true, true]
        );
        // A child inside a Normal item: level 3.
        assert_eq!(
            row(&|root| root.new_child("1", None).new_child("1-1", None)),
            [false, false, false, true]
        );
        // A child inside a Flat item of a top-level map: level 2.
        assert_eq!(
            row(&|root| root
                .new_child("1", None)
                .new_flat_child("1-1", "parent", None)),
            [false, false, true, true]
        );
        // A child inside a top-level child: level 2.
        assert_eq!(
            row(&|root| root.new_child("1", None)),
            [false, false, true, true]
        );
        // Additions saturate instead of overflowing.
        let (root, _client) = DurableContext::test_context_with_depth(u32::MAX);
        assert!(root.new_child("1", None).preserves_children(u32::MAX));
    }

    /// Test 19: each combinator method collects the operations its
    /// iterator creates, which claim their IDs, and claims no ID itself.
    #[test]
    fn combinator_methods_collect_without_claiming() {
        let ctx = DurableContext::new_root(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
        );
        // A lazy iterator: each step claims its ID when the combinator
        // method collects it.
        let inputs = || {
            let ctx = &ctx;
            (0..2).map(move |n| ctx.step(move |_| async move { Ok(n) }).future())
        };
        let try_join_all = ctx.try_join_all(inputs());
        let join_all = ctx.join_all(inputs());
        let select_ok = ctx.select_ok(inputs());
        let race = ctx.race(inputs());
        assert_eq!(
            ctx.mint_id().positional(),
            "9",
            "the eight steps claimed 1 to 8, the combinators nothing"
        );
        drop((try_join_all, join_all, select_ok, race));
    }

    /// Panics with "value drop" when dropped.
    struct DropPanics;

    impl Drop for DropPanics {
        #[expect(clippy::panic)]
        // reason: the test checks that the guard records this panic.
        fn drop(&mut self) {
            panic!("value drop");
        }
    }

    /// Wraps the receiving half of a fresh channel in an
    /// [`UnreadResultGuard`].
    fn guarded<T>(
        coordinator: &Arc<InvocationCoordinator>,
    ) -> (tokio::sync::oneshot::Sender<T>, UnreadResultGuard<T>) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let guard = UnreadResultGuard {
            receiver,
            coordinator: Arc::clone(coordinator),
        };
        (sender, guard)
    }

    /// A result sent but never read is dropped inside a catch when the
    /// guard is dropped. Its destructor's panic does not escape the drop,
    /// and it is recorded on the coordinator.
    #[test]
    fn unread_result_guard_records_a_destructor_panic() {
        let coordinator = InvocationCoordinator::new();
        let (sender, guard) = guarded::<DropPanics>(&coordinator);
        assert!(sender.send(DropPanics).is_ok());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(guard))).is_ok()
        );
        let payload = coordinator.take_panic().expect("recorded");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"value drop"));
    }

    /// A guard with no unread result records nothing. After the guard is
    /// dropped, the sender's `send` fails, so the sender disposes of the
    /// result itself.
    #[test]
    fn unread_result_guard_closes_the_channel() {
        let coordinator = InvocationCoordinator::new();
        let (sender, guard) = guarded::<u32>(&coordinator);
        drop(guard);
        assert!(!coordinator.has_panic());
        assert_eq!(sender.send(7), Err(7));
    }

    /// A claim outside any `run_body` scope checks only ownership, and the
    /// owner thread passes it.
    #[test]
    fn claim_without_active_body_records_nothing() {
        let ctx = DurableContext::test_context();
        let id = ctx.claim_operation_id("step");
        assert_eq!(id.positional(), "1");
        assert!(ctx.coordinator().fatal_error().is_none());
    }

    /// A claim from a thread that does not own the context, with no body
    /// active, records the ownership error and still claims the next ID.
    /// The owner's claim before it records nothing.
    #[test]
    fn claim_from_foreign_thread_records_fatal() {
        let ctx = DurableContext::test_context();
        assert_eq!(ctx.claim_operation_id("step").positional(), "1");
        assert!(ctx.coordinator().fatal_error().is_none());

        let id = std::thread::scope(|scope| {
            scope
                .spawn(|| ctx.claim_operation_id("step"))
                .join()
                .expect("the claim does not panic")
        });
        assert_eq!(id.positional(), "2");
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert_eq!(fatal.error_type, "Error");
        assert_eq!(fatal.error_message, foreign_caller_message("step"));
    }

    /// `enforce_task_ownership` passes for the owner thread and records
    /// nothing. From a thread that does not own the context it returns an
    /// `ExecutionFailed` step error and records the ownership error.
    #[test]
    fn enforce_task_ownership_refuses_foreign_thread() {
        let ctx = DurableContext::test_context();
        assert!(ctx.enforce_task_ownership().is_ok());
        assert!(ctx.coordinator().fatal_error().is_none());

        let result = std::thread::scope(|scope| {
            scope
                .spawn(|| ctx.enforce_task_ownership())
                .join()
                .expect("the check does not panic")
        });
        let err = result.expect_err("a foreign thread is refused");
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Step(step) if matches!(step.kind(), StepErrorKind::ExecutionFailed)
        ));
        // The `StepError` is the operation error's source, and the ownership
        // message is the step error's source.
        let source = std::error::Error::source(&err)
            .and_then(std::error::Error::source)
            .map(ToString::to_string);
        assert!(
            source
                .as_deref()
                .is_some_and(|msg| msg.contains("which does not own the context")),
            "unexpected source: {source:?}"
        );
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert_eq!(fatal.error_type, "Error");
        assert_eq!(fatal.error_message, foreign_caller_message("execute"));
    }

    /// A claim on the context whose body is active passes.
    #[tokio::test]
    async fn claim_in_own_body_records_nothing() {
        let ctx = DurableContext::test_context();
        let id = run_body(ActiveBody::Context(String::new()), async {
            ctx.claim_operation_id("step")
        })
        .await;
        assert_eq!(id.positional(), "1");
        assert!(ctx.coordinator().fatal_error().is_none());
    }

    /// A claim on the context whose body is active, made by a raw task that
    /// does not own the context, still runs the ownership check: the
    /// matching body does not authorize the caller. It records the
    /// ownership error and still claims the next ID.
    #[tokio::test]
    async fn claim_in_own_body_on_raw_task_records_fatal() {
        let ctx = DurableContext::test_context();
        assert_eq!(ctx.claim_operation_id("step").positional(), "1");
        let foreign = ctx.clone();
        let id = tokio::spawn(run_body(ActiveBody::Context(String::new()), async move {
            foreign.claim_operation_id("step")
        }))
        .await
        .expect("the claim does not panic");
        assert_eq!(id.positional(), "2");
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert_eq!(fatal.error_type, "Error");
        assert_eq!(fatal.error_message, foreign_caller_message("step"));
    }

    /// A claim on another context than the active body records message A
    /// and still claims the ID.
    #[tokio::test]
    async fn claim_in_other_body_records_fatal() {
        let ctx = DurableContext::test_context();
        let id = run_body(ActiveBody::Context("3".to_owned()), async {
            ctx.claim_operation_id("wait")
        })
        .await;
        assert_eq!(id.positional(), "1");
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert_eq!(fatal.error_type, "Error");
        assert_eq!(
            fatal.error_message,
            "context usage error in \"wait\": the operation was called on context \"root\", \
             but the SDK is running the body of context \"3\". Inside run_in_child_context, \
             with_retry, map and parallel, call operations on the context passed to the closure."
        );
    }

    /// A claim inside a step body records message B.
    #[tokio::test]
    async fn claim_in_step_body_records_fatal() {
        let ctx = DurableContext::test_context();
        let _id = run_body(ActiveBody::Step("4".to_owned()), async {
            ctx.claim_operation_id("map")
        })
        .await;
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert_eq!(fatal.error_type, "Error");
        assert_eq!(
            fatal.error_message,
            "context usage error in \"map\": the operation was called inside the body of \
             operation \"4\". A step body or a wait_for_condition check cannot call durable \
             operations; use run_in_child_context to group operations."
        );
    }

    /// Helper: builds a Step operation.
    fn make_step_op(id: &str, result: &str) -> Operation {
        Operation::builder()
            .id(id)
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .step_details(
                aws_sdk_lambda::types::StepDetails::builder()
                    .result(result)
                    .build(),
            )
            .build()
            .unwrap()
    }

    /// Tests that `checkpoint_updates` paginates when the checkpoint response
    /// has a `next_marker`: calling `get_state` to fetch all remaining
    /// operations and merging them into the checkpoint log.
    #[tokio::test]
    async fn checkpoint_updates_paginates_on_marker() {
        // The full state has 3 operations (what get_state returns).
        let all_ops = vec![
            make_step_op("step-1", "\"r1\""),
            make_step_op("step-2", "\"r2\""),
            make_step_op("step-3", "\"r3\""),
        ];
        let client = Arc::new(InMemoryExecutionClient::new(all_ops));

        // Enqueue a paginated checkpoint response: returns only step-1
        // but signals more pages via next_marker.
        let page1_ops = vec![make_step_op("step-1", "\"r1\"")];
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            page1_ops,
            "page-2-token".to_owned(),
        ));

        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            client.clone(),
            "initial-token".to_owned(),
        );

        // Perform a checkpoint with no updates (we just want to exercise pagination).
        let result = ctx.checkpoint_updates(Vec::new()).await;
        assert!(result.is_ok());

        // The checkpoint log should now contain ALL operations from get_state.
        assert!(log.get("step-1").is_some(), "step-1 must be in the log");
        assert!(
            log.get("step-2").is_some(),
            "step-2 must be in the log (paginated)"
        );
        assert!(
            log.get("step-3").is_some(),
            "step-3 must be in the log (paginated)"
        );

        // get_state should have been called once for pagination.
        let get_state_count = *client.get_state_call_count.lock().unwrap();
        assert_eq!(
            get_state_count, 1,
            "get_state must be called for pagination"
        );
    }

    /// Mock client for the concurrent pagination/token race test: every
    /// `checkpoint` validates and rotates the token, always returns a
    /// pagination marker, and `get_state` records whether the token it was
    /// handed is still the client's CURRENT token.
    #[derive(Debug)]
    struct PaginatedTokenValidatingClient {
        current_token: std::sync::Mutex<String>,
        counter: std::sync::atomic::AtomicU32,
        stale_checkpoint_tokens: std::sync::atomic::AtomicU32,
        stale_get_state_tokens: std::sync::atomic::AtomicU32,
        get_state_calls: std::sync::atomic::AtomicU32,
    }

    impl PaginatedTokenValidatingClient {
        fn new(initial_token: &str) -> Self {
            Self {
                current_token: std::sync::Mutex::new(initial_token.to_owned()),
                counter: std::sync::atomic::AtomicU32::new(0),
                stale_checkpoint_tokens: std::sync::atomic::AtomicU32::new(0),
                stale_get_state_tokens: std::sync::atomic::AtomicU32::new(0),
                get_state_calls: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl ExecutionClient for PaginatedTokenValidatingClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            let submitted = checkpoint_token.to_owned();
            Box::pin(async move {
                // Widen the race window like the real network call would.
                tokio::task::yield_now().await;
                let mut current = self
                    .current_token
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *current != submitted {
                    self.stale_checkpoint_tokens
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    return Err(ExecutionClientError::non_retryable(format!(
                        "stale checkpoint token: expected {current}, got {submitted}"
                    )));
                }
                let n = self
                    .counter
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let next = format!("token-{}", n + 1);
                current.clone_from(&next);
                drop(current);
                Ok(CheckpointOutput {
                    checkpoint_token: Some(next),
                    updated_operations: Vec::new(),
                    // Every response paginated: forces the marker-triggered
                    // get_state on every checkpoint.
                    next_marker: Some("more-pages".to_owned()),
                })
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            let submitted = checkpoint_token.to_owned();
            Box::pin(async move {
                self.get_state_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Yield so a concurrent branch would have every chance to
                // checkpoint (and rotate the token) if the caller released
                // the token lock too early.
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                let current = self
                    .current_token
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *current != submitted {
                    self.stale_get_state_tokens
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                drop(current);
                Ok(GetStateOutput::new(Vec::new(), None))
            })
        }
    }

    /// Regression test (issue #5): concurrent checkpoints whose responses
    /// carry a pagination marker must perform the marker-triggered
    /// `get_state` while STILL holding the checkpoint-token lock. If the
    /// lock is dropped first, a concurrent branch checkpoints, consumes and
    /// rotates the token, and the paginated `get_state` runs with a stale
    /// token. The mock's `get_state` validates the token it receives
    /// against the client's current token.
    #[tokio::test]
    async fn concurrent_paginated_checkpoints_get_state_uses_current_token() {
        let client = Arc::new(PaginatedTokenValidatingClient::new("token-0"));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
        );

        // Race several concurrent checkpoint_updates calls, each of which
        // paginates. Every one must succeed (no stale checkpoint token) and
        // every get_state must observe the then-current token.
        let (r1, r2, r3, r4) = tokio::join!(
            ctx.checkpoint_updates(Vec::new()),
            ctx.checkpoint_updates(Vec::new()),
            ctx.checkpoint_updates(Vec::new()),
            ctx.checkpoint_updates(Vec::new()),
        );
        assert!(r1.is_ok(), "checkpoint 1 failed: {r1:?}");
        assert!(r2.is_ok(), "checkpoint 2 failed: {r2:?}");
        assert!(r3.is_ok(), "checkpoint 3 failed: {r3:?}");
        assert!(r4.is_ok(), "checkpoint 4 failed: {r4:?}");

        assert_eq!(
            client
                .get_state_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            4,
            "every paginated checkpoint must trigger a get_state"
        );
        assert_eq!(
            client
                .stale_checkpoint_tokens
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "no checkpoint may run with a stale token"
        );
        assert_eq!(
            client
                .stale_get_state_tokens
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the marker-triggered get_state must run with the current token \
             (token lock held through the paginated fetch)"
        );
    }

    /// Tests that `checkpoint_updates` does NOT call `get_state` when there
    /// is no pagination marker.
    #[tokio::test]
    async fn checkpoint_updates_no_pagination_when_no_marker() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        // Default response has no marker.

        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "initial-token".to_owned(),
        );

        let result = ctx.checkpoint_updates(Vec::new()).await;
        assert!(result.is_ok());

        let get_state_count = *client.get_state_call_count.lock().unwrap();
        assert_eq!(
            get_state_count, 0,
            "get_state must NOT be called without a marker"
        );
    }

    /// Tests that the bootstrap pagination path works: when the initial
    /// state has a `NextMarker`, `get_state` is called to fetch the full
    /// operation set, and all operations appear in the checkpoint log.
    #[tokio::test]
    async fn bootstrap_pagination_fetches_all_pages() {
        // Simulate a paginated initial state: page 1 has step-1,
        // and get_state returns the full set (step-1 + step-2 + step-3).
        let all_ops = vec![
            make_step_op("step-1", "\"result-1\""),
            make_step_op("step-2", "\"result-2\""),
            make_step_op("step-3", "\"result-3\""),
        ];
        let client: Arc<dyn ExecutionClient> = Arc::new(InMemoryExecutionClient::new(all_ops));

        // The full log comes from get_state when the initial state is paginated.
        let full_state = client.get_state("arn:test", "token", None).await;
        assert!(full_state.is_ok());
        let full_state = full_state.unwrap();

        let log = operations_to_checkpoint_log(&full_state.operations);
        assert!(log.get("step-1").is_some());
        assert!(log.get("step-2").is_some());
        assert!(log.get("step-3").is_some());

        // Verify that the results are correct.
        let r2 = log.get("step-2").unwrap();
        assert_eq!(r2.result.as_deref(), Some("\"result-2\""));
        assert_eq!(r2.status, CheckpointStatus::Succeeded);
    }

    // Non-determinism detection tests

    #[test]
    fn validate_replay_identity_passes_when_types_match() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: Some("42".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: Some("my-step".to_owned()),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("my-step"));
        assert!(result.is_ok());
    }

    // Targeted checkpoint accessors

    /// Builds a context whose checkpoint log holds one record at positional
    /// ID `"1"` (keyed by its wire ID, matching the production log shape).
    fn ctx_with_record_at_1(record: CheckpointRecord) -> DurableContext {
        let wire_id = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_id, record)]));
        DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        )
    }

    #[test]
    fn checkpoint_view_validated_returns_view_on_identity_match() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: Some("42".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 2,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: true,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: Some("my-step".to_owned()),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        });

        let result =
            ctx.checkpoint_view_validated("1", "wire-1", "Step", Some("Step"), Some("my-step"));
        assert!(result.is_ok());
        let view = result.unwrap();
        assert!(view.is_some());
        let view = view.unwrap();
        assert_eq!(view.status, CheckpointStatus::Succeeded);
        assert_eq!(view.attempt, 2);
        assert!(view.replay_children);

        // No record at another position: Ok(None), nothing to validate.
        let live = ctx.checkpoint_view_validated("2", "wire-2", "Step", Some("Step"), None);
        assert!(matches!(live, Ok(None)));
    }

    #[test]
    fn checkpoint_view_validated_errors_on_identity_mismatch() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        });

        // Claimed as Wait but checkpointed as Step → the same
        // NonDeterministicExecution error `validate_replay_identity` builds.
        let result = ctx.checkpoint_view_validated("1", "wire-1", "Wait", Some("Wait"), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("Step") && display.contains("Wait"),
            "expected both identities in error: {display}"
        );
    }

    /// Replay reaching a record whose status this SDK version does not
    /// recognize must fail the execution naming the raw status (issue
    /// #45), never return the view for the operation to act on.
    #[test]
    #[expect(clippy::panic)] // reason: test assertions over non_exhaustive kinds
    fn checkpoint_view_validated_fails_execution_on_unrecognized_status() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Unknown("PAUSED".to_owned()),
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        });

        assert!(
            ctx.coordinator().fatal_error().is_none(),
            "no fatal recorded before the replay check"
        );

        // Identity matches: the failure is the status, not the identity.
        let result = ctx.checkpoint_view_validated("1", "wire-1", "Step", Some("Step"), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let OperationErrorKind::NonDeterministicExecution(nde) = err.kind() else {
            panic!("expected NonDeterministicExecution, got {:?}", err.kind());
        };
        let NonDeterministicExecutionErrorKind::UnrecognizedStatus(details) = nde.kind() else {
            panic!("expected UnrecognizedStatus, got {:?}", nde.kind());
        };
        assert_eq!(details.wire_id(), "wire-1");
        assert_eq!(details.status(), "PAUSED");
        let display = format!("{err:#}");
        assert!(
            display.contains("PAUSED"),
            "error must name the raw status: {display}"
        );

        // The fatal slot is written so the invocation driver fails the
        // execution even if the returned `Err` is swallowed downstream.
        let fatal = ctx
            .coordinator()
            .fatal_error()
            .expect("unrecognized status must record a fatal error");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        assert!(
            fatal.error_message.contains("PAUSED"),
            "fatal message must carry the raw status: {}",
            fatal.error_message
        );
    }

    /// A record with a recognized status and matching identity records no
    /// fatal and returns the view: the #45 guard fires only on `Unknown`.
    #[test]
    fn checkpoint_view_validated_known_status_records_no_fatal() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::TimedOut,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        });

        let result = ctx.checkpoint_view_validated("1", "wire-1", "Step", Some("Step"), None);
        assert!(matches!(result, Ok(Some(_))));
        assert!(ctx.coordinator().fatal_error().is_none());
    }

    #[test]
    fn targeted_getters_return_single_fields() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Failed,
            result: Some(r#""state""#.to_owned()),
            error_type: Some("SomeError".to_owned()),
            error_message: Some("it broke".to_owned()),
            error_data: None,
            stack_trace: None,
            attempt: 5,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: Some("cb-42".to_owned()),
            op_type: None,
            sub_type: None,
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        });

        assert!(ctx.has_checkpoint_record("1"));
        assert!(!ctx.has_checkpoint_record("2"));

        assert_eq!(
            ctx.checkpoint_result_payload("1").as_deref(),
            Some(r#""state""#)
        );
        assert!(ctx.checkpoint_result_payload("2").is_none());

        assert_eq!(ctx.checkpoint_callback_id("1").as_deref(), Some("cb-42"));

        let wire = ctx.checkpoint_wire_error("1");
        let wire = wire.expect("record 1 exists");
        assert_eq!(wire.error_type(), Some("SomeError"));
        assert_eq!(wire.error_message(), Some("it broke"));
        assert!(ctx.checkpoint_wire_error("2").is_none());

        let view = ctx.checkpoint_status_view("1");
        assert!(view.is_some());
        let view = view.unwrap();
        assert_eq!(view.status, CheckpointStatus::Failed);
        assert_eq!(view.attempt, 5);
    }

    #[test]
    fn checkpoint_terminal_replay_projects_payload_and_error_fields() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Failed,
            result: Some(r#""summary""#.to_owned()),
            error_type: Some("BatchError".to_owned()),
            error_message: Some("batch broke".to_owned()),
            error_data: Some(r#"{"code":7}"#.to_owned()),
            stack_trace: Some(vec!["frame-a".to_owned()]),
            attempt: 3,
            invoke_result: Some("never-projected".to_owned()),
            invoke_error_type: Some("never-projected".to_owned()),
            invoke_error_message: Some("never-projected".to_owned()),
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: true,
            callback_id: Some("cb-99".to_owned()),
            op_type: Some("Context".to_owned()),
            sub_type: Some("Map".to_owned()),
            due_at: None,
            op_name: Some("batch".to_owned()),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        });

        let snapshot = ctx.checkpoint_terminal_replay("1");
        assert_eq!(
            snapshot,
            Some(TerminalReplaySnapshot {
                status: CheckpointStatus::Failed,
                replay_children: true,
                result: Some(r#""summary""#.to_owned()),
                error_message: Some("batch broke".to_owned()),
                error_type: Some("BatchError".to_owned()),
                error_data: Some(r#"{"code":7}"#.to_owned()),
                stack_trace: Some(vec!["frame-a".to_owned()]),
            })
        );

        // No record → no snapshot.
        assert!(ctx.checkpoint_terminal_replay("2").is_none());
    }

    #[test]
    fn validate_replay_identity_fails_on_type_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        // Claimed as Wait but checkpointed as Step → error
        let result = ctx.validate_replay_identity(&record, "wire-1", "Wait", Some("Wait"), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("Step"),
            "expected Step in error: {display}"
        );
        assert!(
            display.contains("Wait"),
            "expected Wait in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_fails_on_subtype_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        // Same op_type but different sub_type → error
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("WaitForCondition"), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
    }

    #[test]
    fn validate_replay_identity_fails_on_name_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: Some("fetch-user".to_owned()),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        // Same type but different name → error
        let result = ctx.validate_replay_identity(
            &record,
            "wire-1",
            "Step",
            Some("Step"),
            Some("fetch-order"),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let display = format!("{err:#}");
        assert!(
            display.contains("fetch-user"),
            "expected name in error: {display}"
        );
        assert!(
            display.contains("fetch-order"),
            "expected claimed name in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_skips_when_no_identity_stored() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: Some("1".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        // Even though claimed type differs, validation passes because the
        // record has no identity to compare against.
        let result = ctx.validate_replay_identity(&record, "wire-1", "Wait", Some("Wait"), None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_replay_identity_case_insensitive_type_match() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("STEP".to_owned()),
            sub_type: Some("STEP".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        // Wire format uses UPPER_CASE: validation is case-insensitive.
        let result = ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), None);
        assert!(result.is_ok());
    }

    /// Regression test (issue #6): the inline JSON envelope path stores the
    /// raw wire value `CHAINED_INVOKE` (with underscore) while the SDK
    /// claims `ChainedInvoke` (`PascalCase`). A case-insensitive comparison
    /// alone rejects every non-paginated replayed invoke as
    /// non-deterministic; canonicalization must accept it.
    #[test]
    fn validate_replay_identity_inline_chained_invoke_replay_matches() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: Some("\"ok\"".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            // Exactly what parse_single_operation stores from the inline
            // envelope: the raw wire `Type` value.
            op_type: Some("CHAINED_INVOKE".to_owned()),
            sub_type: Some("ChainedInvoke".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        let result = ctx.validate_replay_identity(
            &record,
            "wire-1",
            "ChainedInvoke",
            Some("ChainedInvoke"),
            None,
        );
        assert!(
            result.is_ok(),
            "inline CHAINED_INVOKE record must match claimed ChainedInvoke: {result:?}"
        );
    }

    /// Canonicalization must not weaken detection: a genuinely different
    /// type still mismatches after canonicalization.
    #[test]
    fn validate_replay_identity_canonicalized_types_still_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("CHAINED_INVOKE".to_owned()),
            sub_type: Some("ChainedInvoke".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("ChainedInvoke"), None);
        assert!(
            result.is_err(),
            "Step claimed against CHAINED_INVOKE record must mismatch"
        );
    }

    /// Both spellings of every known operation type canonicalize to the
    /// same wire constant, and unknown types stay distinguishable.
    #[test]
    fn canonical_op_type_bridges_wire_and_pascal_spellings() {
        for (pascal, wire) in [
            ("Callback", "CALLBACK"),
            ("ChainedInvoke", "CHAINED_INVOKE"),
            ("Context", "CONTEXT"),
            ("Execution", "EXECUTION"),
            ("Step", "STEP"),
            ("Wait", "WAIT"),
        ] {
            assert_eq!(
                canonical_op_type(pascal),
                canonical_op_type(wire),
                "{pascal} and {wire} must canonicalize identically"
            );
        }
        assert_ne!(
            canonical_op_type("Step"),
            canonical_op_type("ChainedInvoke"),
            "distinct types must stay distinct"
        );
    }

    /// Builds a checkpoint record carrying full identity fields for the
    /// Some↔None conformance tests below.
    fn record_with_identity(
        op_type: Option<&str>,
        sub_type: Option<&str>,
        op_name: Option<&str>,
    ) -> CheckpointRecord {
        CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: op_type.map(ToOwned::to_owned),
            sub_type: sub_type.map(ToOwned::to_owned),
            due_at: None,
            op_name: op_name.map(ToOwned::to_owned),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        }
    }

    fn test_ctx() -> DurableContext {
        DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
        )
    }

    #[test]
    fn validate_replay_identity_fails_when_stored_name_removed() {
        // Record was checkpointed with a name; the claim carries none.
        // Removing `.name(...)` changes replay identity and must be flagged,
        // otherwise a reordered unnamed same-type operation could consume
        // this checkpoint silently.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("my-step"));

        let result = ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), None);
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("my-step"),
            "expected stored name in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_fails_when_name_added() {
        // Record was checkpointed without a name (but with identity); the
        // claim now carries one. The None↔Some direction is a mismatch too.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), None);

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("new-name"));
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("new-name"),
            "expected claimed name in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_fails_when_subtype_dropped() {
        // Sub-type is compared as a complete Option in both directions.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Context"), Some("Map"), None);

        let result = ctx.validate_replay_identity(&record, "wire-1", "Context", None, None);
        assert!(
            result.is_err(),
            "expected mismatch when stored sub-type is dropped from the claim"
        );

        // And the reverse direction: record without a sub-type, claim with one.
        let record = record_with_identity(Some("Context"), None, None);
        let result = ctx.validate_replay_identity(&record, "wire-1", "Context", Some("Map"), None);
        assert!(
            result.is_err(),
            "expected mismatch when a sub-type is added to the claim"
        );
    }

    #[test]
    fn validate_replay_identity_untyped_record_skips_name_and_subtype() {
        // A record genuinely lacking identity (no op_type) is the ONLY
        // lenient path: nothing is compared, whatever the claim carries.
        let ctx = test_ctx();
        let record = record_with_identity(None, None, None);

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("any-name"));
        assert!(result.is_ok());
    }

    #[test]
    fn validate_replay_identity_empty_claimed_name_matches_stored_none() {
        // An empty claimed name and a record with no `Name` are the same
        // identity. An unchanged handler must NOT be rejected on resume.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Context"), Some("MapIteration"), None);

        let result = ctx.validate_replay_identity(
            &record,
            "wire-1",
            "Context",
            Some("MapIteration"),
            Some(""),
        );
        assert!(
            result.is_ok(),
            "claimed empty name must match a stored None: {result:?}"
        );
    }

    #[test]
    fn validate_replay_identity_empty_stored_name_matches_claimed_none() {
        // The reverse direction: a backend that stored an empty name string
        // must match a claim carrying no name.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some(""));

        let result = ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), None);
        assert!(
            result.is_ok(),
            "stored empty name must match a claimed None: {result:?}"
        );
    }

    #[test]
    fn validate_replay_identity_empty_name_still_mismatches_real_name() {
        // Normalization must not weaken detection: an empty claim against a
        // real stored name (and the reverse) is still a mismatch.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("real"));
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some(""));
        assert!(
            result.is_err(),
            "empty claim vs stored 'real' must mismatch"
        );

        let record = record_with_identity(Some("Step"), Some("Step"), None);
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("real"));
        assert!(
            result.is_err(),
            "claimed 'real' vs stored None must mismatch"
        );
    }

    #[test]
    fn validate_replay_identity_mismatch_records_fatal_error() {
        // A mismatch must record the execution-fatal error on the shared
        // suspension-signal slot so the invocation driver fails the execution
        // even when the returned `Err` is swallowed downstream.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("alpha"));

        assert!(
            ctx.coordinator().fatal_error().is_none(),
            "no fatal recorded before validation"
        );
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("beta"));
        assert!(result.is_err());

        let fatal = ctx
            .coordinator()
            .fatal_error()
            .expect("mismatch must record a fatal error");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        assert!(
            fatal.error_message.contains("alpha") && fatal.error_message.contains("beta"),
            "fatal message must name both identities: {}",
            fatal.error_message
        );
    }

    #[test]
    fn validate_replay_identity_pass_records_no_fatal() {
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("same"));

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("same"));
        assert!(result.is_ok());
        assert!(
            ctx.coordinator().fatal_error().is_none(),
            "a passing validation must not record a fatal error"
        );
    }

    // Debug redaction tests

    #[test]
    fn debug_output_redacts_checkpoint_token() {
        let secret_token = "super-secret-credential-value-12345";
        let log = Arc::new(CheckpointLog::empty());
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:my-fn".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            secret_token.to_owned(),
        );

        let debug_output = format!("{ctx:?}");

        // The actual token value MUST NOT appear in debug output.
        assert!(
            !debug_output.contains(secret_token),
            "checkpoint_token value leaked in Debug output: {debug_output}"
        );

        // The redacted placeholder MUST appear.
        assert!(
            debug_output.contains("<redacted>"),
            "expected '<redacted>' in Debug output: {debug_output}"
        );

        // Useful fields MUST be present.
        assert!(
            debug_output.contains("arn:aws:lambda:us-east-1:123456789012:function:my-fn"),
            "expected execution_arn in Debug output: {debug_output}"
        );
    }

    /// Every context derived from the root shares the root's plugins: the
    /// child of `new_child` and the flat child of `new_flat_child`. The test
    /// compares the `Arc`s by address.
    #[test]
    fn derived_contexts_share_the_root_plugins() {
        let plugins = crate::plugin::test_support::invocation_plugins();
        let root = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::new(InMemoryExecutionClient::new(Vec::new())),
            "token-0".to_owned(),
            None,
            0,
            Some(Arc::clone(&plugins)),
        );
        let same = |ctx: &DurableContext| {
            ctx.plugins()
                .is_some_and(|shared| std::ptr::eq(shared, Arc::as_ptr(&plugins)))
        };
        assert!(same(&root), "the root holds the plugins");
        let child = root.new_child("1", None);
        assert!(same(&child), "new_child copies the plugins");
        assert!(
            same(&child.new_child("1-1", None)),
            "a grandchild shares them"
        );
        assert!(
            same(&root.new_flat_child("2", "parent", None)),
            "new_flat_child copies the plugins"
        );
        assert!(
            DurableContext::test_context().plugins().is_none(),
            "a root constructor without the parameter sets none"
        );
    }

    // Checkpoint coalescing tests

    /// Builds a root context with a checkpoint-coalescing delay, backed by
    /// the in-memory client.
    fn coalescing_ctx(client: Arc<dyn ExecutionClient>, delay: Duration) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-0".to_owned(),
            Some(delay),
            0,
            None,
        )
    }

    /// Helper: builds a bare step START update with the given wire ID.
    fn make_update(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id.to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(OperationAction::Start)
            .build()
            .expect("all required OperationUpdate fields set")
    }

    /// Without a Tokio runtime the batch flusher cannot start: every
    /// contributor receives an invocation-scoped error instead of waiting
    /// forever, and nothing is written.
    #[test]
    fn batch_flush_without_runtime_returns_invocation_error() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = coalescing_ctx(client.clone(), Duration::ZERO);
        let polled = crate::task::test_support::poll_once_without_runtime(move || async move {
            ctx.checkpoint_updates(vec![make_update("op-a")]).await
        });
        let std::task::Poll::Ready(Err(err)) = polled else {
            unreachable!("expected an immediate error, got {polled:?}");
        };
        assert_eq!(err.scope(), ExecutionClientErrorScope::Invocation);
        assert!(
            err.to_string().contains(crate::task::NO_RUNTIME_MESSAGE),
            "unexpected error: {err}"
        );
        assert!(client.recorded_updates().is_empty(), "nothing is written");
    }

    /// Two concurrent checkpoint calls inside the delay window coalesce
    /// into ONE client call carrying both updates.
    #[tokio::test(start_paused = true)]
    async fn coalesced_checkpoints_batch_into_one_call() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = coalescing_ctx(client.clone(), Duration::from_millis(50));

        let (a, b) = tokio::join!(
            ctx.checkpoint_updates(vec![make_update("op-a")]),
            ctx.checkpoint_updates(vec![make_update("op-b")]),
        );
        assert!(a.is_ok(), "first coalesced caller succeeds");
        assert!(b.is_ok(), "second coalesced caller succeeds");

        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 1, "both updates must share one API call");

        let ids: Vec<String> = client
            .recorded_updates()
            .iter()
            .map(|u| u.id.clone())
            .collect();
        assert_eq!(
            ids,
            vec!["op-a".to_owned(), "op-b".to_owned()],
            "the single call carries both updates in join order"
        );
    }

    /// An urgent checkpoint (the callback-creation flush point) does not
    /// wait out the delay window: it flushes immediately, carrying any
    /// previously buffered updates with it.
    #[tokio::test(start_paused = true)]
    async fn urgent_checkpoint_flushes_without_waiting_the_window() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        // A window long enough that waiting it out would be visible in
        // paused time.
        let ctx = coalescing_ctx(client.clone(), Duration::from_hours(1));

        let start = tokio::time::Instant::now();
        ctx.submit_checkpoint_updates_urgent(vec![make_update("op-urgent")])
            .await
            .expect("urgent checkpoint succeeds");

        assert_eq!(
            start.elapsed(),
            Duration::ZERO,
            "urgent flush must not wait for the coalescing window"
        );
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 1);
    }

    /// `flush_pending_checkpoints` (the suspension / end-of-invocation
    /// flush point) drains a buffered batch long before its window
    /// elapses, and the buffered caller observes the flushed result.
    #[tokio::test(start_paused = true)]
    async fn flush_pending_checkpoints_drains_the_buffer() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = coalescing_ctx(client.clone(), Duration::from_hours(1));

        let start = tokio::time::Instant::now();
        let buffered = tokio::spawn({
            let ctx = ctx.clone();
            async move {
                ctx.checkpoint_updates(vec![make_update("op-buffered")])
                    .await
            }
        });
        // Let the spawned caller join the batch (well under the window).
        tokio::time::sleep(Duration::from_millis(1)).await;

        ctx.flush_pending_checkpoints()
            .await
            .expect("flush succeeds");

        let result = buffered.await.expect("buffered caller task completes");
        assert!(result.is_ok(), "buffered caller observes the flush result");
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "flush must not wait for the one-hour window"
        );
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 1);
    }

    /// Without a configured delay, `checkpoint_updates` writes immediately:
    /// one call per checkpoint, exactly the pre-knob behavior.
    #[tokio::test(start_paused = true)]
    async fn no_delay_checkpoints_write_immediately() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client.clone(),
            "token-0".to_owned(),
        );

        let start = tokio::time::Instant::now();
        ctx.checkpoint_updates(vec![make_update("op-1")])
            .await
            .expect("first write succeeds");
        ctx.checkpoint_updates(vec![make_update("op-2")])
            .await
            .expect("second write succeeds");

        assert_eq!(start.elapsed(), Duration::ZERO);
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 2, "no coalescing without a configured delay");
    }

    /// A test client whose `checkpoint` calls block until released,
    /// delegating to an [`InMemoryExecutionClient`] once the gate opens.
    /// Lets tests hold a batched write "in flight" deterministically.
    #[derive(Debug)]
    struct GatedClient {
        inner: InMemoryExecutionClient,
        gate: tokio::sync::Semaphore,
        /// Snapshot of update-ID lists, one entry per checkpoint call, in
        /// call order.
        calls: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl GatedClient {
        fn new() -> Self {
            Self {
                inner: InMemoryExecutionClient::new(Vec::new()),
                gate: tokio::sync::Semaphore::new(0),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Allows one blocked (or future) checkpoint call to proceed.
        fn release_one(&self) {
            self.gate.add_permits(1);
        }

        fn call_ids(&self) -> Vec<Vec<String>> {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl ExecutionClient for GatedClient {
        fn checkpoint(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            let arn = execution_arn.to_owned();
            let token = checkpoint_token.to_owned();
            Box::pin(async move {
                // Block until the test releases the gate. The permit is
                // consumed (forgotten) so each call needs its own release.
                let permit = self.gate.acquire().await.map_err(|e| {
                    ExecutionClientError::non_retryable(format!("gate closed: {e}"))
                })?;
                permit.forget();
                self.calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(updates.iter().map(|u| u.id.clone()).collect());
                self.inner.checkpoint(&arn, &token, updates).await
            })
        }

        fn get_state(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            self.inner
                .get_state(execution_arn, checkpoint_token, marker)
        }
    }

    /// REGRESSION (flush contract): once a delay timer claims a batch and
    /// its write is in flight, `flush_pending_checkpoints` must NOT return
    /// until that write finishes: the wrapper reports
    /// PENDING/SUCCEEDED/FAILED right after the flush, and an unawaited
    /// in-flight checkpoint would cross that boundary.
    #[tokio::test(start_paused = true)]
    async fn flush_waits_for_in_flight_timer_claimed_batch() {
        let client = Arc::new(GatedClient::new());
        let ctx = coalescing_ctx(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Duration::from_millis(10),
        );

        // A buffered contributor: its delay timer fires at +10ms, claiming
        // the batch and starting a write that blocks on the gate.
        let contributor = tokio::spawn({
            let ctx = ctx.clone();
            async move {
                ctx.checkpoint_updates(vec![make_update("op-in-flight")])
                    .await
            }
        });
        // Advance past the window so the timer claims the batch and the
        // write task blocks inside the gated client.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // The end-of-invocation flush must now WAIT for that in-flight
        // write. Prove it stays pending while the gate is closed: in
        // paused time, the timeout can only fire because every task is
        // blocked on the (non-timer) gate.
        let mut flush = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.flush_pending_checkpoints().await }
        });
        let raced = tokio::time::timeout(Duration::from_mins(1), &mut flush).await;
        assert!(
            raced.is_err(),
            "flush returned while the timer-claimed batch write was still in flight — \
             the flush barrier is broken"
        );

        // Release the write; now both the contributor and the flush finish.
        client.release_one();
        flush
            .await
            .expect("flush task completes")
            .expect("flush succeeds");
        contributor
            .await
            .expect("contributor task completes")
            .expect("contributor observes the published write");
        assert_eq!(
            client.call_ids(),
            vec![vec!["op-in-flight".to_owned()]],
            "exactly the claimed batch was written, once"
        );
    }

    /// A test client that FAILS every checkpoint call retryably, holding
    /// the FIRST call at a gate so the test can order a second batch's
    /// queuing against the in-flight failing write. Calls after the first
    /// fail immediately (no gate), so a regression that performs an extra
    /// write shows up as an extra recorded call instead of a hang.
    #[derive(Debug)]
    struct GatedRetryableFailClient {
        gate: tokio::sync::Semaphore,
        first: std::sync::atomic::AtomicBool,
        /// Snapshot of update-ID lists, one entry per checkpoint call, in
        /// call order.
        calls: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl GatedRetryableFailClient {
        fn new() -> Self {
            Self {
                gate: tokio::sync::Semaphore::new(0),
                first: std::sync::atomic::AtomicBool::new(true),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Releases the gated first call so its rejection lands.
        fn release_first(&self) {
            self.gate.add_permits(1);
        }

        fn call_ids(&self) -> Vec<Vec<String>> {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }

        fn call_count(&self) -> usize {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len()
        }
    }

    impl ExecutionClient for GatedRetryableFailClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async move {
                self.calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(updates.iter().map(|u| u.id.clone()).collect());
                if self.first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    // Hold the first call in flight until the test
                    // releases it; the permit is consumed so a second
                    // call never blocks here.
                    if let Ok(permit) = self.gate.acquire().await {
                        permit.forget();
                    }
                }
                Err(ExecutionClientError::from_retryable(
                    "injected retryable checkpoint failure".to_owned(),
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                Err(ExecutionClientError::new_non_retryable(
                    "get_state not used in this test",
                ))
            })
        }
    }

    /// REGRESSION (issue #43 review): a flusher QUEUED behind a failing
    /// batch write must not perform another checkpoint call. Batch A's
    /// write is in flight (held at the client's gate) when batch B opens
    /// and queues its flusher on the writer lock; A then fails retryably.
    /// "Retryable exhaustion fails the invocation with no further
    /// writes": B's flusher must observe the failure latch under the
    /// writer lock, publish the prior error to its contributors, and
    /// retain its updates as unwritten: without calling the backend.
    /// Before the latch, B's flusher acquired the lock after A's failure
    /// and performed another write, persisting replay-visible transitions
    /// after the invocation was already doomed.
    #[tokio::test(start_paused = true)]
    async fn queued_batch_flush_after_retryable_failure_writes_nothing() {
        let client = Arc::new(GatedRetryableFailClient::new());
        let ctx = coalescing_ctx(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Duration::ZERO,
        );

        // Batch A: its zero-delay flusher claims the batch under the
        // writer lock and starts the write, which is held at the gate.
        let contributor_a = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-a")]).await }
        });
        while client.call_count() < 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // Batch B opens while A's write is in flight; its flusher queues
        // on the writer lock behind A.
        let contributor_b = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-b")]).await }
        });
        // Let B join the buffer and its flusher reach the writer lock.
        tokio::time::sleep(Duration::from_millis(5)).await;

        // A's held write now fails retryably; B's queued flusher runs
        // right after, under the same writer lock.
        client.release_first();

        let a = contributor_a.await.expect("contributor A task completes");
        let b = contributor_b.await.expect("contributor B task completes");
        let a_err = a.expect_err("batch A observes its own write failure");
        let b_err = b.expect_err("batch B observes the latched prior failure");
        assert!(a_err.is_retryable(), "A's injected failure is retryable");
        assert!(
            b_err.is_retryable(),
            "the latch republishes the prior retryable error, so B's \
             contributors route through the same fail-the-invocation path"
        );

        assert_eq!(
            client.call_ids(),
            vec![vec!["op-a".to_owned()]],
            "batch B must cause NO checkpoint call after batch A's \
             retryable failure — the channel is down and the invocation \
             is already doomed"
        );

        // Both failures are retained for the flush-point classification:
        // A's from the failed write, B's from the latch hit, each with
        // its own unwritten updates.
        let retained = ctx
            .take_retained_flush_failures()
            .await
            .expect("both failures are retained");
        let unwritten: Vec<Vec<String>> = retained
            .failures
            .iter()
            .map(|f| f.unwritten.iter().map(|u| u.id.clone()).collect())
            .collect();
        assert_eq!(
            unwritten,
            vec![vec!["op-a".to_owned()], vec!["op-b".to_owned()]],
            "each batch's updates are retained as unwritten, in order"
        );
    }

    /// A fan-out exceeding one request's operation-count limit is split
    /// into multiple ordered requests rather than one oversized call.
    #[tokio::test(start_paused = true)]
    async fn coalesced_fanout_splits_into_size_capped_requests() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let limits = BatchLimits {
            max_operations: 2,
            max_payload_bytes: usize::MAX,
        };
        let ctx = DurableContext::new_root_with_client_and_coalescer(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client.clone(),
            "token-0".to_owned(),
            CheckpointCoalescer::with_limits(Duration::from_millis(50), limits),
        );

        // Five contributors join one coalescing window.
        let results = tokio::join!(
            ctx.checkpoint_updates(vec![make_update("op-1")]),
            ctx.checkpoint_updates(vec![make_update("op-2")]),
            ctx.checkpoint_updates(vec![make_update("op-3")]),
            ctx.checkpoint_updates(vec![make_update("op-4")]),
            ctx.checkpoint_updates(vec![make_update("op-5")]),
        );
        for outcome in [results.0, results.1, results.2, results.3, results.4] {
            assert!(outcome.is_ok(), "every contributor observes success");
        }

        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(
            call_count, 3,
            "five updates under a two-op cap must go out as 2+2+1 requests"
        );
        let ids: Vec<String> = client
            .recorded_updates()
            .iter()
            .map(|u| u.id.clone())
            .collect();
        assert_eq!(
            ids,
            vec!["op-1", "op-2", "op-3", "op-4", "op-5"],
            "join order is preserved across the split requests"
        );
    }

    /// Pure batching mode (zero window): checkpoints arriving while an
    /// earlier write is in flight batch into ONE follow-up call, and no
    /// artificial delay is added anywhere.
    #[tokio::test(start_paused = true)]
    async fn batching_mode_batches_writes_behind_in_flight_call() {
        let client = Arc::new(GatedClient::new());
        let ctx = coalescing_ctx(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Duration::ZERO,
        );

        // First checkpoint drives immediately and blocks in the client.
        let first = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-first")]).await }
        });
        tokio::task::yield_now().await;

        // Two more checkpoints arrive while the first write is in flight:
        // they join the open batch behind the writer lock.
        let second = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-second")]).await }
        });
        let third = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-third")]).await }
        });
        // Let both join and block on the writer lock / gate.
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Release both calls (first write, then the batched follow-up).
        client.release_one();
        client.release_one();

        for task in [first, second, third] {
            task.await
                .expect("contributor task completes")
                .expect("contributor observes success");
        }

        assert_eq!(
            client.call_ids(),
            vec![
                vec!["op-first".to_owned()],
                vec!["op-second".to_owned(), "op-third".to_owned()],
            ],
            "the two checkpoints that arrived during the in-flight write \
             must batch into one follow-up call"
        );
    }

    // Lifecycle-event ownership regression tests
    //
    // The documented contract (`crate::observability`) guarantees that
    // every transition the service records emits its lifecycle event.
    // These tests pin the two ways the old contributor-owned emission
    // violated that: a contributor dropped after joining a buffered batch,
    // and a split batch whose earlier chunk persists before a later chunk
    // fails.

    /// A `MakeWriter` that captures subscriber output in a shared buffer.
    #[derive(Clone)]
    struct EventCaptureWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for EventCaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if let Ok(mut inner) = self.0.lock() {
                inner.extend_from_slice(buf);
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for EventCaptureWriter {
        type Writer = EventCaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Installs a JSON capture subscriber (all levels, flattened fields,
    /// span list included) and returns the shared buffer plus the guard.
    fn lifecycle_capture() -> (
        Arc<std::sync::Mutex<Vec<u8>>>,
        tracing::subscriber::DefaultGuard,
    ) {
        use tracing_subscriber::layer::SubscriberExt as _;
        let buffer = Arc::new(std::sync::Mutex::new(Vec::new()));
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(true)
            .with_writer(EventCaptureWriter(Arc::clone(&buffer)));
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let guard = tracing::subscriber::set_default(subscriber);
        (buffer, guard)
    }

    /// Parses the captured output into the JSON lifecycle-event lines with
    /// the given event name, returning each line's parsed JSON.
    fn captured_lifecycle_events(
        buffer: &Arc<std::sync::Mutex<Vec<u8>>>,
        event_name: &str,
    ) -> Vec<serde_json::Value> {
        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );
        output
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|v| {
                v.get("message").and_then(serde_json::Value::as_str) == Some(event_name)
                    && v.get("target").and_then(serde_json::Value::as_str)
                        == Some(crate::observability::TARGET)
            })
            .collect()
    }

    /// REGRESSION (event ownership, dropped contributor): a buffered
    /// contributor dropped after joining a batch, explicitly supported for
    /// `race`/`select_ok` losers, must not take its lifecycle events with
    /// it. The flush task persists the update, so the write path must emit
    /// the event, inside the span the contributor captured.
    ///
    /// The scenario retries a few times because `tracing`'s global
    /// max-level hint and callsite-interest caches are rebuilt whenever any
    /// concurrently running test registers or drops a subscriber, and a
    /// rebuild racing this test's `DEBUG`-level emission can drop the event
    /// before it reaches this test's (thread-local) subscriber. The
    /// persistence invariant is asserted on every attempt; a genuine
    /// emission regression fails all attempts.
    #[tokio::test(start_paused = true)]
    async fn dropped_buffered_contributor_still_emits_events_for_persisted_updates() {
        use crate::observability::event_names;

        let mut emitted = false;
        for _attempt in 0..5 {
            let (buffer, _guard) = lifecycle_capture();
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let ctx = coalescing_ctx(client.clone(), Duration::from_hours(1));

            // The contributor polls once inside its own span, far enough
            // to capture its events and join the coalescing batch, then is
            // dropped, exactly like a lost `race` branch.
            {
                let span = tracing::info_span!("contributor-span");
                let _entered = span.enter();
                let mut fut = Box::pin(ctx.checkpoint_updates(vec![make_update("op-dropped")]));
                let mut poll_cx = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(
                    Future::poll(fut.as_mut(), &mut poll_cx).is_pending(),
                    "the buffered contributor must be awaiting the batch"
                );
            } // `fut` dropped here, before any flush.

            // The end-of-invocation flush persists the batch anyway.
            ctx.flush_pending_checkpoints()
                .await
                .expect("flush persists the dropped contributor's update");

            let ids: Vec<String> = client
                .recorded_updates()
                .iter()
                .map(|u| u.id.clone())
                .collect();
            assert_eq!(
                ids,
                vec!["op-dropped".to_owned()],
                "the dropped contributor's update must still be persisted"
            );

            let events = captured_lifecycle_events(&buffer, event_names::OPERATION_STARTED);
            if events.is_empty() {
                // A concurrent subscriber rebuild dropped the event before
                // it reached this test's subscriber; retry the scenario.
                continue;
            }

            let matching: Vec<&serde_json::Value> = events
                .iter()
                .filter(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-dropped")
                })
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "the persisted transition must emit exactly one operation_started \
                 event even though its contributor was dropped; got {events:?}"
            );
            let spans = format!("{:?}", matching.first().and_then(|v| v.get("spans")));
            assert!(
                spans.contains("contributor-span"),
                "the event must carry the originating contributor's span context, \
                 got spans: {spans}"
            );
            emitted = true;
            break;
        }
        assert!(
            emitted,
            "no attempt emitted operation_started for the dropped contributor's \
             persisted update — telemetry for a persisted transition was lost"
        );
    }

    /// A client whose second (and any later) `checkpoint` call fails,
    /// delegating the first to an [`InMemoryExecutionClient`]. Lets tests
    /// persist an early batch chunk and then fail a later one.
    #[derive(Debug)]
    struct FailSecondCallClient {
        inner: InMemoryExecutionClient,
        calls: std::sync::Mutex<u32>,
    }

    impl FailSecondCallClient {
        fn new() -> Self {
            Self {
                inner: InMemoryExecutionClient::new(Vec::new()),
                calls: std::sync::Mutex::new(0),
            }
        }
    }

    impl ExecutionClient for FailSecondCallClient {
        fn checkpoint(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            let arn = execution_arn.to_owned();
            let token = checkpoint_token.to_owned();
            Box::pin(async move {
                let call_index = {
                    let mut calls = self
                        .calls
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *calls += 1;
                    *calls
                };
                if call_index >= 2 {
                    return Err(ExecutionClientError::new_non_retryable(
                        "injected failure on the second chunk",
                    ));
                }
                self.inner.checkpoint(&arn, &token, updates).await
            })
        }

        fn get_state(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            self.inner
                .get_state(execution_arn, checkpoint_token, marker)
        }
    }

    /// REGRESSION (event ownership, split batch): when a sealed batch splits
    /// into several requests and an earlier request persists before a later
    /// one fails, the persisted chunk's events must be emitted: the
    /// aggregate batch error published to contributors must not suppress
    /// telemetry for transitions the service actually recorded.
    ///
    /// Retries for the same reason as
    /// [`dropped_buffered_contributor_still_emits_events_for_persisted_updates`]:
    /// concurrent tests rebuilding `tracing`'s global caches can race a
    /// `DEBUG` emission. Hard invariants assert on every attempt.
    #[tokio::test(start_paused = true)]
    async fn split_batch_emits_events_for_chunks_persisted_before_a_failure() {
        use crate::observability::event_names;

        let mut emitted = false;
        for _attempt in 0..5 {
            let (buffer, _guard) = lifecycle_capture();
            let client = Arc::new(FailSecondCallClient::new());
            let limits = BatchLimits {
                max_operations: 1,
                max_payload_bytes: usize::MAX,
            };
            let ctx = DurableContext::new_root_with_client_and_coalescer(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::empty()),
                Arc::clone(&client) as Arc<dyn ExecutionClient>,
                "token-0".to_owned(),
                CheckpointCoalescer::with_limits(Duration::from_millis(50), limits),
            );

            // Two contributors share one batch; the one-op cap splits it
            // into two requests. The first persists, the second fails.
            let (first, second) = tokio::join!(
                ctx.checkpoint_updates(vec![make_update("op-persisted")]),
                ctx.checkpoint_updates(vec![make_update("op-rejected")]),
            );
            assert!(
                first.is_err() && second.is_err(),
                "every contributor observes the aggregate batch error"
            );

            let ids: Vec<String> = client
                .inner
                .recorded_updates()
                .iter()
                .map(|u| u.id.clone())
                .collect();
            assert_eq!(
                ids,
                vec!["op-persisted".to_owned()],
                "exactly the first chunk was persisted"
            );

            let events = captured_lifecycle_events(&buffer, event_names::OPERATION_STARTED);
            if events.is_empty() {
                // A concurrent subscriber rebuild dropped the event before
                // it reached this test's subscriber; retry the scenario.
                continue;
            }

            let persisted: Vec<&serde_json::Value> = events
                .iter()
                .filter(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-persisted")
                })
                .collect();
            assert_eq!(
                persisted.len(),
                1,
                "the persisted chunk must emit its operation_started event \
                 despite the later chunk's failure; got {events:?}"
            );
            assert!(
                !events.iter().any(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-rejected")
                }),
                "the rejected chunk recorded nothing, so it must emit nothing; \
                 got {events:?}"
            );
            emitted = true;
            break;
        }
        assert!(
            emitted,
            "no attempt emitted operation_started for the persisted chunk — \
             telemetry for a persisted transition was lost"
        );
    }

    /// REGRESSION (event ownership, failed pagination hydration): the
    /// checkpoint call can persist the transitions and *then* fail while
    /// hydrating paginated state through `get_state`. The persisted
    /// transitions' events must already be emitted by then, emission
    /// happens immediately after the service accepts the write, before the
    /// fallible pagination fetch, even though the caller observes an error.
    ///
    /// Retries for the same reason as
    /// [`dropped_buffered_contributor_still_emits_events_for_persisted_updates`]:
    /// concurrent tests rebuilding `tracing`'s global caches can race a
    /// `DEBUG` emission. Hard invariants assert on every attempt.
    #[tokio::test]
    async fn persisted_checkpoint_emits_events_when_pagination_hydration_fails() {
        use crate::observability::event_names;

        let mut emitted = false;
        for _attempt in 0..5 {
            let (buffer, _guard) = lifecycle_capture();
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            // The write itself succeeds but signals more pages, and the
            // follow-up get_state fetch fails.
            client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
                Vec::new(),
                "page-2-token".to_owned(),
            ));
            client.fail_get_state("injected get_state failure");

            let ctx = DurableContext::new_root_with_client(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::empty()),
                Arc::clone(&client) as Arc<dyn ExecutionClient>,
                "token-0".to_owned(),
            );

            let result = ctx
                .checkpoint_updates(vec![make_update("op-hydration")])
                .await;
            assert!(
                result.is_err(),
                "the caller must observe the pagination-hydration failure"
            );
            let ids: Vec<String> = client
                .recorded_updates()
                .iter()
                .map(|u| u.id.clone())
                .collect();
            assert_eq!(
                ids,
                vec!["op-hydration".to_owned()],
                "the transition was persisted before get_state failed"
            );

            let events = captured_lifecycle_events(&buffer, event_names::OPERATION_STARTED);
            if events.is_empty() {
                // A concurrent subscriber rebuild dropped the event before
                // it reached this test's subscriber; retry the scenario.
                continue;
            }

            let matching: Vec<&serde_json::Value> = events
                .iter()
                .filter(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-hydration")
                })
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "the persisted transition must emit exactly one operation_started \
                 event even though pagination hydration failed; got {events:?}"
            );
            emitted = true;
            break;
        }
        assert!(
            emitted,
            "no attempt emitted operation_started for the persisted transition — \
             telemetry was suppressed by a failed pagination fetch"
        );
    }

    // Checkpoint admission tests (abandoned owners).

    /// The bound on every await in the admission tests. A rejected write
    /// never resolves, so these awaits time out.
    const ADMISSION_BOUND: Duration = Duration::from_secs(1);

    /// A root context with batching on (`Some(Duration::ZERO)`), a
    /// coalescing delay, or no buffering (`None`).
    fn admission_root(
        client: &Arc<InMemoryExecutionClient>,
        window: Option<Duration>,
    ) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
            window,
            0,
            None,
        )
    }

    /// The wire IDs the client recorded, in order.
    fn recorded_ids(client: &InMemoryExecutionClient) -> Vec<String> {
        client
            .recorded_updates()
            .iter()
            .map(|u| u.id.clone())
            .collect()
    }

    /// Writes `id` from `item` after abandoning it, and asserts that the
    /// write never resolves and that nothing is recorded.
    async fn assert_abandoned_write_not_sent(window: Option<Duration>) {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, window);
        let item = root.new_child("1", None);
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let written = tokio::time::timeout(
            ADMISSION_BOUND,
            item.checkpoint_updates(vec![make_update("u")]),
        )
        .await;
        assert!(written.is_err(), "a rejected write never resolves");
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
    }

    #[tokio::test]
    async fn abandoned_owner_update_is_not_sent_buffered() {
        assert_abandoned_write_not_sent(Some(Duration::ZERO)).await;
    }

    #[tokio::test]
    async fn abandoned_owner_update_is_not_sent_direct() {
        assert_abandoned_write_not_sent(None).await;
    }

    #[tokio::test]
    async fn abandoned_owner_direct_write_is_rejected() {
        // Batching is on, but `checkpoint_updates_direct` skips the buffer.
        // So this rejection comes from the check under the token lock.
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, Some(Duration::ZERO));
        let item = root.new_child("1", None);
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let written = tokio::time::timeout(
            ADMISSION_BOUND,
            item.checkpoint_updates_direct(vec![make_update("u")]),
        )
        .await;
        assert!(written.is_err(), "a rejected direct write never resolves");
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
    }

    #[tokio::test]
    async fn nested_owner_follows_its_outer_item() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, Some(Duration::ZERO));
        let item = root.new_child("1", None);
        let inner = item.new_child("1-1", None);
        assert_eq!(
            inner.owners(),
            [
                crate::engine::compute_wire_id_public("1"),
                crate::engine::compute_wire_id_public("1-1"),
            ],
            "a nested context runs in its outer item, outermost first"
        );
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        let written = tokio::time::timeout(
            ADMISSION_BOUND,
            inner.checkpoint_updates(vec![make_update("u")]),
        )
        .await;
        assert!(written.is_err(), "the nested write never resolves");
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
    }

    #[tokio::test(start_paused = true)]
    async fn admitted_update_precedes_parent_final_update() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, Some(Duration::from_millis(50)));
        let item = root.new_child("1", None);
        let writer = tokio::spawn({
            let item = item.clone();
            async move { item.checkpoint_updates(vec![make_update("a")]).await }
        });
        // Yield until `a` is buffered. The delay window holds it there, so
        // it is admitted but not yet sent when the item is abandoned.
        let coalescer = Arc::clone(root.inner.coalescer.as_ref().expect("buffering is on"));
        let mut yields = 0;
        while coalescer.pending_len() == 0 {
            assert!(yields < 1_000, "the item update was never buffered");
            yields += 1;
            tokio::task::yield_now().await;
        }
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let parent = tokio::spawn({
            let root = root.clone();
            async move { root.checkpoint_updates(vec![make_update("p")]).await }
        });
        tokio::time::advance(Duration::from_millis(100)).await;
        tokio::time::timeout(ADMISSION_BOUND, writer)
            .await
            .expect("the admitted write resolves")
            .expect("the writer task completes")
            .expect("the admitted write succeeds");
        tokio::time::timeout(ADMISSION_BOUND, parent)
            .await
            .expect("the parent write resolves")
            .expect("the parent task completes")
            .expect("the parent write succeeds");
        assert_eq!(
            recorded_ids(&client),
            ["a", "p"],
            "the admitted item update reaches the service before the parent's"
        );
    }

    /// A client that holds its first checkpoint call until the test opens a
    /// gate, and records the token and the update IDs of every call. Call
    /// `n` returns the token `token-n`.
    #[derive(Debug, Default)]
    struct TokenGateClient {
        entered: tokio::sync::Notify,
        gate: tokio::sync::Notify,
        calls: std::sync::Mutex<Vec<(String, Vec<String>)>>,
    }

    impl TokenGateClient {
        fn calls(&self) -> Vec<(String, Vec<String>)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl ExecutionClient for TokenGateClient {
        fn checkpoint(
            &self,
            _arn: &str,
            token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            let token = token.to_owned();
            Box::pin(async move {
                let n = {
                    let mut calls = self.calls.lock().unwrap();
                    calls.push((token, updates.iter().map(|u| u.id.clone()).collect()));
                    calls.len()
                };
                if n == 1 {
                    self.entered.notify_one();
                    self.gate.notified().await;
                }
                Ok(CheckpointOutput {
                    checkpoint_token: Some(format!("token-{n}")),
                    updated_operations: Vec::new(),
                    next_marker: None,
                })
            })
        }

        fn get_state(
            &self,
            _arn: &str,
            _token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(GetStateOutput {
                    operations: Vec::new(),
                    next_marker: None,
                })
            })
        }
    }

    #[tokio::test]
    async fn admitted_direct_write_survives_cancelled_caller() {
        let client = Arc::new(TokenGateClient::default());
        let root = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
            None,
            0,
            None,
        );
        let item = root.new_child("1", None);
        let entered = client.entered.notified();
        let writer = tokio::spawn({
            let item = item.clone();
            async move { item.checkpoint_updates(vec![make_update("a")]).await }
        });
        tokio::time::timeout(ADMISSION_BOUND, entered)
            .await
            .expect("the client holds the item's write");
        writer.abort();
        assert!(
            writer
                .await
                .expect_err("the caller was aborted")
                .is_cancelled(),
            "the caller is cancelled while its write is in flight"
        );
        client.gate.notify_one();
        tokio::time::timeout(
            ADMISSION_BOUND,
            root.checkpoint_updates(vec![make_update("p")]),
        )
        .await
        .expect("the parent write resolves")
        .expect("the parent write succeeds");
        assert_eq!(
            client.calls(),
            [
                ("token-0".to_owned(), vec!["a".to_owned()]),
                ("token-1".to_owned(), vec!["p".to_owned()]),
            ],
            "the cancelled caller's write completed and rotated the token"
        );
    }

    #[tokio::test]
    async fn abandoned_owner_starts_no_operation() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, Some(Duration::ZERO));
        let item = root.new_child("1", None);
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        let stepped = tokio::time::timeout(
            ADMISSION_BOUND,
            item.step(move |_| async move {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(1_u32)
            })
            .into_future(),
        )
        .await;
        assert!(stepped.is_err(), "the operation never resolves");
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "the step body never ran"
        );
        assert!(
            client.recorded_updates().is_empty(),
            "no update is recorded"
        );
    }

    /// An abandoned owner does not even replay a recorded operation, which
    /// writes nothing: the builder future is never polled past the check.
    #[tokio::test]
    async fn abandoned_owner_replays_no_operation() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let wire = crate::engine::compute_wire_id_public("1-1");
        let record = CheckpointRecord {
            id: wire.clone(),
            status: CheckpointStatus::Succeeded,
            result: Some("7".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        let log = CheckpointLog::from_records(vec![(wire, record)]);
        let root = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
            Some(Duration::ZERO),
            0,
            None,
        );
        let item = root.new_child("1", None);
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let stepped = tokio::time::timeout(
            ADMISSION_BOUND,
            item.step(|_| async { Ok(0_u32) }).into_future(),
        )
        .await;
        assert!(
            stepped.is_err(),
            "the recorded step is not replayed for an abandoned owner: {stepped:?}"
        );
    }

    /// A builder created before its owner is abandoned, and finalized after,
    /// is not checked against a contradicting record. The record says the
    /// operation was a wait, and the builder claims a step. For an owner that
    /// is not abandoned, the same claim records a fatal error and resolves.
    /// For the abandoned owner, nothing is recorded, nothing is written, and
    /// the future never resolves.
    #[tokio::test]
    async fn abandoned_owner_skips_replay_preflight() {
        let wait_record = |positional: &str| {
            let wire = crate::engine::compute_wire_id_public(positional);
            let record = CheckpointRecord {
                id: wire.clone(),
                status: CheckpointStatus::Succeeded,
                result: None,
                error_type: None,
                error_message: None,
                error_data: None,
                stack_trace: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: None,
                op_type: Some("Wait".to_owned()),
                sub_type: Some("Wait".to_owned()),
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
            };
            (wire, record)
        };
        let build_root = || {
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let log = CheckpointLog::from_records(vec![wait_record("1-1")]);
            let root = DurableContext::new_root_with_client_and_defaults(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(log),
                Arc::clone(&client) as Arc<dyn ExecutionClient>,
                "token-0".to_owned(),
                Some(Duration::ZERO),
                0,
                None,
            );
            (client, root)
        };

        // Control: an owner that is not abandoned records the mismatch.
        let (_, control_root) = build_root();
        let control_item = control_root.new_child("1", None);
        let control = tokio::time::timeout(
            ADMISSION_BOUND,
            control_item.step(|_| async { Ok(0_u32) }).into_future(),
        )
        .await
        .expect("the mismatch resolves at once");
        assert!(control.is_err(), "the mismatch is an error: {control:?}");
        assert!(
            control_item.coordinator().fatal_error().is_some(),
            "the mismatch records a fatal error"
        );

        // The builder exists before the abandonment and is finalized after it.
        let (client, root) = build_root();
        let item = root.new_child("1", None);
        let builder = item.step(|_| async { Ok(0_u32) });
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let stepped = tokio::time::timeout(ADMISSION_BOUND, builder.into_future()).await;
        assert!(
            stepped.is_err(),
            "the abandoned owner's future never resolves: {stepped:?}"
        );
        assert!(
            item.coordinator().fatal_error().is_none(),
            "no fatal error is recorded for the abandoned owner"
        );
        assert!(
            root.coordinator().fatal_error().is_none(),
            "no fatal error reaches the root"
        );
        assert!(client.recorded_updates().is_empty(), "nothing is written");
    }

    /// An abandoned owner does not replay a recorded callback either. The
    /// recorded `Started` callback resolves without a write, so only the
    /// admission check at the terminal can hold it back.
    #[tokio::test]
    async fn abandoned_owner_replays_no_callback() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let wire = crate::engine::compute_wire_id_public("1-1");
        let record = CheckpointRecord {
            id: wire.clone(),
            status: CheckpointStatus::Started,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: Some("cb-1".to_owned()),
            op_type: None,
            sub_type: None,
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire, record)]));
        let root = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
            Some(Duration::ZERO),
            0,
            None,
        );
        // The control: an owner that is not abandoned replays the callback.
        let control = root.new_child("1", None);
        let replayed = tokio::time::timeout(
            ADMISSION_BOUND,
            control.create_callback::<String>().into_future(),
        )
        .await
        .expect("the recorded callback replays")
        .expect("the replay succeeds");
        assert_eq!(replayed.id(), "cb-1");

        let item = root.new_child("1", None);
        root.admission()
            .abandon(item.owners().iter().map(String::as_str));
        let created = tokio::time::timeout(
            ADMISSION_BOUND,
            item.create_callback::<String>().into_future(),
        )
        .await;
        assert!(
            created.is_err(),
            "the recorded callback is not replayed for an abandoned owner"
        );
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
    }

    #[tokio::test]
    async fn admitted_owner_writes_as_before() {
        for window in [Some(Duration::ZERO), None] {
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let root = admission_root(&client, window);
            let item = root.new_child("1", None);
            tokio::time::timeout(
                ADMISSION_BOUND,
                item.checkpoint_updates(vec![make_update("u")]),
            )
            .await
            .expect("the write resolves")
            .expect("the write succeeds");
            assert_eq!(recorded_ids(&client), ["u"], "window {window:?}");
        }
    }

    /// Without a Tokio runtime a direct write cannot start its task. The
    /// caller gets the no-runtime error at once, and nothing is sent.
    #[test]
    fn direct_write_without_runtime_returns_invocation_error() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = admission_root(&client, None);
        let polled = crate::task::test_support::poll_once_without_runtime(move || async move {
            ctx.checkpoint_updates(vec![make_update("u")]).await
        });
        let std::task::Poll::Ready(Err(err)) = polled else {
            unreachable!("expected an immediate error, got {polled:?}");
        };
        assert_eq!(err.scope(), ExecutionClientErrorScope::Invocation);
        assert!(
            err.to_string().contains(crate::task::NO_RUNTIME_MESSAGE),
            "unexpected error: {err}"
        );
        assert!(
            std::error::Error::source(&err)
                .is_some_and(<dyn std::error::Error>::is::<crate::task::NoRuntime>),
            "the error carries the no-runtime marker"
        );
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
    }

    /// An operation whose direct write finds no Tokio runtime returns the
    /// error `.spawn()` returns without a runtime, at once. Nothing is
    /// sent, and no invocation fault is recorded.
    #[test]
    fn operation_without_runtime_returns_spawn_error() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let thread_client = Arc::clone(&client);
        // The context is built on the polling thread, so that thread owns it.
        let polled = crate::task::test_support::poll_once_without_runtime(move || async move {
            let ctx = admission_root(&thread_client, None);
            let signal = Arc::clone(ctx.coordinator());
            (
                ctx.step(|_| async { Ok(1_u32) }).into_future().await,
                signal,
            )
        });
        let std::task::Poll::Ready((Err(err), signal)) = polled else {
            unreachable!("expected an immediate error, got {polled:?}");
        };
        let spawn_error = crate::task::NoRuntime::operation_error(crate::task::NoRuntime);
        assert_eq!(format!("{err:#}"), format!("{spawn_error:#}"));
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
        assert!(signal.fatal_error().is_none(), "no fault is recorded");
    }

    /// A replayed `Started` child skips its START write, so its body runs
    /// and its first write is the terminal `FAIL`. When that direct write
    /// finds no Tokio runtime, the child returns the error `.spawn()`
    /// returns without a runtime. Nothing is sent, and no invocation fault
    /// is recorded.
    #[test]
    fn replayed_child_failure_without_runtime_returns_spawn_error() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let wire = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire.clone(),
            status: CheckpointStatus::Started,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Context".to_owned()),
            sub_type: Some(crate::child::CHILD_SUB_TYPE.to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        let thread_client = Arc::clone(&client);
        // The context is built on the polling thread, so that thread owns it.
        let polled = crate::task::test_support::poll_once_without_runtime(move || async move {
            let ctx = DurableContext::new_root_with_client_and_defaults(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::from_records(vec![(wire, record)])),
                thread_client as Arc<dyn ExecutionClient>,
                "token-0".to_owned(),
                None,
                0,
                None,
            );
            let signal = Arc::clone(ctx.coordinator());
            let outcome = ctx
                .run_in_child_context(|_child| async { Err::<u32, BoxError>("boom".into()) })
                .into_future()
                .await;
            (outcome, signal)
        });
        let std::task::Poll::Ready((Err(err), signal)) = polled else {
            unreachable!("expected an immediate error, got {polled:?}");
        };
        let spawn_error = crate::task::NoRuntime::operation_error(crate::task::NoRuntime);
        assert_eq!(format!("{err:#}"), format!("{spawn_error:#}"));
        assert!(client.recorded_updates().is_empty(), "nothing is sent");
        assert!(signal.fatal_error().is_none(), "no fault is recorded");
    }

    /// A client whose checkpoint call panics.
    #[derive(Debug)]
    struct PanickingClient;

    impl ExecutionClient for PanickingClient {
        fn checkpoint(
            &self,
            _arn: &str,
            _token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            #[expect(clippy::panic)] // reason: the panic is the behavior under test
            Box::pin(async { panic!("boom in checkpoint") })
        }

        fn get_state(
            &self,
            _arn: &str,
            _token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(GetStateOutput {
                    operations: Vec::new(),
                    next_marker: None,
                })
            })
        }
    }

    /// Writes one update directly through `client`, and returns the
    /// coordinator's kept panic after the write task stopped. The caller
    /// must stay pending: the write channel carries no panic payload.
    async fn direct_write_kept_panic(
        client: Arc<dyn ExecutionClient>,
    ) -> Box<dyn std::any::Any + Send + 'static> {
        let root = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-0".to_owned(),
            None,
            0,
            None,
        );
        let coordinator = Arc::clone(root.coordinator());
        let mut write = std::pin::pin!(root.checkpoint_updates(vec![make_update("u")]));
        let waited = tokio::time::timeout(ADMISSION_BOUND, async {
            tokio::select! {
                _ = &mut write => false,
                () = async {
                    // Wait until the write task has stopped, then poll the
                    // caller once more.
                    tokio::task::yield_now().await;
                    coordinator.tasks().wait_idle().await;
                } => true,
            }
        })
        .await
        .expect("the write task stops");
        assert!(waited, "the caller resolved");
        let polled =
            std::future::poll_fn(|cx| std::task::Poll::Ready(write.as_mut().poll(cx))).await;
        assert!(
            polled.is_pending(),
            "the caller stays pending after the client panicked"
        );
        coordinator.take_panic().expect("the panic is kept")
    }

    /// An error source whose destructor panics with "source drop".
    #[derive(Debug)]
    struct DropPanickingSource;

    impl std::fmt::Display for DropPanickingSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("drop-panicking source")
        }
    }

    impl std::error::Error for DropPanickingSource {}

    impl Drop for DropPanickingSource {
        #[expect(clippy::panic)] // reason: the panic is the behavior under test
        fn drop(&mut self) {
            panic!("source drop");
        }
    }

    /// A client whose checkpoint call waits for `release`, then fails with
    /// an error whose source panics when dropped.
    #[derive(Debug)]
    struct GatedFailingClient {
        release: tokio::sync::Notify,
    }

    impl ExecutionClient for GatedFailingClient {
        fn checkpoint(
            &self,
            _arn: &str,
            _token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                self.release.notified().await;
                Err(ExecutionClientError::with_source(
                    ExecutionClientErrorScope::Invocation,
                    "gated failure",
                    DropPanickingSource,
                ))
            })
        }

        fn get_state(
            &self,
            _arn: &str,
            _token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(GetStateOutput {
                    operations: Vec::new(),
                    next_marker: None,
                })
            })
        }
    }

    /// The caller of a direct write is dropped before the write returns. The
    /// write task drops the unread result inside a catch, and the panic of
    /// its destructor is recorded on the coordinator.
    #[tokio::test]
    async fn unread_direct_write_result_drop_panic_is_recorded() {
        let client = Arc::new(GatedFailingClient {
            release: tokio::sync::Notify::new(),
        });
        let root = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
            None,
            0,
            None,
        );
        let coordinator = Arc::clone(root.coordinator());
        let mut write = Box::pin(root.checkpoint_updates(vec![make_update("u")]));
        let polled =
            std::future::poll_fn(|cx| std::task::Poll::Ready(write.as_mut().poll(cx))).await;
        assert!(polled.is_pending());
        drop(write);
        client.release.notify_one();
        tokio::time::timeout(ADMISSION_BOUND, coordinator.tasks().wait_idle())
            .await
            .expect("the write task stops");
        let payload = coordinator
            .take_panic()
            .expect("the destructor's panic is recorded");
        assert_eq!(
            crate::task::test_support::str_payload(payload.as_ref()),
            Some("source drop")
        );
    }

    /// A panic of the checkpoint client in a direct write's task is kept
    /// by the coordinator with its original payload, and the caller stays
    /// pending.
    #[tokio::test]
    async fn direct_write_panic_is_recorded_and_caller_stays_pending() {
        let payload = direct_write_kept_panic(Arc::new(PanickingClient)).await;
        assert_eq!(
            crate::task::test_support::str_payload(payload.as_ref()),
            Some("boom in checkpoint")
        );
    }

    /// A checkpoint future that returns `Ready` and then panics in its
    /// destructor.
    struct ReadyThenDropPanic;

    impl Future for ReadyThenDropPanic {
        type Output = Result<CheckpointOutput, ExecutionClientError>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Ready(Ok(CheckpointOutput {
                checkpoint_token: Some("token-1".to_owned()),
                updated_operations: Vec::new(),
                next_marker: None,
            }))
        }
    }

    impl Drop for ReadyThenDropPanic {
        #[expect(clippy::panic)] // reason: the panic is the behavior under test
        fn drop(&mut self) {
            panic!("boom in checkpoint drop");
        }
    }

    /// A client whose checkpoint future panics in its destructor.
    #[derive(Debug)]
    struct DropPanickingClient;

    impl ExecutionClient for DropPanickingClient {
        fn checkpoint(
            &self,
            _arn: &str,
            _token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(ReadyThenDropPanic)
        }

        fn get_state(
            &self,
            _arn: &str,
            _token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(GetStateOutput {
                    operations: Vec::new(),
                    next_marker: None,
                })
            })
        }
    }

    /// A panic from the destructor of a checkpoint future that returned
    /// `Ready` is kept by the coordinator with its original payload, and the
    /// caller of the direct write stays pending.
    #[tokio::test]
    async fn direct_write_drop_panic_is_recorded_and_caller_stays_pending() {
        let payload = direct_write_kept_panic(Arc::new(DropPanickingClient)).await;
        assert_eq!(
            crate::task::test_support::str_payload(payload.as_ref()),
            Some("boom in checkpoint drop")
        );
    }

    #[tokio::test]
    async fn child_context_owner_covers_its_descendants() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, Some(Duration::ZERO));
        let child = root.new_child("1", None);
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        let written = tokio::time::timeout(
            ADMISSION_BOUND,
            child.checkpoint_updates(vec![make_update("u")]),
        )
        .await;
        assert!(written.is_err(), "the child's write never resolves");
        tokio::time::timeout(
            ADMISSION_BOUND,
            root.checkpoint_updates(vec![make_update("p")]),
        )
        .await
        .expect("the root write resolves")
        .expect("the root write succeeds");
        assert_eq!(recorded_ids(&client), ["p"], "only the root update is sent");
    }

    #[tokio::test]
    async fn finished_child_rejects_escaped_descendant_write() {
        use aws_sdk_lambda::types::OperationAction;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = admission_root(&client, Some(Duration::ZERO));
        let slot: Arc<std::sync::Mutex<Option<DurableFuture<u32>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let release = Arc::new(tokio::sync::Notify::new());
        let step_wire = crate::engine::compute_wire_id_public("1-1");
        let child_wire = crate::engine::compute_wire_id_public("1");

        let returned = tokio::time::timeout(ADMISSION_BOUND, {
            let slot = Arc::clone(&slot);
            let release = Arc::clone(&release);
            let client = Arc::clone(&client);
            let step_wire = step_wire.clone();
            root.run_in_child_context(move |child| async move {
                let handle = child
                    .step(move |_| async move {
                        release.notified().await;
                        Ok(7_u32)
                    })
                    .name("s")
                    .spawn();
                *slot.lock().unwrap() = Some(handle);
                let mut yields = 0;
                while !client
                    .recorded_updates()
                    .iter()
                    .any(|u| u.id == step_wire && u.action == OperationAction::Start)
                {
                    if yields == 1_000 {
                        return Err("the step START was never recorded".into());
                    }
                    yields += 1;
                    tokio::task::yield_now().await;
                }
                Ok(1_u32)
            })
            .into_future()
        })
        .await
        .expect("the child returns")
        .expect("the child succeeds");
        assert_eq!(returned, 1);

        release.notify_one();
        let handle = slot
            .lock()
            .unwrap()
            .take()
            .expect("the body stored the handle");
        let awaited = tokio::time::timeout(ADMISSION_BOUND, handle).await;
        assert!(awaited.is_err(), "the escaped operation never resolves");

        let updates = client.recorded_updates();
        let has = |id: &str, action: OperationAction| {
            updates.iter().any(|u| u.id == id && u.action == action)
        };
        assert!(
            has(&step_wire, OperationAction::Start),
            "s START is recorded"
        );
        assert!(
            has(&child_wire, OperationAction::Succeed),
            "the child's SUCCEED is recorded"
        );
        assert!(
            !has(&step_wire, OperationAction::Succeed),
            "s SUCCEED is never sent after the child returned"
        );
    }

    // Tenant ID on the root and child spans.

    /// Emits one record under the root span of a context built from a
    /// Lambda context with `tenant_id`, and one under a child body's span,
    /// and returns the span lists of the two records.
    fn tenant_span_lists(
        tenant_id: Option<&str>,
    ) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
        use crate::tracing_layer::tests::{json_records, json_span_subscriber, spans_of};

        let buffer = Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = json_span_subscriber(&buffer);
        tracing::subscriber::with_default(subscriber, || {
            let mut lambda_context = lambda_runtime::Context::default();
            lambda_context.tenant_id = tenant_id.map(str::to_owned);
            let root = DurableContext::new_root(
                "arn:test".to_owned(),
                lambda_context,
                Arc::new(CheckpointLog::empty()),
            );
            root.replay_span().in_scope(|| tracing::info!("in-root"));
            let child = root.new_child("1", Some("child"));
            child.replay_span().in_scope(|| tracing::info!("in-child"));
        });
        let records = json_records(&buffer);
        (
            spans_of(&records, "in-root"),
            spans_of(&records, "in-child"),
        )
    }

    #[test]
    fn tenant_id_on_root_and_child_spans() {
        use crate::tracing_layer::tests::has_field;

        let tenant = serde_json::Value::String("t-1".to_owned());
        let (root, child) = tenant_span_lists(Some("t-1"));
        assert!(
            has_field(&root, crate::tracing_layer::fields::TENANT_ID, &tenant),
            "{root:?}"
        );
        assert!(
            has_field(&child, crate::tracing_layer::fields::TENANT_ID, &tenant),
            "{child:?}"
        );
    }

    #[test]
    fn no_tenant_id_without_tenant() {
        use crate::tracing_layer::tests::any_field;

        let (root, child) = tenant_span_lists(None);
        assert!(
            !any_field(&root, crate::tracing_layer::fields::TENANT_ID),
            "{root:?}"
        );
        assert!(
            !any_field(&child, crate::tracing_layer::fields::TENANT_ID),
            "{child:?}"
        );
        // The record is still under the SDK's spans.
        assert!(
            any_field(&child, crate::tracing_layer::fields::OPERATION_ID),
            "{child:?}"
        );
    }
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod coordinator_tests {
    use super::{DurableContext, remaining_until};
    use crate::client::ExecutionClient;
    use crate::coordinator::{InvocationCoordinator, PermitKind, Phase};
    use crate::driver::test_support::{ScriptedClient, operation_like};
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use crate::task::test_support::{Schedule, ScheduleState};
    use aws_sdk_lambda::types::{OperationAction, OperationStatus, OperationType, OperationUpdate};
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, SystemTime};

    fn ctx_with(log: Arc<CheckpointLog>, client: &Arc<ScriptedClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
        )
    }

    #[test]
    fn remaining_time_cases() {
        let now = SystemTime::now();
        let now_ms = |offset: i64| {
            let base = now
                .duration_since(std::time::UNIX_EPOCH)
                .expect("after the epoch")
                .as_millis();
            u64::try_from(i128::try_from(base).expect("fits") + i128::from(offset))
                .expect("non-negative")
        };
        assert_eq!(remaining_until(0, now), None);
        let ahead = remaining_until(now_ms(5_000), now).expect("a deadline");
        assert!(ahead > Duration::from_secs(4) && ahead <= Duration::from_secs(5));
        assert_eq!(remaining_until(now_ms(-1_000), now), Some(Duration::ZERO));

        // The context reads its own Lambda deadline.
        let ctx = DurableContext::test_context();
        assert_eq!(ctx.remaining_time(), None, "the default deadline is 0");
    }

    /// A wait on a missing record starts its schedule as a `STARTED` record
    /// with no end time: a first refresh after 1 s, and a backoff when the
    /// refresh finds the record `STARTED`.
    #[tokio::test(start_paused = true)]
    async fn wait_for_record_without_record_starts_as_started() {
        let ctx = DurableContext::test_context();
        let coordinator = Arc::clone(ctx.coordinator());
        let registered = tokio::time::Instant::now();
        let wire = "missing".to_owned();
        let mut wait = Box::pin(ctx.wait_for_record(&wire, |s| *s == CheckpointStatus::Succeeded));
        let polled = futures_poll_once(wait.as_mut());
        assert!(polled.is_pending());
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(registered + Duration::from_secs(1))
        );
        tokio::time::advance(Duration::from_secs(1)).await;
        let crate::coordinator::DueRefresh::Due { keys, .. } =
            coordinator.take_due(tokio::time::Instant::now())
        else {
            unreachable!("the registration is due");
        };
        let records =
            std::collections::BTreeMap::from([(wire.clone(), (CheckpointStatus::Started, None))]);
        coordinator.after_refresh(&keys, &records);
        assert_eq!(
            coordinator.next_refresh_at(),
            Some(registered + Duration::from_secs(2)),
            "an unchanged STARTED record backs off"
        );
    }

    /// Polls `fut` once with a no-op waker.
    fn futures_poll_once<F: Future + ?Sized>(
        fut: std::pin::Pin<&mut F>,
    ) -> std::task::Poll<F::Output> {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        fut.poll(&mut cx)
    }

    #[test]
    fn wait_for_record_ready_before_registration() {
        let wire = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire.clone(),
            CheckpointRecord::with_status(&wire, CheckpointStatus::Succeeded),
        )]));
        let client = Arc::new(ScriptedClient::empty());
        let ctx = ctx_with(log, &client);
        let coordinator = Arc::clone(ctx.coordinator());
        let schedule = Schedule::idle(&coordinator);
        let mut wait = Box::pin(ctx.wait_for_record(&wire, |s| *s != CheckpointStatus::Started));
        assert!(schedule.poll(wait.as_mut()).is_ready());
        assert_eq!(coordinator.live_waits(), 0);
        assert_eq!(client.call_count(), 0);
    }

    /// What the readiness hook of the next test changes on its first call:
    /// the log, the coordinator, and the wire ID.
    type Hook = Option<(Arc<CheckpointLog>, Arc<InvocationCoordinator>, String)>;

    thread_local! {
        static HOOK: RefCell<Hook> = const { RefCell::new(None) };
    }

    /// On its first call: marks the record `SUCCEEDED`, wakes its
    /// registration, as a merge between the read and the end of the poll
    /// does, and returns `false`. Later calls: ready once the record left
    /// `STARTED`.
    fn merge_during_first_read(status: &CheckpointStatus) -> bool {
        if let Some((log, coordinator, wire)) = HOOK.with(|hook| hook.borrow_mut().take()) {
            log.insert(
                wire.clone(),
                CheckpointRecord::with_status(&wire, CheckpointStatus::Succeeded),
            );
            coordinator.wake_operations(&[wire]);
            return false;
        }
        *status != CheckpointStatus::Started
    }

    #[test]
    fn wait_for_record_merge_after_read_is_not_lost() {
        let wire = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire.clone(),
            CheckpointRecord::with_status(&wire, CheckpointStatus::Started),
        )]));
        let client = Arc::new(ScriptedClient::empty());
        let ctx = ctx_with(Arc::clone(&log), &client);
        let coordinator = Arc::clone(ctx.coordinator());
        HOOK.with(|hook| {
            *hook.borrow_mut() = Some((Arc::clone(&log), Arc::clone(&coordinator), wire.clone()));
        });
        let schedule = Schedule::idle(&coordinator);
        let mut wait = Box::pin(ctx.wait_for_record(&wire, merge_during_first_read));
        assert!(schedule.poll(wait.as_mut()).is_pending());
        assert_eq!(
            schedule.state(),
            ScheduleState::Queued,
            "the wake during the poll keeps the task queued"
        );
        assert_eq!(coordinator.permits(PermitKind::Activity), 1);
        assert_eq!(coordinator.phase(), Phase::Running);
        assert!(schedule.poll(wait.as_mut()).is_ready());
        assert_eq!(
            client.call_count(),
            0,
            "no checkpoint call between the polls"
        );
        assert_eq!(coordinator.live_waits(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_write_holds_token_lock_through_response_merge() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let ctx_slot: Arc<std::sync::OnceLock<DurableContext>> =
            Arc::new(std::sync::OnceLock::new());
        let a_merged_at_b = Arc::new(AtomicBool::new(false));
        let responder_ctx = Arc::clone(&ctx_slot);
        let responder_flag = Arc::clone(&a_merged_at_b);
        let client = Arc::new(
            ScriptedClient::new(move |updates, history| {
                if history.len() == updates.len() {
                    // A's call: the response carries operation `a`.
                    return updates
                        .iter()
                        .map(|u| {
                            operation_like(u, OperationStatus::Succeeded)
                                .build()
                                .expect("all required fields set")
                        })
                        .collect();
                }
                let merged = responder_ctx
                    .get()
                    .is_some_and(|ctx| ctx.has_checkpoint_record("1"));
                responder_flag.store(merged, Ordering::SeqCst);
                Vec::new()
            })
            .with_first_call_gate(Arc::clone(&gate)),
        );
        let ctx = ctx_with(Arc::new(CheckpointLog::empty()), &client);
        ctx_slot.set(ctx.clone()).expect("set once");
        let update = |positional: &str| {
            OperationUpdate::builder()
                .id(crate::engine::compute_wire_id_public(positional))
                .r#type(OperationType::Step)
                .sub_type("Step")
                .action(OperationAction::Start)
                .build()
                .expect("all required fields set")
        };
        let a_ctx = ctx.clone();
        let a_update = update("1");
        let a = tokio::spawn(async move { a_ctx.checkpoint_updates_direct(vec![a_update]).await });
        for _ in 0..1000 {
            if client.call_count() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(client.call_count(), 1, "A's call was received");
        let b_ctx = ctx.clone();
        let b_update = update("2");
        let b = tokio::spawn(async move { b_ctx.checkpoint_updates_direct(vec![b_update]).await });
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(client.call_count(), 1, "B waits for A's merge");
        gate.notify_one();
        a.await.expect("A does not panic").expect("A succeeds");
        b.await.expect("B does not panic").expect("B succeeds");
        let calls = client.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls.get(1).map(|(token, _)| token.as_str()),
            Some("t1"),
            "B's call carries the token A's response rotated in"
        );
        assert!(
            a_merged_at_b.load(Ordering::SeqCst),
            "operation a was merged before B's call was sent"
        );
    }

    #[tokio::test]
    async fn hold_activity_after_suspension_never_resolves() {
        let ctx = DurableContext::test_context();
        ctx.coordinator().finish();
        let ran = Arc::new(AtomicBool::new(false));
        let held = Arc::clone(&ran);
        let waited = tokio::time::timeout(
            Duration::from_millis(100),
            ctx.hold_activity(async move { held.store(true, Ordering::SeqCst) }),
        )
        .await;
        assert!(waited.is_err(), "the held future never resolves");
        assert!(!ran.load(Ordering::SeqCst), "the work never ran");
    }

    #[tokio::test]
    async fn direct_write_after_suspension_is_rejected() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = ctx_with(Arc::new(CheckpointLog::empty()), &client);
        ctx.coordinator().finish();
        let update = OperationUpdate::builder()
            .id("u")
            .r#type(OperationType::Step)
            .action(OperationAction::Start)
            .build()
            .expect("all required fields set");
        let written = tokio::time::timeout(
            Duration::from_millis(100),
            ctx.checkpoint_updates_direct(vec![update]),
        )
        .await;
        assert!(written.is_err(), "the rejected write never resolves");
        assert_eq!(client.call_count(), 0, "nothing was sent");
    }

    #[tokio::test]
    async fn wait_for_record_after_suspension_never_resolves() {
        let wire = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire.clone(),
            CheckpointRecord::with_status(&wire, CheckpointStatus::Succeeded),
        )]));
        let ctx = ctx_with(log, &Arc::new(ScriptedClient::empty()));
        ctx.coordinator().finish();
        let waited = tokio::time::timeout(
            Duration::from_millis(100),
            ctx.wait_for_record(&wire, |s| *s != CheckpointStatus::Started),
        )
        .await;
        assert!(waited.is_err(), "a refused registration never resolves");
    }

    /// A `RETRY` update of `positional` with sub-type `sub_type`.
    fn retry_update(positional: &str, sub_type: &str, payload: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(crate::engine::compute_wire_id_public(positional))
            .r#type(OperationType::Step)
            .sub_type(sub_type)
            .action(OperationAction::Retry)
            .payload(payload)
            .build()
            .expect("all required fields set")
    }

    #[tokio::test]
    async fn wfc_retry_keeps_status_the_response_carries() {
        let wfc = crate::wait_for_condition::WFC_SUB_TYPE;
        for (carried, expected) in [
            (OperationStatus::Ready, CheckpointStatus::Ready),
            (OperationStatus::Succeeded, CheckpointStatus::Succeeded),
            (OperationStatus::Failed, CheckpointStatus::Failed),
        ] {
            let wire = crate::engine::compute_wire_id_public("1");
            let log = Arc::new(CheckpointLog::from_records(vec![(
                wire.clone(),
                CheckpointRecord::with_status(&wire, CheckpointStatus::Started),
            )]));
            let status = carried.clone();
            let client = Arc::new(ScriptedClient::new(move |updates, _| {
                updates
                    .iter()
                    .map(|u| {
                        operation_like(u, status.clone())
                            .build()
                            .expect("all required fields set")
                    })
                    .collect()
            }));
            let ctx = ctx_with(Arc::clone(&log), &client);
            ctx.checkpoint_updates_direct(vec![retry_update("1", wfc, "1")])
                .await
                .expect("the write succeeds");
            assert_eq!(
                log.get(&wire).expect("present").status,
                expected,
                "a RETRY response that carries {carried:?} keeps it"
            );
        }
    }

    #[tokio::test]
    async fn wfc_retry_omitted_from_response_is_pending() {
        let wfc = crate::wait_for_condition::WFC_SUB_TYPE;
        let ready = crate::engine::compute_wire_id_public("1");
        let pending = crate::engine::compute_wire_id_public("2");
        let step = crate::engine::compute_wire_id_public("3");
        let succeeded = crate::engine::compute_wire_id_public("5");
        let mut pending_record = CheckpointRecord::with_status(&pending, CheckpointStatus::Pending);
        pending_record.result = Some("old".to_owned());
        let log = Arc::new(CheckpointLog::from_records(vec![
            (
                ready.clone(),
                CheckpointRecord::with_status(&ready, CheckpointStatus::Ready),
            ),
            (pending.clone(), pending_record),
            (
                step.clone(),
                CheckpointRecord::with_status(&step, CheckpointStatus::Started),
            ),
            (
                succeeded.clone(),
                CheckpointRecord::with_status(&succeeded, CheckpointStatus::Started),
            ),
        ]));
        let client = Arc::new(ScriptedClient::empty());
        let ctx = ctx_with(Arc::clone(&log), &client);
        ctx.checkpoint_updates_direct(vec![
            retry_update("1", wfc, "next"),
            retry_update("2", wfc, "new"),
            retry_update("3", "Step", "x"),
            retry_update("4", wfc, "x"),
            OperationUpdate::builder()
                .id(crate::engine::compute_wire_id_public("5"))
                .r#type(OperationType::Step)
                .sub_type(wfc)
                .action(OperationAction::Succeed)
                .payload("y")
                .build()
                .expect("all required fields set"),
        ])
        .await
        .expect("the write succeeds");

        let record = log.get(&ready).expect("present");
        assert_eq!(
            record.status,
            CheckpointStatus::Pending,
            "a READY record from the previous cycle takes the PENDING the service recorded"
        );
        assert_eq!(record.result.as_deref(), Some("next"));
        assert_eq!(
            log.get(&pending).expect("present").result.as_deref(),
            Some("old"),
            "a record the service already marked PENDING is kept"
        );
        assert_eq!(
            log.get(&step).expect("present").status,
            CheckpointStatus::Started,
            "a step RETRY is not a wait_for_condition RETRY"
        );
        assert_eq!(
            log.get(&succeeded).expect("present").status,
            CheckpointStatus::Started,
            "a wait_for_condition SUCCEED is not a RETRY"
        );
        let missing = crate::engine::compute_wire_id_public("4");
        assert!(
            log.get(&missing).is_none(),
            "a missing record stays missing"
        );
    }
}

/// The write order of unbuffered submissions and of reserved writes.
#[cfg(test)]
// The tests submit from a task on a runtime worker, which needs a spawn.
#[expect(clippy::disallowed_methods)]
mod write_order_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    use super::DurableContext;
    use crate::client::{
        CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput,
        InMemoryExecutionClient,
    };
    use crate::engine::CheckpointLog;

    fn start(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id.to_owned())
            .r#type(OperationType::Step)
            .action(OperationAction::Start)
            .build()
            .expect("all required OperationUpdate fields set")
    }

    /// A root context over `client`, with the coalescer `window` (`None`:
    /// no coalescer, so every write is direct).
    fn context(client: &Arc<InMemoryExecutionClient>, window: Option<Duration>) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
            window,
            0,
            None,
        )
    }

    fn ids(client: &InMemoryExecutionClient) -> Vec<String> {
        client
            .recorded_updates()
            .iter()
            .map(|u| u.id().to_owned())
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn unbuffered_submissions_reach_client_in_submission_order() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = context(&client, None);
        tokio::spawn(async move {
            let a = ctx.submit_checkpoint_updates(vec![start("A")]);
            let b = ctx.submit_checkpoint_updates(vec![start("B")]);
            a.await.expect("A is written");
            b.await.expect("B is written");
        })
        .await
        .expect("the submitting task completes");
        assert_eq!(ids(&client), ["A", "B"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn child_and_root_unbuffered_submissions_share_fifo() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = context(&client, None);
        let hold_a = Arc::new(tokio::sync::Notify::new());
        let gate_notify = Arc::clone(&hold_a);
        root.set_pre_lock_gate(Arc::new(move |updates: &[OperationUpdate]| {
            updates
                .iter()
                .any(|u| u.id() == "A")
                .then(|| Arc::clone(&gate_notify))
        }));
        let child = root.new_child("1", None);
        let a = child.submit_checkpoint_updates(vec![start("A")]);
        let b = root.submit_checkpoint_updates(vec![start("B")]);
        let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
        while tokio::time::Instant::now() < deadline {
            assert!(
                client.recorded_updates().is_empty(),
                "a write reached the client while A was held: {:?}",
                ids(&client)
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        hold_a.notify_one();
        a.await.expect("A is written");
        b.await.expect("B is written");
        assert_eq!(ids(&client), ["A", "B"]);
    }

    #[tokio::test]
    async fn dropping_reserved_write_releases_in_order() {
        for window in [None, Some(Duration::ZERO)] {
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let ctx = context(&client, window);
            let a = ctx.reserve_checkpoint_updates(vec![start("A")]);
            let b = ctx.submit_checkpoint_updates(vec![start("B")]);
            drop(a);
            tokio::time::timeout(Duration::from_secs(2), b)
                .await
                .map_err(|_| format!("B is blocked behind the dropped A ({window:?})"))
                .unwrap()
                .expect("B is written");
            assert_eq!(ids(&client), ["A", "B"], "{window:?}");
        }
    }

    /// A released reservation is written in its place: before a write
    /// submitted after it, and only once released.
    #[tokio::test]
    async fn released_reservation_is_written_before_later_submissions() {
        for window in [None, Some(Duration::ZERO)] {
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let ctx = context(&client, window);
            let a = ctx.reserve_checkpoint_updates(vec![start("A")]);
            let b = ctx.submit_checkpoint_updates(vec![start("B")]);
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert!(
                client.recorded_updates().is_empty(),
                "nothing is sent while A is held ({window:?})"
            );
            let a = a.release();
            a.await.expect("A is written");
            b.await.expect("B is written");
            assert_eq!(ids(&client), ["A", "B"], "{window:?}");
        }
    }

    /// A reservation the admission or the coordinator rejects holds
    /// nothing, sends nothing, and never resolves.
    #[tokio::test]
    async fn rejected_reservation_sends_nothing() {
        for window in [None, Some(Duration::ZERO)] {
            // The phase left `Running`: the coordinator refuses the permit.
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let ctx = context(&client, window);
            ctx.coordinator().finish();
            let write = ctx.reserve_checkpoint_updates(vec![start("A")]).release();
            assert!(
                tokio::time::timeout(Duration::from_millis(50), write)
                    .await
                    .is_err(),
                "a refused reservation never resolves ({window:?})"
            );
            assert!(client.recorded_updates().is_empty(), "{window:?}");
        }
        // An abandoned owner: the batch join is rejected.
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let root = context(&client, Some(Duration::ZERO));
        let child = root.new_child("1", None);
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        let write = child.reserve_checkpoint_updates(vec![start("A")]).release();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), write)
                .await
                .is_err(),
            "a rejected join never resolves"
        );
        assert_eq!(client.recorded_updates(), Vec::new());
    }

    /// A client that panics on a request carrying `A`, and records the IDs
    /// of the other requests.
    #[derive(Default)]
    struct PanicsOnA(std::sync::Mutex<Vec<String>>);

    impl ExecutionClient for PanicsOnA {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            // The client panic under test.
            assert!(!updates.iter().any(|u| u.id() == "A"), "client boom");
            self.0
                .lock()
                .unwrap()
                .extend(updates.iter().map(|u| u.id().to_owned()));
            Box::pin(async {
                Ok(CheckpointOutput::new(
                    Some("t".to_owned()),
                    Vec::new(),
                    None,
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async { Ok(GetStateOutput::new(Vec::new(), None)) })
        }
    }

    /// A checkpoint client panic unwinds the direct writer. The next write
    /// starts a new writer, so it is still sent.
    #[tokio::test]
    async fn write_after_writer_panic_starts_a_new_writer() {
        let client = Arc::new(PanicsOnA::default());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
        );
        let a = ctx.submit_checkpoint_updates(vec![start("A")]);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), a)
                .await
                .is_err(),
            "the panicked write never resolves"
        );
        assert!(
            ctx.coordinator().has_panic(),
            "the gateway recorded the panic"
        );
        let b = ctx.checkpoint_updates_after_drive(vec![start("B")]);
        tokio::time::timeout(Duration::from_secs(2), b)
            .await
            .expect("a new writer sends B")
            .expect("B is written");
        assert_eq!(*client.0.lock().unwrap(), ["B"]);
    }
}

/// The context usage error for an operation claimed, spawned or executed on
/// a task or thread that does not own the context.
pub(crate) fn foreign_caller_message(method: &str) -> String {
    format!(
        "context usage error in \"{method}\": the operation was called on a task or thread \
         that does not own the context. Start concurrent operations with .spawn(), map, \
         parallel or a combinator, not with tokio::spawn or std::thread::spawn."
    )
}

/// Tests of the change sets the write path queues for
/// `on_operation_change`, and of the points that deliver them.
#[cfg(test)]
mod change_hook_tests {
    use super::{CheckpointCoalescer, DurableContext};
    use crate::OperationStatus;
    use crate::client::{ExecutionClient, InMemoryExecutionClient, TestResponse};
    use crate::driver::test_support::{
        RefreshClient, ScriptedClient, drive_handler, named, operation_like,
    };
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use crate::plugin::test_support::Changes;
    use crate::plugin::{InvocationPlugins, OperationInfo, PluginFactory};
    use aws_sdk_lambda::types::{
        Operation, OperationAction, OperationStatus as WireStatus, OperationType, OperationUpdate,
    };
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;

    /// A root context on `client` over `log`, with `plugins`.
    fn ctx_on(
        client: Arc<dyn ExecutionClient>,
        log: Arc<CheckpointLog>,
        window: Option<Duration>,
        plugins: Arc<InvocationPlugins>,
    ) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "t0".to_owned(),
            window,
            0,
            Some(plugins),
        )
    }

    /// A client whose response carries each update's operation, with the
    /// status its action records.
    fn echo_client() -> Arc<ScriptedClient> {
        Arc::new(ScriptedClient::new(|updates, _history| {
            updates
                .iter()
                .map(|u| {
                    let status = match u.action() {
                        OperationAction::Succeed => WireStatus::Succeeded,
                        OperationAction::Fail => WireStatus::Failed,
                        OperationAction::Retry => WireStatus::Pending,
                        _ => WireStatus::Started,
                    };
                    operation_like(u, status).build().expect("all fields set")
                })
                .collect()
        }))
    }

    /// A step operation `id` with `status`.
    fn step_op(id: &str, status: WireStatus) -> Operation {
        Operation::builder()
            .id(id)
            .r#type(OperationType::Step)
            .status(status)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .build()
            .expect("all fields set")
    }

    /// A step `START` update of `id`.
    fn start(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id)
            .r#type(OperationType::Step)
            .action(OperationAction::Start)
            .build()
            .expect("all fields set")
    }

    /// The change hook runs with the token mutex and the coalescer's writer
    /// lock free, with and without batching.
    #[tokio::test]
    async fn change_hook_runs_outside_locks() {
        type Locks = (
            Arc<tokio::sync::Mutex<String>>,
            Option<Arc<CheckpointCoalescer>>,
        );
        for window in [None, Some(Duration::ZERO)] {
            let locks: Arc<OnceLock<Locks>> = Arc::default();
            let free: Arc<Mutex<Vec<(bool, bool)>>> = Arc::default();
            let mut recorder = Changes::default();
            {
                let locks = Arc::clone(&locks);
                let free = Arc::clone(&free);
                recorder.on_change = Some(Arc::new(move |_info| {
                    let (token, coalescer) = locks.get().expect("set before the run");
                    let token_free = token.try_lock().is_ok();
                    let writer_free = coalescer
                        .as_ref()
                        .is_none_or(|c| c.writer_lock().try_lock().is_ok());
                    free.lock().unwrap().push((token_free, writer_free));
                }));
            }
            let ctx = ctx_on(
                echo_client(),
                Arc::new(CheckpointLog::empty()),
                window,
                recorder.plugins(),
            );
            locks
                .set((ctx.checkpoint_token_mutex(), ctx.inner.coalescer.clone()))
                .map_err(|_| "set twice")
                .expect("set once");
            drive_handler(ctx, None, |ctx| async move {
                ctx.step(|_| async { Ok(1_u32) })
                    .name("s")
                    .await
                    .map(|v| v.to_string())
                    .map_err(|e| crate::error::WireError::new(None::<String>, Some(e.to_string())))
            })
            .await
            .expect("no panic");
            let free = free.lock().unwrap().clone();
            assert!(!free.is_empty(), "window {window:?}: at least one call");
            assert!(
                free.iter().all(|&(token, writer)| token && writer),
                "window {window:?}: {free:?}"
            );
        }
    }

    /// The retained `START` write of a default step, which its start step
    /// reserved, delivers the `STARTED` change when the step awaits it: the
    /// change is reported before the step sends its `SUCCEED`, with and
    /// without batching.
    #[tokio::test]
    async fn retained_start_write_reports_its_change() {
        for window in [None, Some(Duration::ZERO)] {
            let recorder = Changes::default();
            let seen: Arc<Mutex<Vec<(OperationAction, usize)>>> = Arc::default();
            let client = {
                let calls = Arc::clone(&recorder.calls);
                let seen = Arc::clone(&seen);
                Arc::new(ScriptedClient::new(move |updates, _history| {
                    let delivered = calls.lock().unwrap().len();
                    updates
                        .iter()
                        .map(|u| {
                            seen.lock().unwrap().push((u.action().clone(), delivered));
                            let status = if u.action() == &OperationAction::Succeed {
                                WireStatus::Succeeded
                            } else {
                                WireStatus::Started
                            };
                            operation_like(u, status).build().expect("all fields set")
                        })
                        .collect()
                }))
            };
            let ctx = ctx_on(
                client,
                Arc::new(CheckpointLog::empty()),
                window,
                recorder.plugins(),
            );
            drive_handler(ctx, None, |ctx| async move {
                ctx.step(|_| async { Ok(1_u32) })
                    .name("s")
                    .await
                    .map(|v| v.to_string())
                    .map_err(|e| crate::error::WireError::new(None::<String>, Some(e.to_string())))
            })
            .await
            .expect("no panic");
            let seen = seen.lock().unwrap().clone();
            assert_eq!(
                seen,
                [(OperationAction::Start, 0), (OperationAction::Succeed, 1)],
                "window {window:?}: {:?}",
                recorder.timeline()
            );
        }
    }

    /// A context whose plugin's first change hook, for change set
    /// `earlier`, is held open on another thread, and a completed write
    /// whose own change set `own` is queued behind it.
    struct HeldWrite {
        ctx: DurableContext,
        recorder: Changes,
        write: std::pin::Pin<Box<super::SubmittedWrite>>,
        release: std::sync::mpsc::Sender<()>,
        holder: Option<std::thread::JoinHandle<()>>,
    }

    impl HeldWrite {
        fn start() -> Self {
            Self::start_with(Ok(crate::client::CheckpointOutput::new(
                None,
                Vec::new(),
                None,
            )))
        }

        /// As [`Self::start`], with `result` as the write's result.
        fn start_with(
            result: Result<crate::client::CheckpointOutput, crate::client::ExecutionClientError>,
        ) -> Self {
            let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
            let (release, release_rx) = std::sync::mpsc::channel::<()>();
            let release_rx = Mutex::new(release_rx);
            let first = std::sync::atomic::AtomicBool::new(true);
            let recorder = Changes {
                on_change: Some(Arc::new(move |_info| {
                    if first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                    }
                })),
                ..Changes::default()
            };
            let ctx = ctx_on(
                Arc::new(InMemoryExecutionClient::new(Vec::new())),
                Arc::new(CheckpointLog::empty()),
                None,
                recorder.plugins(),
            );
            let entry = |id: &str| {
                vec![OperationInfo::from_record(
                    id,
                    &CheckpointRecord::with_status(id, CheckpointStatus::Started),
                )]
            };
            let plugins = ctx.shared_plugins().expect("the context has plugins");
            plugins.push_operation_change(entry("earlier"));
            let holder = {
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    ctx.plugins()
                        .expect("the context has plugins")
                        .drain_operation_changes(&ctx.inner.engine.checkpoint_log);
                })
            };
            entered_rx.recv().unwrap();
            // The write's own response queued its change set.
            plugins.push_operation_change(entry("own"));
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let write = Box::pin(super::SubmittedWrite::from_receiver(receiver).reported_by(&ctx));
            assert!(sender.send(result).is_ok(), "the write is waiting");
            Self {
                ctx,
                recorder,
                write,
                release,
                holder: Some(holder),
            }
        }

        /// Waits until the holder delivered both change sets and returned.
        fn join_holder(&mut self) {
            self.holder
                .take()
                .expect("the holder is joined once")
                .join()
                .unwrap();
        }

        fn waiter_count(&self) -> usize {
            self.ctx
                .plugins()
                .expect("the context has plugins")
                .delivery_waiter_count()
        }
    }

    /// A completed write whose change set another thread is still
    /// delivering, held open by an earlier change hook, returns its
    /// result only after that change set was delivered. So the operation
    /// that awaits it fires its next hook after its change. Repeated
    /// polls keep one waker registration, and the delivery wakes the
    /// registered waker by itself: the test sees the wake before it polls
    /// the write again.
    #[test]
    fn write_waits_for_change_delivered_by_another_task() {
        let mut held = HeldWrite::start();
        let (signal, signals) = crate::plugin::test_support::SignalWaker::new();
        let waker = std::task::Waker::from(Arc::clone(&signal));
        let mut cx = std::task::Context::from_waker(&waker);
        for _ in 0..100 {
            assert!(
                held.write.as_mut().poll(&mut cx).is_pending(),
                "the write waits while its change set is undelivered"
            );
        }
        assert_eq!(held.waiter_count(), 1);
        assert_eq!(held.recorder.updated_ids(), [["earlier"]]);
        assert_eq!(signal.woken(), 0);
        held.release.send(()).unwrap();
        signals
            .recv_timeout(Duration::from_secs(30))
            .expect("a delivery woke the write's waker");
        held.join_holder();
        match held.write.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(result) => result.expect("the write succeeded"),
            std::task::Poll::Pending => unreachable!("the write is ready once delivered"),
        };
        assert_eq!(held.recorder.updated_ids(), [["earlier"], ["own"]]);
        assert_eq!(held.waiter_count(), 0);
    }

    /// A write dropped while it waits for a delivery removes its waker
    /// registration.
    #[test]
    fn dropped_write_removes_its_registration() {
        let mut held = HeldWrite::start();
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(held.write.as_mut().poll(&mut cx).is_pending());
        assert_eq!(held.waiter_count(), 1);
        held.write = Box::pin(super::SubmittedWrite::rejected());
        assert_eq!(held.waiter_count(), 0);
        held.release.send(()).unwrap();
        held.join_holder();
        assert_eq!(held.recorder.updated_ids(), [["earlier"], ["own"]]);
        assert_eq!(held.waiter_count(), 0);
    }

    /// An error source whose destructor panics with "source drop".
    #[derive(Debug)]
    struct DropPanickingSource;

    impl std::fmt::Display for DropPanickingSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("drop-panicking source")
        }
    }

    impl std::error::Error for DropPanickingSource {}

    impl Drop for DropPanickingSource {
        #[expect(clippy::panic)] // reason: the panic is the behavior under test
        fn drop(&mut self) {
            panic!("source drop");
        }
    }

    /// A write dropped while it waits for a delivery drops the result it
    /// retained inside a catch, and records the panic of that result's
    /// destructor on the invocation coordinator.
    #[test]
    fn dropped_write_records_the_panic_of_its_retained_result() {
        let mut held =
            HeldWrite::start_with(Err(crate::client::ExecutionClientError::with_source(
                crate::client::ExecutionClientErrorScope::Invocation,
                "failed write",
                DropPanickingSource,
            )));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(held.write.as_mut().poll(&mut cx).is_pending());
        assert!(!held.ctx.coordinator().has_panic());
        held.write = Box::pin(super::SubmittedWrite::rejected());
        assert_eq!(held.waiter_count(), 0);
        let payload = held
            .ctx
            .coordinator()
            .take_panic()
            .expect("the retained result's destructor panic is recorded");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"source drop"));
        held.release.send(()).unwrap();
        held.join_holder();
        assert_eq!(held.recorder.updated_ids(), [["earlier"], ["own"]]);
    }

    /// The operations maps and change infos one run of a step and a child
    /// context builds, with `plugins` or without any.
    async fn change_work_of_run(plugins: Option<Arc<InvocationPlugins>>) -> (usize, usize) {
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            echo_client(),
            "t0".to_owned(),
            Some(Duration::ZERO),
            0,
            plugins,
        );
        let projections = crate::engine::projections();
        let infos = crate::plugin::infos_from_records();
        drive_handler(ctx, None, |ctx| async move {
            let one = ctx.step(|_| async { Ok(1_u32) }).name("s").await;
            let two = ctx
                .run_in_child_context(|child| async move {
                    Ok(child.step(|_| async { Ok(2_u32) }).name("inner").await?)
                })
                .name("c")
                .await;
            Ok(format!("{one:?} {two:?}"))
        })
        .await
        .expect("no panic");
        (
            crate::engine::projections() - projections,
            crate::plugin::infos_from_records() - infos,
        )
    }

    /// Without a plugin, the checkpoint responses of a run build no
    /// operations map and no operation info. With one, the same run builds
    /// both, so the counters observe the change work.
    #[tokio::test]
    async fn no_plugin_builds_no_change_info() {
        assert_eq!(change_work_of_run(None).await, (0, 0));
        let (projected, infos) = change_work_of_run(Some(Changes::default().plugins())).await;
        assert!(projected > 0, "a plugin's run projects the log");
        assert!(infos > 0, "a plugin's run builds infos");
    }

    /// A direct write delivers its change on the caller's task once the
    /// detached write task returned the result, `Ok` or `Err`.
    #[tokio::test]
    async fn direct_write_reports_change_on_caller() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::Success(vec![step_op(
            "d",
            WireStatus::Started,
        )]));
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            vec![step_op("d", WireStatus::Succeeded)],
            "page-2-token".to_owned(),
        ));
        client.fail_get_state("injected get_state failure");
        let recorder = Changes::default();
        let ctx = ctx_on(
            client,
            Arc::new(CheckpointLog::empty()),
            Some(Duration::ZERO),
            recorder.plugins(),
        );
        ctx.checkpoint_updates_direct(vec![start("d")])
            .await
            .expect("the write succeeds");
        assert_eq!(recorder.updated_ids(), [["d"]]);
        assert!(
            ctx.checkpoint_updates_direct(vec![start("d")])
                .await
                .is_err()
        );
        assert_eq!(recorder.updated_ids(), [["d"], ["d"]]);
    }

    /// A response that changes no status queues nothing, so no call is
    /// made.
    #[tokio::test]
    async fn unchanged_response_queues_nothing() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::Success(vec![step_op(
            "s",
            WireStatus::Started,
        )]));
        let recorder = Changes::default();
        let log = Arc::new(CheckpointLog::from_records(vec![(
            "s".to_owned(),
            CheckpointRecord::with_status("s", CheckpointStatus::Started),
        )]));
        let ctx = ctx_on(client, log, None, recorder.plugins());
        ctx.checkpoint_updates(vec![start("s")])
            .await
            .expect("the write succeeds");
        assert!(recorder.calls().is_empty(), "{:?}", recorder.updated_ids());
    }

    /// An `EXECUTION` operation that changes in the response and again in
    /// the state pages is listed once, with its last status.
    #[tokio::test]
    async fn execution_listed_once_per_change_set() {
        let execution = |status: WireStatus| {
            Operation::builder()
                .id("exec-1")
                .r#type(OperationType::Execution)
                .status(status)
                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                .build()
                .expect("all fields set")
        };
        let client = Arc::new(InMemoryExecutionClient::new(vec![execution(
            WireStatus::Succeeded,
        )]));
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            vec![execution(WireStatus::Started)],
            "page-2-token".to_owned(),
        ));
        let recorder = Changes::default();
        let ctx = ctx_on(
            client,
            Arc::new(CheckpointLog::empty()),
            None,
            recorder.plugins(),
        );
        ctx.checkpoint_updates(Vec::new())
            .await
            .expect("the write succeeds");
        let calls = recorder.calls();
        assert_eq!(calls.len(), 1, "{:?}", recorder.updated_ids());
        let info = calls.first().expect("one call");
        let statuses: Vec<_> = info
            .updated_operations()
            .iter()
            .map(OperationInfo::status)
            .collect();
        assert_eq!(statuses, [Some(OperationStatus::Succeeded)]);
    }

    /// When reading the state pages fails, the caller sees the error and
    /// the change the response merged is still reported.
    #[tokio::test]
    async fn change_reported_when_pagination_hydration_fails() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            vec![step_op("op-hydration", WireStatus::Started)],
            "page-2-token".to_owned(),
        ));
        client.fail_get_state("injected get_state failure");
        let recorder = Changes::default();
        let ctx = ctx_on(
            client,
            Arc::new(CheckpointLog::empty()),
            None,
            recorder.plugins(),
        );
        let result = ctx.checkpoint_updates(vec![start("op-hydration")]).await;
        assert!(result.is_err());
        assert_eq!(recorder.updated_ids(), [["op-hydration"]]);
    }

    /// The change that completes a waiting operation, merged from a
    /// refresh, is reported before the operation's end hook. Without the
    /// drain when the wait returns, the sibling step's write at 3 s would
    /// deliver it, after the end hook.
    #[tokio::test(start_paused = true)]
    async fn refresh_reports_wait_change() {
        let client = Arc::new(RefreshClient::new(|req| {
            if req.empty_number == 1
                && let Some(start) = named(req.history, "w", &OperationAction::Start)
            {
                let record = operation_like(start, WireStatus::Succeeded)
                    .build()
                    .expect("all required fields set");
                return Ok(vec![record]);
            }
            Ok(Vec::new())
        }));
        let recorder = Changes::default();
        let ctx = ctx_on(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Arc::new(CheckpointLog::empty()),
            None,
            recorder.plugins(),
        );
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let wait = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .spawn();
            let (wait, _step) = tokio::join!(wait, step);
            Ok(format!("{}", wait.is_ok()))
        })
        .await
        .expect("no panic");
        assert_eq!(
            outcome,
            crate::driver::InvocationOutcome::Complete("true".to_owned())
        );
        let succeeded = recorder.calls().iter().any(|info| {
            info.updated_operations()
                .iter()
                .any(|op| op.name() == Some("w") && op.status() == Some(OperationStatus::Succeeded))
        });
        assert!(succeeded, "{:?}", recorder.timeline());
        let timeline = recorder.timeline();
        let change = timeline
            .iter()
            .position(|e| e == "change:w:SUCCEEDED")
            .expect("the change is recorded");
        let end = timeline
            .iter()
            .position(|e| e == "end:w")
            .expect("the end hook ran");
        assert!(change < end, "{timeline:?}");
    }

    /// The task that sends a refresh delivers the change the refresh
    /// merged when the call returns, also for an operation nobody waits
    /// on. Without that drain, the change waits for the next drain point:
    /// the wait's completion at the second refresh, or the sibling step's
    /// write at 3 s.
    #[tokio::test(start_paused = true)]
    async fn refresh_sender_reports_change() {
        let client = Arc::new(RefreshClient::new(|req| {
            let Some(start) = named(req.history, "w", &OperationAction::Start) else {
                return Ok(Vec::new());
            };
            let record = match req.empty_number {
                1 => step_op("x", WireStatus::Succeeded),
                2 => operation_like(start, WireStatus::Succeeded)
                    .build()
                    .expect("all required fields set"),
                _ => return Ok(Vec::new()),
            };
            Ok(vec![record])
        }));
        let at: Arc<Mutex<Vec<Duration>>> = Arc::default();
        let mut recorder = Changes::default();
        {
            let at = Arc::clone(&at);
            let started = tokio::time::Instant::now();
            recorder.on_change = Some(Arc::new(move |info| {
                if info.updated_operations().iter().any(|op| op.id() == "x") {
                    at.lock().unwrap().push(started.elapsed());
                }
            }));
        }
        let ctx = ctx_on(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Arc::new(CheckpointLog::empty()),
            None,
            recorder.plugins(),
        );
        drive_handler(ctx, None, |ctx| async move {
            let wait = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .spawn();
            let (wait, _step) = tokio::join!(wait, step);
            Ok(format!("{}", wait.is_ok()))
        })
        .await
        .expect("no panic");
        let at = at.lock().unwrap().clone();
        assert_eq!(at.len(), 1, "{:?}", recorder.timeline());
        assert!(
            at.first()
                .is_some_and(|t| *t < Duration::from_millis(1_500)),
            "the change of x is reported at the first refresh: {at:?}, {:?}",
            client.empty_answers()
        );
    }

    /// Runs 50 new steps over a log of 2 000 succeeded steps with the
    /// given number of recording plugins, and returns the projections the
    /// run made and the number of change calls the first plugin received.
    async fn projections_of_run(plugin_count: usize) -> (usize, usize) {
        let records = (0..2_000)
            .map(|i| {
                let id = format!("old-{i}");
                let mut record = CheckpointRecord::with_status(&id, CheckpointStatus::Succeeded);
                record.op_type = Some("STEP".to_owned());
                (id, record)
            })
            .collect();
        let recorders: Vec<Changes> = (0..plugin_count).map(|_| Changes::default()).collect();
        let plugins = crate::plugin::test_support::plugins_of_all(
            recorders
                .iter()
                .map(|r| Arc::new(r.clone()) as Arc<dyn PluginFactory>)
                .collect(),
        );
        let ctx = ctx_on(
            echo_client(),
            Arc::new(CheckpointLog::from_records(records)),
            None,
            plugins,
        );
        let before = crate::engine::projections();
        drive_handler(ctx, None, |ctx| async move {
            for i in 0..50_u32 {
                ctx.step(move |_| async move { Ok(i) })
                    .name(format!("new-{i}"))
                    .await
                    .map_err(|e| {
                        crate::error::WireError::new(None::<String>, Some(e.to_string()))
                    })?;
            }
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        let projected = crate::engine::projections() - before;
        let calls = recorders.first().expect("a plugin").calls().len();
        (projected, calls)
    }

    /// Each change event projects the log once, whatever the number of
    /// plugins.
    #[tokio::test]
    async fn change_hooks_project_once_per_response() {
        let (one_projected, one_calls) = projections_of_run(1).await;
        let (three_projected, three_calls) = projections_of_run(3).await;
        assert!(one_calls > 0);
        assert_eq!(one_projected, one_calls);
        assert_eq!(three_projected, three_calls);
        assert_eq!(one_calls, three_calls);
    }

    /// An `EXECUTION` operation in a response replaces the plugins' slot,
    /// is listed as a change, and does not enter the log.
    #[tokio::test]
    async fn execution_update_replaces_slot() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let execution = Operation::builder()
            .id("exec-1")
            .r#type(OperationType::Execution)
            .status(WireStatus::Succeeded)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .build()
            .expect("all fields set");
        client.enqueue_checkpoint_response(TestResponse::Success(vec![
            execution,
            step_op("s", WireStatus::Started),
        ]));
        let recorder = Changes::default();
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
            vec![("test::Changes", Arc::new(recorder.clone()))];
        let slot = OperationInfo::execution(
            "exec-1",
            "EXECUTION",
            &CheckpointStatus::Started,
            None,
            None,
            None,
        );
        let plugins =
            crate::plugin::create_plugins(&factories, slot, crate::plugin::tests_support_info)
                .expect("one factory");
        let log = Arc::new(CheckpointLog::empty());
        let ctx = ctx_on(client, Arc::clone(&log), None, plugins);
        ctx.checkpoint_updates(vec![start("s")])
            .await
            .expect("the write succeeds");
        let calls = recorder.calls();
        let info = calls.first().expect("one call");
        assert_eq!(
            info.operations()
                .get("exec-1")
                .and_then(OperationInfo::status),
            Some(OperationStatus::Succeeded)
        );
        assert!(
            info.updated_operations()
                .iter()
                .any(|op| op.id() == "exec-1")
        );
        assert!(!log.contains("exec-1"));
    }

    /// A paginated response and its state pages give one change set, which
    /// lists each changed ID once, with the status after both merges.
    #[tokio::test]
    async fn paginated_response_is_one_change_set() {
        let client = Arc::new(InMemoryExecutionClient::new(vec![
            step_op("step-1", WireStatus::Succeeded),
            step_op("step-2", WireStatus::Succeeded),
        ]));
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            vec![step_op("step-1", WireStatus::Started)],
            "page-2-token".to_owned(),
        ));
        let recorder = Changes::default();
        let ctx = ctx_on(
            client,
            Arc::new(CheckpointLog::empty()),
            None,
            recorder.plugins(),
        );
        ctx.checkpoint_updates(Vec::new())
            .await
            .expect("the write succeeds");
        assert_eq!(recorder.updated_ids(), [["step-1", "step-2"]]);
        let calls = recorder.calls();
        let step_1 = calls
            .first()
            .and_then(|info| info.updated_operations().first())
            .expect("step-1 is listed");
        assert_eq!(step_1.status(), Some(OperationStatus::Succeeded));
    }
}
