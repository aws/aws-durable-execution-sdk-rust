//! Checkpoint write buffering for the `checkpoint_delay` and
//! `checkpoint_batching` options.
//!
//! When [`Options`](crate::Options) configures a
//! [`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay) and/or
//! [`checkpoint_batching`](crate::OptionsBuilder::checkpoint_batching),
//! checkpoint writes from concurrently running operations are held in a
//! shared buffer, then written together in fewer `CheckpointDurableExecution`
//! calls. A non-zero delay holds a write for up to the configured window so
//! neighbors can join it; a zero delay (pure batching mode) requests the
//! write immediately, and batching emerges from accumulation while an
//! earlier write is in flight. This module owns the buffer, the batch
//! handshake, the single-writer lock, and the size-capped batch splitting;
//! the actual API call stays in `DurableContext::checkpoint_updates_direct`,
//! which the flusher invokes with the drained buffer.
//!
//! # Invariants
//!
//! - Every buffered update belongs to the currently open batch: joining
//!   after a batch was taken opens a fresh batch, and taking a batch drains
//!   the buffer under the same lock that guards joins.
//! - Batches are sealed and written **only while holding the
//!   [`CheckpointCoalescer::writer_lock`]**, which totally orders every
//!   buffered write. A flush point that acquires the writer lock therefore
//!   cannot proceed while any earlier claimed batch is still in flight:
//!   this is what makes `DurableContext::flush_pending_checkpoints` a true
//!   barrier rather than a best-effort drain.
//! - A batch is flushed by whoever gets to it first: a contributor whose
//!   delay window elapsed, an urgent contributor (callback creation), or an
//!   unconditional flush point (suspension / end of invocation). The
//!   [`CheckpointCoalescer::take_batch`] pointer check makes a stale flusher
//!   a no-op instead of letting it steal a newer batch.
//! - A sealed batch is split into one or more requests by
//!   [`split_into_requests`], each within [`BatchLimits`] (operation count
//!   and estimated payload bytes), preserving join order across the splits.
//! - Once any buffered write fails, the **failure latch** is set (under
//!   the writer lock, by the failing flusher) and no buffered write
//!   reaches the backend again for the rest of the invocation: every
//!   flusher, spawned or flush-point, checks the latch under the writer
//!   lock before writing, and on a hit publishes the latched error to its
//!   contributors and retains its updates as unwritten (issue #43,
//!   "retryable exhaustion fails the invocation with no further writes").
//!   The terminal `FAIL` writes of the non-retryable classification are
//!   unaffected: they go through the direct (uncoalesced) path.
//! - When a checkpoint response has no token, the SDK sends nothing more
//!   this invocation (see `InvocationCoordinator::withdraw`). The
//!   withdrawal drops the open batch unpublished with its permits (see
//!   [`CheckpointCoalescer::abandon_pending`]), so its
//!   contributors never resolve. A flusher that already claimed a batch
//!   drops it unpublished when it reaches the checkpoint-token lock.
//! - The flush itself runs on a spawned task (see
//!   `DurableContext::spawn_batch_flush`), so a contributor that is dropped
//!   mid-await (a lost `race`, for example) cannot cancel an in-flight batch
//!   write and strand the other contributors.
//! - Each buffered update carries its lifecycle event with it (see
//!   [`TrackedUpdate`]): the flusher that persists a chunk emits that
//!   chunk's events immediately after the chunk's write succeeds. Telemetry
//!   for a persisted transition therefore cannot be lost to a dropped
//!   contributor, and a chunk persisted before a later chunk fails still
//!   emits its events even though the batch as a whole publishes the error.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use aws_sdk_lambda::types::OperationUpdate;
use tokio::sync::Notify;
use tokio::sync::futures::Notified;
use tokio::task::AbortHandle;

use crate::client::{CheckpointOutput, ExecutionClientError};
use crate::coordinator::{InvocationCoordinator, Permit, PermitKind};
use crate::tracing_layer::PendingTransitionEvent;

/// An operation update paired with the lifecycle event describing the
/// transition it records (see
/// [`PendingTransitionEvent`](crate::tracing_layer::PendingTransitionEvent)).
///
/// The pair travels together through the whole write path, into the
/// coalescing buffer, across batch splits, and into the flush task, so
/// the code that actually persists the update also owns its event and
/// emits it the moment that write succeeds. Keeping the event with the
/// update (rather than in the contributor future awaiting the batch) is
/// what upholds the documented guarantee that every persisted transition
/// emits telemetry: a contributor dropped after joining a batch (a lost
/// `race`/`select_ok` branch) cannot take its events down with it, and a
/// chunk persisted before a later chunk fails still emits its events.
#[derive(Debug)]
pub(crate) struct TrackedUpdate {
    /// The update to checkpoint.
    pub(crate) update: OperationUpdate,
    /// The lifecycle event to emit once the update is persisted.
    pub(crate) event: PendingTransitionEvent,
}

impl TrackedUpdate {
    /// Pairs an update with the lifecycle event captured from it. The
    /// capture snapshots the current [`tracing::Span`], so call this from
    /// the originating operation's context (the checkpoint chokepoint),
    /// not from the flush task.
    pub(crate) fn capture(update: OperationUpdate, attempt: u32) -> Self {
        let event = PendingTransitionEvent::capture(&update, attempt);
        Self { update, event }
    }
}

/// The scopes of one invocation: child contexts, `with_retry` blocks with
/// their own record, the attempts of every `with_retry` block (an
/// unwrapped block has no scope of its own), and batch items, flat items
/// included. Tracks which scopes
/// have closed, and the operation tasks started inside each scope.
///
/// A scope is named by the wire ID of its child context. Every context
/// carries the scopes it runs in, outermost first (see
/// `DurableContext::owners`). The root context runs in no scope.
///
/// A closed scope does two things:
///
/// 1. It rejects checkpoint updates. An update is admitted only when no
///    owner of the context that writes it is closed. So once a scope is
///    closed, no later update from it or from any context nested in it is
///    sent.
/// 2. It stops tasks. An operation task started with `.spawn()` is
///    registered with the owners of the context that started it (see
///    [`Self::register_task`]). Closing a scope aborts every registered task
///    whose owners include it, so nested work stops too. An aborted task
///    stops at its next `.await`.
///
/// The admission check runs where an update enters the ordered write path:
/// inside [`CheckpointCoalescer::join`] under the buffer lock, and after a
/// direct write takes the checkpoint-token lock. An update admitted before
/// a scope closes is already in the buffer or already holds the token
/// lock. The parent's final update is issued after the close, so it is
/// written after that update. [`Self::abandon`] takes only the admission
/// lock, and the buffer lock is never taken inside it.
#[derive(Debug, Default)]
pub(crate) struct Admission {
    state: Mutex<AdmissionState>,
}

/// The closed scopes and the registered tasks of an [`Admission`].
#[derive(Debug, Default)]
struct AdmissionState {
    /// The wire IDs of the closed scopes.
    abandoned: BTreeSet<String>,
    /// Each registered task that no close has aborted yet, with the owners
    /// of the context that started it.
    tasks: Vec<(Vec<String>, AbortHandle)>,
}

impl Admission {
    /// Creates an admission with no closed scope.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, AdmissionState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Closes these scopes. Later updates from them are rejected, and every
    /// registered task started inside one of them is aborted.
    pub(crate) fn abandon<'a>(&self, owners: impl IntoIterator<Item = &'a str>) {
        let stopped = {
            let mut guard = self.lock();
            let state = &mut *guard;
            state
                .abandoned
                .extend(owners.into_iter().map(str::to_owned));
            let abandoned = &state.abandoned;
            let (stopped, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut state.tasks)
                .into_iter()
                .partition(|(owners, _)| owners.iter().any(|owner| abandoned.contains(owner)));
            state.tasks = kept;
            stopped
        };
        // Tokio is not asked to abort under the lock.
        for (_, task) in stopped {
            task.abort();
        }
    }

    /// True when no owner in `owners` is closed.
    pub(crate) fn admits(&self, owners: &[String]) -> bool {
        let state = self.lock();
        !owners.iter().any(|owner| state.abandoned.contains(owner))
    }

    /// Whether a task started inside `owner` is registered and not aborted.
    #[cfg(test)]
    pub(crate) fn has_task_in(&self, owner: &str) -> bool {
        self.lock()
            .tasks
            .iter()
            .any(|(owners, _)| owners.iter().any(|o| o == owner))
    }

    /// Registers an operation task started inside `owners`, so the close
    /// of any of them aborts it. Aborts it at once when one of them is
    /// already closed. Forgets the registered tasks that have finished.
    pub(crate) fn register_task(&self, owners: &[String], task: AbortHandle) {
        let late = {
            let mut state = self.lock();
            if owners.iter().any(|owner| state.abandoned.contains(owner)) {
                Some(task)
            } else {
                state.tasks.retain(|(_, task)| !task.is_finished());
                state.tasks.push((owners.to_vec(), task));
                None
            }
        };
        if let Some(task) = late {
            task.abort();
        }
    }

    /// The number of registered tasks.
    #[cfg(test)]
    fn registered(&self) -> usize {
        self.lock().tasks.len()
    }
}

/// Closes one scope of an [`Admission`] when dropped.
///
/// A child body, an item body and a `with_retry` attempt hold one while
/// they run. So the scope closes when the body ends, and also when the
/// body's future is dropped because the operation that owns it is
/// stopped.
pub(crate) struct ScopeClose {
    admission: Arc<Admission>,
    scope: String,
}

impl ScopeClose {
    /// A guard that closes the scope `scope` of `admission` on drop.
    pub(crate) fn new(admission: &Arc<Admission>, scope: &str) -> Self {
        Self {
            admission: Arc::clone(admission),
            scope: scope.to_owned(),
        }
    }
}

impl Drop for ScopeClose {
    fn drop(&mut self) {
        self.admission.abandon([self.scope.as_str()]);
    }
}

/// Size caps for one `CheckpointDurableExecution` request. A sealed batch
/// larger than either cap is split into multiple requests, preserving
/// order. The defaults match the limits the durable execution service
/// enforces per request (and the values peer durable-execution SDKs use
/// for their checkpoint batchers).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchLimits {
    /// Maximum number of operation updates per request.
    pub(crate) max_operations: usize,
    /// Maximum estimated payload size per request, in bytes.
    pub(crate) max_payload_bytes: usize,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_operations: 250,
            max_payload_bytes: 750 * 1024,
        }
    }
}

/// Number of UTF-8 bytes `s` occupies once JSON string escaping is applied
/// on the wire (excluding the surrounding quotes). Quotes, backslashes, and
/// control characters expand under escaping, a `"` becomes the two bytes
/// `\"`, so raw `str::len` under-estimates escape-heavy payloads and would
/// let a batch that fits the raw-byte cap serialize past the service's
/// request limit.
pub(crate) fn json_escaped_len(s: &str) -> usize {
    s.chars()
        .map(|c| match c {
            // Escaped as a two-byte sequence: \" \\ \b \t \n \f \r.
            '"' | '\\' | '\u{08}' | '\t' | '\n' | '\u{0C}' | '\r' => 2,
            // Remaining control characters escape as six-byte \uXXXX.
            c if (c as u32) < 0x20 => 6,
            c => c.len_utf8(),
        })
        .sum()
}

/// Conservatively estimates the wire size of one operation update: the sum
/// of every string field it carries **as JSON-escaped on the wire** (see
/// [`json_escaped_len`]) plus a fixed overhead for structure, enums, and
/// field names. Used only to decide batch splits, so erring a little high
/// is safe (it splits earlier) while erring low is not.
pub(crate) fn estimated_update_size(update: &OperationUpdate) -> usize {
    /// Per-update allowance for JSON structure, field names, enum values,
    /// timestamps, string quotes, and the nested option structs.
    const STRUCTURAL_OVERHEAD: usize = 256;

    let opt_len = |s: &Option<String>| s.as_deref().map_or(0, json_escaped_len);
    let error_len = update.error.as_ref().map_or(0, |e| {
        e.error_message.as_deref().map_or(0, json_escaped_len)
            + e.error_type.as_deref().map_or(0, json_escaped_len)
            + e.error_data.as_deref().map_or(0, json_escaped_len)
            + e.stack_trace()
                .iter()
                .map(|s| json_escaped_len(s))
                .sum::<usize>()
    });

    json_escaped_len(&update.id)
        + opt_len(&update.parent_id)
        + opt_len(&update.name)
        + opt_len(&update.sub_type)
        + opt_len(&update.payload)
        + error_len
        + STRUCTURAL_OVERHEAD
}

/// Splits a sealed batch into request-sized chunks, each within `limits`,
/// preserving order. A single update whose estimated size alone exceeds the
/// payload cap is emitted as its own one-update request (it cannot be split
/// further; the service applies its own limit to it exactly as it would to
/// an immediate write). Each update's lifecycle event stays paired with it
/// across the split, so the writer emits exactly the events of the chunk it
/// just persisted.
pub(crate) fn split_into_requests(
    updates: Vec<TrackedUpdate>,
    limits: &BatchLimits,
) -> Vec<Vec<TrackedUpdate>> {
    let mut requests: Vec<Vec<TrackedUpdate>> = Vec::new();
    let mut current: Vec<TrackedUpdate> = Vec::new();
    let mut current_bytes = 0_usize;

    for tracked in updates {
        let size = estimated_update_size(&tracked.update);
        let over_count = current.len() >= limits.max_operations.max(1);
        let over_bytes = !current.is_empty() && current_bytes + size > limits.max_payload_bytes;
        if over_count || over_bytes {
            requests.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current_bytes += size;
        current.push(tracked);
    }
    if !current.is_empty() {
        requests.push(current);
    }
    requests
}

/// Shared coalescing buffer for checkpoint writes. One per execution
/// invocation, shared (via `Arc`) by the root context and every child
/// context, so updates from all namespaces coalesce together.
#[derive(Debug)]
pub(crate) struct CheckpointCoalescer {
    /// The coalescing window: how long a contributor waits for co-batched
    /// updates before flushing the batch itself. `Duration::ZERO` means no
    /// window (pure `checkpoint_batching` mode): the write is requested
    /// immediately, and batching comes from accumulation behind the
    /// writer lock while an earlier write is in flight.
    delay: Duration,
    state: Mutex<CoalescerState>,
    /// The single-writer lock: every sealed batch is claimed and written
    /// while holding it, which totally orders buffered writes and lets the
    /// flush points wait for in-flight writes by acquiring it.
    writer: tokio::sync::Mutex<()>,
    /// Per-request size caps applied when a sealed batch is written.
    limits: BatchLimits,
    /// Wakes a flusher waiting for the open batch's holds to be released.
    unheld: Notify,
}

/// Buffer state guarded by the coalescer lock.
#[derive(Debug, Default)]
struct CoalescerState {
    /// Updates (each paired with its lifecycle event) awaiting the next
    /// flush, in join order.
    pending: Vec<TrackedUpdate>,
    /// One checkpoint permit per admitted join of the pending updates. The
    /// permits travel with the updates to the flusher, which releases them
    /// after the response is merged and published.
    permits: Vec<Permit>,
    /// The batch the pending updates belong to, if one is open.
    batch: Option<Arc<CheckpointBatch>>,
    /// Write failures observed by flushers, retained for the
    /// end-of-invocation flush point (issue #43).
    ///
    /// A spawned batch flush publishes its error only to the batch's
    /// contributors, and every contributor may already be dropped (a
    /// lost `race`/`select_ok` branch, a dropped `DurableFuture`). Without
    /// this retention such a failure would be fully discarded, leaving
    /// the affected operations' records claiming less than what executed.
    /// The flush point drains these via
    /// [`CheckpointCoalescer::take_failed_flushes`] and applies the #43
    /// classification (retryable fails the invocation; non-retryable
    /// persists terminal `FAIL` records, then fails the execution).
    failed: Vec<FailedFlush>,
    /// The FIRST write failure this invocation's buffered channel
    /// suffered: the failure latch (issue #43).
    ///
    /// Once set, no buffered write may reach the backend again this
    /// invocation: "retryable exhaustion fails the invocation with no
    /// further writes", and a write queued behind the failure would
    /// otherwise persist replay-visible transitions after the invocation
    /// is already doomed. Every flusher checks the latch under the
    /// [`CheckpointCoalescer::writer_lock`] before writing (the latch is
    /// always set by the failing flusher while it still holds that lock,
    /// so a queued flusher cannot race past it); on a hit it publishes
    /// the latched error to its contributors and retains its updates as
    /// unwritten instead of calling the backend. Deliberately sticky:
    /// [`CheckpointCoalescer::take_failed_flushes`] drains the retained
    /// failures but never clears the latch: the coalescer lives for one
    /// invocation, and its write channel does not come back within it.
    latched: Option<ExecutionClientError>,
    /// The reservations that hold the open batch (see
    /// [`CheckpointCoalescer::join_owned`] with `reserve`). A held batch is
    /// not claimed for a flush.
    holds: usize,
}

/// The checkpoint permits of one claimed batch. Dropping them releases the
/// batch's hold on the invocation.
#[derive(Debug)]
pub(crate) struct BatchPermits {
    /// Held until the batch is released, then dropped. Read only by
    /// tests; the field exists for its drop.
    #[cfg_attr(not(test), expect(dead_code))]
    // reason: the permits are held for their drop, which releases them.
    held: Vec<Permit>,
}

impl BatchPermits {
    /// The number of permits.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.held.len()
    }
}

/// One checkpoint write failure retained by the coalescer: the error plus
/// the updates the write did not persist (issue #43).
#[derive(Debug)]
pub(crate) struct FailedFlush {
    /// The client error the write failed with.
    pub(crate) error: ExecutionClientError,
    /// The updates that were NOT persisted by the failed write: the
    /// rejected chunk and everything after it, never a chunk that already
    /// succeeded (a terminal `FAIL` must not be written over a persisted
    /// outcome).
    pub(crate) unwritten: Vec<OperationUpdate>,
}

/// The rendezvous for one coalesced checkpoint call: contributors await its
/// published result, and exactly one flusher publishes it.
#[derive(Debug, Default)]
pub(crate) struct CheckpointBatch {
    notify: Notify,
    result: Mutex<Option<Result<CheckpointOutput, ExecutionClientError>>>,
    /// The wire IDs whose classification a failed write of this batch
    /// records. A refresh joins with its first due registration's wire ID,
    /// because the driver can drop the refresh's caller before the batch
    /// is written. The publisher records the failure for each ID, so the
    /// failure decides the invocation's outcome with or without a reader.
    failure_owners: Mutex<Vec<String>>,
}

impl CheckpointBatch {
    /// Returns a clone of the published batch result, if the flush has
    /// completed.
    pub(crate) fn result_clone(&self) -> Option<Result<CheckpointOutput, ExecutionClientError>> {
        self.result
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Publishes the flush result and wakes every waiting contributor.
    pub(crate) fn publish(&self, result: Result<CheckpointOutput, ExecutionClientError>) {
        *self.result.lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
        self.notify.notify_waiters();
    }

    /// Takes the wire IDs whose classification a failed write of this
    /// batch records (see [`CheckpointCoalescer::join_owned`]).
    pub(crate) fn take_failure_owners(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .failure_owners
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// Returns a future that resolves when the batch result is published.
    /// Callers must `enable()` the future *before* checking
    /// [`Self::result_clone`], so a publish between the check and the await
    /// cannot be missed.
    pub(crate) fn notified(&self) -> Notified<'_> {
        self.notify.notified()
    }
}

impl CheckpointCoalescer {
    /// Creates a coalescer with the given delay window and the default
    /// per-request [`BatchLimits`].
    pub(crate) fn new(delay: Duration) -> Self {
        Self::with_limits(delay, BatchLimits::default())
    }

    /// Creates a coalescer with explicit per-request size caps. Tests use
    /// this to force batch splits with small fan-outs.
    pub(crate) fn with_limits(delay: Duration, limits: BatchLimits) -> Self {
        Self {
            delay,
            state: Mutex::new(CoalescerState::default()),
            writer: tokio::sync::Mutex::new(()),
            limits,
            unheld: Notify::new(),
        }
    }

    /// The coalescing window.
    pub(crate) fn delay(&self) -> Duration {
        self.delay
    }

    /// The per-request size caps applied when a sealed batch is written.
    pub(crate) fn limits(&self) -> BatchLimits {
        self.limits
    }

    /// The single-writer lock. Every claimed batch must be written while
    /// holding it (see the module invariants); a flush point acquires it to
    /// wait for any in-flight write before returning.
    pub(crate) fn writer_lock(&self) -> &tokio::sync::Mutex<()> {
        &self.writer
    }

    /// [`Self::join_owned`] without a failure owner. Tests use it.
    #[cfg(test)]
    pub(crate) fn join(
        &self,
        admission: &Admission,
        coordinator: &Arc<InvocationCoordinator>,
        owners: &[String],
        updates: Vec<TrackedUpdate>,
    ) -> Option<(Arc<CheckpointBatch>, bool)> {
        self.join_owned(admission, coordinator, owners, updates, None, false)
    }

    /// Adds `updates` to the buffer and returns the batch they joined,
    /// opening a new batch if none is pending, and whether this join opened
    /// it. The buffer takes ownership of each update's lifecycle event along
    /// with it: from this point the flusher that persists the update, not
    /// the (cancellable) contributor, is responsible for emitting it.
    ///
    /// An admitted join takes a checkpoint permit of `coordinator`, under
    /// the buffer lock, and buffers it with the updates. So the invocation
    /// does not suspend while an admitted update waits in the buffer.
    ///
    /// Returns `None` and buffers nothing when `admission` rejects one of
    /// `owners` (see [`Admission`]), or when the coordinator refuses the
    /// permit. The checks run under the buffer lock. So an update is either
    /// buffered before its owner is abandoned, and written before any update
    /// buffered later, or it is rejected.
    ///
    /// With `failure_owner` set, the join adds it to the batch's failure
    /// owners under the buffer lock. A batch is claimed under the same
    /// lock, so the owner is in place before the batch can be written and
    /// its result published.
    ///
    /// With `reserve`, the join also holds the batch: no flusher claims it
    /// until [`Self::release_hold`] is called once for this join.
    pub(crate) fn join_owned(
        &self,
        admission: &Admission,
        coordinator: &Arc<InvocationCoordinator>,
        owners: &[String],
        updates: Vec<TrackedUpdate>,
        failure_owner: Option<&str>,
        reserve: bool,
    ) -> Option<(Arc<CheckpointBatch>, bool)> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if !admission.admits(owners) {
            return None;
        }
        let permit = coordinator.acquire(PermitKind::Checkpoint)?;
        state.pending.extend(updates);
        state.permits.push(permit);
        if reserve {
            state.holds = state.holds.saturating_add(1);
        }
        let opened = state.batch.is_none();
        let batch = Arc::clone(state.batch.get_or_insert_with(Arc::default));
        if let Some(owner) = failure_owner {
            batch
                .failure_owners
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(owner.to_owned());
        }
        Some((batch, opened))
    }

    /// Releases one hold of the open batch, and wakes a flusher waiting
    /// for it (see [`Self::join_owned`]).
    pub(crate) fn release_hold(&self) {
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.holds = state.holds.saturating_sub(1);
        }
        self.unheld.notify_waiters();
    }

    /// Waits until no reservation holds the open batch, then runs `take`
    /// under the buffer lock. A hold is taken under the same lock, so no
    /// hold is added between the check and the claim.
    async fn when_unheld<T>(&self, take: impl Fn(&mut CoalescerState) -> T) -> T {
        loop {
            // `enable()` registers the waiter before the check, so a
            // release between the check and the await still wakes it.
            let mut unheld = std::pin::pin!(self.unheld.notified());
            unheld.as_mut().enable();
            {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if state.holds == 0 {
                    return take(&mut state);
                }
            }
            unheld.await;
        }
    }

    /// Claims `target`'s buffered updates for flushing, if `target` is still
    /// the open batch, once no reservation holds it. Returns `None` when
    /// another flusher already claimed it (its result will be published by
    /// that flusher).
    ///
    /// The returned permits are the batch's checkpoint permits: the flusher
    /// drops them after the response is merged and published.
    pub(crate) async fn take_batch_unheld(
        &self,
        target: &Arc<CheckpointBatch>,
    ) -> Option<(Vec<TrackedUpdate>, BatchPermits)> {
        self.when_unheld(|state| take_target(state, target)).await
    }

    /// [`Self::take_batch_unheld`] without the wait for holds. Tests use it.
    #[cfg(test)]
    pub(crate) fn take_batch(
        &self,
        target: &Arc<CheckpointBatch>,
    ) -> Option<(Vec<TrackedUpdate>, BatchPermits)> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        take_target(&mut state, target)
    }

    /// Claims whatever batch is open, once no reservation holds it. Used by
    /// the flush points (suspension, end of invocation) that must drain the
    /// buffer regardless of which batch is pending. Returns `None` when the
    /// buffer is idle.
    pub(crate) async fn take_any_unheld(
        &self,
    ) -> Option<(Arc<CheckpointBatch>, Vec<TrackedUpdate>, BatchPermits)> {
        self.when_unheld(take_open).await
    }

    /// [`Self::take_any_unheld`] without the wait for holds. Tests use it.
    #[cfg(test)]
    pub(crate) fn take_any(
        &self,
    ) -> Option<(Arc<CheckpointBatch>, Vec<TrackedUpdate>, BatchPermits)> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        take_open(&mut state)
    }

    /// Drops the open batch without publishing it, with its updates and
    /// its permits. The withdrawal of the checkpoint token calls it (see
    /// `InvocationCoordinator::withdraw`), so the batch's contributors
    /// never resolve. A hold on the batch stays counted until its
    /// reservation is released, as every reservation is when it drops; a
    /// flusher waiting for the holds then finds no batch. Returns the
    /// permits, so the caller releases them outside the buffer lock.
    pub(crate) fn abandon_pending(&self) -> BatchPermits {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.batch = None;
        state.pending.clear();
        BatchPermits {
            held: std::mem::take(&mut state.permits),
        }
    }

    /// Retains a checkpoint write failure for the end-of-invocation flush
    /// point (issue #43): `unwritten` are the updates the failed write did
    /// not persist. Also latches the failure (first error wins) so no
    /// later buffered write reaches the backend this invocation: see
    /// [`CoalescerState::latched`].
    pub(crate) fn record_failed_flush(
        &self,
        error: ExecutionClientError,
        unwritten: Vec<OperationUpdate>,
    ) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.latched.is_none() {
            state.latched = Some(error.clone());
        }
        state.failed.push(FailedFlush { error, unwritten });
    }

    /// The first write failure this invocation's buffered channel
    /// suffered, if any: the failure latch (issue #43). A flusher that
    /// observes it (under the writer lock) must not call the backend: it
    /// publishes this error to its contributors and retains its updates
    /// via [`Self::record_failed_flush`] instead.
    pub(crate) fn latched_failure(&self) -> Option<ExecutionClientError> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.latched.clone()
    }

    /// The number of buffered updates. Tests use it to wait until an
    /// update is buffered.
    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pending
            .len()
    }

    /// The IDs of the buffered updates, in join order.
    #[cfg(test)]
    pub(crate) fn pending_ids(&self) -> Vec<String> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pending
            .iter()
            .map(|tracked| tracked.update.id().to_owned())
            .collect()
    }

    /// Drains the retained write failures, in occurrence order. The
    /// failure latch is deliberately NOT cleared (see
    /// [`CoalescerState::latched`]): the write channel stays poisoned for
    /// the rest of the invocation.
    pub(crate) fn take_failed_flushes(&self) -> Vec<FailedFlush> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut state.failed)
    }
}

/// Claims `target`'s updates and permits when it is the open batch.
fn take_target(
    state: &mut CoalescerState,
    target: &Arc<CheckpointBatch>,
) -> Option<(Vec<TrackedUpdate>, BatchPermits)> {
    match &state.batch {
        Some(open) if Arc::ptr_eq(open, target) => {
            state.batch = None;
            Some((
                std::mem::take(&mut state.pending),
                BatchPermits {
                    held: std::mem::take(&mut state.permits),
                },
            ))
        }
        _ => None,
    }
}

/// Claims the open batch, its updates and its permits, if one is open.
fn take_open(
    state: &mut CoalescerState,
) -> Option<(Arc<CheckpointBatch>, Vec<TrackedUpdate>, BatchPermits)> {
    let batch = state.batch.take()?;
    Some((
        batch,
        std::mem::take(&mut state.pending),
        BatchPermits {
            held: std::mem::take(&mut state.permits),
        },
    ))
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;

    fn dummy_update(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id.to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Start)
            .build()
            .expect("all required OperationUpdate fields set")
    }

    /// Pairs a test update with a captured lifecycle event, as the
    /// checkpoint chokepoint does in production.
    fn track(update: OperationUpdate) -> TrackedUpdate {
        TrackedUpdate::capture(update, 1)
    }

    /// Joins one update from the root, which no abandonment can reject.
    fn join(coalescer: &CheckpointCoalescer, id: &str) -> Arc<CheckpointBatch> {
        coalescer
            .join(
                &Admission::new(),
                &InvocationCoordinator::new(),
                &[],
                vec![track(dummy_update(id))],
            )
            .expect("the root is always admitted")
            .0
    }

    #[test]
    fn admission_rejects_an_abandoned_owner_and_its_descendants() {
        let admission = Admission::new();
        let outer = vec!["item".to_owned()];
        let nested = vec!["item".to_owned(), "inner".to_owned()];
        let sibling = vec!["other".to_owned()];
        assert!(admission.admits(&outer), "nothing is abandoned yet");
        admission.abandon(["item"]);
        assert!(!admission.admits(&outer), "the abandoned owner is rejected");
        assert!(
            !admission.admits(&nested),
            "a context nested in the abandoned owner is rejected"
        );
        assert!(admission.admits(&sibling), "a sibling owner is admitted");
        assert!(admission.admits(&[]), "the root is admitted");
    }

    /// A task that never finishes, and its abort handle.
    fn parked() -> (tokio::task::JoinHandle<()>, AbortHandle) {
        #[expect(clippy::disallowed_methods)]
        // reason: a plain task stands in for an operation task; only its
        // abort handle is registered.
        let task = tokio::spawn(std::future::pending::<()>());
        let abort = task.abort_handle();
        (task, abort)
    }

    async fn is_cancelled(task: tokio::task::JoinHandle<()>) -> bool {
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .is_ok_and(|joined| joined.is_err_and(|err| err.is_cancelled()))
    }

    #[tokio::test]
    async fn closing_a_scope_aborts_its_nested_tasks_only() {
        let admission = Admission::new();
        let (nested, nested_abort) = parked();
        let (sibling, sibling_abort) = parked();
        admission.register_task(&["item".to_owned(), "inner".to_owned()], nested_abort);
        admission.register_task(&["other".to_owned()], sibling_abort);
        admission.abandon(["item"]);
        assert!(is_cancelled(nested).await, "the nested task is aborted");
        assert!(!sibling.is_finished(), "the sibling task keeps running");
        assert_eq!(admission.registered(), 1, "the sibling stays registered");
        sibling.abort();
    }

    #[tokio::test]
    async fn a_task_registered_in_a_closed_scope_is_aborted_at_once() {
        let admission = Admission::new();
        admission.abandon(["item"]);
        let (task, abort) = parked();
        admission.register_task(&["item".to_owned()], abort);
        assert!(is_cancelled(task).await, "the late task is aborted");
        assert_eq!(admission.registered(), 0);
    }

    #[tokio::test]
    async fn registration_forgets_finished_tasks() {
        let admission = Admission::new();
        let (finished, finished_abort) = parked();
        finished.abort();
        assert!(is_cancelled(finished).await);
        admission.register_task(&["a".to_owned()], finished_abort);
        let (running, running_abort) = parked();
        admission.register_task(&["b".to_owned()], running_abort);
        assert_eq!(admission.registered(), 1, "the finished task is forgotten");
        running.abort();
    }

    #[tokio::test]
    async fn scope_close_closes_its_scope_on_drop() {
        let admission = Arc::new(Admission::new());
        let guard = ScopeClose::new(&admission, "item");
        assert!(admission.admits(&["item".to_owned()]));
        drop(guard);
        assert!(!admission.admits(&["item".to_owned()]));
    }

    #[test]
    fn join_buffers_nothing_for_an_abandoned_owner() {
        let coalescer = CheckpointCoalescer::new(Duration::ZERO);
        let admission = Admission::new();
        admission.abandon(["item"]);
        let rejected = coalescer.join(
            &admission,
            &InvocationCoordinator::new(),
            &["item".to_owned()],
            vec![track(dummy_update("a"))],
        );
        assert!(rejected.is_none(), "the rejected join returns no batch");
        assert!(
            coalescer.take_any().is_none(),
            "the rejected join opened no batch and buffered nothing"
        );
    }

    fn dummy_output() -> CheckpointOutput {
        CheckpointOutput {
            checkpoint_token: Some("token-1".to_owned()),
            updated_operations: Vec::new(),
            next_marker: None,
        }
    }

    #[test]
    fn joins_accumulate_into_one_batch() {
        let coalescer = CheckpointCoalescer::new(Duration::from_millis(10));
        let first = join(&coalescer, "a");
        let second = join(&coalescer, "b");
        assert!(
            Arc::ptr_eq(&first, &second),
            "concurrent joins share one batch"
        );

        let (updates, permits) = coalescer.take_batch(&first).expect("batch is open");
        assert_eq!(updates.len(), 2, "both joins' updates drain together");
        assert_eq!(permits.len(), 2, "each join's permit drains with them");
    }

    #[test]
    fn take_batch_is_a_no_op_for_a_stale_batch() {
        let coalescer = CheckpointCoalescer::new(Duration::from_millis(10));
        let old = join(&coalescer, "a");
        assert!(coalescer.take_batch(&old).is_some());

        // A new batch opens after the take; the stale handle cannot claim it.
        let new = join(&coalescer, "b");
        assert!(!Arc::ptr_eq(&old, &new), "a fresh batch opened");
        assert!(
            coalescer.take_batch(&old).is_none(),
            "stale flusher must not steal the newer batch"
        );
        assert_eq!(
            coalescer.take_any().map(|(_, u, _)| u.len()),
            Some(1),
            "the newer batch still holds its update"
        );
    }

    #[test]
    fn take_any_on_idle_buffer_is_none() {
        let coalescer = CheckpointCoalescer::new(Duration::from_millis(10));
        assert!(coalescer.take_any().is_none());
    }

    #[test]
    fn split_respects_operation_count_cap_and_preserves_order() {
        let limits = BatchLimits {
            max_operations: 2,
            max_payload_bytes: usize::MAX,
        };
        let updates = vec![
            track(dummy_update("a")),
            track(dummy_update("b")),
            track(dummy_update("c")),
            track(dummy_update("d")),
            track(dummy_update("e")),
        ];
        let requests = split_into_requests(updates, &limits);
        assert_eq!(
            requests.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![2, 2, 1],
            "five updates under a two-op cap split into 2+2+1"
        );
        let flat: Vec<&str> = requests
            .iter()
            .flatten()
            .map(|t| t.update.id.as_str())
            .collect();
        assert_eq!(
            flat,
            vec!["a", "b", "c", "d", "e"],
            "join order is preserved across splits"
        );
    }

    #[test]
    fn split_respects_payload_byte_cap() {
        let payload_update = |id: &str, payload_len: usize| {
            OperationUpdate::builder()
                .id(id.to_owned())
                .r#type(aws_sdk_lambda::types::OperationType::Step)
                .action(aws_sdk_lambda::types::OperationAction::Succeed)
                .payload("x".repeat(payload_len))
                .build()
                .expect("all required OperationUpdate fields set")
        };
        // Each update estimates to a bit over 1 KiB; a 2.5 KiB cap fits two.
        let limits = BatchLimits {
            max_operations: usize::MAX,
            max_payload_bytes: 2560,
        };
        let updates = vec![
            track(payload_update("a", 1024)),
            track(payload_update("b", 1024)),
            track(payload_update("c", 1024)),
        ];
        let requests = split_into_requests(updates, &limits);
        assert_eq!(
            requests.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1, 1],
            "1 KiB payloads plus overhead exceed the cap pairwise, so each goes alone"
        );

        let small = vec![
            track(payload_update("a", 10)),
            track(payload_update("b", 10)),
        ];
        let requests = split_into_requests(small, &limits);
        assert_eq!(
            requests.len(),
            1,
            "small updates within the cap stay in one request"
        );
    }

    #[test]
    fn oversized_single_update_goes_alone() {
        let big = OperationUpdate::builder()
            .id("big".to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Succeed)
            .payload("x".repeat(4096))
            .build()
            .expect("all required OperationUpdate fields set");
        let limits = BatchLimits {
            max_operations: 10,
            max_payload_bytes: 1024,
        };
        let requests = split_into_requests(
            vec![
                track(dummy_update("a")),
                track(big),
                track(dummy_update("b")),
            ],
            &limits,
        );
        assert_eq!(
            requests.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1, 1],
            "an update larger than the cap is emitted as its own request"
        );
        let middle = requests
            .get(1)
            .and_then(|r| r.first())
            .expect("the middle request holds the oversized update");
        assert_eq!(middle.update.id, "big");
    }

    #[test]
    fn estimated_size_counts_payload_and_error_strings() {
        let with_payload = OperationUpdate::builder()
            .id("p".to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Succeed)
            .payload("y".repeat(500))
            .build()
            .expect("all required OperationUpdate fields set");
        let bare = dummy_update("p");
        assert!(
            estimated_update_size(&with_payload) >= estimated_update_size(&bare) + 500,
            "the payload string must count toward the estimate"
        );
    }

    #[test]
    fn json_escaped_len_matches_serde_json_output() {
        let cases = [
            "plain alphanumeric text",
            r#"quote-heavy "" \" \\ payload"#,
            "control\tchars\nand\rmore\u{08}\u{0C}",
            "\u{01}\u{1F}", // non-shorthand control chars escape as \uXXXX
            "unicode: héllo wörld – 日本語 🦀",
            "",
        ];
        for s in cases {
            let wire = serde_json::to_string(s).expect("string serializes");
            assert_eq!(
                json_escaped_len(s),
                wire.len() - 2, // minus the surrounding quotes
                "escaped length must match serde_json wire bytes for {s:?}"
            );
        }
    }

    #[test]
    fn split_accounts_for_json_escaping_of_payloads() {
        // Each payload is 1000 raw bytes of `"`, which escapes to 2000 wire
        // bytes. Under raw-byte counting, two updates estimate to
        // 2 * (1000 + overhead) and fit a 3000-byte cap in one oversized
        // request; escaped counting sees 2 * (2000 + overhead) and splits.
        let quote_update = |id: &str| {
            OperationUpdate::builder()
                .id(id.to_owned())
                .r#type(aws_sdk_lambda::types::OperationType::Step)
                .action(aws_sdk_lambda::types::OperationAction::Succeed)
                .payload("\"".repeat(1000))
                .build()
                .expect("all required OperationUpdate fields set")
        };
        let limits = BatchLimits {
            max_operations: usize::MAX,
            max_payload_bytes: 3000,
        };
        let updates = vec![track(quote_update("a")), track(quote_update("b"))];
        let requests = split_into_requests(updates, &limits);
        assert_eq!(
            requests.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 1],
            "escape-heavy payloads must split on their wire size, not raw bytes"
        );
    }

    #[test]
    fn estimated_size_uses_escaped_bytes_for_all_string_fields() {
        let escaped = OperationUpdate::builder()
            .id("\"\"".to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Succeed)
            .name("\\\\".to_owned())
            .payload("\n\n".to_owned())
            .build()
            .expect("all required OperationUpdate fields set");
        let plain = OperationUpdate::builder()
            .id("xx".to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Succeed)
            .name("xx".to_owned())
            .payload("xx".to_owned())
            .build()
            .expect("all required OperationUpdate fields set");
        assert_eq!(
            estimated_update_size(&escaped),
            estimated_update_size(&plain) + 6,
            "each of the six escapable characters must count as two wire bytes"
        );
    }

    #[tokio::test]
    async fn publish_wakes_an_enabled_waiter() {
        let batch = Arc::new(CheckpointBatch::default());

        let waiter = {
            let batch = Arc::clone(&batch);
            tokio::spawn(async move {
                loop {
                    let mut notified = std::pin::pin!(batch.notified());
                    notified.as_mut().enable();
                    if let Some(result) = batch.result_clone() {
                        return result;
                    }
                    notified.await;
                }
            })
        };

        batch.publish(Ok(dummy_output()));
        let result = waiter.await.expect("waiter task completes");
        assert_eq!(
            result.expect("published Ok").checkpoint_token(),
            Some("token-1"),
            "waiter observes the published result"
        );
    }

    /// The failure latch (issue #43): the FIRST recorded failure latches
    /// and stays latched: later failures do not replace it, and draining
    /// the retained failures does not clear it, because the buffered
    /// write channel does not come back within an invocation.
    #[test]
    fn failure_latch_is_first_error_and_sticky() {
        let coalescer = CheckpointCoalescer::new(Duration::ZERO);
        assert!(
            coalescer.latched_failure().is_none(),
            "no latch before any failure"
        );

        coalescer.record_failed_flush(
            ExecutionClientError::from_retryable("first failure".to_owned()),
            vec![dummy_update("a")],
        );
        coalescer.record_failed_flush(
            ExecutionClientError::new_non_retryable("second failure"),
            vec![dummy_update("b")],
        );

        let latched = coalescer.latched_failure().expect("latch is set");
        assert!(
            latched.to_string().contains("first failure"),
            "the first failure wins the latch: {latched}"
        );

        let drained = coalescer.take_failed_flushes();
        assert_eq!(drained.len(), 2, "both failures retained in order");
        assert!(
            coalescer.take_failed_flushes().is_empty(),
            "retained failures drain once"
        );
        assert!(
            coalescer.latched_failure().is_some(),
            "the latch survives the drain — the channel stays poisoned"
        );
    }

    #[test]
    fn admitted_join_holds_a_checkpoint_permit_until_the_batch_is_released() {
        let coordinator = InvocationCoordinator::new();
        let coalescer = CheckpointCoalescer::new(Duration::ZERO);
        let (batch, opened) = coalescer
            .join(
                &Admission::new(),
                &coordinator,
                &[],
                vec![track(dummy_update("a"))],
            )
            .expect("admitted");
        assert!(opened, "the first join opens the batch");
        let (_, second_opened) = coalescer
            .join(
                &Admission::new(),
                &coordinator,
                &[],
                vec![track(dummy_update("b"))],
            )
            .expect("admitted");
        assert!(!second_opened, "a later join joins the open batch");
        assert_eq!(coordinator.permits(PermitKind::Checkpoint), 2);
        let (_, permits) = coalescer.take_batch(&batch).expect("open");
        assert_eq!(coordinator.permits(PermitKind::Checkpoint), 2);
        drop(permits);
        assert_eq!(coordinator.permits(PermitKind::Checkpoint), 0);
    }

    #[test]
    fn join_after_suspension_buffers_nothing() {
        let coordinator = InvocationCoordinator::new();
        coordinator.finish();
        let coalescer = CheckpointCoalescer::new(Duration::ZERO);
        let refused = coalescer.join(
            &Admission::new(),
            &coordinator,
            &[],
            vec![track(dummy_update("a"))],
        );
        assert!(refused.is_none());
        assert!(coalescer.take_any().is_none(), "nothing was buffered");
    }
}
