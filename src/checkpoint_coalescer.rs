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
//!   cannot proceed while any earlier claimed batch is still in flight —
//!   this is what makes `DurableContext::flush_pending_checkpoints` a true
//!   barrier rather than a best-effort drain.
//! - A batch is flushed by whoever gets to it first — a contributor whose
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
//!   flusher — spawned or flush-point — checks the latch under the writer
//!   lock before writing, and on a hit publishes the latched error to its
//!   contributors and retains its updates as unwritten (issue #43,
//!   "retryable exhaustion fails the invocation with no further writes").
//!   The terminal `FAIL` writes of the non-retryable classification are
//!   unaffected: they go through the direct (uncoalesced) path.
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

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use aws_sdk_lambda::types::OperationUpdate;
use tokio::sync::Notify;
use tokio::sync::futures::Notified;

use crate::client::{CheckpointOutput, ClientError};
use crate::tracing_layer::PendingTransitionEvent;

/// An operation update paired with the lifecycle event describing the
/// transition it records (see
/// [`PendingTransitionEvent`](crate::tracing_layer::PendingTransitionEvent)).
///
/// The pair travels together through the whole write path — into the
/// coalescing buffer, across batch splits, and into the flush task — so
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
/// control characters expand under escaping — a `"` becomes the two bytes
/// `\"` — so raw `str::len` under-estimates escape-heavy payloads and would
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
}

/// Buffer state guarded by the coalescer lock.
#[derive(Debug, Default)]
struct CoalescerState {
    /// Updates (each paired with its lifecycle event) awaiting the next
    /// flush, in join order.
    pending: Vec<TrackedUpdate>,
    /// The batch the pending updates belong to, if one is open.
    batch: Option<Arc<CheckpointBatch>>,
    /// Write failures observed by flushers, retained for the
    /// end-of-invocation flush point (issue #43).
    ///
    /// A spawned batch flush publishes its error only to the batch's
    /// contributors — and every contributor may already be dropped (a
    /// lost `race`/`select_ok` branch, a dropped `DurableFuture`). Without
    /// this retention such a failure would be fully discarded, leaving
    /// the affected operations' records claiming less than what executed.
    /// The flush point drains these via
    /// [`CheckpointCoalescer::take_failed_flushes`] and applies the #43
    /// classification (retryable fails the invocation; non-retryable
    /// persists terminal `FAIL` records, then fails the execution).
    failed: Vec<FailedFlush>,
    /// The FIRST write failure this invocation's buffered channel
    /// suffered — the failure latch (issue #43).
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
    /// failures but never clears the latch — the coalescer lives for one
    /// invocation, and its write channel does not come back within it.
    latched: Option<ClientError>,
}

/// One checkpoint write failure retained by the coalescer: the error plus
/// the updates the write did not persist (issue #43).
#[derive(Debug)]
pub(crate) struct FailedFlush {
    /// The client error the write failed with.
    pub(crate) error: ClientError,
    /// The updates that were NOT persisted by the failed write — the
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
    result: Mutex<Option<Result<CheckpointOutput, ClientError>>>,
}

impl CheckpointBatch {
    /// Returns a clone of the published batch result, if the flush has
    /// completed.
    pub(crate) fn result_clone(&self) -> Option<Result<CheckpointOutput, ClientError>> {
        self.result
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Publishes the flush result and wakes every waiting contributor.
    pub(crate) fn publish(&self, result: Result<CheckpointOutput, ClientError>) {
        *self.result.lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
        self.notify.notify_waiters();
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

    /// Adds `updates` to the buffer and returns the batch they joined,
    /// opening a new batch if none is pending. The buffer takes ownership
    /// of each update's lifecycle event along with it: from this point the
    /// flusher that persists the update — not the (cancellable)
    /// contributor — is responsible for emitting it.
    pub(crate) fn join(&self, updates: Vec<TrackedUpdate>) -> Arc<CheckpointBatch> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.pending.extend(updates);
        Arc::clone(state.batch.get_or_insert_with(Arc::default))
    }

    /// Claims `target`'s buffered updates for flushing, if `target` is still
    /// the open batch. Returns `None` when another flusher already claimed
    /// it (its result will be published by that flusher).
    pub(crate) fn take_batch(&self, target: &Arc<CheckpointBatch>) -> Option<Vec<TrackedUpdate>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match &state.batch {
            Some(open) if Arc::ptr_eq(open, target) => {
                state.batch = None;
                Some(std::mem::take(&mut state.pending))
            }
            _ => None,
        }
    }

    /// Claims whatever batch is open, unconditionally. Used by the flush
    /// points (suspension, end of invocation) that must drain the buffer
    /// regardless of which batch is pending. Returns `None` when the buffer
    /// is idle.
    pub(crate) fn take_any(&self) -> Option<(Arc<CheckpointBatch>, Vec<TrackedUpdate>)> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let batch = state.batch.take()?;
        Some((batch, std::mem::take(&mut state.pending)))
    }

    /// Retains a checkpoint write failure for the end-of-invocation flush
    /// point (issue #43): `unwritten` are the updates the failed write did
    /// not persist. Also latches the failure (first error wins) so no
    /// later buffered write reaches the backend this invocation — see
    /// [`CoalescerState::latched`].
    pub(crate) fn record_failed_flush(&self, error: ClientError, unwritten: Vec<OperationUpdate>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.latched.is_none() {
            state.latched = Some(error.clone());
        }
        state.failed.push(FailedFlush { error, unwritten });
    }

    /// The first write failure this invocation's buffered channel
    /// suffered, if any — the failure latch (issue #43). A flusher that
    /// observes it (under the writer lock) must not call the backend: it
    /// publishes this error to its contributors and retains its updates
    /// via [`Self::record_failed_flush`] instead.
    pub(crate) fn latched_failure(&self) -> Option<ClientError> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.latched.clone()
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

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/future.rs helpers.
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

    fn dummy_output() -> CheckpointOutput {
        CheckpointOutput {
            checkpoint_token: "token-1".to_owned(),
            updated_operations: Vec::new(),
            next_marker: None,
        }
    }

    #[test]
    fn joins_accumulate_into_one_batch() {
        let coalescer = CheckpointCoalescer::new(Duration::from_millis(10));
        let first = coalescer.join(vec![track(dummy_update("a"))]);
        let second = coalescer.join(vec![track(dummy_update("b"))]);
        assert!(
            Arc::ptr_eq(&first, &second),
            "concurrent joins share one batch"
        );

        let updates = coalescer.take_batch(&first).expect("batch is open");
        assert_eq!(updates.len(), 2, "both joins' updates drain together");
    }

    #[test]
    fn take_batch_is_a_no_op_for_a_stale_batch() {
        let coalescer = CheckpointCoalescer::new(Duration::from_millis(10));
        let old = coalescer.join(vec![track(dummy_update("a"))]);
        assert!(coalescer.take_batch(&old).is_some());

        // A new batch opens after the take; the stale handle cannot claim it.
        let new = coalescer.join(vec![track(dummy_update("b"))]);
        assert!(!Arc::ptr_eq(&old, &new), "a fresh batch opened");
        assert!(
            coalescer.take_batch(&old).is_none(),
            "stale flusher must not steal the newer batch"
        );
        assert_eq!(
            coalescer.take_any().map(|(_, u)| u.len()),
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
            result.expect("published Ok").checkpoint_token,
            "token-1",
            "waiter observes the published result"
        );
    }

    /// The failure latch (issue #43): the FIRST recorded failure latches
    /// and stays latched — later failures do not replace it, and draining
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
            ClientError::from_retryable("first failure".to_owned()),
            vec![dummy_update("a")],
        );
        coalescer.record_failed_flush(
            ClientError::new_non_retryable("second failure"),
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
}
