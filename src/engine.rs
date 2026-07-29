//! Internal replay engine: positional ID minting, wire ID hashing,
//! checkpoint-log pairing, and replay-mode detection.
//!
//! This module is a private engine concern — nothing here is part of the
//! public API surface.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

// ────────────────────────────────────────────────────────────────────────────
// Operation ID
// ────────────────────────────────────────────────────────────────────────────

/// A minted operation identity — the positional path and its SHA-256 wire
/// form, computed once at mint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperationId {
    /// The human-readable positional path (e.g. `"1"`, `"2-1"`, `"2-1-3"`).
    positional: String,
    /// SHA-256 hex digest (64 chars) of the positional string, computed once.
    wire: String,
}

impl OperationId {
    /// Returns the positional string.
    #[allow(dead_code)] // reason: used by operation execution
    pub(crate) fn positional(&self) -> &str {
        &self.positional
    }

    /// Returns the 64-hex-char wire ID (SHA-256 of the positional string).
    #[allow(dead_code)] // reason: used by the checkpoint client
    pub(crate) fn wire(&self) -> &str {
        &self.wire
    }
}

/// Computes the SHA-256 hex digest of the given positional ID string.
///
/// The hash input is the raw UTF-8 bytes of the formatted positional string
/// (matching the JS/Go model where `hashId(input)` hashes the string as-is).
fn compute_wire_id(positional: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(positional.as_bytes());
    let digest = hasher.finalize();
    // Full 64-char hex digest — fits the 64-char OperationId wire cap exactly.
    hex::encode_sha256(digest.as_slice())
}

/// Computes the wire ID from a positional string (crate-internal accessor).
///
/// Used by context.rs to look up checkpoint records by positional ID when the
/// log is keyed by wire ID.
pub(crate) fn compute_wire_id_public(positional: &str) -> String {
    compute_wire_id(positional)
}

/// Minimal hex encoder for a SHA-256 digest (32 bytes → 64 hex chars).
/// Avoids pulling in the `hex` crate — the encoding is trivial.
mod hex {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

    pub(super) fn encode_sha256(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            // SAFETY argument: (b >> 4) is in [0, 15] and (b & 0x0f) is in
            // [0, 15], both within HEX_CHARS' length of 16.
            #[allow(clippy::indexing_slicing)] // reason: index ≤ 15 for any u8 half-byte
            {
                out.push(HEX_CHARS[(b >> 4) as usize] as char);
                out.push(HEX_CHARS[(b & 0x0f) as usize] as char);
            }
        }
        out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ID Counter (shared across context clones via Arc)
// ────────────────────────────────────────────────────────────────────────────

/// The per-context operation-ID counter.
///
/// Wrapped in `Arc` so that `DurableContext` clones (which share the same
/// operation namespace) share a single counter. The atomic provides interior
/// mutability without a lock — minting is a single `fetch_add(1, SeqCst)`.
///
/// `SeqCst` is deliberate: the spec's defining invariant is that ID
/// assignment follows program order. `SeqCst` is the only ordering that
/// prevents the compiler and CPU from reordering atomic operations with
/// respect to other memory accesses on the same thread, guaranteeing the
/// "minted at the call site" contract.
#[derive(Debug)]
pub(crate) struct IdCounter {
    prefix: String,
    counter: AtomicU64,
}

impl IdCounter {
    /// Creates a new counter with the given prefix. Counter starts below 1
    /// so the first `mint()` yields position 1.
    pub(crate) fn new(prefix: String) -> Self {
        Self {
            prefix,
            counter: AtomicU64::new(0),
        }
    }

    /// Mints the next operation ID. Synchronous, lock-free.
    ///
    /// This is the call-site claim: every invocation advances the counter by
    /// exactly one position and returns the operation identity.
    pub(crate) fn mint(&self) -> OperationId {
        let n = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        let positional = self.format(n);
        let wire = compute_wire_id(&positional);
        OperationId { positional, wire }
    }

    /// Returns the counter value (number of IDs minted so far).
    #[allow(dead_code)] // reason: used in tests and future phases
    pub(crate) fn count(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Creates a child counter whose prefix is the given positional ID.
    /// Used when spawning child contexts.
    #[allow(dead_code)] // reason: used by EngineState::new_child
    pub(crate) fn child(positional_id: &str) -> Self {
        Self::new(positional_id.to_owned())
    }

    fn format(&self, n: u64) -> String {
        if self.prefix.is_empty() {
            n.to_string()
        } else {
            format!("{}-{n}", self.prefix)
        }
    }

    /// Returns the prefix of this counter (empty string for root).
    #[allow(dead_code)] // reason: used in tests and future phases
    pub(crate) fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Advances the counter by `n` positions without minting IDs.
    ///
    /// Used when replaying a terminal batch: the child IDs that were minted
    /// during the original execution must be skipped so subsequent operations
    /// receive the correct positional ID.
    pub(crate) fn advance(&self, n: usize) {
        self.counter.fetch_add(n as u64, Ordering::SeqCst);
    }

    /// Peeks at the positional ID that the next `mint()` will produce,
    /// without advancing the counter.
    pub(crate) fn peek_next(&self) -> String {
        let next_n = self.counter.load(Ordering::SeqCst) + 1;
        self.format(next_n)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Checkpoint Log (replay data structure)
// ────────────────────────────────────────────────────────────────────────────

/// The status of a checkpointed operation, as recorded by the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // reason: variants constructed by the checkpoint client
pub(crate) enum CheckpointStatus {
    /// Operation started but not yet resolved.
    Started,
    /// Operation is awaiting an external event.
    Pending,
    /// Operation result is ready to be consumed.
    Ready,
    /// Operation completed successfully.
    Succeeded,
    /// Operation failed.
    Failed,
    /// Operation was cancelled.
    Cancelled,
    /// Operation timed out.
    TimedOut,
    /// Operation was stopped.
    Stopped,
}

impl CheckpointStatus {
    /// Returns true if this status represents a terminal (settled) outcome
    /// that replay can return without re-executing.
    pub(crate) fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::TimedOut | Self::Stopped
        )
    }
}

/// A stored operation record from the checkpoint log.
///
/// Models the checkpoint API response shape. Fields are added as the engine
/// consumes them in later slices.
#[derive(Debug, Clone)]
#[allow(dead_code)] // reason: fields read by operation execution
pub(crate) struct CheckpointRecord {
    /// The wire operation ID (SHA-256 hex) used as the map key.
    pub(crate) id: String,
    /// The operation's status.
    pub(crate) status: CheckpointStatus,
    /// The serialized result payload (for succeeded step operations).
    pub(crate) result: Option<String>,
    /// Error type identifier (for failed/timed-out operations).
    pub(crate) error_type: Option<String>,
    /// Error message (for failed/timed-out operations).
    pub(crate) error_message: Option<String>,
    /// The attempt number from the backend's step details (0 if unavailable).
    pub(crate) attempt: u32,
    /// The serialized result payload from a succeeded chained invoke.
    pub(crate) invoke_result: Option<String>,
    /// Error type from a failed chained invoke.
    pub(crate) invoke_error_type: Option<String>,
    /// Error message from a failed chained invoke.
    pub(crate) invoke_error_message: Option<String>,
    /// Whether the child context result was too large and must be
    /// reconstructed by re-executing the child body (`ReplayChildren` mode).
    pub(crate) replay_children: bool,
    /// The callback ID assigned by the backend (for callback operations).
    pub(crate) callback_id: Option<String>,
}

/// The checkpoint log: maps positional operation IDs to stored records.
///
/// The log is populated from the backend on invocation start and is
/// immutable during execution (new checkpoint results are only visible
/// on the next invocation).
#[derive(Debug)]
pub(crate) struct CheckpointLog {
    /// Records keyed by wire operation ID for fast lookup during replay.
    /// Uses interior mutability so that checkpoint responses can merge
    /// backend-assigned fields (e.g. `callback_id`) into the live log.
    records: RwLock<HashMap<String, CheckpointRecord>>,
}

impl CheckpointLog {
    /// Creates an empty checkpoint log (first-ever invocation).
    pub(crate) fn empty() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a checkpoint log from a list of records.
    ///
    /// Records are stored keyed by their wire ID. The caller must provide
    /// the wire-ID → record mapping (the inline parser uses the `Id` field
    /// from the backend directly as the key).
    #[allow(dead_code)] // reason: used by the checkpoint client and tests
    pub(crate) fn from_records(records: Vec<(String, CheckpointRecord)>) -> Self {
        Self {
            records: RwLock::new(records.into_iter().collect()),
        }
    }

    /// Looks up a stored record by wire operation ID.
    ///
    /// Returns `Some` with the frozen result for replayed operations;
    /// `None` for live (not-yet-checkpointed) operations.
    pub(crate) fn get(&self, wire_id: &str) -> Option<CheckpointRecord> {
        let guard = self
            .records
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get(wire_id).cloned()
    }

    /// Inserts or replaces a record in the log.
    ///
    /// Used to merge backend-assigned fields (e.g. `callback_id`) from
    /// checkpoint responses into the live log so that subsequent reads
    /// in the same invocation observe the updated state.
    pub(crate) fn insert(&self, wire_id: String, record: CheckpointRecord) {
        let mut guard = self
            .records
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(wire_id, record);
    }

    /// Returns the number of terminal records in the log (the high-water
    /// mark for replay detection).
    #[allow(dead_code)] // reason: used for progress reporting
    pub(crate) fn terminal_count(&self) -> usize {
        let guard = self
            .records
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.values().filter(|r| r.status.is_terminal()).count()
    }

    /// Returns true if the log has any records at all.
    pub(crate) fn has_records(&self) -> bool {
        let guard = self
            .records
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        !guard.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Engine State (shared inner for DurableContext)
// ────────────────────────────────────────────────────────────────────────────

/// The shared engine state carried by every `DurableContext` clone.
///
/// The `Arc<EngineState>` is what makes `DurableContext` cheap to clone and
/// share across async boundaries. The `IdCounter` uses interior mutability
/// (atomic) so minting works through a shared reference.
#[derive(Debug)]
pub(crate) struct EngineState {
    /// The ID counter for this context's namespace.
    pub(crate) id_counter: IdCounter,
    /// The checkpoint log for replay pairing.
    pub(crate) checkpoint_log: Arc<CheckpointLog>,
}

impl EngineState {
    /// Creates engine state for a root context.
    pub(crate) fn new_root(checkpoint_log: Arc<CheckpointLog>) -> Self {
        Self {
            id_counter: IdCounter::new(String::new()),
            checkpoint_log,
        }
    }

    /// Creates engine state for a child context whose prefix is the parent
    /// operation's positional ID.
    pub(crate) fn new_child(
        parent_positional_id: &str,
        checkpoint_log: Arc<CheckpointLog>,
    ) -> Self {
        Self {
            id_counter: IdCounter::child(parent_positional_id),
            checkpoint_log,
        }
    }

    /// Mints the next operation ID for this context.
    pub(crate) fn mint_id(&self) -> OperationId {
        self.id_counter.mint()
    }

    /// Determines whether the context is in replay mode for the given
    /// positional operation ID.
    ///
    /// An operation is replaying if the checkpoint log contains a terminal
    /// record for it — meaning the result was frozen in a prior invocation.
    pub(crate) fn is_replaying_at(&self, positional_id: &str) -> bool {
        let wire_id = compute_wire_id(positional_id);
        self.checkpoint_log
            .get(&wire_id)
            .is_some_and(|r| r.status.is_terminal())
    }

    /// Returns whether the context is currently in replay mode (there are
    /// checkpointed records and we have not yet passed the high-water mark).
    ///
    /// The context starts in replay mode if the checkpoint log has any
    /// terminal records, and transitions to live mode when the next minted
    /// ID has no corresponding checkpoint record.
    pub(crate) fn is_replaying(&self) -> bool {
        if !self.checkpoint_log.has_records() {
            return false;
        }
        // The context is replaying as long as the NEXT operation to be
        // claimed would have a checkpoint record (keyed by wire ID).
        let next_positional = self.id_counter.peek_next();
        let next_wire = compute_wire_id(&next_positional);
        self.checkpoint_log
            .get(&next_wire)
            .is_some_and(|r| r.status.is_terminal())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Counter basics ──────────────────────────────────────────────────

    #[test]
    fn counter_starts_at_one() {
        let counter = IdCounter::new(String::new());
        let id = counter.mint();
        assert_eq!(id.positional(), "1");
    }

    #[test]
    fn counter_increments_per_call() {
        let counter = IdCounter::new(String::new());
        let id1 = counter.mint();
        let id2 = counter.mint();
        let id3 = counter.mint();
        assert_eq!(id1.positional(), "1");
        assert_eq!(id2.positional(), "2");
        assert_eq!(id3.positional(), "3");
    }

    #[test]
    fn counter_with_prefix() {
        let counter = IdCounter::new("2".to_owned());
        let id1 = counter.mint();
        let id2 = counter.mint();
        assert_eq!(id1.positional(), "2-1");
        assert_eq!(id2.positional(), "2-2");
    }

    #[test]
    fn counter_advance_skips_positions() {
        // Simulates batch replay: map consumes positions 2,3 during original
        // execution; on replay, advance(2) skips them so the next operation
        // gets position 4.
        let counter = IdCounter::new(String::new());
        let map_id = counter.mint(); // position 1 (the map parent)
        assert_eq!(map_id.positional(), "1");
        counter.advance(2); // skip 2 child positions
        let wait_id = counter.mint(); // should be position 4
        assert_eq!(wait_id.positional(), "4");
    }

    // ── Child prefix chaining produces full positional paths ────────────

    #[test]
    fn child_prefix_chaining() {
        // Root mints "1", "2"; child of "2" mints "2-1", "2-2";
        // grandchild of "2-1" mints "2-1-1", "2-1-2".
        let root = IdCounter::new(String::new());
        let _id1 = root.mint(); // "1"
        let id2 = root.mint(); // "2"
        assert_eq!(id2.positional(), "2");

        let child = IdCounter::child(id2.positional());
        let child_id1 = child.mint();
        let child_id2 = child.mint();
        assert_eq!(child_id1.positional(), "2-1");
        assert_eq!(child_id2.positional(), "2-2");

        let grandchild = IdCounter::child(child_id1.positional());
        let gc_id1 = grandchild.mint();
        let gc_id2 = grandchild.mint();
        assert_eq!(gc_id1.positional(), "2-1-1");
        assert_eq!(gc_id2.positional(), "2-1-2");
    }

    // ── SHA-256 wire ID: known-answer tests ─────────────────────────────

    #[test]
    fn wire_id_known_answer_simple() {
        // SHA-256 of "1" (the string, UTF-8 bytes)
        // Independently computed: echo -n "1" | sha256sum
        // = 6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b
        let id = compute_wire_id("1");
        assert_eq!(
            id,
            "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b"
        );
    }

    #[test]
    fn wire_id_known_answer_prefixed() {
        // SHA-256 of "2-1" (positional string for first child of op 2)
        // echo -n "2-1" | sha256sum
        // = e2433ac3278279ac90b51abed3d4dd9057d9e1f66541ad9cdfd13762a30e5a43
        let id = compute_wire_id("2-1");
        assert_eq!(
            id,
            "e2433ac3278279ac90b51abed3d4dd9057d9e1f66541ad9cdfd13762a30e5a43"
        );
    }

    #[test]
    fn wire_id_known_answer_deep_path() {
        // SHA-256 of "2-1-3"
        // echo -n "2-1-3" | sha256sum
        // = c00876d4d917a061ad5a377270e059994251eea9818733e6abfef69c6080625b
        let id = compute_wire_id("2-1-3");
        assert_eq!(
            id,
            "c00876d4d917a061ad5a377270e059994251eea9818733e6abfef69c6080625b"
        );
    }

    #[test]
    fn wire_id_length_and_charset() {
        let counter = IdCounter::new(String::new());
        for _ in 0..20 {
            let id = counter.mint();
            let wire = id.wire();
            // Must be exactly 64 hex characters.
            assert_eq!(wire.len(), 64);
            assert!(
                wire.chars().all(|c| c.is_ascii_hexdigit()),
                "wire ID contains non-hex character: {wire}"
            );
            // All lowercase hex per SHA-256 convention.
            assert!(
                wire.chars().all(|c| !c.is_ascii_uppercase()),
                "wire ID should be lowercase: {wire}"
            );
        }
    }

    #[test]
    fn wire_id_at_mint_matches_standalone() {
        // Verify the wire ID carried in OperationId matches compute_wire_id.
        let counter = IdCounter::new("prefix".to_owned());
        let id = counter.mint();
        assert_eq!(id.wire(), compute_wire_id(id.positional()));
    }

    // ── IDs minted at call site (creation order, not await order) ───────

    #[tokio::test]
    async fn ids_minted_at_creation_not_await() {
        // The defining test: creating two builders claims IDs in CREATION
        // order. Awaiting in reverse order does not change the IDs.
        let counter = Arc::new(IdCounter::new(String::new()));

        // Simulate two builder creations by minting synchronously.
        let id_first = counter.mint();
        let id_second = counter.mint();

        // Assert creation order is preserved regardless of "await" order.
        assert_eq!(id_first.positional(), "1");
        assert_eq!(id_second.positional(), "2");
    }

    #[tokio::test]
    async fn ids_stable_under_interleaved_join() {
        // join! over builder creations: IDs are trivially ordered because
        // mint is synchronous and executes within the owning task.
        let counter = Arc::new(IdCounter::new(String::new()));

        let c1 = Arc::clone(&counter);
        let c2 = Arc::clone(&counter);

        // tokio::join! polls futures but mint happens before any poll.
        let (id_a, id_b) = tokio::join!(async move { c1.mint() }, async move { c2.mint() },);

        // Both mints happened in the single task running this test (tokio
        // join! does not spawn). The first branch mints first.
        assert_eq!(id_a.positional(), "1");
        assert_eq!(id_b.positional(), "2");
    }

    // ── Checkpoint-log pairing ──────────────────────────────────────────

    #[test]
    fn checkpoint_log_returns_stored_result() {
        let log = CheckpointLog::from_records(vec![(
            "1".to_owned(),
            CheckpointRecord {
                id: "wire-1".to_owned(),
                status: CheckpointStatus::Succeeded,
                result: Some(r#""hello""#.to_owned()),
                error_type: None,
                error_message: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
            },
        )]);

        let record = log.get("1");
        assert!(record.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — value verified present above
        let record = record.unwrap();
        assert_eq!(record.status, CheckpointStatus::Succeeded);
        assert_eq!(record.result.as_deref(), Some(r#""hello""#));
    }

    #[test]
    fn checkpoint_log_miss_for_unknown_id() {
        let log = CheckpointLog::from_records(vec![(
            "1".to_owned(),
            CheckpointRecord {
                id: "wire-1".to_owned(),
                status: CheckpointStatus::Succeeded,
                result: Some(r"42".to_owned()),
                error_type: None,
                error_message: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
            },
        )]);

        assert!(log.get("2").is_none());
        assert!(log.get("unknown").is_none());
    }

    #[test]
    fn checkpoint_log_empty_always_miss() {
        let log = CheckpointLog::empty();
        assert!(log.get("1").is_none());
        assert!(log.get("anything").is_none());
    }

    // ── Replay-mode detection ───────────────────────────────────────────

    #[test]
    fn replay_mode_with_checkpointed_operations() {
        // Log has records keyed by wire IDs (hashes) for positions 1 and 2.
        let wire1 = compute_wire_id("1");
        let wire2 = compute_wire_id("2");
        let log = Arc::new(CheckpointLog::from_records(vec![
            (
                wire1.clone(),
                CheckpointRecord {
                    id: wire1,
                    status: CheckpointStatus::Succeeded,
                    result: Some("a".to_owned()),
                    error_type: None,
                    error_message: None,
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: None,
                },
            ),
            (
                wire2.clone(),
                CheckpointRecord {
                    id: wire2,
                    status: CheckpointStatus::Succeeded,
                    result: Some("b".to_owned()),
                    error_type: None,
                    error_message: None,
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: None,
                },
            ),
        ]));

        let engine = EngineState::new_root(Arc::clone(&log));

        // Before any mint, next is "1" which has a record → replaying.
        assert!(engine.is_replaying());

        // Mint "1" → the record for "1" is terminal → replaying at "1".
        let id1 = engine.mint_id();
        assert_eq!(id1.positional(), "1");
        assert!(engine.is_replaying_at(id1.positional()));

        // Next would be "2" → still replaying.
        assert!(engine.is_replaying());

        // Mint "2" → replaying at "2".
        let id2 = engine.mint_id();
        assert_eq!(id2.positional(), "2");
        assert!(engine.is_replaying_at(id2.positional()));

        // Next would be "3" → no record → NOT replaying (live mode).
        assert!(!engine.is_replaying());
    }

    #[test]
    fn replay_mode_empty_log_never_replaying() {
        let log = Arc::new(CheckpointLog::empty());
        let engine = EngineState::new_root(log);

        assert!(!engine.is_replaying());
        let id = engine.mint_id();
        assert!(!engine.is_replaying_at(id.positional()));
    }

    #[test]
    fn replay_mode_non_terminal_not_replaying() {
        // A "Started" status is non-terminal — operation needs re-execution.
        let wire1 = compute_wire_id("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire1.clone(),
            CheckpointRecord {
                id: wire1,
                status: CheckpointStatus::Started,
                result: None,
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

        let engine = EngineState::new_root(log);
        // Non-terminal record: not replaying.
        assert!(!engine.is_replaying());
    }

    #[test]
    fn replay_mode_child_context() {
        // Child context "2" has operations "2-1" and "2-2" checkpointed.
        let wire_2_1 = compute_wire_id("2-1");
        let wire_2_2 = compute_wire_id("2-2");
        let log = Arc::new(CheckpointLog::from_records(vec![
            (
                wire_2_1.clone(),
                CheckpointRecord {
                    id: wire_2_1,
                    status: CheckpointStatus::Succeeded,
                    result: Some("x".to_owned()),
                    error_type: None,
                    error_message: None,
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: None,
                },
            ),
            (
                wire_2_2.clone(),
                CheckpointRecord {
                    id: wire_2_2,
                    status: CheckpointStatus::Failed,
                    result: None,
                    error_type: Some("StepError".to_owned()),
                    error_message: Some("oops".to_owned()),
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: None,
                },
            ),
        ]));

        let engine = EngineState::new_child("2", log);

        // "2-1" is next → replaying.
        assert!(engine.is_replaying());

        let id1 = engine.mint_id();
        assert_eq!(id1.positional(), "2-1");
        assert!(engine.is_replaying_at(id1.positional()));

        // "2-2" next → still replaying (Failed is terminal).
        assert!(engine.is_replaying());

        let id2 = engine.mint_id();
        assert_eq!(id2.positional(), "2-2");
        assert!(engine.is_replaying_at(id2.positional()));

        // "2-3" next → no record → live.
        assert!(!engine.is_replaying());
    }
}
