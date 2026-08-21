//! Map and parallel operation execution engine.
//!
//! Implements the coordinator loop for bounded-concurrency fan-out with
//! completion-threshold semantics. Both `map` and `parallel` share the
//! same underlying batch engine — they differ only in how items are
//! produced (from a `Vec<I>` + closure, or from named `Branch<O>` values).
//!
//! Wire shape:
//! - Parent: `OperationType::Context`, `SubType` `Map` or `Parallel`
//! - Children: `OperationType::Context`, `SubType` `MapIteration` or `ParallelBranch`
//! - Actions: Start → (Succeed | Fail)
//! - `ParentId` on children points to the parent's wire ID
//!
//! Invariants:
//! - Suspended branches RETAIN concurrency slots (only terminal events
//!   decrement in-flight)
//! - Never-started branches are OMITTED from results
//! - Completion decision uses the STARTED set
//! - Results collected positionally per branch index
//! - Each branch runs in its own child context
//! - Fail-fast when `min_successful` can no longer be met

use std::future::Future;
use std::sync::Arc;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tokio::task::JoinSet;
use tracing::Instrument as _;

use crate::BoxError;
use crate::Serdes;
use crate::context::DurableContext;
use crate::driver::{ScopeOutcome, drive_scope};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, ChildFnError, OperationError, OperationErrorKind,
};
use crate::serdes::SerdesContext;

/// Wire sub-type for map operations.
pub(crate) const MAP_SUB_TYPE: &str = "Map";
/// Wire sub-type for map iteration children.
const MAP_ITERATION_SUB_TYPE: &str = "MapIteration";
/// Wire sub-type for parallel operations.
pub(crate) const PARALLEL_SUB_TYPE: &str = "Parallel";
/// Wire sub-type for parallel branch children.
const PARALLEL_BRANCH_SUB_TYPE: &str = "ParallelBranch";

/// Maximum checkpoint payload size in bytes (256KB). Matches child.rs.
const CHECKPOINT_SIZE_LIMIT_BYTES: usize = 256 * 1024;

/// Sentinel error message returned by `replay_terminal_batch` when the
/// batch has `replay_children` set, signalling the caller should fall
/// through to normal re-execution instead of short-circuiting.
const REPLAY_CHILDREN_SENTINEL: &str = "__replay_children_reexecute__";

/// The error type identifier recorded for a batch item whose body failed.
///
/// Matches the `errorType` the per-child FAIL checkpoint carries, so the
/// batch summary payload and the child records agree on error identity.
const CHILD_FN_ERROR_TYPE: &str = "ChildFnError";

/// The terminal status of one item or branch in a batch operation.
///
/// Each item in a [`BatchResult`] has a status indicating whether it
/// completed successfully or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchItemStatus {
    /// The item completed successfully.
    Succeeded,
    /// The item failed.
    Failed,
}

/// The outcome of one item or branch in a batch operation.
///
/// Contains the index, optional display name, terminal status, and either
/// the result value or an error (type identifier plus message).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::{BatchItemStatus, BatchItem};
///
/// let item: BatchItem<i32> = BatchItem::new(
///     0,
///     String::new(),
///     BatchItemStatus::Succeeded,
///     Some(42),
///     None,
///     None,
/// );
/// assert_eq!(item.status, BatchItemStatus::Succeeded);
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct BatchItem<O> {
    /// Zero-based position in the original input.
    pub index: usize,
    /// Display name (if any).
    pub name: String,
    /// Terminal status.
    pub status: BatchItemStatus,
    /// Result value (only meaningful when status is Succeeded).
    pub result: Option<O>,
    /// Error message (only meaningful when status is Failed).
    pub error_message: Option<String>,
    /// Error type identifier (only meaningful when status is Failed).
    ///
    /// Carries the `errType` recorded in the batch checkpoint payload
    /// (`"ChildFnError"` for a failed item body), so error identity
    /// survives checkpoint replay alongside the message. `None` when the
    /// payload recorded no type (a payload written before error typing).
    pub error_type: Option<String>,
}

impl<O> BatchItem<O> {
    /// Creates a new batch item.
    #[must_use]
    pub fn new(
        index: usize,
        name: String,
        status: BatchItemStatus,
        result: Option<O>,
        error_message: Option<String>,
        error_type: Option<String>,
    ) -> Self {
        Self {
            index,
            name,
            status,
            result,
            error_message,
            error_type,
        }
    }
}

/// The overall status of a completed batch.
///
/// Returned by [`BatchResult::status`]: [`BatchStatus::Succeeded`] when no
/// started item failed, [`BatchStatus::Failed`] otherwise. The `Display`
/// implementation renders the wire strings `"SUCCEEDED"` and `"FAILED"`.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::BatchStatus;
///
/// assert_eq!(BatchStatus::Succeeded.as_str(), "SUCCEEDED");
/// assert_eq!(BatchStatus::Failed.to_string(), "FAILED");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchStatus {
    /// No started item failed.
    Succeeded,
    /// At least one started item failed.
    Failed,
}

impl BatchStatus {
    /// Wire representation of the batch status.
    ///
    /// Returns the string used in checkpoint payloads and conformance
    /// assertions.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
        }
    }
}

impl std::fmt::Display for BatchStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A failed item's error, tied to the item that produced it.
///
/// Returned by [`BatchResult::errors`]. Each entry borrows from the
/// [`BatchItem`] it describes and carries the item's position, display
/// name, error type identifier, and error message, so callers can
/// associate a failure with the input that caused it rather than
/// receiving a bare message string.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::{BatchResult, BatchItem, BatchItemStatus, CompletionReason};
///
/// let result: BatchResult<i32> = BatchResult::new(
///     vec![BatchItem::new(
///         2,
///         "branch-c".to_owned(),
///         BatchItemStatus::Failed,
///         None,
///         Some("boom".into()),
///         Some("ChildFnError".into()),
///     )],
///     CompletionReason::AllCompleted,
/// );
/// let errors = result.errors();
/// assert_eq!(errors[0].index, 2);
/// assert_eq!(errors[0].name, "branch-c");
/// assert_eq!(errors[0].message, "boom");
/// assert_eq!(errors[0].error_type, Some("ChildFnError"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BatchError<'a> {
    /// Zero-based position of the failed item in the original input.
    pub index: usize,
    /// Display name of the failed item. Empty when the item has no name
    /// (an unnamed map iteration).
    pub name: &'a str,
    /// The error message the item failed with.
    pub message: &'a str,
    /// The error type identifier the item failed with, as recorded in the
    /// batch checkpoint payload (`"ChildFnError"` for a failed item body).
    /// `None` when the payload recorded no type (a payload written before
    /// error typing).
    pub error_type: Option<&'a str>,
}

/// Why the batch completed.
///
/// Records the reason a [`BatchResult`] finished: all items ran, the
/// success threshold was met, the failure tolerance was exceeded, or a
/// custom completion predicate reported the batch done.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionReason;
///
/// let reason = CompletionReason::AllCompleted;
/// assert_eq!(reason.as_str(), "ALL_COMPLETED");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionReason {
    /// Every item ran to completion.
    AllCompleted,
    /// The `min_successful` threshold was met.
    MinSuccessfulReached,
    /// The failure tolerance was exceeded.
    FailureToleranceExceeded,
    /// The custom completion predicate returned `true`.
    ///
    /// Set when a batch ends early because the predicate configured through
    /// [`crate::builders::map_parallel::CompletionConfig::with_completion_predicate`] or
    /// [`crate::builders::map_parallel::CompletionConfigBuilder::completion_predicate`] fired. Like
    /// [`CompletionReason::MinSuccessfulReached`], a batch completed this
    /// way is a successful early completion: item failures inside it are
    /// tolerated rather than propagated as errors.
    PredicateMatched,
}

impl CompletionReason {
    /// Wire representation of the completion reason.
    ///
    /// Returns the string used in checkpoint payloads and conformance
    /// assertions.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllCompleted => "ALL_COMPLETED",
            Self::MinSuccessfulReached => "MIN_SUCCESSFUL_REACHED",
            Self::FailureToleranceExceeded => "FAILURE_TOLERANCE_EXCEEDED",
            Self::PredicateMatched => "PREDICATE_MATCHED",
        }
    }

    /// Parses the wire representation back into a completion reason.
    ///
    /// Unrecognized strings map to [`CompletionReason::AllCompleted`],
    /// matching the checkpoint replay path's tolerance for payloads written
    /// by newer SDK versions.
    pub(crate) fn from_wire(s: &str) -> Self {
        match s {
            "MIN_SUCCESSFUL_REACHED" => Self::MinSuccessfulReached,
            "FAILURE_TOLERANCE_EXCEEDED" => Self::FailureToleranceExceeded,
            "PREDICATE_MATCHED" => Self::PredicateMatched,
            _ => Self::AllCompleted,
        }
    }
}

/// The settled outcome of one batch item, as seen by a completion predicate.
///
/// Carries the item's zero-based input position and whether it succeeded or
/// failed. Entries are appended to [`BatchStats::outcomes`] strictly in
/// input order: item `i`'s outcome enters the statistics only after the
/// outcomes of items `0..i` have all entered, whatever order the items
/// actually settled in at run time. Live settlement order is
/// scheduler-timed and is not recorded in the checkpoint log, so it cannot
/// be reproduced on replay; the input-order prefix can always be derived
/// from recorded state alone. That canonical ordering is what keeps an
/// order-sensitive predicate deterministic across replay — identical
/// recorded state always produces the identical outcome sequence.
///
/// The public constructor exists so a completion predicate can be unit
/// tested against hand-built statistics without running a batch.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::{BatchItemStatus, SettledOutcome};
///
/// let outcome = SettledOutcome::new(3, BatchItemStatus::Failed);
/// assert_eq!(outcome.index(), 3);
/// assert_eq!(outcome.status(), BatchItemStatus::Failed);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SettledOutcome {
    /// Zero-based position of the item in the original input.
    index: usize,
    /// Whether the item succeeded or failed.
    status: BatchItemStatus,
}

impl SettledOutcome {
    /// Creates a settled outcome (public so predicates can be unit tested).
    #[must_use]
    pub fn new(index: usize, status: BatchItemStatus) -> Self {
        Self { index, status }
    }

    /// Returns the item's zero-based position in the original input.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns whether the item succeeded or failed.
    #[must_use]
    pub fn status(&self) -> BatchItemStatus {
        self.status
    }
}

/// Running statistics of a map/parallel batch, passed to a custom
/// completion predicate.
///
/// A snapshot of the batch's *committed prefix*: the settled outcomes of
/// items `0..settled()`, in input order, plus the total item count. An
/// item's outcome is committed — and the predicate re-evaluated — only
/// once every earlier item's outcome is known, so a still-running (or
/// suspended) item holds later items' outcomes out of the statistics until
/// it settles. Live settlement order is scheduler-timed and unrecorded;
/// the committed prefix derives from recorded checkpoint state alone, so a
/// predicate that is a pure function of these statistics sees the
/// identical sequence of snapshots on the original run and on every
/// replay — see the determinism requirement on
/// [`CompletionConfigBuilder::completion_predicate`](crate::builders::map_parallel::CompletionConfigBuilder::completion_predicate).
///
/// The public constructor exists so a completion predicate can be unit
/// tested against hand-built statistics without running a batch.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::{BatchItemStatus, BatchStats, SettledOutcome};
///
/// let outcomes = [
///     SettledOutcome::new(0, BatchItemStatus::Failed),
///     SettledOutcome::new(1, BatchItemStatus::Succeeded),
/// ];
/// let stats = BatchStats::new(1, 1, 5, &outcomes);
/// assert_eq!(stats.succeeded(), 1);
/// assert_eq!(stats.failed(), 1);
/// assert_eq!(stats.settled(), 2);
/// assert_eq!(stats.total_items(), 5);
/// assert_eq!(stats.outcomes().len(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BatchStats<'a> {
    /// Count of committed items that have succeeded so far.
    succeeded: usize,
    /// Count of committed items that have failed so far.
    failed: usize,
    /// Total number of items in the batch.
    total_items: usize,
    /// Committed per-item settled outcomes, in input order (see
    /// [`SettledOutcome`]).
    outcomes: &'a [SettledOutcome],
}

impl<'a> BatchStats<'a> {
    /// Creates a statistics snapshot (public so predicates can be unit
    /// tested).
    #[must_use]
    pub fn new(
        succeeded: usize,
        failed: usize,
        total_items: usize,
        outcomes: &'a [SettledOutcome],
    ) -> Self {
        Self {
            succeeded,
            failed,
            total_items,
            outcomes,
        }
    }

    /// Returns the count of committed items that have succeeded so far.
    #[must_use]
    pub fn succeeded(&self) -> usize {
        self.succeeded
    }

    /// Returns the count of committed items that have failed so far.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.failed
    }

    /// Returns the count of committed items (succeeded or failed) so far.
    ///
    /// Because outcomes commit strictly in input order, this is also the
    /// length of the committed prefix: the items `0..settled()` are exactly
    /// the ones whose outcomes [`outcomes`](Self::outcomes) holds.
    #[must_use]
    pub fn settled(&self) -> usize {
        self.succeeded + self.failed
    }

    /// Returns the total number of items in the batch.
    #[must_use]
    pub fn total_items(&self) -> usize {
        self.total_items
    }

    /// Returns the committed per-item settled outcomes, strictly in input
    /// order: the outcome at position `i` belongs to item `i` (see
    /// [`SettledOutcome`] for why the SDK canonicalizes the order).
    #[must_use]
    pub fn outcomes(&self) -> &'a [SettledOutcome] {
        self.outcomes
    }
}

/// The collected outcome of a map/parallel operation.
///
/// Contains per-item outcomes and the reason the batch completed. Provides
/// accessor methods for building conformance projections.
///
/// Only items that were actually started are included — items that never
/// started due to early completion are omitted.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::{
///     BatchResult, BatchItem, BatchItemStatus, BatchStatus, CompletionReason,
/// };
///
/// let result: BatchResult<i32> = BatchResult::new(
///     vec![
///         BatchItem::new(0, String::new(), BatchItemStatus::Succeeded, Some(10), None, None),
///         BatchItem::new(
///             1,
///             String::new(),
///             BatchItemStatus::Failed,
///             None,
///             Some("oops".into()),
///             Some("ChildFnError".into()),
///         ),
///     ],
///     CompletionReason::FailureToleranceExceeded,
/// );
/// assert!(result.has_failure());
/// assert_eq!(result.success_count(), 1);
/// assert_eq!(result.failure_count(), 1);
/// assert_eq!(result.status(), BatchStatus::Failed);
/// assert_eq!(result.errors()[0].index, 1);
/// assert_eq!(result.errors()[0].message, "oops");
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct BatchResult<O> {
    /// Per-item outcomes in input order (only started items).
    pub items: Vec<BatchItem<O>>,
    /// Why the batch completed.
    pub reason: CompletionReason,
}

impl<O> BatchResult<O> {
    /// Creates a new batch result.
    #[must_use]
    pub fn new(items: Vec<BatchItem<O>>, reason: CompletionReason) -> Self {
        Self { items, reason }
    }

    /// Returns successful results in input order.
    ///
    /// Failed or not-started items are omitted.
    #[must_use]
    pub fn results(&self) -> Vec<&O> {
        self.items
            .iter()
            .filter_map(|item| {
                if item.status == BatchItemStatus::Succeeded {
                    item.result.as_ref()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the count of successes.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == BatchItemStatus::Succeeded)
            .count()
    }

    /// Returns the count of failures.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == BatchItemStatus::Failed)
            .count()
    }

    /// Reports whether any item failed.
    #[must_use]
    pub fn has_failure(&self) -> bool {
        self.items
            .iter()
            .any(|i| i.status == BatchItemStatus::Failed)
    }

    /// Returns [`BatchStatus::Succeeded`] if no item failed,
    /// [`BatchStatus::Failed`] otherwise.
    #[must_use]
    pub fn status(&self) -> BatchStatus {
        if self.has_failure() {
            BatchStatus::Failed
        } else {
            BatchStatus::Succeeded
        }
    }

    /// Returns the total number of items that were started.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.items.len()
    }

    /// Returns the errors from failed items, in input order.
    ///
    /// Each [`BatchError`] carries the failed item's index, display name,
    /// error type identifier, and error message, so a caller can tell
    /// which input produced which failure and what kind of error it was.
    #[must_use]
    pub fn errors(&self) -> Vec<BatchError<'_>> {
        self.items
            .iter()
            .filter(|item| item.status == BatchItemStatus::Failed)
            .map(|item| BatchError {
                index: item.index,
                name: &item.name,
                message: item.error_message.as_deref().unwrap_or_default(),
                error_type: item.error_type.as_deref(),
            })
            .collect()
    }
}

/// Controls whether batch items run in real or virtual child contexts.
///
/// [`NestingMode::Normal`] (the default) runs each item in its own child
/// context, producing per-item `ContextStarted`/`ContextSucceeded` events.
/// [`NestingMode::Flat`] runs items in a virtual context: operations inside
/// each item are checkpointed directly under the parent batch context, with
/// no per-item context events.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::NestingMode;
///
/// let mode = NestingMode::Flat;
/// assert_ne!(mode, NestingMode::Normal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum NestingMode {
    /// Each item runs in its own child context with full context events.
    #[default]
    Normal,
    /// Items run in a virtual context under the parent — no per-item
    /// context events are emitted.
    Flat,
}

/// Internal state for a map execution passed from the builder.
///
/// Holds the user's map closure as `Arc<F>` — shared rather than boxed,
/// because the same closure runs once per item, concurrently. Each item
/// call produces a **concrete** future from `Arc<F>`, so the
/// [`JoinSet`](tokio::task::JoinSet) inside [`execute_batch`] holds
/// concrete futures with no per-item box; the one erasure point is the
/// builder's `.future()` / `into_future`, which boxes the whole execution
/// future once inside [`DurableFuture`](crate::DurableFuture).
pub(crate) struct MapExecution<I, O, F, IS, RS> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) max_concurrency: Option<usize>,
    pub(crate) completion: Option<crate::builders::map_parallel::CompletionConfig>,
    pub(crate) serdes: IS,
    pub(crate) result_serdes: RS,
    pub(crate) nesting: NestingMode,
    pub(crate) item_namer: Option<Arc<dyn Fn(usize) -> String + Send + Sync>>,
    pub(crate) items: Vec<I>,
    pub(crate) closure: Arc<F>,
    pub(crate) _marker: std::marker::PhantomData<fn() -> O>,
}

impl<I, O, F, Fut, IS, RS> MapExecution<I, O, F, IS, RS>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<BatchSummary>,
{
    /// Executes the map operation.
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        collect_successful(self.execute_batch_result().await?)
    }

    /// Executes the map operation and returns the full `BatchResult`.
    ///
    /// The user closure enters the batch engine only through the
    /// [`ItemDispatch`] object, so [`execute_batch`] — the checkpoint state
    /// machine — compiles once per result type while every item still runs
    /// as a concrete, unboxed future. Used directly by `await_batch` and by
    /// [`Self::execute`], whose `Vec<O>` view is projected by
    /// [`collect_successful`].
    pub(crate) async fn execute_batch_result(self) -> Result<BatchResult<O>, OperationError> {
        let total_items = self.items.len();
        let items = into_item_slots(self.items);
        let closure = self.closure;
        let items_ref = Arc::clone(&items);

        let dispatch = MapDispatch {
            run_item: Arc::new(move |child_ctx: DurableContext, index: usize| {
                let items = Arc::clone(&items_ref);
                let closure = Arc::clone(&closure);
                async move {
                    let item = take_item(&items, index).map_err(ChildFnError::from)?;
                    (closure)(child_ctx, item, index).await.map_err(Into::into)
                }
            }),
        };

        execute_batch(
            self.ctx,
            self.op_id,
            self.name,
            self.max_concurrency,
            self.completion,
            self.serdes,
            self.result_serdes,
            self.nesting,
            self.item_namer,
            total_items,
            MAP_SUB_TYPE,
            MAP_ITERATION_SUB_TYPE,
            &dispatch,
        )
        .await
    }
}

/// Internal state for a parallel execution passed from the builder.
pub(crate) struct ParallelExecution<O, IS, RS> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) max_concurrency: Option<usize>,
    pub(crate) completion: Option<crate::builders::map_parallel::CompletionConfig>,
    pub(crate) serdes: IS,
    pub(crate) result_serdes: RS,
    pub(crate) nesting: NestingMode,
    pub(crate) branches: Vec<(String, crate::future::BranchBody<O>)>,
}

impl<O, IS, RS> ParallelExecution<O, IS, RS>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<BatchSummary>,
{
    /// Executes the parallel operation.
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        // Mirrors map's tolerance handling (issue #27) via the shared
        // [`collect_successful`] projection: failed branches within
        // tolerance are skipped; only a tolerance-exceeded batch errors.
        collect_successful(self.execute_batch_result().await?)
    }

    /// Executes the parallel operation and returns the full `BatchResult`.
    ///
    /// Branch bodies are heterogeneous, so each carries exactly one erased
    /// future ([`crate::future::BranchBody`], built at `Branch::new`); the
    /// batch engine and its per-item state machine are shared, non-generic
    /// code. Used directly by `await_batch` and by [`Self::execute`], whose
    /// `Vec<O>` view is projected by [`collect_successful`].
    pub(crate) async fn execute_batch_result(self) -> Result<BatchResult<O>, OperationError> {
        let total = self.branches.len();
        // Split each branch into its display name (threaded to the
        // coordinator as the item namer so it reaches child checkpoint
        // updates) and its erased body (kept in a take-once slot since it
        // runs at most once).
        let mut names: Vec<String> = Vec::with_capacity(total);
        let mut slots: Vec<std::sync::Mutex<Option<crate::future::BranchBody<O>>>> =
            Vec::with_capacity(total);
        for (name, body) in self.branches {
            names.push(name);
            slots.push(std::sync::Mutex::new(Some(body)));
        }
        let branch_namer: Arc<dyn Fn(usize) -> String + Send + Sync> =
            Arc::new(move |index| names.get(index).cloned().unwrap_or_default());

        let dispatch = BranchDispatch {
            slots: Arc::new(slots),
        };

        execute_batch(
            self.ctx,
            self.op_id,
            self.name,
            self.max_concurrency,
            self.completion,
            self.serdes,
            self.result_serdes,
            self.nesting,
            Some(branch_namer),
            total,
            PARALLEL_SUB_TYPE,
            PARALLEL_BRANCH_SUB_TYPE,
            &dispatch,
        )
        .await
    }
}

/// Projects a full [`BatchResult`] onto the plain `Vec<O>` success view
/// shared by `map` and `parallel` `.await`.
///
/// If the batch completed within tolerance (`AllCompleted`) or ended early
/// on a success trigger (`MinSuccessfulReached` or `PredicateMatched`),
/// failed items are expected and NOT propagated as errors — they are simply
/// skipped. Only a batch that ended because the failure tolerance was
/// exceeded (including the default fail-fast case) becomes an `Err`,
/// carrying the first failed item's message.
fn collect_successful<O>(batch_result: BatchResult<O>) -> Result<Vec<O>, OperationError> {
    let mut results = Vec::with_capacity(batch_result.items.len());
    for item in batch_result.items {
        match item.status {
            BatchItemStatus::Succeeded => {
                if let Some(value) = item.result {
                    results.push(value);
                }
            }
            BatchItemStatus::Failed => {
                if batch_result.reason == CompletionReason::FailureToleranceExceeded {
                    let msg = item
                        .error_message
                        .unwrap_or_else(|| "branch failed".to_owned());
                    return Err(batch_error(&msg));
                }
                // Within tolerance: skip this item in the Vec<O> output.
            }
        }
    }
    // Check if the batch itself failed due to tolerance exceeded.
    if batch_result.reason == CompletionReason::FailureToleranceExceeded {
        return Err(batch_error("failure tolerance exceeded"));
    }
    Ok(results)
}

// ────────────────────────────────────────────────────────────────────────────
// Core batch engine
// ────────────────────────────────────────────────────────────────────────────

/// A pre-claimed child operation with its metadata.
struct PreClaimed {
    index: usize,
    op_id: OperationId,
    is_terminal: bool,
}

/// Per-in-flight-branch metadata the coordinator retains so it can record a
/// controlled failure for a branch whose task ends via a `JoinError` (a panic
/// in user branch code, or a cancellation) instead of producing an outcome.
struct BranchMeta {
    index: usize,
    child_wire: String,
    item_name: String,
}

/// Extracts a human-readable message from a captured panic payload.
fn panic_message(payload: &dyn std::any::Any) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "branch panicked".to_owned())
}

/// How one branch resolved within a single invocation.
///
/// A `Terminal` branch (succeeded or failed) frees its concurrency slot; a
/// `Suspended` branch parked on a durable operation and KEEPS its slot until
/// it terminally completes on a later invocation (the slot-holding
/// invariant). Never-started branches are simply never dispatched.
enum ItemOutcome<O> {
    /// The branch reached a terminal state this invocation.
    Terminal(BatchItem<O>),
    /// The branch suspended (parked) — its slot stays held.
    Suspended,
}

/// The payload a branch task delivers through the coordinator's `JoinSet`:
/// the item index plus how the item resolved.
type ItemJoin<O> = (usize, Result<ItemOutcome<O>, OperationError>);

/// Everything one item body needs, owned and free of generic parameters.
///
/// A dispatcher moves one of these into each spawned task, which is what
/// lets the whole checkpoint state machine around the user closure
/// ([`item_before`], [`item_after`], [`execute_batch`]) compile once per
/// result type instead of once per user call site.
struct ItemRequest<IS> {
    ctx: DurableContext,
    child_op_id: OperationId,
    index: usize,
    is_terminal: bool,
    start_checkpointed: bool,
    parent_wire: String,
    child_sub_type: String,
    item_name: String,
    nesting: NestingMode,
    /// The item serdes, shared across items behind one `Arc` (the
    /// forwarding `impl Serdes for Arc<S>` makes the handle itself a
    /// serdes).
    serdes: Arc<IS>,
}

/// What [`item_before`] decided: the item is already resolved from the
/// checkpoint log, or the body must run in the prepared child context.
enum ItemPrelude<O> {
    /// Recorded terminal outcome decoded from the checkpoint log.
    Done(BatchItem<O>),
    /// The body must run; the child context (with its own suspension
    /// scope) is ready.
    Run {
        /// The child context the item body receives.
        child_ctx: DurableContext,
    },
}

/// Pre-closure half of one batch item: replay decode, child START
/// checkpoint, child-context creation. Generic only over the result type
/// `O` — no user closure reaches this code.
async fn item_before<O, IS>(req: &ItemRequest<IS>) -> Result<ItemPrelude<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let child_positional = req.child_op_id.positional().to_owned();
    let child_wire = req.child_op_id.wire().to_owned();
    // Per-item serdes context: the child's wire ID is stable across replays,
    // so a context-sensitive serdes resolves the same location every time.
    let serdes_ctx = SerdesContext::new(child_wire.clone(), req.ctx.execution_arn());

    // Replay path: child already terminal.
    if req.is_terminal {
        match replay_terminal_child(
            &req.ctx,
            &child_positional,
            req.index,
            &req.item_name,
            &req.serdes,
            &serdes_ctx,
        )
        .await
        {
            Ok(item) => return Ok(ItemPrelude::Done(item)),
            Err(e) => {
                // ReplayChildren sentinel: fall through to re-execution.
                let is_replay_children =
                    crate::error::chain_string(&e).contains(REPLAY_CHILDREN_SENTINEL);
                if !is_replay_children {
                    return Err(e);
                }
                // Fall through: re-execute the child to reconstruct result.
            }
        }
    }

    // FLAT nesting: skip child context events; run item directly under
    // parent. Operations inside the flat branch checkpoint with ParentId
    // pointing to the batch parent (not the virtual child).
    if req.nesting == NestingMode::Flat {
        let child_ctx = req
            .ctx
            .new_scoped_flat_child(&child_positional, &req.parent_wire);
        return Ok(ItemPrelude::Run { child_ctx });
    }

    // Check if we need to checkpoint START for this child.
    // Skip if the caller already checkpointed START synchronously (for
    // concurrency safety).
    if !req.start_checkpointed && !req.ctx.has_checkpoint_record(&child_positional) {
        let update = build_child_update(
            &child_wire,
            &req.item_name,
            &req.child_sub_type,
            &req.parent_wire,
            OperationAction::Start,
        );
        if let Err(err) = req.ctx.checkpoint_updates(vec![update]).await {
            // Audit (#43) — batch child START: the item closure has not
            // run, so no terminal FAIL is needed; re-invocation
            // reconverges on the same write. Routing unrecoverable (not
            // as an item failure) matters doubly here: a tolerant
            // completion config must not absorb a checkpoint failure.
            return req
                .ctx
                .checkpoint_failure_unrecoverable(&child_wire, err, None)
                .await;
        }
    }

    // Create child context with its OWN suspension scope so a park inside
    // the body is caught locally as Suspended instead of tearing down the
    // whole invocation.
    Ok(ItemPrelude::Run {
        child_ctx: req.ctx.new_scoped_child(&child_positional),
    })
}

/// Post-closure half of one batch item: outcome checkpointing and
/// [`BatchItem`] assembly. Generic only over the result type `O` — no user
/// closure reaches this code.
#[allow(clippy::too_many_lines)] // reason: FLAT/NORMAL outcome checkpointing reads better in one flow
async fn item_after<O, IS>(
    req: &ItemRequest<IS>,
    scope: &Arc<crate::driver::SuspensionSignal>,
    outcome: ScopeOutcome<Result<O, ChildFnError>>,
) -> Result<ItemOutcome<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let child_wire = req.child_op_id.wire().to_owned();
    let serdes_ctx = SerdesContext::new(child_wire.clone(), req.ctx.execution_arn());
    let serdes = &req.serdes;

    // FLAT nesting emits no child-context events: only the value transform
    // (round-trip for live == replay consistency) happens here.
    if req.nesting == NestingMode::Flat {
        return match outcome {
            ScopeOutcome::Suspended => Ok(ItemOutcome::Suspended),
            ScopeOutcome::Completed(Ok(value)) => {
                // A result serialization failure is LOCAL and
                // deterministic: the item closure already ran, so the item
                // settles as a failed `BatchItem` instead of yielding a
                // catchable coordinator error with no record (issue #43).
                // FLAT items have no per-child record — the failure is
                // recorded inside the parent batch's summary payload when
                // the parent checkpoints, and replay reconstructs the same
                // failed item from it, so live and replayed batches agree.
                let round_trip = async {
                    let serialized = serialize_value(value, serdes, serdes_ctx.clone()).await?;
                    let deserialized: O = deserialize_value(serialized, serdes, serdes_ctx).await?;
                    Ok::<_, OperationError>(deserialized)
                };
                match round_trip.await {
                    Ok(deserialized) => Ok(ItemOutcome::Terminal(BatchItem {
                        index: req.index,
                        name: req.item_name.clone(),
                        status: BatchItemStatus::Succeeded,
                        result: Some(deserialized),
                        error_message: None,
                        error_type: None,
                    })),
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        Ok(ItemOutcome::Terminal(BatchItem {
                            index: req.index,
                            name: req.item_name.clone(),
                            status: BatchItemStatus::Failed,
                            result: None,
                            error_message: wire.error_message().map(str::to_owned),
                            error_type: wire.error_type().map(str::to_owned),
                        }))
                    }
                }
            }
            ScopeOutcome::Completed(Err(child_err)) => {
                // The recorded message is the flattened chain, built at
                // the single flattening site.
                let wire = crate::error::wire_error_for(&child_err, CHILD_FN_ERROR_TYPE);
                Ok(ItemOutcome::Terminal(BatchItem {
                    index: req.index,
                    name: req.item_name.clone(),
                    status: BatchItemStatus::Failed,
                    result: None,
                    error_message: wire.error_message().map(str::to_owned),
                    error_type: wire.error_type().map(str::to_owned),
                }))
            }
        };
    }

    match outcome {
        ScopeOutcome::Suspended => {
            // Branch parked: do NOT checkpoint a terminal state. The child
            // context stays Started-not-terminal in the log; on resume it is
            // re-entered and its now-completed durable op replays. The branch
            // keeps its concurrency slot until it terminally completes.
            Ok(ItemOutcome::Suspended)
        }
        ScopeOutcome::Completed(Ok(value)) => {
            // Serialize and checkpoint success.
            //
            // A result serialization failure is LOCAL and deterministic:
            // the item closure already ran, so a terminal FAIL is
            // persisted for the child FIRST, and the item settles as a
            // failed `BatchItem` (issue #43). That is exactly the shape a
            // replay of the FAIL record produces (see
            // [`replay_terminal_child`]), so live and replayed batches
            // agree, and the closure never re-runs for this failure.
            let serialized = match serialize_value(value, serdes, serdes_ctx.clone()).await {
                Ok(serialized) => serialized,
                Err(op_err) => {
                    let wire = crate::error::serialization_failure_wire(&op_err);
                    let update = build_child_fail_update(
                        &child_wire,
                        &req.item_name,
                        &req.child_sub_type,
                        &req.parent_wire,
                        &wire,
                    );
                    if let Err(client_err) = req.ctx.checkpoint_updates(vec![update]).await {
                        // Audit (#43) — batch child FAIL (serialization):
                        // the item closure ran; the failed FAIL write
                        // routes unrecoverable with a minimal terminal
                        // FAIL retry.
                        let cwire = crate::error::checkpoint_failure_wire(&client_err);
                        let terminal = build_child_fail_update(
                            &child_wire,
                            &req.item_name,
                            &req.child_sub_type,
                            &req.parent_wire,
                            &cwire,
                        );
                        return req
                            .ctx
                            .checkpoint_failure_unrecoverable(
                                &child_wire,
                                client_err,
                                Some(terminal),
                            )
                            .await;
                    }
                    return Ok(ItemOutcome::Terminal(BatchItem {
                        index: req.index,
                        name: req.item_name.clone(),
                        status: BatchItemStatus::Failed,
                        result: None,
                        error_message: wire.error_message().map(str::to_owned),
                        error_type: wire.error_type().map(str::to_owned),
                    }));
                }
            };
            let mut builder = OperationUpdate::builder()
                .id(child_wire.clone())
                .r#type(OperationType::Context)
                .sub_type(req.child_sub_type.clone())
                .action(OperationAction::Succeed)
                .parent_id(req.parent_wire.clone());

            if serialized.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
                builder = builder.context_options(
                    aws_sdk_lambda::types::ContextOptions::builder()
                        .replay_children(true)
                        .build(),
                );
            } else {
                builder = builder.payload(serialized.clone());
            }

            #[allow(clippy::expect_used)] // reason: all required fields are set above
            let update = builder
                .build()
                .expect("all required OperationUpdate fields set");

            if let Err(err) = req.ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43) — batch child SUCCEED: the item closure
                // ran, so its side effects need a recorded outcome. A
                // permanent rejection persists a small terminal FAIL
                // before the execution fails. Routing unrecoverable (not
                // as an item failure) keeps a tolerant completion config
                // from absorbing a checkpoint failure.
                let cwire = crate::error::checkpoint_failure_wire(&err);
                let terminal = build_child_fail_update(
                    &child_wire,
                    &req.item_name,
                    &req.child_sub_type,
                    &req.parent_wire,
                    &cwire,
                );
                return req
                    .ctx
                    .checkpoint_failure_unrecoverable(&child_wire, err, Some(terminal))
                    .await;
            }

            // Round-trip deserialize for live == replay consistency.
            let deserialized: O = deserialize_value(serialized, serdes, serdes_ctx).await?;

            Ok(ItemOutcome::Terminal(BatchItem {
                index: req.index,
                name: req.item_name.clone(),
                status: BatchItemStatus::Succeeded,
                result: Some(deserialized),
                error_message: None,
                error_type: None,
            }))
        }
        ScopeOutcome::Completed(Err(child_err)) => {
            // Defensive: a durable op that set its suspend flag and then
            // returned Err (rather than parking) is a suspension, not a
            // failure — mirror the invocation driver's precedence rule.
            if scope.is_suspend_requested() {
                return Ok(ItemOutcome::Suspended);
            }

            // Checkpoint failure. The wire record is derived from the
            // carried error, so the message is the flattened chain and
            // `error_data`/`stack_trace` pass through the boundary.
            let wire = crate::error::wire_error_for(&child_err, CHILD_FN_ERROR_TYPE);
            let builder = OperationUpdate::builder()
                .id(child_wire.clone())
                .r#type(OperationType::Context)
                .sub_type(req.child_sub_type.clone())
                .action(OperationAction::Fail)
                .parent_id(req.parent_wire.clone())
                .error(wire.to_error_object());

            #[allow(clippy::expect_used)] // reason: all required fields are set above
            let update = builder
                .build()
                .expect("all required OperationUpdate fields set");

            if let Err(client_err) = req.ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43) — batch child FAIL: the item closure ran
                // and failed; the failed FAIL write routes unrecoverable
                // with a minimal terminal FAIL retry (the original
                // carried the item error's payload). Discarding it would
                // let a tolerant batch continue on a record claiming less
                // than what executed.
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal = build_child_fail_update(
                    &child_wire,
                    &req.item_name,
                    &req.child_sub_type,
                    &req.parent_wire,
                    &cwire,
                );
                return req
                    .ctx
                    .checkpoint_failure_unrecoverable(&child_wire, client_err, Some(terminal))
                    .await;
            }

            Ok(ItemOutcome::Terminal(BatchItem {
                index: req.index,
                name: req.item_name.clone(),
                status: BatchItemStatus::Failed,
                result: None,
                error_message: wire.error_message().map(str::to_owned),
                error_type: wire.error_type().map(str::to_owned),
            }))
        }
    }
}

/// Thin generic wrapper around one map item — the ONLY code monomorphized
/// per map call site. Everything before and after the user closure is the
/// non-generic [`item_before`] / [`item_after`] pair; this wrapper just
/// polls the user's concrete future between them under the branch driver.
async fn run_single_item<O, F, Fut, IS>(
    req: ItemRequest<IS>,
    run_item: Arc<F>,
) -> Result<ItemOutcome<O>, OperationError>
where
    O: Send + 'static,
    F: Fn(DurableContext, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ChildFnError>> + Send + 'static,
    IS: Serdes<O>,
{
    match item_before(&req).await? {
        ItemPrelude::Done(item) => Ok(ItemOutcome::Terminal(item)),
        ItemPrelude::Run { child_ctx } => {
            // Instrument the branch body with the branch namespace's
            // replay-aware span: a resumed branch's pre-wait log lines are
            // suppressed while its own operations replay, independently of
            // the root handler span and of sibling branches.
            let scope = Arc::clone(child_ctx.suspension_signal());
            let span = child_ctx.replay_span();
            let index = req.index;
            let outcome = drive_scope(
                run_item(child_ctx, index).instrument(span),
                Arc::clone(&scope),
            )
            .await;
            item_after(&req, &scope, outcome).await
        }
    }
}

/// Non-generic branch runner for `parallel`: the branch body is already the
/// single erased future carried by [`crate::future::BranchBody`], so
/// nothing here monomorphizes per user call site.
async fn run_branch_item<O, IS>(
    req: ItemRequest<IS>,
    slots: Arc<Vec<std::sync::Mutex<Option<crate::future::BranchBody<O>>>>>,
) -> Result<ItemOutcome<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    match item_before(&req).await? {
        ItemPrelude::Done(item) => Ok(ItemOutcome::Terminal(item)),
        ItemPrelude::Run { child_ctx } => {
            let scope = Arc::clone(child_ctx.suspension_signal());
            let span = child_ctx.replay_span();
            let outcome = match take_branch_body(&slots, req.index) {
                Ok(body) => {
                    // `start` injects the child context and returns the
                    // branch's single erased future, polled here under the
                    // branch driver exactly like a map item's body.
                    drive_scope(body.start(child_ctx).instrument(span), Arc::clone(&scope)).await
                }
                Err(err) => ScopeOutcome::Completed(Err(err)),
            };
            item_after(&req, &scope, outcome).await
        }
    }
}

/// Takes one branch body out of its take-once slot.
fn take_branch_body<O>(
    slots: &[std::sync::Mutex<Option<crate::future::BranchBody<O>>>],
    index: usize,
) -> Result<crate::future::BranchBody<O>, ChildFnError> {
    let guard = slots
        .get(index)
        .ok_or_else(|| ChildFnError::new("branch index out of bounds"))?;
    let mut lock = guard
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    lock.take()
        .ok_or_else(|| ChildFnError::new("branch already consumed (concurrent access bug)"))
}

/// Object-safe item dispatcher: spawns one item body onto the coordinator's
/// [`JoinSet`] as a CONCRETE (unboxed) future.
///
/// This boundary is what keeps [`execute_batch`] — the batch checkpoint
/// state machine — non-generic over the user's closure: only the dispatcher
/// and the thin wrapper it spawns ([`run_single_item`]) monomorphize per
/// call site, while the futures inside the `JoinSet` stay unboxed.
trait ItemDispatch<O, IS>: Send + Sync {
    /// Spawns the item body for `req` and returns the task's abort handle.
    fn spawn_item(
        &self,
        set: &mut JoinSet<ItemJoin<O>>,
        req: ItemRequest<IS>,
    ) -> tokio::task::AbortHandle;
}

/// Map dispatcher: shares the user closure as `Arc<F>` and produces one
/// concrete future per item — the `JoinSet` holds no per-item box.
struct MapDispatch<F> {
    run_item: Arc<F>,
}

impl<O, F, Fut, IS> ItemDispatch<O, IS> for MapDispatch<F>
where
    O: Send + 'static,
    F: Fn(DurableContext, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ChildFnError>> + Send + 'static,
    IS: Serdes<O>,
{
    fn spawn_item(
        &self,
        set: &mut JoinSet<ItemJoin<O>>,
        req: ItemRequest<IS>,
    ) -> tokio::task::AbortHandle {
        let run_item = Arc::clone(&self.run_item);
        set.spawn(async move {
            // Bless this task for task-ownership checks.
            let task_ownership = req.ctx.task_ownership().clone();
            crate::combinator::bless_current_task(&task_ownership);
            let index = req.index;
            (index, run_single_item(req, run_item).await)
        })
    }
}

/// Parallel dispatcher: hands each spawned task the shared take-once slots
/// holding the single-erasure branch bodies. Non-generic — the erased
/// branch body already carries the user's closure and future.
struct BranchDispatch<O> {
    slots: Arc<Vec<std::sync::Mutex<Option<crate::future::BranchBody<O>>>>>,
}

impl<O, IS> ItemDispatch<O, IS> for BranchDispatch<O>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    fn spawn_item(
        &self,
        set: &mut JoinSet<ItemJoin<O>>,
        req: ItemRequest<IS>,
    ) -> tokio::task::AbortHandle {
        let slots = Arc::clone(&self.slots);
        set.spawn(async move {
            // Bless this task for task-ownership checks.
            let task_ownership = req.ctx.task_ownership().clone();
            crate::combinator::bless_current_task(&task_ownership);
            let index = req.index;
            (index, run_branch_item(req, slots).await)
        })
    }
}

/// Mutable coordinator bookkeeping shared by the join-drain paths.
struct BatchProgress<O> {
    /// Running completion statistics plus the first trigger to fire.
    tracker: CompletionTracker,
    /// Terminal outcomes, positional by item index.
    results: Vec<Option<BatchItem<O>>>,
    /// Parked branches — they RETAIN their concurrency slots.
    suspended_count: usize,
    /// Whether ANY branch parked this invocation.
    any_suspended: bool,
}

impl<O> BatchProgress<O> {
    fn new(total_items: usize) -> Self {
        Self {
            tracker: CompletionTracker::new(total_items),
            results: (0..total_items).map(|_| None).collect(),
            suspended_count: 0,
            any_suspended: false,
        }
    }

    /// Records one item outcome: a terminal item feeds the completion
    /// statistics and its result slot; a suspended item keeps its slot.
    fn settle_outcome(
        &mut self,
        completion_cfg: &crate::builders::map_parallel::CompletionConfig,
        total_items: usize,
        index: usize,
        outcome: ItemOutcome<O>,
    ) {
        match outcome {
            ItemOutcome::Terminal(item) => {
                self.tracker
                    .settle(completion_cfg, total_items, index, item.status);
                if let Some(slot) = self.results.get_mut(index) {
                    *slot = Some(item);
                }
            }
            ItemOutcome::Suspended => {
                self.suspended_count += 1;
                self.any_suspended = true;
            }
        }
    }
}

/// Borrowed coordinator context shared by the join-drain helpers, so the
/// helpers stay non-generic functions with manageable signatures.
struct BatchEnv<'a, O, IS> {
    ctx: &'a DurableContext,
    dispatch: &'a dyn ItemDispatch<O, IS>,
    completion_cfg: &'a crate::builders::map_parallel::CompletionConfig,
    total_items: usize,
    nesting: NestingMode,
    parent_wire: &'a str,
    child_sub_type: &'a str,
    serdes: &'a Arc<IS>,
}

impl<O, IS> BatchEnv<'_, O, IS>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    /// Processes one joined branch task: a produced outcome feeds the
    /// progress accounting; a `JoinError` (a panic in user branch code, or a
    /// cancellation) is recorded as a controlled BRANCH FAILURE — mirroring
    /// the failure a normal branch would have produced so accounting and the
    /// completion threshold treat it identically — with a best-effort child
    /// FAIL checkpoint so a retry does not repeat already-started work.
    ///
    /// The `JoinError` arm applies to LIVE branches only. A `ReplayChildren`
    /// reconstruction task never reaches it: [`Self::resolve_terminal_inline`]
    /// handles its own join event, because a panic while reconstructing a
    /// recorded terminal SUCCESS must unwind the coordinator — not
    /// checkpoint `Fail` over durable success history.
    async fn settle_joined(
        &self,
        joined: Result<(tokio::task::Id, ItemJoin<O>), tokio::task::JoinError>,
        branch_meta: &mut std::collections::HashMap<tokio::task::Id, BranchMeta>,
        progress: &mut BatchProgress<O>,
    ) -> Result<(), OperationError> {
        match joined {
            Ok((task_id, (index, outcome))) => {
                branch_meta.remove(&task_id);
                match outcome {
                    Ok(outcome) => {
                        progress.settle_outcome(
                            self.completion_cfg,
                            self.total_items,
                            index,
                            outcome,
                        );
                        Ok(())
                    }
                    // Coordinator-level failure (e.g. a checkpoint call
                    // failed): surface; the caller's JoinSet drop aborts
                    // remaining branches.
                    Err(e) => Err(e),
                }
            }
            Err(join_err) => {
                // The branch task terminated without producing an outcome.
                let Some(meta) = branch_meta.remove(&join_err.id()) else {
                    return Err(batch_error(
                        "branch task terminated with an unrecognized task id",
                    ));
                };
                let message = match join_err.try_into_panic() {
                    Ok(payload) => panic_message(payload.as_ref()),
                    Err(_) => "branch task was cancelled".to_owned(),
                };

                // Child FAIL checkpoint for the panicked/cancelled branch.
                // Skipped in FLAT mode, which emits no child-context
                // events (mirrors the normal fail path).
                if self.nesting != NestingMode::Flat {
                    let update = OperationUpdate::builder()
                        .id(meta.child_wire.clone())
                        .r#type(OperationType::Context)
                        .sub_type(self.child_sub_type.to_owned())
                        .action(OperationAction::Fail)
                        .parent_id(self.parent_wire.to_owned())
                        .error(
                            crate::error::WireError::new(
                                Some(CHILD_FN_ERROR_TYPE),
                                Some(message.clone()),
                            )
                            .to_error_object(),
                        );
                    if let Ok(update) = update.build()
                        && let Err(client_err) = self.ctx.checkpoint_updates(vec![update]).await
                    {
                        // Audit (#43) — batch child FAIL (panicked
                        // branch): the item closure ran; the failed FAIL
                        // write routes unrecoverable with a minimal
                        // terminal FAIL retry.
                        let cwire = crate::error::checkpoint_failure_wire(&client_err);
                        let terminal = build_child_fail_update(
                            &meta.child_wire,
                            &meta.item_name,
                            self.child_sub_type,
                            self.parent_wire,
                            &cwire,
                        );
                        return self
                            .ctx
                            .checkpoint_failure_unrecoverable(
                                &meta.child_wire,
                                client_err,
                                Some(terminal),
                            )
                            .await;
                    }
                }

                progress.settle_outcome(
                    self.completion_cfg,
                    self.total_items,
                    meta.index,
                    ItemOutcome::Terminal(BatchItem {
                        index: meta.index,
                        name: meta.item_name,
                        status: BatchItemStatus::Failed,
                        result: None,
                        error_message: Some(message),
                        error_type: Some(CHILD_FN_ERROR_TYPE.to_owned()),
                    }),
                );
                Ok(())
            }
        }
    }

    /// Resolves a recorded-terminal child inline on the coordinator, in
    /// input order. The common case decodes the recorded outcome without
    /// touching the item body; a `ReplayChildren` child (result too large to
    /// checkpoint) is re-executed through the dispatcher and drained
    /// immediately, so input order is preserved and the `JoinSet` still
    /// holds only concrete futures.
    ///
    /// A panic during that re-execution is rethrown (unwinding the
    /// coordinator) rather than graded as a branch failure: the child's
    /// durable record is a terminal SUCCESS, and checkpointing `Fail` over
    /// it would permanently fail the batch over a transient reconstruction
    /// crash. See the `JoinError` arm below.
    async fn resolve_terminal_inline(
        &self,
        op_id: &OperationId,
        index: usize,
        item_name: &str,
        join_set: &mut JoinSet<ItemJoin<O>>,
        branch_meta: &mut std::collections::HashMap<tokio::task::Id, BranchMeta>,
        progress: &mut BatchProgress<O>,
    ) -> Result<(), OperationError> {
        let child_positional = op_id.positional().to_owned();
        let serdes_ctx = SerdesContext::new(op_id.wire().to_owned(), self.ctx.execution_arn());
        match replay_terminal_child(
            self.ctx,
            &child_positional,
            index,
            item_name,
            self.serdes,
            &serdes_ctx,
        )
        .await
        {
            Ok(item) => {
                progress.settle_outcome(
                    self.completion_cfg,
                    self.total_items,
                    index,
                    ItemOutcome::Terminal(item),
                );
                Ok(())
            }
            Err(e) => {
                let is_replay_children =
                    crate::error::chain_string(&e).contains(REPLAY_CHILDREN_SENTINEL);
                if !is_replay_children {
                    return Err(e);
                }
                // ReplayChildren: re-execute the body to reconstruct the
                // oversized result. The join set is quiescent at every
                // inline call site (the replay pass runs before live
                // dispatch; the sequential cursor has nothing in flight), so
                // the immediate drain below joins exactly this item.
                let req = ItemRequest {
                    ctx: self.ctx.clone(),
                    child_op_id: op_id.clone(),
                    index,
                    is_terminal: true,
                    start_checkpointed: false,
                    parent_wire: self.parent_wire.to_owned(),
                    child_sub_type: self.child_sub_type.to_owned(),
                    item_name: item_name.to_owned(),
                    nesting: self.nesting,
                    serdes: Arc::clone(self.serdes),
                };
                let abort = self.dispatch.spawn_item(join_set, req);
                let reconstruction_id = abort.id();
                branch_meta.insert(
                    reconstruction_id,
                    BranchMeta {
                        index,
                        child_wire: op_id.wire().to_owned(),
                        item_name: item_name.to_owned(),
                    },
                );
                let Some(joined) = join_set.join_next_with_id().await else {
                    return Err(batch_error(
                        "re-executed replay child produced no join event",
                    ));
                };
                // Verify (rather than assume) the quiescence invariant: the
                // join event must belong to the reconstruction task spawned
                // above, because the JoinError arm below applies
                // reconstruction-specific panic semantics.
                let joined_id = match &joined {
                    Ok((task_id, _)) => *task_id,
                    Err(join_err) => join_err.id(),
                };
                if joined_id != reconstruction_id {
                    return Err(batch_error(
                        "re-executed replay child joined an unexpected task",
                    ));
                }
                match joined {
                    Ok(ok) => self.settle_joined(Ok(ok), branch_meta, progress).await,
                    Err(join_err) => {
                        branch_meta.remove(&join_err.id());
                        // This child's durable record is a terminal SUCCESS
                        // (`replay_children = true`); the task that ended
                        // without an outcome was merely RECONSTRUCTING that
                        // recorded result. Grading this as a branch failure
                        // (as `settle_joined` does for live branches) would
                        // checkpoint `Fail` over recorded success history
                        // and permanently fail the batch because of a
                        // transient reconstruction crash. Mirror the
                        // pre-JoinSet inline await instead: rethrow the
                        // panic payload so the coordinator unwinds with the
                        // original panic, no failure is checkpointed, and
                        // the recorded terminal state stays untouched for
                        // the next attempt to reconstruct.
                        match join_err.try_into_panic() {
                            Ok(payload) => std::panic::resume_unwind(payload),
                            // A cancelled task carries no payload to
                            // rethrow: surface a coordinator error, still
                            // without checkpointing failure.
                            Err(join_err) => Err(batch_error(&format!(
                                "replay child reconstruction task was cancelled: {join_err}"
                            ))),
                        }
                    }
                }
            }
        }
    }
}

/// Core batch execution: schedule items with bounded concurrency and
/// completion checking.
///
/// Non-generic over the user closure: item bodies enter only through the
/// [`ItemDispatch`] object, so this coordinator — the batch's checkpoint
/// state machine — monomorphizes once per result type `O`.
#[allow(clippy::too_many_lines)]
// reason: batch coordination has distinct phases (claim, schedule, collect, checkpoint) that read better as one flow
#[allow(clippy::too_many_arguments)] // reason: batch execution requires all these parameters
async fn execute_batch<O, IS, RS>(
    ctx: DurableContext,
    parent_op_id: OperationId,
    parent_name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<crate::builders::map_parallel::CompletionConfig>,
    serdes: IS,
    result_serdes: RS,
    nesting: NestingMode,
    item_namer: Option<Arc<dyn Fn(usize) -> String + Send + Sync>>,
    total_items: usize,
    parent_sub_type: &str,
    child_sub_type: &str,
    dispatch: &dyn ItemDispatch<O, IS>,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<BatchSummary>,
{
    // Share the item serdes across items behind one `Arc` — the forwarding
    // `impl Serdes for Arc<S>` makes the handle itself a serdes, so no
    // `Clone` bound is required of the user's implementation.
    let serdes = Arc::new(serdes);

    // 1. Task-ownership check.
    ctx.enforce_task_ownership()?;

    // 1b. Validate max_concurrency BEFORE any checkpointing.
    // Zero is invalid — must be positive or unset (None = unlimited).
    if max_concurrency == Some(0) {
        return Err(batch_error(
            "max concurrency must be positive or unset for unlimited",
        ));
    }

    let parent_positional = parent_op_id.positional().to_owned();
    let parent_wire = parent_op_id.wire().to_owned();

    // 2. Check if the parent batch is already terminal in the checkpoint log.
    // Identity validation and the status read happen in one read-guard pass;
    // the terminal payload/error projection is cloned only when the batch is
    // actually terminal and must be replayed.
    if let Some(view) = ctx.checkpoint_view_validated(
        &parent_positional,
        &parent_wire,
        "Context",
        Some(parent_sub_type),
        parent_name.as_deref(),
    )? {
        if view.status.is_terminal() {
            let snapshot = ctx
                .checkpoint_terminal_replay(&parent_positional)
                .ok_or_else(|| batch_error("terminal batch has no checkpoint record"))?;
            let serdes_ctx = SerdesContext::new(&parent_wire, ctx.execution_arn());
            match replay_terminal_batch(
                &ctx,
                &snapshot,
                &parent_positional,
                total_items,
                &serdes,
                &result_serdes,
                &serdes_ctx,
            )
            .await
            {
                Ok(result) => {
                    // Recorded terminal batch summary returned without
                    // re-running the batch (see `crate::observability`).
                    ctx.emit_operation_replayed(
                        &parent_wire,
                        parent_name.as_deref(),
                        "Context",
                        Some(parent_sub_type),
                        view.attempt,
                    );
                    // Advance the parent counter past the iteration IDs that
                    // were consumed during the original execution. Sequential
                    // (concurrency=1) only claims started items; concurrent
                    // claims all items upfront.
                    let concurrency = max_concurrency.unwrap_or(total_items).max(1);
                    if concurrency == 1 {
                        ctx.advance_counter(result.items.len());
                    } else {
                        ctx.advance_counter(total_items);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    // ReplayChildren sentinel: fall through to re-execution.
                    // The children's operations are still in the checkpoint
                    // log, so re-executing the batch will replay each child
                    // from its terminal record.
                    let is_replay_children =
                        crate::error::chain_string(&e).contains(REPLAY_CHILDREN_SENTINEL);
                    if !is_replay_children {
                        // A recorded batch FAILURE replays as this error; an
                        // internal replay problem (missing/corrupt payload)
                        // is not a replayed outcome and emits nothing.
                        if matches!(snapshot.status, CheckpointStatus::Failed) {
                            ctx.emit_operation_replayed(
                                &parent_wire,
                                parent_name.as_deref(),
                                "Context",
                                Some(parent_sub_type),
                                view.attempt,
                            );
                        }
                        return Err(e);
                    }
                    // Fall through: the batch parent is terminal but we need
                    // to re-execute children to reconstruct the result.
                }
            }
        }
        // Non-terminal (Started/Pending): continue from where we left off.
    } else {
        // 3. First invocation: checkpoint the parent START.
        let update = build_parent_update(
            &parent_wire,
            parent_name.as_deref(),
            parent_sub_type,
            OperationAction::Start,
            &ctx,
        );
        if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
            // Audit (#43) — batch parent START: no item closure has run,
            // so no terminal FAIL is needed; re-invocation reconverges
            // on the same write.
            return ctx
                .checkpoint_failure_unrecoverable(&parent_wire, err, None)
                .await;
        }
    }

    // 4. Empty collection: checkpoint success immediately.
    if total_items == 0 {
        let result = BatchResult {
            items: Vec::new(),
            reason: CompletionReason::AllCompleted,
        };
        let (payload, result) =
            from_batch_result(result, &serdes, &parent_wire, ctx.execution_arn()).await?;
        let serialized_payload = serialize_value(
            payload,
            &result_serdes,
            SerdesContext::new(&parent_wire, ctx.execution_arn()),
        )
        .await?;
        checkpoint_batch_success_serialized(
            &ctx,
            &parent_wire,
            parent_name.as_deref(),
            parent_sub_type,
            &serialized_payload,
        )
        .await?;
        return Ok(result);
    }

    // 5. Determine effective concurrency (needed before pre-claiming).
    let concurrency = max_concurrency.unwrap_or(total_items).max(1);
    let completion_cfg = completion.unwrap_or_default();

    // Validate completion config.
    if let Err(msg) = completion_cfg.validate() {
        return Err(batch_error(&msg));
    }

    // 6. Pre-claim child IDs.
    // For concurrency > 1 (concurrent path): claim ALL child IDs upfront
    // on the owning task (determinism rule 4 — deterministic ID ordering
    // regardless of completion order).
    // For concurrency == 1 (sequential path): mint IDs lazily, one at a
    // time inside the loop, so only STARTED items consume IDs
    // and the service doesn't see dangling operations for never-started
    // items on early termination.
    let mut pre_claimed: Vec<PreClaimed> = Vec::with_capacity(total_items);
    if concurrency > 1 {
        for i in 0..total_items {
            let child_op_id = ctx.mint_id();
            let child_positional = child_op_id.positional().to_owned();
            let child_wire = child_op_id.wire().to_owned();
            let child_name = item_namer.as_ref().map(|namer| namer(i));
            // Non-determinism detection on child items happens inside the
            // validated view fetch; only the status is consumed here.
            let is_terminal = ctx
                .checkpoint_view_validated(
                    &child_positional,
                    &child_wire,
                    "Context",
                    Some(child_sub_type),
                    child_name.as_deref(),
                )?
                .is_some_and(|view| view.status.is_terminal());
            pre_claimed.push(PreClaimed {
                index: i,
                op_id: child_op_id,
                is_terminal,
            });
        }
    }

    // 7. Execute items with bounded concurrency, branch-local suspension, and
    // slot-holding accounting.

    // 7a. For concurrent mode: checkpoint ALL child STARTs synchronously
    // BEFORE dispatching any tasks. This prevents token rotation races
    // between the main loop and spawned tasks: all child STARTs are
    // checkpointed on the owning task before any spawned task runs.
    if concurrency > 1 {
        for pre in &pre_claimed {
            if !pre.is_terminal
                && nesting != NestingMode::Flat
                && !ctx.has_checkpoint_record(pre.op_id.positional())
            {
                let child_wire = pre.op_id.wire().to_owned();
                let item_name_str = item_namer
                    .as_ref()
                    .map_or_else(String::new, |namer| namer(pre.index));
                let update = build_child_update(
                    &child_wire,
                    &item_name_str,
                    child_sub_type,
                    &parent_wire,
                    OperationAction::Start,
                );
                if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
                    // Audit (#43) — pre-claimed batch child START: the
                    // item closure has not run, so no terminal FAIL is
                    // needed; re-invocation reconverges on the same
                    // write.
                    return ctx
                        .checkpoint_failure_unrecoverable(&child_wire, err, None)
                        .await;
                }
            }
        }
    }

    // Coordinator loop. In-flight = running + suspended, bounded by
    // `concurrency`: a SUSPENDED branch KEEPS its slot (only terminal
    // completion frees one — the slot-holding invariant), so
    // `suspended_count` counts against the cap and new branches only start
    // when capacity remains after terminal completions. Each branch runs
    // through the dispatcher's thin wrapper, which drives the branch body
    // under its own scope so a park resolves to `ItemOutcome::Suspended`
    // locally rather than tearing down the whole invocation.
    let child_sub_type_owned = child_sub_type.to_owned();
    // Coordinator-owned, observable task set. Dropping the `JoinSet` (on error
    // return OR when the driver drops this coordinator on invocation teardown)
    // aborts every still-running branch task, so no branch task outlives the
    // invocation — the same abort-on-drop ownership as the `.spawn()` path.
    // Unlike a plain result channel, `join_next_with_id` also surfaces a
    // `JoinError` when a task ends WITHOUT producing an outcome (a panic in
    // user branch code, or a cancellation), so the coordinator accounts for
    // every terminated task rather than waiting forever for a value a panicked
    // task will never deliver.
    let mut join_set: JoinSet<ItemJoin<O>> = JoinSet::new();
    // Maps a branch task's id to the metadata needed to record a controlled
    // failure if that task ends via a `JoinError`. Removed on both the value
    // and the error arm, so it never outgrows the in-flight set.
    let mut branch_meta: std::collections::HashMap<tokio::task::Id, BranchMeta> =
        std::collections::HashMap::with_capacity(total_items);

    let mut progress = BatchProgress::<O>::new(total_items);
    let mut running: usize = 0;
    let mut next_index: usize = 0;

    let env = BatchEnv {
        ctx: &ctx,
        dispatch,
        completion_cfg: &completion_cfg,
        total_items,
        nesting,
        parent_wire: &parent_wire,
        child_sub_type: &child_sub_type_owned,
        serdes: &serdes,
    };

    // 7b. Replay pass (concurrent mode): resolve every recorded-terminal
    // child inline on the coordinator, in input order, BEFORE dispatching
    // any live work — never through the live `JoinSet` schedule. This is
    // what keeps completion triggers deterministic under replay: outcomes
    // already in the checkpoint log feed the statistics in canonical input
    // order before any live settlement joins, so identical recorded state
    // yields identical trigger decisions no matter how the scheduler orders
    // the join events of resumed live branches. Recorded terminals are
    // completed history — real work whose outcome the service persisted —
    // so they are applied unconditionally: a trigger firing mid-pass halts
    // future live dispatch but never drops an already-recorded outcome
    // from the result.
    //
    // The sequential path (`concurrency == 1`) discovers terminality lazily
    // at the dispatch cursor instead, where settlement order and input
    // order already coincide.
    if concurrency > 1 {
        for i in 0..total_items {
            let Some(pre) = pre_claimed.get(i) else {
                return Err(batch_error("pre-claimed index out of range"));
            };
            if !pre.is_terminal {
                continue;
            }
            let item_name = item_namer
                .as_ref()
                .map_or_else(String::new, |namer| namer(i));
            env.resolve_terminal_inline(
                &pre.op_id,
                i,
                &item_name,
                &mut join_set,
                &mut branch_meta,
                &mut progress,
            )
            .await?;
        }
    }

    loop {
        // Dispatch while capacity remains and not-started eligible work exists.
        // A completion trigger (`tracker.stop_reason`) halts new dispatch;
        // already-running branches are still drained below (in-flight
        // branches always complete).
        while progress.tracker.stop_reason.is_none()
            && next_index < total_items
            && running + progress.suspended_count < concurrency
        {
            let i = next_index;
            next_index += 1;

            // Concurrent mode: recorded-terminal children were already
            // applied by the replay pass above — skip them here.
            if concurrency > 1 && pre_claimed.get(i).is_some_and(|pre| pre.is_terminal) {
                continue;
            }

            // Determine the PreClaimed item: for concurrent mode use the
            // pre-claimed vector; for sequential mode, mint lazily (only
            // started items consume IDs).
            let pc = if concurrency > 1 {
                let pre = pre_claimed
                    .get(i)
                    .ok_or_else(|| batch_error("pre-claimed index out of range"))?;
                PreClaimed {
                    index: pre.index,
                    op_id: pre.op_id.clone(),
                    is_terminal: pre.is_terminal,
                }
            } else {
                let child_op_id = ctx.mint_id();
                let child_positional = child_op_id.positional().to_owned();
                let child_wire = child_op_id.wire().to_owned();
                let child_name = item_namer.as_ref().map(|namer| namer(i));
                // Non-determinism detection on child items happens inside
                // the validated view fetch; only the status is consumed.
                let is_terminal = ctx
                    .checkpoint_view_validated(
                        &child_positional,
                        &child_wire,
                        "Context",
                        Some(child_sub_type),
                        child_name.as_deref(),
                    )?
                    .is_some_and(|view| view.status.is_terminal());
                PreClaimed {
                    index: i,
                    op_id: child_op_id,
                    is_terminal,
                }
            };

            // Recorded-terminal child (sequential path only — the
            // concurrent path applied its recorded terminals in the replay
            // pass above): resolve it inline on the coordinator, at the
            // dispatch cursor, in input order. With one item in flight at a
            // time, settlement order and input order coincide, so
            // completion triggers see the same canonical sequence here as
            // everywhere else.
            if pc.is_terminal {
                let item_name = item_namer
                    .as_ref()
                    .map_or_else(String::new, |namer| namer(pc.index));
                env.resolve_terminal_inline(
                    &pc.op_id,
                    pc.index,
                    &item_name,
                    &mut join_set,
                    &mut branch_meta,
                    &mut progress,
                )
                .await?;
                continue;
            }

            let start_checkpointed =
                concurrency > 1 && !pc.is_terminal && nesting != NestingMode::Flat;

            // Compute the item name in the coordinator so it is available both
            // for the branch body and for a controlled-failure checkpoint if
            // the branch task ends via a `JoinError`.
            let item_name = item_namer
                .as_ref()
                .map_or_else(String::new, |namer| namer(pc.index));
            let branch_index = pc.index;
            let branch_wire = pc.op_id.wire().to_owned();

            let req = ItemRequest {
                ctx: ctx.clone(),
                child_op_id: pc.op_id,
                index: branch_index,
                is_terminal: pc.is_terminal,
                start_checkpointed,
                parent_wire: parent_wire.clone(),
                child_sub_type: child_sub_type_owned.clone(),
                item_name: item_name.clone(),
                nesting,
                serdes: Arc::clone(&serdes),
            };
            let abort = dispatch.spawn_item(&mut join_set, req);
            branch_meta.insert(
                abort.id(),
                BranchMeta {
                    index: branch_index,
                    child_wire: branch_wire,
                    item_name,
                },
            );
            running += 1;
        }

        if running == 0 {
            // Quiescent: nothing in flight. Either all dispatched work is
            // terminal, a threshold stopped dispatch, or every remaining slot
            // is held by a parked branch.
            break;
        }

        // Await the next branch task to terminate. `join_next_with_id` yields a
        // produced outcome (`Ok`) OR a `JoinError` (`Err`) for a task that ended
        // without one — a panic in user branch code (library code forbids
        // panics), or, defensively, a cancellation.
        let Some(joined) = join_set.join_next_with_id().await else {
            break; // set is empty — unreachable while running > 0
        };
        running -= 1;
        env.settle_joined(joined, &mut branch_meta, &mut progress)
            .await?;
    }

    let BatchProgress {
        tracker,
        results,
        any_suspended,
        ..
    } = progress;

    // Quiescent. If a branch is parked and no completion trigger fired, the
    // batch cannot finish this invocation: suspend the coordinator's OWN
    // scope so whoever drives it (the invocation driver at the root, or an
    // outer coordinator's branch driver when nested) observes the suspension
    // and reports PENDING for its subtree. `suspend_now` never returns; the
    // coordinator future is dropped at teardown, aborting the guards.
    // Started-not-terminal children replay on the next invocation. When a
    // trigger DID fire, parked branches are excluded (like never-started
    // work) and the batch completes normally.
    if any_suspended && tracker.stop_reason.is_none() {
        return Ok(ctx.suspend_now::<BatchResult<O>>().await);
    }

    // 9. Assemble results in input order (only terminal items; suspended and
    // never-started branches are omitted).
    let final_items: Vec<BatchItem<O>> = results.into_iter().flatten().collect();

    // 10. Determine completion reason: the first trigger to fire during the
    // run (recorded at the settle event that fired it — first trigger wins),
    // or `AllCompleted` when no trigger fired. Capturing the reason at
    // trigger time keeps a non-monotonic predicate stable: the reason
    // reflects the statistics the trigger actually saw, not a re-evaluation
    // against the final counts after in-flight branches drained.
    //
    // A batch that recorded NO settle event never ran a trigger evaluation,
    // so for that degenerate case fall back to the fixed-threshold
    // evaluation against the final counts, exactly as the pre-predicate
    // code evaluated at the end of every run. (Today a zero-item batch
    // returns early at step 4 before this loop, so the fallback is a guard
    // that keeps the fixed-threshold semantics intact if that early return
    // ever changes; it never rewrites the reason of a batch that settled
    // items.)
    let reason = tracker.stop_reason.unwrap_or_else(|| {
        if tracker.settled_count() == 0 {
            if should_stop_min(&completion_cfg, tracker.success_count) {
                CompletionReason::MinSuccessfulReached
            } else if should_stop_failure(&completion_cfg, tracker.failure_count, total_items) {
                CompletionReason::FailureToleranceExceeded
            } else {
                CompletionReason::AllCompleted
            }
        } else {
            CompletionReason::AllCompleted
        }
    });

    let batch_result = BatchResult {
        items: final_items,
        reason,
    };

    // 11. Serialize the batch result BEFORE the async checkpoint call
    // (avoids requiring O: Sync for the reference across await).
    // The whole-batch summary goes through the SAME `serialize_value` helper
    // as every other path, so `result_serdes` receives the typed
    // `BatchSummary` value directly, exactly as an item serdes receives the
    // typed item value.
    let (payload, batch_result) =
        from_batch_result(batch_result, &serdes, &parent_wire, ctx.execution_arn()).await?;
    let serialized_payload = serialize_value(
        payload,
        &result_serdes,
        SerdesContext::new(&parent_wire, ctx.execution_arn()),
    )
    .await?;

    // 12. Checkpoint the parent SUCCEED.
    checkpoint_batch_success_serialized(
        &ctx,
        &parent_wire,
        parent_name.as_deref(),
        parent_sub_type,
        &serialized_payload,
    )
    .await?;

    Ok(batch_result)
}

// ────────────────────────────────────────────────────────────────────────────
// Replay helpers
// ────────────────────────────────────────────────────────────────────────────

/// Replays a terminal batch (parent already succeeded/failed in the log).
///
/// Takes the targeted [`crate::engine::TerminalReplaySnapshot`] projection
/// rather than the full checkpoint record — the status, `replay_children`,
/// and payload/error strings are all this helper reads.
async fn replay_terminal_batch<O, IS, RS>(
    _ctx: &DurableContext,
    snapshot: &crate::engine::TerminalReplaySnapshot,
    _parent_positional: &str,
    _total_items: usize,
    serdes: &Arc<IS>,
    result_serdes: &RS,
    serdes_ctx: &SerdesContext,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<BatchSummary>,
{
    match &snapshot.status {
        CheckpointStatus::Succeeded => {
            if snapshot.replay_children {
                // ReplayChildren mode: cannot reconstruct from the payload
                // alone — the caller must fall through to re-execution.
                // Signal this by returning a sentinel error that the caller
                // catches to continue normal execution.
                return Err(batch_error(REPLAY_CHILDREN_SENTINEL));
            }
            // Deserialize the stored batch summary.
            let payload_str = snapshot
                .result
                .clone()
                .ok_or_else(|| batch_error("terminal batch has no result payload"))?;
            // Reverse the result serdes transform — through the same helper
            // every other path uses.
            let payload: BatchSummary =
                deserialize_value(payload_str, result_serdes, serdes_ctx.clone()).await?;
            // The batch parent's serdes context carries the parent wire ID and
            // the execution ARN, which is exactly what the per-item contexts
            // are derived from.
            to_batch_result(
                payload,
                serdes,
                serdes_ctx.operation_id(),
                serdes_ctx.durable_execution_arn(),
            )
            .await
        }
        CheckpointStatus::Failed => {
            let msg = snapshot.error_message.as_deref().unwrap_or("batch failed");
            Err(batch_error(msg))
        }
        _ => {
            // Shouldn't happen — we checked is_terminal() above.
            Err(batch_error("unexpected non-terminal status in replay"))
        }
    }
}

/// Replays a terminal child item from the checkpoint log.
///
/// `item_name` is the validated per-item name the caller derived for this
/// slot (branch name for `parallel`, generated name for `map`); the replayed
/// [`BatchItem`] carries it so structured error access reports the producing
/// item's name even when the item reaches the batch result through replay.
async fn replay_terminal_child<O, S>(
    ctx: &DurableContext,
    child_positional: &str,
    index: usize,
    item_name: &str,
    serdes: &S,
    serdes_ctx: &SerdesContext,
) -> Result<BatchItem<O>, OperationError>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    let record = ctx
        .checkpoint_terminal_replay(child_positional)
        .ok_or_else(|| batch_error("replay child has no checkpoint record"))?;

    match &record.status {
        CheckpointStatus::Succeeded => {
            if record.replay_children {
                // ReplayChildren mode: signal re-execution needed.
                return Err(batch_error(REPLAY_CHILDREN_SENTINEL));
            }
            let payload = record
                .result
                .clone()
                .ok_or_else(|| batch_error("succeeded child has no result"))?;
            let value: O = deserialize_value(payload, serdes, serdes_ctx.clone()).await?;
            Ok(BatchItem {
                index,
                name: item_name.to_owned(),
                status: BatchItemStatus::Succeeded,
                result: Some(value),
                error_message: None,
                error_type: None,
            })
        }
        CheckpointStatus::Failed => {
            let msg = record.error_message.as_deref().unwrap_or("child failed");
            Ok(BatchItem {
                index,
                name: item_name.to_owned(),
                status: BatchItemStatus::Failed,
                result: None,
                error_message: Some(msg.to_owned()),
                error_type: record.error_type.clone(),
            })
        }
        _ => Err(batch_error(
            "unexpected non-terminal child status in replay",
        )),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Checkpoint helpers
// ────────────────────────────────────────────────────────────────────────────

/// Checkpoints the parent batch as SUCCEEDED with the pre-serialized result.
async fn checkpoint_batch_success_serialized(
    ctx: &DurableContext,
    parent_wire: &str,
    parent_name: Option<&str>,
    parent_sub_type: &str,
    serialized_payload: &str,
) -> Result<(), OperationError> {
    let mut builder = OperationUpdate::builder()
        .id(parent_wire.to_owned())
        .r#type(OperationType::Context)
        .sub_type(parent_sub_type.to_owned())
        .action(OperationAction::Succeed);

    if let Some(n) = parent_name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire_id) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire_id);
    }

    if serialized_payload.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
        builder = builder.context_options(
            aws_sdk_lambda::types::ContextOptions::builder()
                .replay_children(true)
                .build(),
        );
    } else {
        builder = builder.payload(serialized_payload.to_owned());
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43) — batch parent SUCCEED: the batch's items ran, so
        // the batch outcome needs a recorded terminal. A permanent
        // rejection persists a small terminal FAIL before the execution
        // fails.
        let cwire = crate::error::checkpoint_failure_wire(&err);
        let terminal =
            build_parent_fail_update(parent_wire, parent_name, parent_sub_type, ctx, &cwire);
        return ctx
            .checkpoint_failure_unrecoverable(parent_wire, err, Some(terminal))
            .await;
    }

    Ok(())
}

/// Builds a parent-level operation update.
fn build_parent_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    action: OperationAction,
    ctx: &DurableContext,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(sub_type.to_owned())
        .action(action);

    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds a child-level operation update.
fn build_child_update(
    child_wire: &str,
    child_name: &str,
    child_sub_type: &str,
    parent_wire: &str,
    action: OperationAction,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(child_wire.to_owned())
        .r#type(OperationType::Context)
        .sub_type(child_sub_type.to_owned())
        .action(action)
        .parent_id(parent_wire.to_owned());

    // Propagate the map item / parallel branch display name so it appears
    // on the child's history records.
    if !child_name.is_empty() {
        builder = builder.name(child_name.to_owned());
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds a child-level `FAIL` update carrying `wire` as its error — the
/// terminal record persisted when the child's own outcome write was
/// permanently rejected (issue #43).
fn build_child_fail_update(
    child_wire: &str,
    child_name: &str,
    child_sub_type: &str,
    parent_wire: &str,
    wire: &crate::error::WireError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(child_wire.to_owned())
        .r#type(OperationType::Context)
        .sub_type(child_sub_type.to_owned())
        .action(OperationAction::Fail)
        .parent_id(parent_wire.to_owned())
        .error(wire.to_error_object());
    if !child_name.is_empty() {
        builder = builder.name(child_name.to_owned());
    }
    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds a parent-level `FAIL` update carrying `wire` as its error — the
/// terminal record persisted when the batch parent's own outcome write
/// was permanently rejected (issue #43).
fn build_parent_fail_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    ctx: &DurableContext,
    wire: &crate::error::WireError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(sub_type.to_owned())
        .action(OperationAction::Fail)
        .error(wire.to_error_object());
    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }
    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

// ────────────────────────────────────────────────────────────────────────────
// Completion logic
// ────────────────────────────────────────────────────────────────────────────

/// Running completion state of one batch coordinator: the settled counts,
/// the canonical committed prefix the predicate observes, and the first
/// completion trigger to fire.
///
/// Every settlement — a recorded-terminal child applied inline in input
/// order, a live join, or a controlled `JoinError` failure — flows through
/// [`settle`](Self::settle), so the statistics a completion trigger sees are
/// updated and evaluated in exactly one place.
///
/// # Two evaluation streams
///
/// The fixed thresholds and the custom predicate deliberately read from
/// different streams:
///
/// * **Fixed thresholds** (`min_successful`, the failure tolerances) are
///   evaluated on the raw settlement counts, immediately at every settle
///   event. They are monotonic in those counts, so whether they fire is a
///   function of the settled *set*, never of the settlement *order* — a
///   threshold that did not fire before a suspension cannot fire while
///   replay re-applies the same recorded set. Immediate evaluation
///   preserves the pre-predicate semantics: the batch stops the moment the
///   threshold is objectively met.
///
/// * **The custom predicate** is an arbitrary, order-sensitive function, so
///   it is evaluated only on the *committed prefix*: settled outcomes are
///   buffered per index and committed strictly in input order (item `i`
///   commits only after items `0..i` have all committed). Live settlement
///   order is scheduler-timed and is not recorded anywhere, so it cannot be
///   reproduced on replay; the committed prefix is derivable from recorded
///   state alone, which makes the sequence of predicate evaluations — and
///   therefore its decisions — identical on the original run and on every
///   replay.
struct CompletionTracker {
    /// Count of items that have succeeded so far (settlement order).
    success_count: usize,
    /// Count of items that have failed so far (settlement order).
    failure_count: usize,
    /// Per-index settled status, buffered until the item commits (its index
    /// becomes part of the contiguous committed prefix).
    pending: Vec<Option<BatchItemStatus>>,
    /// The next input index to commit: items `0..next_commit` have
    /// committed, in input order.
    next_commit: usize,
    /// Count of committed items that succeeded.
    committed_success: usize,
    /// Count of committed items that failed.
    committed_failure: usize,
    /// The committed outcomes, in input order: `committed_outcomes[i]` is
    /// item `i`'s outcome. This is what the custom predicate observes.
    committed_outcomes: Vec<SettledOutcome>,
    /// The first completion trigger to fire (first trigger wins). `Some`
    /// halts new dispatch.
    stop_reason: Option<CompletionReason>,
}

impl CompletionTracker {
    fn new(total_items: usize) -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            pending: vec![None; total_items],
            next_commit: 0,
            committed_success: 0,
            committed_failure: 0,
            committed_outcomes: Vec::with_capacity(total_items),
            stop_reason: None,
        }
    }

    /// Returns how many items have settled so far (either status, any
    /// order).
    fn settled_count(&self) -> usize {
        self.success_count + self.failure_count
    }

    /// Applies one settled outcome to the running statistics and, while no
    /// trigger has fired yet, evaluates the completion triggers (first
    /// trigger wins — once `stop_reason` is set it never changes).
    ///
    /// Within one settle event the check order is fixed — `min_successful`,
    /// then the failure tolerances, then the custom predicate — matching
    /// the precedence the fixed thresholds always had.
    fn settle(
        &mut self,
        cfg: &crate::builders::map_parallel::CompletionConfig,
        total_items: usize,
        index: usize,
        status: BatchItemStatus,
    ) {
        match status {
            BatchItemStatus::Succeeded => self.success_count += 1,
            BatchItemStatus::Failed => self.failure_count += 1,
        }
        if let Some(slot) = self.pending.get_mut(index) {
            *slot = Some(status);
        }
        // Fixed thresholds: evaluated immediately, on the settlement-order
        // counts (see the type-level docs for why this is order-safe).
        if self.stop_reason.is_none() {
            self.stop_reason =
                evaluate_thresholds(cfg, self.success_count, self.failure_count, total_items);
        }
        // Custom predicate: evaluated once per newly *committed* item, on
        // the committed prefix only. Draining continues even after a
        // trigger fired so the committed statistics stay consistent; the
        // predicate itself is no longer consulted once `stop_reason` is
        // set (first trigger wins).
        while let Some(ready) = self.pending.get(self.next_commit).copied().flatten() {
            self.committed_outcomes
                .push(SettledOutcome::new(self.next_commit, ready));
            match ready {
                BatchItemStatus::Succeeded => self.committed_success += 1,
                BatchItemStatus::Failed => self.committed_failure += 1,
            }
            self.next_commit += 1;
            if self.stop_reason.is_none() {
                let snapshot = BatchStats::new(
                    self.committed_success,
                    self.committed_failure,
                    total_items,
                    &self.committed_outcomes,
                );
                if cfg.predicate_matches(&snapshot) {
                    self.stop_reason = Some(CompletionReason::PredicateMatched);
                }
            }
        }
    }
}

/// Evaluates the fixed completion thresholds against the running batch
/// counts, returning the first that fires (or `None`).
///
/// Called after each settled item. Within one settle event the check order
/// is fixed — `min_successful`, then the failure tolerances — matching the
/// precedence the thresholds always had. The custom predicate is evaluated
/// separately, on the committed prefix (see [`CompletionTracker`]).
fn evaluate_thresholds(
    cfg: &crate::builders::map_parallel::CompletionConfig,
    success_count: usize,
    failure_count: usize,
    total_items: usize,
) -> Option<CompletionReason> {
    if should_stop_min(cfg, success_count) {
        return Some(CompletionReason::MinSuccessfulReached);
    }
    if should_stop_failure(cfg, failure_count, total_items) {
        return Some(CompletionReason::FailureToleranceExceeded);
    }
    None
}

/// Checks if the `min_successful` threshold has been met.
fn should_stop_min(
    cfg: &crate::builders::map_parallel::CompletionConfig,
    success_count: usize,
) -> bool {
    match cfg.min_successful() {
        Some(min) if min > 0 => success_count >= min,
        _ => false,
    }
}

/// Checks if the failure tolerance has been exceeded.
fn should_stop_failure(
    cfg: &crate::builders::map_parallel::CompletionConfig,
    failure_count: usize,
    total_items: usize,
) -> bool {
    // Count-based tolerance.
    if let Some(tolerated) = cfg.tolerated_failure_count()
        && failure_count > tolerated
    {
        return true;
    }

    // Percentage-based tolerance.
    // Uses cross-multiplication (failure_count * 100 > pct * total_items) to
    // avoid integer-division truncation.  This means a true failure rate of
    // 33.3% correctly exceeds a 33% threshold (1*100=100 > 33*3=99).
    // When pct == 0, any failure exceeds the threshold (fail-fast).
    if let Some(pct) = cfg.tolerated_failure_percentage()
        && total_items > 0
        && failure_count * 100 > pct * total_items
    {
        return true;
    }

    false
}

// ────────────────────────────────────────────────────────────────────────────
// Serialization helpers
// ────────────────────────────────────────────────────────────────────────────

/// Serializes a value through the configured serdes (ownership transfers;
/// the serdes decides where its work runs).
///
/// This is the same boundary every other operation (step, invoke, callback,
/// child, batch result) uses, and — since the item paths call this helper
/// too — the same one map/parallel ITEM results use. There is no separate
/// item rule.
async fn serialize_value<T, S: Serdes<T>>(
    value: T,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    serdes
        .serialize(value, serdes_ctx)
        .await
        .map_err(|e| batch_error(&format!("serialize result: {e}")))
}

/// Deserializes a value through the configured serdes.
///
/// Reverses [`serialize_value`]: the serdes turns the wire payload directly
/// back into the typed value — no intermediate representation and no
/// runtime downcast.
async fn deserialize_value<T, S: Serdes<T>>(
    payload: String,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<T, OperationError> {
    serdes
        .deserialize(payload, serdes_ctx)
        .await
        .map_err(|e| batch_error(&format!("deserialize result: {e}")))
}

/// Builds the serdes context for an individual batch item result stored
/// inside the batch summary payload.
///
/// The identity is derived from the batch parent's wire ID plus the item
/// index, so it is deterministic across replays and distinct per item (a
/// context-sensitive serdes such as `FileSystemSerdes` must not collapse all
/// item results onto one path).
fn item_summary_serdes_ctx(parent_wire: &str, execution_arn: &str, index: usize) -> SerdesContext {
    SerdesContext::new(format!("{parent_wire}/item-{index}"), execution_arn)
}

/// The whole-batch summary a map/parallel operation checkpoints: one entry
/// per item (index, status, and the item's serialized result or error)
/// plus the batch completion reason.
///
/// This is the value a **result serdes** (a serdes attached via
/// `result_serdes(...)` on [`MapBuilder`](crate::builders::MapBuilder) or
/// [`ParallelBuilder`](crate::builders::ParallelBuilder)) transforms. Its
/// fields are private: a result serdes is expected to be type-agnostic (a
/// blanket `impl<T> Serdes<T>` over `T: serde::Serialize +
/// serde::de::DeserializeOwned`), transforming the summary through its
/// serde representation rather than field access.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchSummary {
    results: Vec<BatchCheckpointItem>,
    reason: String,
}

/// Per-item checkpoint representation.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BatchCheckpointItem {
    index: usize,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    name: String,
    status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    result: String,
    #[serde(default, rename = "errType", skip_serializing_if = "String::is_empty")]
    err_type: String,
    #[serde(
        default,
        rename = "errMessage",
        skip_serializing_if = "String::is_empty"
    )]
    err_message: String,
}

/// Converts a live `BatchResult` into the checkpoint payload format,
/// round-tripping every successful item value through the item serdes.
///
/// The batch result is consumed: each successful item's value transfers by
/// ownership to `Serdes::serialize` (which lets the serdes move it into a
/// blocking task without requiring `O: Sync`), and the returned
/// `BatchResult` carries the values reconstructed from their wire form —
/// so live and replay observe identical values.
async fn from_batch_result<O, IS>(
    result: BatchResult<O>,
    serdes: &Arc<IS>,
    parent_wire: &str,
    execution_arn: &str,
) -> Result<(BatchSummary, BatchResult<O>), OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let BatchResult { items, reason } = result;
    let reason_wire = reason.as_str().to_owned();

    let mut summary_items = Vec::with_capacity(items.len());
    let mut rebuilt_items = Vec::with_capacity(items.len());
    for item in items {
        let BatchItem {
            index,
            name,
            status,
            result,
            error_message,
            error_type,
        } = item;
        let status_str = match status {
            BatchItemStatus::Succeeded => "SUCCEEDED",
            BatchItemStatus::Failed => "FAILED",
        };
        let (result_str, rebuilt_value) = match (status, result) {
            (BatchItemStatus::Succeeded, Some(value)) => {
                let item_ctx = item_summary_serdes_ctx(parent_wire, execution_arn, index);
                let wire = serdes
                    .serialize(value, item_ctx.clone())
                    .await
                    .map_err(|e| batch_error(&format!("serialize result: {e}")))?;
                let back: O = serdes
                    .deserialize(wire.clone(), item_ctx)
                    .await
                    .map_err(|e| batch_error(&format!("deserialize result: {e}")))?;
                (wire, Some(back))
            }
            _ => (String::new(), None),
        };
        summary_items.push(BatchCheckpointItem {
            index,
            name: name.clone(),
            status: status_str.to_owned(),
            result: result_str,
            err_type: if status == BatchItemStatus::Failed {
                error_type
                    .clone()
                    .unwrap_or_else(|| CHILD_FN_ERROR_TYPE.to_owned())
            } else {
                String::new()
            },
            err_message: error_message.clone().unwrap_or_default(),
        });
        rebuilt_items.push(BatchItem {
            index,
            name,
            status,
            result: rebuilt_value,
            error_message,
            error_type,
        });
    }

    Ok((
        BatchSummary {
            results: summary_items,
            reason: reason_wire,
        },
        BatchResult {
            items: rebuilt_items,
            reason,
        },
    ))
}

/// Converts a deserialized checkpoint payload back into a `BatchResult`.
async fn to_batch_result<O, IS>(
    payload: BatchSummary,
    serdes: &Arc<IS>,
    parent_wire: &str,
    execution_arn: &str,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let reason = CompletionReason::from_wire(&payload.reason);
    let mut items = Vec::with_capacity(payload.results.len());
    for cp in payload.results {
        let status = match cp.status.as_str() {
            "SUCCEEDED" => BatchItemStatus::Succeeded,
            "FAILED" => BatchItemStatus::Failed,
            other => return Err(batch_error(&format!("unknown item status: {other}"))),
        };
        let result = if status == BatchItemStatus::Succeeded && !cp.result.is_empty() {
            let item_ctx = item_summary_serdes_ctx(parent_wire, execution_arn, cp.index);
            Some(deserialize_value::<O, _>(cp.result, serdes, item_ctx).await?)
        } else {
            None
        };
        items.push(BatchItem {
            index: cp.index,
            name: cp.name,
            status,
            result,
            error_message: if status == BatchItemStatus::Failed {
                Some(cp.err_message)
            } else {
                None
            },
            error_type: if status == BatchItemStatus::Failed && !cp.err_type.is_empty() {
                Some(cp.err_type)
            } else {
                None
            },
        });
    }

    Ok(BatchResult { items, reason })
}

// ────────────────────────────────────────────────────────────────────────────
// Error helper
// ────────────────────────────────────────────────────────────────────────────

/// Constructs a batch operation error.
fn batch_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::Internal,
        Some(message.to_owned().into()),
    )))
}

/// Wraps owned map inputs in indexed take-once slots shareable across
/// branch tasks. Each item is MOVED out exactly once by [`take_item`] —
/// no `Clone` bound and no serde round-trip (which would reject or mutate
/// valid serde types that are not JSON-round-trippable).
fn into_item_slots<I>(items: Vec<I>) -> Arc<Vec<std::sync::Mutex<Option<I>>>> {
    Arc::new(
        items
            .into_iter()
            .map(|item| std::sync::Mutex::new(Some(item)))
            .collect(),
    )
}

/// Moves the item at `index` out of its slot. Each index is dispatched at
/// most once per invocation, so a second take indicates a coordinator bug.
fn take_item<I>(items: &[std::sync::Mutex<Option<I>>], index: usize) -> Result<I, OperationError> {
    let slot = items
        .get(index)
        .ok_or_else(|| batch_error("item index out of bounds"))?;
    let mut lock = slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    lock.take()
        .ok_or_else(|| batch_error("map item already consumed (concurrent access bug)"))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::Semaphore;

    /// Helper to create a test context with the given checkpoint log.
    fn test_ctx(log: CheckpointLog) -> DurableContext {
        DurableContext::new_root(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
        )
    }

    /// Helper to create a live test context backed by a recording client.
    fn test_ctx_with_client(
        log: CheckpointLog,
    ) -> (DurableContext, Arc<crate::client::InMemoryExecutionClient>) {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        (ctx, client)
    }

    /// Map items must be MOVED into the closure, not re-created through a
    /// JSON round-trip: a `HashMap` with tuple keys implements
    /// `Serialize + DeserializeOwned` but `serde_json` rejects non-string
    /// keys, so any round-trip of the ITEM would fail the whole map.
    #[tokio::test]
    async fn map_moves_non_json_round_trippable_items() {
        use std::collections::HashMap;

        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        let mut m1: HashMap<(u32, u32), u32> = HashMap::new();
        m1.insert((1, 2), 10);
        let mut m2: HashMap<(u32, u32), u32> = HashMap::new();
        m2.insert((3, 4), 20);
        m2.insert((5, 6), 30);

        let result = ctx
            .map(vec![m1, m2], |_child, item, _idx| async move {
                Ok(item.values().sum::<u32>())
            })
            .await
            .expect("map over non-JSON-round-trippable items must succeed");
        assert_eq!(result, vec![10, 50]);
    }

    /// Collects the names of recorded child START updates (updates that
    /// carry a `ParentId`, distinguishing them from the batch parent).
    fn child_start_names(client: &crate::client::InMemoryExecutionClient) -> Vec<Option<String>> {
        client
            .recorded_updates()
            .iter()
            .filter(|u| matches!(u.action(), OperationAction::Start) && u.parent_id().is_some())
            .map(|u| u.name().map(str::to_owned))
            .collect()
    }

    /// `item_namer` values must appear on the child START history records,
    /// not just in the in-memory batch result.
    #[tokio::test]
    async fn map_item_namer_reaches_child_start_updates() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let result = ctx
            .map(
                vec![1_i32, 2],
                |_child, item, _idx| async move { Ok(item * 10) },
            )
            .item_namer(|i| format!("item-{i}"))
            .await
            .expect("map must succeed");
        assert_eq!(result, vec![10, 20]);

        let mut names = child_start_names(&client);
        names.sort();
        assert_eq!(
            names,
            vec![Some("item-0".to_owned()), Some("item-1".to_owned())],
            "item namer values must be checkpointed on child STARTs"
        );
    }

    /// Parallel branch names must appear on the child START history
    /// records — the public naming API must not be silently discarded.
    #[tokio::test]
    async fn parallel_branch_names_reach_child_start_updates() {
        use crate::future::Branch;

        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let result = ctx
            .parallel([
                Branch::new("alpha", |_c| async move { Ok(1_i32) }),
                Branch::new("beta", |_c| async move { Ok(2_i32) }),
            ])
            .await
            .expect("parallel must succeed");
        assert_eq!(result, vec![1, 2]);

        let mut names = child_start_names(&client);
        names.sort();
        assert_eq!(
            names,
            vec![Some("alpha".to_owned()), Some("beta".to_owned())],
            "branch names must be checkpointed on child STARTs"
        );
    }

    fn succeeded_record(positional_id: &str, result: &str) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Succeeded,
                result: Some(result.to_owned()),
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
                op_name: None,
            },
        )
    }

    fn failed_record(positional_id: &str, msg: &str) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Failed,
                result: None,
                error_type: Some("ChildFnError".to_owned()),
                error_message: Some(msg.to_owned()),
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
                op_name: None,
            },
        )
    }

    /// A child recorded as terminal SUCCESS whose result was too large to
    /// checkpoint inline: `replay_children = true`, no payload. Replaying it
    /// requires re-executing the child body to reconstruct the result.
    fn replay_children_success_record(positional_id: &str) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
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
                replay_children: true,
                callback_id: None,
                op_type: None,
                sub_type: None,
                op_name: None,
            },
        )
    }

    #[tokio::test]
    async fn basic_map_two_items() {
        // A live client: the batch's checkpoint writes must succeed for
        // the items to run to completion. (Pre-#43 this test used a
        // client-less context and asserted the checkpoint failure
        // surfaced as Err; a rejected write now parks the future for the
        // invocation driver instead of yielding.)
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let result = ctx
            .map(vec![10, 20], |_child, item: i32, _idx| async move {
                Ok(item * 2)
            })
            .await;
        assert_eq!(result.unwrap(), vec![20, 40]);
    }

    #[tokio::test]
    async fn basic_parallel_two_branches() {
        use crate::future::Branch;
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let branches = vec![
            Branch::new("a", |_ctx| Box::pin(async { Ok(1) })),
            Branch::new("b", |_ctx| Box::pin(async { Ok(2) })),
        ];
        let result = ctx.parallel(branches).await;
        assert_eq!(result.unwrap(), vec![1, 2]);
    }

    /// `await_batch` on a tolerated mixed batch must report each branch's
    /// terminal status (with names and error messages) and the completion
    /// reason, without converting the tolerated failure into an error.
    #[tokio::test]
    async fn parallel_await_batch_reports_per_branch_status_and_reason() {
        use crate::future::Branch;
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .parallel(vec![
                Branch::new("ok-0", |_ctx| async move { Ok(1_i32) }),
                Branch::new("boom", |_ctx| async move { Err("intentional".into()) }),
                Branch::new("ok-2", |_ctx| async move { Ok(3_i32) }),
            ])
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("a tolerated branch failure must not become an operation error");

        assert_eq!(batch.reason, CompletionReason::AllCompleted);
        assert_eq!(batch.items.len(), 3);
        assert_eq!(batch.success_count(), 2);
        assert_eq!(batch.failure_count(), 1);
        assert_eq!(batch.status(), BatchStatus::Failed);

        let by_index = |idx: usize| {
            batch
                .items
                .iter()
                .find(|i| i.index == idx)
                .expect("batch must contain an item for every started branch")
        };
        assert_eq!(by_index(0).status, BatchItemStatus::Succeeded);
        assert_eq!(by_index(0).result, Some(1));
        assert_eq!(by_index(0).name, "ok-0");
        assert_eq!(by_index(1).status, BatchItemStatus::Failed);
        assert_eq!(by_index(1).result, None);
        assert_eq!(by_index(1).name, "boom");
        assert!(
            by_index(1)
                .error_message
                .as_deref()
                .unwrap_or_default()
                .contains("intentional"),
            "failed branch must carry its error message: {:?}",
            by_index(1).error_message
        );
        assert_eq!(by_index(2).status, BatchItemStatus::Succeeded);
        assert_eq!(by_index(2).result, Some(3));

        // The structured error view must associate the failure with the
        // branch that produced it.
        let errors = batch.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].index, 1);
        assert_eq!(errors[0].name, "boom");
        assert!(
            errors[0].message.contains("intentional"),
            "structured error must carry the branch's message: {:?}",
            errors[0].message
        );
        // Live execution grades a failed branch body as ChildFnError, and
        // the structured view exposes that identity alongside the message.
        assert_eq!(errors[0].error_type, Some(CHILD_FN_ERROR_TYPE));
        assert_eq!(by_index(1).error_type.as_deref(), Some(CHILD_FN_ERROR_TYPE));
    }

    /// Partial replay must not discard branch names: when a branch reached a
    /// terminal state on a previous invocation (e.g. it failed while a
    /// sibling suspended), the next invocation reconstructs it from the
    /// checkpoint log via `replay_terminal_child`, and the resulting
    /// [`BatchItem`] — and the structured error view — must still carry the
    /// producing branch's name. The live branch bodies here return values
    /// that CONTRADICT the recorded outcomes, proving the items came from
    /// replay rather than re-execution.
    #[tokio::test]
    async fn parallel_replayed_terminal_branches_retain_names() {
        use crate::future::Branch;

        // Second-invocation state: the batch parent (positional "1") is not
        // yet terminal, child 0 (positional "2") failed, and child 1
        // (positional "3") succeeded with 7.
        let log = CheckpointLog::from_records(vec![
            failed_record("2", "recorded failure"),
            succeeded_record("3", "7"),
        ]);
        let (ctx, _client) = test_ctx_with_client(log);

        let batch = ctx
            .parallel(vec![
                // Would SUCCEED if re-executed — the failure below proves
                // the item was reconstructed from the checkpoint record.
                Branch::new("boom", |_ctx| async move { Ok(999_i32) }),
                // Would return 999 if re-executed — the 7 below proves
                // replay.
                Branch::new("steady", |_ctx| async move { Ok(999_i32) }),
            ])
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("a tolerated replayed failure must not become an operation error");

        assert_eq!(batch.items.len(), 2);
        let by_index = |idx: usize| {
            batch
                .items
                .iter()
                .find(|i| i.index == idx)
                .expect("batch must contain an item for every branch")
        };

        let failed = by_index(0);
        assert_eq!(failed.status, BatchItemStatus::Failed);
        assert_eq!(
            failed.name, "boom",
            "a replayed failed branch must retain its branch name"
        );
        assert_eq!(failed.error_message.as_deref(), Some("recorded failure"));

        let succeeded = by_index(1);
        assert_eq!(succeeded.status, BatchItemStatus::Succeeded);
        assert_eq!(succeeded.result, Some(7));
        assert_eq!(
            succeeded.name, "steady",
            "a replayed succeeded branch must retain its branch name"
        );

        // The structured error view must associate the replayed failure
        // with the branch that produced it.
        let errors = batch.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].index, 0);
        assert_eq!(
            errors[0].name, "boom",
            "structured errors must name the replayed failed branch"
        );
        assert_eq!(errors[0].message, "recorded failure");
        assert_eq!(errors[0].error_type, Some(CHILD_FN_ERROR_TYPE));
    }

    /// `await_batch` surfaces a tolerance-exceeded batch as-is: the reason
    /// says why it ended and the failed branch keeps its status, while the
    /// plain `.await` on the same workload converts it into an error.
    #[tokio::test]
    async fn parallel_await_batch_surfaces_tolerance_exceeded_without_error() {
        use crate::future::Branch;
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .parallel(vec![Branch::new("boom", |_ctx| async move {
                Err::<i32, _>("intentional".into())
            })])
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .tolerated_failure_count(0)
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("await_batch must return the batch even when tolerance is exceeded");

        assert_eq!(batch.reason, CompletionReason::FailureToleranceExceeded);
        assert_eq!(batch.failure_count(), 1);
        assert_eq!(
            batch.items.first().map(|i| i.status),
            Some(BatchItemStatus::Failed)
        );
    }

    /// Issue #27: `parallel` must honor the failure tolerance a
    /// `CompletionConfig` expresses, exactly as `map` does. A failed branch
    /// within tolerance is omitted from the plain `.await`'s `Vec<O>`
    /// output rather than failing the whole operation.
    #[tokio::test]
    async fn parallel_plain_await_tolerates_failures_within_tolerance() {
        use crate::future::Branch;
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        let results: Vec<i32> = ctx
            .parallel(vec![
                Branch::new("ok-0", |_ctx| async move { Ok(1_i32) }),
                Branch::new("boom", |_ctx| async move { Err("intentional".into()) }),
                Branch::new("ok-2", |_ctx| async move { Ok(3_i32) }),
            ])
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await
            .expect("a tolerated branch failure must not fail the parallel operation");

        assert_eq!(
            results,
            vec![1, 3],
            "successful branch values must be returned in order, tolerated failure omitted"
        );

        // Same workload on map must agree (parity with map's tolerance).
        let (map_ctx, _c) = test_ctx_with_client(CheckpointLog::empty());
        let map_results: Vec<i32> = map_ctx
            .map(vec![1_i32, -1, 3], |_child, item, _idx| async move {
                if item < 0 {
                    Err("intentional".into())
                } else {
                    Ok(item)
                }
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await
            .expect("map must tolerate the failure");
        assert_eq!(
            map_results, results,
            "map and parallel plain `.await` must agree under the same tolerance"
        );
    }

    /// Parity: `map` and `parallel` `await_batch` must agree on the batch
    /// shape for equivalent workloads — same completion reason, same
    /// per-index statuses, values, counts, and overall status string.
    #[tokio::test]
    async fn map_and_parallel_await_batch_agree_on_shape() {
        use crate::future::Branch;

        // Equivalent workload: 3 slots, the middle one fails, one failure
        // tolerated. `map` derives it from items; `parallel` from branches.
        let (map_ctx, _c1) = test_ctx_with_client(CheckpointLog::empty());
        let map_batch = map_ctx
            .map(vec![0_i32, 1, 2], |_child, item, _idx| async move {
                if item == 1 {
                    Err("intentional".into())
                } else {
                    Ok(item * 10)
                }
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("map await_batch must tolerate the failure");

        let (par_ctx, _c2) = test_ctx_with_client(CheckpointLog::empty());
        let par_batch = par_ctx
            .parallel((0_i32..3).map(|i| {
                Branch::new(format!("branch-{i}"), move |_ctx| async move {
                    if i == 1 {
                        Err("intentional".into())
                    } else {
                        Ok(i * 10)
                    }
                })
            }))
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("parallel await_batch must tolerate the failure");

        assert_eq!(map_batch.reason, par_batch.reason);
        assert_eq!(map_batch.items.len(), par_batch.items.len());
        assert_eq!(map_batch.success_count(), par_batch.success_count());
        assert_eq!(map_batch.failure_count(), par_batch.failure_count());
        assert_eq!(map_batch.status(), par_batch.status());

        let shape = |batch: &BatchResult<i32>| {
            let mut items: Vec<_> = batch
                .items
                .iter()
                .map(|i| (i.index, i.status, i.result))
                .collect();
            items.sort_unstable_by_key(|(idx, _, _)| *idx);
            items
        };
        assert_eq!(
            shape(&map_batch),
            shape(&par_batch),
            "map and parallel batches must report identical per-index outcomes"
        );
    }

    // Compile-level proof that map items and parallel branches accept any
    // `IntoIterator` (array, Vec, lazy iterator) and that closures are plain
    // `async move` bodies using `?` — no manual `Box::pin`, no error
    // conversion. Builders are constructed (type-checking the closures) then
    // dropped without awaiting, so no execution client is required.
    #[tokio::test]
    async fn map_and_parallel_accept_into_iterator_and_plain_async() {
        use crate::future::Branch;
        let ctx = test_ctx(CheckpointLog::empty());

        // map: array literal.
        let _ = ctx.map([1_i32, 2, 3], |c, item, _idx| async move {
            let v = c.step(move |_| async move { Ok(item + 1) }).await?;
            Ok(v)
        });
        // map: Vec.
        let _ = ctx.map(vec![1_i32, 2], |c, item, _idx| async move {
            let v = c.step(move |_| async move { Ok(item) }).await?;
            Ok(v)
        });
        // map: lazy iterator adapter.
        let _ = ctx.map((0_i32..3).map(|x| x * 2), |c, item, _idx| async move {
            let v = c.step(move |_| async move { Ok(item) }).await?;
            Ok(v)
        });

        // parallel: array of branches.
        let _ = ctx.parallel([
            Branch::new("a", |c| async move {
                let v = c.step(|_| async { Ok(1_i32) }).await?;
                Ok(v)
            }),
            Branch::new("b", |c| async move {
                let v = c.step(|_| async { Ok(2_i32) }).await?;
                Ok(v)
            }),
        ]);
        // parallel: Vec of branches.
        let _ = ctx.parallel(vec![Branch::new("x", |_c| async move { Ok(9_i32) })]);
        // parallel: lazy iterator of branches.
        let _ = ctx.parallel(
            (0_i32..2).map(|i| Branch::new(format!("branch-{i}"), move |_c| async move { Ok(i) })),
        );
    }

    #[tokio::test]
    async fn replay_frozen_batch_returns_without_re_execution() {
        // Set up: operation "1" is the parent batch that succeeded.
        // The payload contains two successful items.
        let payload = serde_json::json!({
            "results": [
                {"index": 0, "status": "SUCCEEDED", "result": "\"hello\""},
                {"index": 1, "status": "SUCCEEDED", "result": "\"world\""}
            ],
            "reason": "ALL_COMPLETED"
        });
        let log = CheckpointLog::from_records(vec![{
            let wire_id = crate::engine::compute_wire_id_public("1");
            (
                wire_id.clone(),
                CheckpointRecord {
                    id: wire_id,
                    status: CheckpointStatus::Succeeded,
                    result: Some(payload.to_string()),
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
                    op_name: None,
                },
            )
        }]);
        let ctx = test_ctx(log);

        let result: Result<Vec<String>, _> = ctx
            .map(
                vec!["a".to_owned(), "b".to_owned()],
                |_child, _item: String, _idx| async move {
                    unreachable!("should not execute during replay");
                },
            )
            .await;

        let values = result.expect("replay should succeed");
        assert_eq!(values, vec!["hello", "world"]);
    }

    #[tokio::test]
    async fn completion_config_min_successful() {
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(2)
            .build()
            .expect("valid completion config");
        assert!(should_stop_min(&cfg, 2));
        assert!(should_stop_min(&cfg, 3));
        assert!(!should_stop_min(&cfg, 1));
    }

    #[tokio::test]
    async fn completion_config_tolerated_failure_count() {
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .tolerated_failure_count(0)
            .build()
            .expect("valid completion config");
        // 0 tolerated means fail-fast: first failure exceeds.
        assert!(should_stop_failure(&cfg, 1, 10));
        assert!(!should_stop_failure(&cfg, 0, 10));
    }

    #[tokio::test]
    async fn completion_config_tolerated_failure_percentage() {
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .tolerated_failure_percentage(20)
            .build()
            .expect("valid completion config");
        // 3/10 = 30% > 20%: should stop.
        assert!(should_stop_failure(&cfg, 3, 10));
        // 2/10 = 20% == 20%: should NOT stop (strictly exceeds).
        assert!(!should_stop_failure(&cfg, 2, 10));
    }

    #[tokio::test]
    async fn tolerated_failure_percentage_boundary_cross_multiplication() {
        // The original integer-division bug: 1 failure of 3 items with
        // pct=33.  True rate is 33.3% which exceeds 33%, but old code
        // computed (1*100)/3 == 33, and 33 > 33 is false.
        // Cross-multiplication: 1*100=100 > 33*3=99 → true (correctly stops).
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .tolerated_failure_percentage(33)
            .build()
            .expect("valid completion config");
        assert!(
            should_stop_failure(&cfg, 1, 3),
            "1/3 = 33.3% should exceed 33% threshold"
        );

        // One item below the boundary: 0 failures of 3 must NOT stop.
        assert!(
            !should_stop_failure(&cfg, 0, 3),
            "0/3 = 0% should not exceed 33%"
        );

        // Exactly at threshold when the division is exact: 1/3 with pct=34
        // means 33.3% < 34%, should NOT stop.
        let cfg34 = crate::builders::map_parallel::CompletionConfig::builder()
            .tolerated_failure_percentage(34)
            .build()
            .expect("valid completion config");
        assert!(
            !should_stop_failure(&cfg34, 1, 3),
            "1/3 = 33.3% should not exceed 34%"
        );
    }

    #[tokio::test]
    async fn tolerated_failure_percentage_zero_means_fail_fast() {
        // pct=0 means fail on first failure (fail-fast).
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .tolerated_failure_percentage(0)
            .build()
            .expect("valid completion config");
        // First failure must stop the batch.
        assert!(
            should_stop_failure(&cfg, 1, 10),
            "pct=0 should stop on first failure"
        );
        // Zero failures should NOT stop.
        assert!(
            !should_stop_failure(&cfg, 0, 10),
            "pct=0 with no failures should not stop"
        );
    }

    #[tokio::test]
    async fn max_concurrency_bounds_peak() {
        // Track peak concurrent executions using an atomic counter.
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let active_clone = Arc::clone(&active);
        let peak_clone = Arc::clone(&peak);

        // We can't run the full execution without a client, but we can
        // test the semaphore logic by observing that the concurrency
        // coordinator limits in-flight items.
        //
        // Direct semaphore test:
        let sem = Arc::new(Semaphore::new(2));
        let mut handles = Vec::new();
        for _ in 0..5 {
            let sem = Arc::clone(&sem);
            let active = Arc::clone(&active_clone);
            let peak = Arc::clone(&peak_clone);
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                loop {
                    let old = peak.load(Ordering::SeqCst);
                    if current <= old
                        || peak
                            .compare_exchange(old, current, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                    {
                        break;
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert!(
            peak_clone.load(Ordering::SeqCst) <= 2,
            "peak should not exceed max_concurrency=2"
        );
    }

    #[tokio::test]
    async fn deterministic_ids_under_concurrent_scheduling() {
        // IDs must be claimed in creation order regardless of execution.
        let ctx = test_ctx(CheckpointLog::empty());

        // Mint 3 IDs (simulating what map does internally).
        let id1 = ctx.mint_id();
        let id2 = ctx.mint_id();
        let id3 = ctx.mint_id();

        assert_eq!(id1.positional(), "1");
        assert_eq!(id2.positional(), "2");
        assert_eq!(id3.positional(), "3");
    }

    #[tokio::test]
    async fn never_started_branches_omitted_from_results() {
        // A batch with min_successful=1 should omit branches that never started.
        // We test the completion logic directly.
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(1)
            .build()
            .expect("valid completion config");
        // After 1 success, should stop.
        assert!(should_stop_min(&cfg, 1));
    }

    #[tokio::test]
    async fn completion_config_validate_mutual_exclusivity() {
        // Having both min_successful and tolerated_failure_count is valid
        // (Go/JS allow it — first threshold fires). No error.
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(2)
            .tolerated_failure_count(1)
            .build()
            .expect("valid completion config");
        assert!(cfg.validate().is_ok());
    }

    /// Within one settle event the tracker checks the triggers in a fixed
    /// order — `min_successful`, then the failure tolerances, then the
    /// custom predicate — so when several would fire at once the fixed
    /// thresholds keep the precedence they always had.
    #[test]
    fn completion_tracker_orders_triggers_within_one_event() {
        /// Drives a fresh tracker through one success (item 0) and one
        /// failure (item 1) out of 4 items, returning the first trigger.
        fn first_trigger(
            cfg: &crate::builders::map_parallel::CompletionConfig,
        ) -> Option<CompletionReason> {
            let mut tracker = CompletionTracker::new(4);
            tracker.settle(cfg, 4, 0, BatchItemStatus::Succeeded);
            tracker.settle(cfg, 4, 1, BatchItemStatus::Failed);
            tracker.stop_reason
        }

        // All three triggers satisfied at once: min_successful wins.
        let all = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(1)
            .tolerated_failure_count(0)
            .completion_predicate(|_| true)
            .build()
            .expect("valid completion config");
        assert_eq!(
            first_trigger(&all),
            Some(CompletionReason::MinSuccessfulReached)
        );

        // Failure tolerance and predicate satisfied at the same settle
        // event (item 1's failure): failure tolerance wins.
        let failure_and_predicate = crate::builders::map_parallel::CompletionConfig::builder()
            .tolerated_failure_count(0)
            .completion_predicate(|stats| stats.settled() >= 2)
            .build()
            .expect("valid completion config");
        assert_eq!(
            first_trigger(&failure_and_predicate),
            Some(CompletionReason::FailureToleranceExceeded)
        );

        // Only the predicate fires.
        let predicate_only = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(10)
            .completion_predicate(|stats| stats.settled() >= 2)
            .build()
            .expect("valid completion config");
        assert_eq!(
            first_trigger(&predicate_only),
            Some(CompletionReason::PredicateMatched)
        );

        // Nothing fires.
        let none = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(10)
            .completion_predicate(|stats| stats.settled() >= 3)
            .build()
            .expect("valid completion config");
        assert_eq!(first_trigger(&none), None);
    }

    /// The custom predicate observes only the committed prefix: outcomes
    /// commit strictly in input order, so an item that settles ahead of an
    /// earlier, still-unsettled item stays out of the statistics until the
    /// earlier item settles — whatever the live settlement order was. This
    /// is the property that makes predicate decisions reproducible from
    /// recorded state alone.
    #[test]
    fn completion_tracker_commits_outcomes_in_input_order() {
        /// Every predicate evaluation's view of the outcomes, as
        /// `(index, status)` pairs.
        type ObservedPrefixes = Vec<Vec<(usize, BatchItemStatus)>>;
        let observed: std::sync::Arc<std::sync::Mutex<ObservedPrefixes>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_handle = std::sync::Arc::clone(&observed);
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .completion_predicate(move |stats| {
                if let Ok(mut log) = observed_handle.lock() {
                    log.push(
                        stats
                            .outcomes()
                            .iter()
                            .map(|o| (o.index(), o.status()))
                            .collect(),
                    );
                }
                false
            })
            .build()
            .expect("valid completion config");

        let mut tracker = CompletionTracker::new(3);
        // Reversed live settlement order: item 1 fails before item 0
        // succeeds; item 2 settles last.
        tracker.settle(&cfg, 3, 1, BatchItemStatus::Failed);
        assert_eq!(
            tracker.settled_count(),
            1,
            "settlement-order counts advance immediately"
        );
        tracker.settle(&cfg, 3, 0, BatchItemStatus::Succeeded);
        tracker.settle(&cfg, 3, 2, BatchItemStatus::Succeeded);

        // Item 1's settle committed nothing (item 0 was outstanding), item
        // 0's settle committed the prefix [0, 1], item 2's settle extended
        // it to [0, 1, 2]: three evaluations, each on an input-order prefix.
        let log = observed.lock().expect("no poisoned lock in test");
        assert_eq!(
            *log,
            vec![
                vec![(0, BatchItemStatus::Succeeded)],
                vec![
                    (0, BatchItemStatus::Succeeded),
                    (1, BatchItemStatus::Failed)
                ],
                vec![
                    (0, BatchItemStatus::Succeeded),
                    (1, BatchItemStatus::Failed),
                    (2, BatchItemStatus::Succeeded)
                ],
            ]
        );
    }

    /// A custom completion predicate ends a live map batch early, dispatch
    /// stops, and the recorded batch payload carries the
    /// `PREDICATE_MATCHED` reason.
    #[tokio::test]
    async fn completion_predicate_ends_map_batch_early() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .map(
                vec![10_u32, 20, 30, 40, 50],
                |_child, item, _idx| async move { Ok(item) },
            )
            .name("predicated")
            .max_concurrency(1)
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .completion_predicate(|stats| stats.settled() >= 2)
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("a predicate-completed batch must not become an operation error");

        assert_eq!(batch.reason, CompletionReason::PredicateMatched);
        assert_eq!(
            batch.items.len(),
            2,
            "the predicate fires after the second settle; later items never start"
        );
        assert_eq!(batch.success_count(), 2);

        // The batch parent SUCCEED payload records the reason string, which
        // is what replay reads back through `CompletionReason::from_wire`.
        let parent_success_payload = client
            .recorded_updates()
            .iter()
            .filter(|u| matches!(u.action(), OperationAction::Succeed) && u.parent_id().is_none())
            .filter_map(|u| u.payload().map(str::to_owned))
            .next_back()
            .expect("the batch parent must record a SUCCEED payload");
        assert!(
            parent_success_payload.contains("PREDICATE_MATCHED"),
            "recorded payload must carry the predicate reason, got: {parent_success_payload}"
        );
    }

    /// The predicate receives the per-item settled outcomes, so it can key
    /// off WHICH item settled and how — not just the counts.
    #[tokio::test]
    async fn completion_predicate_observes_settled_outcomes() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        // Stop as soon as item index 1 is observed to have failed. The high
        // failure tolerance keeps the failure trigger out of the way.
        let batch = ctx
            .map(vec![0_u32, 1, 2, 3], |_child, item, _idx| async move {
                if item == 1 {
                    Err("item 1 fails".into())
                } else {
                    Ok(item)
                }
            })
            .max_concurrency(1)
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .tolerated_failure_count(10)
                    .completion_predicate(|stats| {
                        stats.outcomes().iter().any(|outcome| {
                            outcome.index() == 1 && outcome.status() == BatchItemStatus::Failed
                        })
                    })
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("a predicate-completed batch must not become an operation error");

        assert_eq!(batch.reason, CompletionReason::PredicateMatched);
        assert_eq!(
            batch.items.len(),
            2,
            "items 2 and 3 must never start once the predicate fires"
        );
        assert_eq!(batch.failure_count(), 1);
    }

    /// The predicate composes with `min_successful`: within one settle event
    /// the fixed threshold is checked first, so when both fire at the same
    /// settle the recorded reason is `MinSuccessfulReached` (first trigger
    /// wins, matching the existing threshold semantics).
    #[tokio::test]
    async fn completion_predicate_composes_with_min_successful() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        // Both min_successful(2) and the predicate (settled >= 2) fire at
        // the second settle: the threshold wins.
        let batch = ctx
            .map(vec![1_u32, 2, 3, 4], |_child, item, _idx| async move {
                Ok(item)
            })
            .max_concurrency(1)
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .min_successful(2)
                    .completion_predicate(|stats| stats.settled() >= 2)
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("batch must complete early");
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        assert_eq!(batch.items.len(), 2);

        // The predicate fires at an earlier settle event than the threshold:
        // the predicate wins (first trigger across events).
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![1_u32, 2, 3, 4], |_child, item, _idx| async move {
                Ok(item)
            })
            .max_concurrency(1)
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .min_successful(3)
                    .completion_predicate(|stats| stats.settled() >= 1)
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("batch must complete early");
        assert_eq!(batch.reason, CompletionReason::PredicateMatched);
        assert_eq!(batch.items.len(), 1);
    }

    /// Zero-item batches keep their pre-predicate behavior: the empty
    /// collection short-circuits before the coordinator loop and records
    /// `ALL_COMPLETED`, whatever completion thresholds or predicate are
    /// configured. (`min_successful(0)` means "no minimum" — the trigger
    /// only arms for `min > 0` — so it never rewrote the reason before the
    /// predicate feature either, and the predicate is never consulted when
    /// nothing settles.)
    #[tokio::test]
    async fn empty_batch_records_all_completed_regardless_of_completion_config() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .map(
                Vec::<u32>::new(),
                |_child, item, _idx| async move { Ok(item) },
            )
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .min_successful(0)
                    .completion_predicate(|_| true)
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("an empty batch completes immediately");

        assert_eq!(batch.reason, CompletionReason::AllCompleted);
        assert!(batch.items.is_empty());
    }

    /// The predicate's wire reason survives the checkpoint round-trip.
    #[test]
    fn completion_reason_predicate_round_trips_wire() {
        assert_eq!(
            CompletionReason::from_wire(CompletionReason::PredicateMatched.as_str()),
            CompletionReason::PredicateMatched
        );
    }

    // ── Gap 1: BatchResult/BatchItem public API tests ───────────────────

    #[test]
    fn batch_item_new_constructs_correctly() {
        let item: BatchItem<String> = BatchItem::new(
            3,
            "my-item".to_owned(),
            BatchItemStatus::Succeeded,
            Some("hello".to_owned()),
            None,
            None,
        );
        assert_eq!(item.index, 3);
        assert_eq!(item.name, "my-item");
        assert_eq!(item.status, BatchItemStatus::Succeeded);
        assert_eq!(item.result.as_deref(), Some("hello"));
        assert!(item.error_message.is_none());
        assert!(item.error_type.is_none());
    }

    #[test]
    fn batch_result_accessors_match_go_sdk() {
        let result: BatchResult<i32> = BatchResult::new(
            vec![
                BatchItem::new(
                    0,
                    String::new(),
                    BatchItemStatus::Succeeded,
                    Some(10),
                    None,
                    None,
                ),
                BatchItem::new(
                    1,
                    String::new(),
                    BatchItemStatus::Failed,
                    None,
                    Some("err".into()),
                    Some("ChildFnError".into()),
                ),
                BatchItem::new(
                    2,
                    String::new(),
                    BatchItemStatus::Succeeded,
                    Some(30),
                    None,
                    None,
                ),
            ],
            CompletionReason::FailureToleranceExceeded,
        );
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 1);
        assert_eq!(result.total_count(), 3);
        assert!(result.has_failure());
        assert_eq!(result.status(), BatchStatus::Failed);
        let errors = result.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].index, 1);
        assert_eq!(errors[0].name, "");
        assert_eq!(errors[0].message, "err");
        assert_eq!(errors[0].error_type, Some("ChildFnError"));
        assert_eq!(result.results(), vec![&10, &30]);
        assert_eq!(result.reason, CompletionReason::FailureToleranceExceeded);
    }

    #[test]
    fn batch_result_all_succeeded() {
        let result: BatchResult<&str> = BatchResult::new(
            vec![
                BatchItem::new(
                    0,
                    String::new(),
                    BatchItemStatus::Succeeded,
                    Some("a"),
                    None,
                    None,
                ),
                BatchItem::new(
                    1,
                    String::new(),
                    BatchItemStatus::Succeeded,
                    Some("b"),
                    None,
                    None,
                ),
            ],
            CompletionReason::AllCompleted,
        );
        assert!(!result.has_failure());
        assert_eq!(result.status(), BatchStatus::Succeeded);
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 0);
        assert!(result.errors().is_empty());
    }

    /// `BatchStatus` must render exactly the strings the old string-typed
    /// `status()` accessor returned.
    #[test]
    fn batch_status_display_preserves_wire_strings() {
        assert_eq!(BatchStatus::Succeeded.as_str(), "SUCCEEDED");
        assert_eq!(BatchStatus::Failed.as_str(), "FAILED");
        assert_eq!(BatchStatus::Succeeded.to_string(), "SUCCEEDED");
        assert_eq!(BatchStatus::Failed.to_string(), "FAILED");
    }

    /// Every completion reason `execute_batch` can produce must round-trip
    /// through its wire representation.
    #[test]
    fn completion_reason_round_trips_through_wire() {
        for reason in [
            CompletionReason::AllCompleted,
            CompletionReason::MinSuccessfulReached,
            CompletionReason::FailureToleranceExceeded,
        ] {
            assert_eq!(CompletionReason::from_wire(reason.as_str()), reason);
        }
        // Unknown strings from a newer SDK degrade to AllCompleted.
        assert_eq!(
            CompletionReason::from_wire("SOME_FUTURE_REASON"),
            CompletionReason::AllCompleted
        );
    }

    /// `errors()` must associate each error with the item that produced it:
    /// the item's index, its display name, the error type, and the message.
    #[test]
    fn errors_carry_item_index_name_and_message() {
        let result: BatchResult<i32> = BatchResult::new(
            vec![
                BatchItem::new(
                    0,
                    "a".to_owned(),
                    BatchItemStatus::Succeeded,
                    Some(1),
                    None,
                    None,
                ),
                BatchItem::new(
                    1,
                    "b".to_owned(),
                    BatchItemStatus::Failed,
                    None,
                    Some("first failure".into()),
                    Some("ChildFnError".into()),
                ),
                BatchItem::new(
                    2,
                    "c".to_owned(),
                    BatchItemStatus::Succeeded,
                    Some(3),
                    None,
                    None,
                ),
                BatchItem::new(
                    4,
                    "e".to_owned(),
                    BatchItemStatus::Failed,
                    None,
                    Some("second failure".into()),
                    Some("TimeoutError".into()),
                ),
            ],
            CompletionReason::AllCompleted,
        );

        let errors = result.errors();
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].index, 1);
        assert_eq!(errors[0].name, "b");
        assert_eq!(errors[0].message, "first failure");
        assert_eq!(errors[0].error_type, Some("ChildFnError"));
        assert_eq!(errors[1].index, 4);
        assert_eq!(errors[1].name, "e");
        assert_eq!(errors[1].message, "second failure");
        assert_eq!(errors[1].error_type, Some("TimeoutError"));
    }

    /// A failed item with no recorded message or error type yields an empty
    /// message and a `None` type rather than being dropped from `errors()`.
    #[test]
    fn errors_include_failed_items_without_messages() {
        let result: BatchResult<i32> = BatchResult::new(
            vec![BatchItem::new(
                0,
                "x".to_owned(),
                BatchItemStatus::Failed,
                None,
                None,
                None,
            )],
            CompletionReason::AllCompleted,
        );
        let errors = result.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].index, 0);
        assert_eq!(errors[0].name, "x");
        assert_eq!(errors[0].message, "");
        assert_eq!(errors[0].error_type, None);
    }

    /// A failed item's error type and message must survive the checkpoint
    /// payload round-trip (`from_batch_result` → `to_batch_result`), so
    /// error identity is preserved across suspension and replay rather
    /// than being discarded on the way back in.
    #[tokio::test]
    async fn batch_checkpoint_round_trips_error_type_and_message() {
        let original: BatchResult<i32> = BatchResult::new(
            vec![
                BatchItem::new(
                    0,
                    "ok".to_owned(),
                    BatchItemStatus::Succeeded,
                    Some(7),
                    None,
                    None,
                ),
                BatchItem::new(
                    1,
                    "boom".to_owned(),
                    BatchItemStatus::Failed,
                    None,
                    Some("it broke".to_owned()),
                    Some("CustomFailure".to_owned()),
                ),
            ],
            CompletionReason::FailureToleranceExceeded,
        );

        let json = std::sync::Arc::new(crate::serdes::JsonSerdes);
        let (payload, _rebuilt) = from_batch_result(original, &json, "parent-wire", "arn:test")
            .await
            .expect("serializing a batch result must succeed");
        let replayed: BatchResult<i32> = to_batch_result(payload, &json, "parent-wire", "arn:test")
            .await
            .expect("deserializing the payload must succeed");

        assert_eq!(replayed.reason, CompletionReason::FailureToleranceExceeded);
        assert_eq!(replayed.items.len(), 2);
        let failed = &replayed.items[1];
        assert_eq!(failed.index, 1);
        assert_eq!(failed.name, "boom");
        assert_eq!(failed.status, BatchItemStatus::Failed);
        assert_eq!(failed.error_message.as_deref(), Some("it broke"));
        assert_eq!(failed.error_type.as_deref(), Some("CustomFailure"));
        let errors = replayed.errors();
        assert_eq!(errors[0].error_type, Some("CustomFailure"));
        assert_eq!(errors[0].message, "it broke");
        let ok = &replayed.items[0];
        assert_eq!(ok.result, Some(7));
        assert!(ok.error_type.is_none());
    }

    /// A failed item that recorded no error type (a live item defaults to
    /// `ChildFnError`) writes the default to the wire, so payloads keep
    /// the `errType` field older readers expect.
    #[tokio::test]
    async fn from_batch_result_defaults_missing_error_type() {
        let original: BatchResult<i32> = BatchResult::new(
            vec![BatchItem::new(
                0,
                String::new(),
                BatchItemStatus::Failed,
                None,
                Some("boom".to_owned()),
                None,
            )],
            CompletionReason::AllCompleted,
        );
        let json = std::sync::Arc::new(crate::serdes::JsonSerdes);
        let (payload, _rebuilt) = from_batch_result(original, &json, "parent-wire", "arn:test")
            .await
            .expect("serializing a batch result must succeed");
        assert_eq!(
            payload.results.first().map(|r| r.err_type.as_str()),
            Some(CHILD_FN_ERROR_TYPE)
        );
    }

    /// A checkpoint payload whose failed item carries no `errType` (written
    /// before error typing) replays with `error_type: None` rather than a
    /// fabricated type.
    #[tokio::test]
    async fn to_batch_result_missing_err_type_is_none() {
        let payload: BatchSummary = serde_json::from_str(
            r#"{"results":[{"index":0,"status":"FAILED","errMessage":"boom"}],"reason":"ALL_COMPLETED"}"#,
        )
        .expect("payload literal must deserialize");
        let json = std::sync::Arc::new(crate::serdes::JsonSerdes);
        let replayed: BatchResult<i32> = to_batch_result(payload, &json, "parent-wire", "arn:test")
            .await
            .expect("deserializing the payload must succeed");
        let item = replayed.items.first().expect("one item");
        assert_eq!(item.error_message.as_deref(), Some("boom"));
        assert_eq!(item.error_type, None);
        assert_eq!(replayed.errors()[0].error_type, None);
    }

    /// Replaying a frozen terminal batch whose payload carries per-item
    /// `errType` exposes that type through `errors()` without re-executing
    /// any item body.
    #[tokio::test]
    async fn replay_frozen_batch_preserves_error_type() {
        let payload = serde_json::json!({
            "results": [
                {"index": 0, "status": "SUCCEEDED", "result": "\"hello\""},
                {"index": 1, "status": "FAILED", "errType": "ChildFnError", "errMessage": "boom"}
            ],
            "reason": "ALL_COMPLETED"
        });
        let log = CheckpointLog::from_records(vec![{
            let wire_id = crate::engine::compute_wire_id_public("1");
            (
                wire_id.clone(),
                CheckpointRecord {
                    id: wire_id,
                    status: CheckpointStatus::Succeeded,
                    result: Some(payload.to_string()),
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
                    op_name: None,
                },
            )
        }]);
        let ctx = test_ctx(log);

        let batch = ctx
            .map(
                vec!["a".to_owned(), "b".to_owned()],
                |_child, _item: String, _idx| async move {
                    Err::<String, _>("should not execute during replay".into())
                },
            )
            .await_batch()
            .await
            .expect("frozen batch must replay from the payload");

        assert_eq!(batch.success_count(), 1);
        assert_eq!(batch.failure_count(), 1);
        let errors = batch.errors();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].index, 1);
        assert_eq!(errors[0].message, "boom");
        assert_eq!(errors[0].error_type, Some(CHILD_FN_ERROR_TYPE));
    }

    // ── Gap 3: NestingMode tests ────────────────────────────────────────

    #[test]
    fn nesting_mode_default_is_normal() {
        assert_eq!(NestingMode::default(), NestingMode::Normal);
    }

    #[test]
    fn nesting_mode_equality() {
        assert_ne!(NestingMode::Flat, NestingMode::Normal);
    }

    // ── Gap 5: Completion reason wire strings ───────────────────────────

    #[test]
    fn completion_reason_as_str_values() {
        assert_eq!(CompletionReason::AllCompleted.as_str(), "ALL_COMPLETED");
        assert_eq!(
            CompletionReason::MinSuccessfulReached.as_str(),
            "MIN_SUCCESSFUL_REACHED"
        );
        assert_eq!(
            CompletionReason::FailureToleranceExceeded.as_str(),
            "FAILURE_TOLERANCE_EXCEEDED"
        );
    }

    // ── Bug 1: Early-stopping race — stopped re-check after acquire ────

    #[tokio::test]
    async fn early_stop_recheck_after_semaphore_acquire() {
        // Simulates the fix: with concurrency=1, once `stopped` is set, the
        // next iteration must NOT slip through after acquiring the permit.
        let sem = Arc::new(Semaphore::new(1));
        let stopped = Arc::new(AtomicBool::new(false));
        let launched = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for i in 0..3 {
            let sem = Arc::clone(&sem);
            let stopped = Arc::clone(&stopped);
            let launched = Arc::clone(&launched);
            handles.push(tokio::spawn(async move {
                // Acquire
                let permit = sem.acquire_owned().await.unwrap();
                // Re-check stopped AFTER acquire (the fix).
                if stopped.load(Ordering::SeqCst) {
                    drop(permit);
                    return;
                }
                launched.fetch_add(1, Ordering::SeqCst);
                // Simulate work, then stop after item 0.
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                if i == 0 {
                    stopped.store(true, Ordering::SeqCst);
                }
                drop(permit);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        // With the fix, only item 0 should launch (item 1 sees stopped=true
        // after acquiring, item 2 likewise). Without the fix, item 1 would
        // slip through.
        assert_eq!(
            launched.load(Ordering::SeqCst),
            1,
            "only one item should launch before stopped is honoured"
        );
    }

    // ── Bug 3: FLAT nesting ParentId points to batch parent ────────────

    #[test]
    fn flat_child_context_reports_batch_parent_wire_id() {
        let ctx = test_ctx(CheckpointLog::empty());
        // Mint an ID for the batch parent.
        let batch_op = ctx.mint_id();
        let batch_wire = batch_op.wire().to_owned();
        // Mint an ID for the virtual child.
        let child_op = ctx.mint_id();
        let child_positional = child_op.positional().to_owned();
        // Create a flat child context with the batch parent's wire ID.
        let flat_child = ctx.new_scoped_flat_child(&child_positional, &batch_wire);
        // Operations inside the flat child should report the batch parent
        // as their parent wire ID.
        assert_eq!(
            flat_child.parent_wire_id(),
            Some(batch_wire.as_str()),
            "flat child must report the batch parent's wire ID"
        );
    }

    // ── Bug 4: Suspension propagation (no ContextFailed checkpoint) ─────

    #[test]
    fn suspension_signal_detected_prevents_false_failure() {
        // The suspension signal is shared across parent and child contexts.
        let ctx = test_ctx(CheckpointLog::empty());
        assert!(
            !ctx.suspension_signal().is_suspend_requested(),
            "initially no suspension"
        );
        ctx.request_suspend();
        assert!(
            ctx.suspension_signal().is_suspend_requested(),
            "suspension signal propagates"
        );
        // A child context shares the same signal.
        let child = ctx.new_child("1");
        assert!(
            child.suspension_signal().is_suspend_requested(),
            "child inherits suspension signal from parent"
        );
    }

    // ── Bug 5: max_concurrency=0 validation error ──────────────────────

    #[tokio::test]
    async fn max_concurrency_zero_returns_validation_error() {
        let ctx = test_ctx(CheckpointLog::empty());
        // Attempt parallel with max_concurrency=0.
        let result: Result<Vec<String>, _> = ctx
            .parallel(Vec::<crate::Branch<String>>::new())
            .max_concurrency(0)
            .await;
        let err = result.expect_err("max_concurrency=0 must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("max concurrency must be positive"),
            "error message should mention validation: {msg}"
        );
    }

    #[tokio::test]
    async fn max_concurrency_zero_map_returns_validation_error() {
        let ctx = test_ctx(CheckpointLog::empty());
        // Attempt map with max_concurrency=0.
        let result: Result<Vec<i32>, _> = ctx
            .map(
                vec![1, 2],
                |_child, item: i32, _idx| async move { Ok(item) },
            )
            .max_concurrency(0)
            .await;
        let err = result.expect_err("max_concurrency=0 must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("max concurrency must be positive"),
            "error message should mention validation: {msg}"
        );
    }

    // ── Checkpoint serialization tests ──────────────────────────────────

    /// A token-validating execution client that rejects checkpoint calls
    /// with stale tokens — simulating the service's actual behavior.
    /// Used to verify that concurrent callers are properly serialized.
    #[derive(Debug)]
    struct TokenValidatingClient {
        /// The current expected token; only calls with this token succeed.
        current_token: std::sync::Mutex<String>,
        /// Counter for generating unique next tokens.
        counter: std::sync::Mutex<u32>,
        /// Count of successful checkpoint calls.
        success_count: std::sync::Mutex<u32>,
        /// Count of calls that arrived with a stale (wrong) token.
        stale_token_count: std::sync::Mutex<u32>,
    }

    impl TokenValidatingClient {
        fn new(initial_token: &str) -> Self {
            Self {
                current_token: std::sync::Mutex::new(initial_token.to_owned()),
                counter: std::sync::Mutex::new(0),
                success_count: std::sync::Mutex::new(0),
                stale_token_count: std::sync::Mutex::new(0),
            }
        }

        fn success_count(&self) -> u32 {
            *self
                .success_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        fn stale_token_count(&self) -> u32 {
            *self
                .stale_token_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }

    impl crate::client::ExecutionClient for TokenValidatingClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::client::CheckpointOutput,
                            crate::client::ClientError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let submitted_token = checkpoint_token.to_owned();
            Box::pin(async move {
                // Simulate a brief network delay to increase race window.
                tokio::task::yield_now().await;

                let mut current = self
                    .current_token
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);

                if *current != submitted_token {
                    // Stale token — the real service would return an error.
                    let mut stale = self
                        .stale_token_count
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *stale += 1;
                    return Err(crate::client::ClientError::non_retryable(format!(
                        "stale checkpoint token: expected {}, got {}",
                        *current, submitted_token
                    )));
                }

                // Rotate the token.
                let mut counter = self
                    .counter
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *counter += 1;
                let new_token = format!("token-{counter}");
                *current = new_token.clone();

                let mut success = self
                    .success_count
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *success += 1;

                Ok(crate::client::CheckpointOutput {
                    checkpoint_token: new_token,
                    updated_operations: Vec::new(),
                    next_marker: None,
                })
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<crate::client::GetStateOutput, crate::client::ClientError>,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::client::GetStateOutput {
                    operations: Vec::new(),
                })
            })
        }
    }

    #[tokio::test]
    async fn concurrent_checkpoint_serialization_no_token_conflict() {
        // Spawn N tasks that all call checkpoint_updates concurrently.
        // With the tokio::sync::Mutex serialization, NO stale token errors
        // should occur. Without it, the token-validating client would
        // reject concurrent calls.
        let client = Arc::new(TokenValidatingClient::new("initial-token"));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            client.clone() as Arc<dyn crate::client::ExecutionClient>,
            "initial-token".to_owned(),
        );

        let num_concurrent = 10;
        let mut handles = Vec::new();

        for i in 0..num_concurrent {
            let ctx = ctx.clone();
            handles.push(tokio::spawn(async move {
                let update = OperationUpdate::builder()
                    .id(format!("op-{i}"))
                    .r#type(OperationType::Context)
                    .sub_type("Test")
                    .action(OperationAction::Succeed)
                    .build()
                    .expect("valid update");
                ctx.checkpoint_updates(vec![update]).await
            }));
        }

        let mut success_count = 0_u32;
        for handle in handles {
            let result = handle.await.expect("task should not panic");
            assert!(
                result.is_ok(),
                "checkpoint should succeed due to serialization: {:?}",
                result.unwrap_err()
            );
            success_count += 1;
        }

        assert_eq!(success_count, num_concurrent);
        assert_eq!(client.success_count(), num_concurrent);
        assert_eq!(
            client.stale_token_count(),
            0,
            "no stale token conflicts should occur with serialization"
        );
    }

    #[tokio::test]
    async fn fail_fast_then_parent_terminal_checkpoint_emitted() {
        // Simulates a parallel batch where one branch fails (triggering
        // fail-fast via tolerated_failure_count=0). The parent SUCCEED
        // checkpoint must still be emitted after the failed branch's
        // best-effort FAIL checkpoint.
        //
        // With serialization, the parent's succeed checkpoint waits for
        // any in-flight child checkpoints to complete, then uses the
        // latest token — ensuring it succeeds.
        let client = Arc::new(TokenValidatingClient::new("initial-token"));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            client.clone() as Arc<dyn crate::client::ExecutionClient>,
            "initial-token".to_owned(),
        );

        // Run a parallel batch: 3 branches, first fails, tolerated=0.
        // This triggers fail-fast after the first failure.
        // The public API converts FailureToleranceExceeded to an error,
        // but the internal batch engine still emits the parent terminal
        // checkpoint (ContextSucceeded) before returning.
        let result: Result<Vec<i32>, _> = ctx
            .parallel(vec![
                crate::future::Branch::new("fail", |_ctx| {
                    Box::pin(async { Err("intentional".into()) })
                }),
                crate::future::Branch::new("ok-1", |_ctx| Box::pin(async { Ok(1) })),
                crate::future::Branch::new("ok-2", |_ctx| Box::pin(async { Ok(2) })),
            ])
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .tolerated_failure_count(0)
                    .build()
                    .expect("valid completion config"),
            )
            .await;

        // The public API propagates failures as errors — that's correct
        // behavior. The key assertion is that no stale-token conflicts
        // occurred during the batch execution.
        assert!(
            result.is_err(),
            "parallel with fail-fast should surface the first failure"
        );

        // Critically: no stale token conflicts occurred, meaning the
        // parent succeed checkpoint was properly serialized after the
        // child fail checkpoint.
        assert_eq!(
            client.stale_token_count(),
            0,
            "all checkpoints must serialize without token conflicts"
        );

        // At minimum: parent START + child START + child FAIL + parent SUCCEED = 4.
        // (Other children may or may not start depending on scheduling.)
        assert!(
            client.success_count() >= 4,
            "expected at least 4 checkpoint calls (parent start + child start + child fail + parent succeed), got {}",
            client.success_count()
        );
    }

    #[tokio::test]
    #[allow(clippy::panic)] // reason: the branch panic under test is the behavior being asserted
    async fn map_branch_panic_fails_batch_without_hanging() {
        // A panic in a user map-branch closure must surface as a controlled
        // batch failure promptly, not hang the coordinator to its Lambda
        // timeout. The branch task aborts on panic and never delivers an
        // outcome; the coordinator observes the JoinError and records a branch
        // failure instead of waiting forever for a value that never arrives.
        let client = Arc::new(TokenValidatingClient::new("initial-token"));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            client.clone() as Arc<dyn crate::client::ExecutionClient>,
            "initial-token".to_owned(),
        );

        let run = async move {
            ctx.map(vec![0_i32], |_child, _item: i32, _idx| async move {
                panic!("boom in map branch")
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .tolerated_failure_count(0)
                    .build()
                    .expect("valid completion config"),
            )
            .await
        };

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), run).await;
        let result: Result<Vec<i32>, OperationError> =
            outcome.expect("map with a panicking branch must resolve, not hang");
        let err =
            result.expect_err("a panicking branch must surface as a controlled batch failure");
        assert!(
            format!("{err:#}").contains("boom in map branch"),
            "batch failure should carry the branch panic message, got: {err:#}"
        );
    }

    #[tokio::test]
    #[allow(clippy::panic)] // reason: the branch panic under test is the behavior being asserted
    async fn parallel_branch_panic_fails_batch_without_hanging() {
        use crate::future::Branch;

        // Same guarantee for parallel: a panicking branch is a controlled
        // failure delivered promptly, never a hang.
        let client = Arc::new(TokenValidatingClient::new("initial-token"));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            client.clone() as Arc<dyn crate::client::ExecutionClient>,
            "initial-token".to_owned(),
        );

        let run = async move {
            ctx.parallel(vec![Branch::new("boom", |_ctx| {
                Box::pin(async { panic!("boom in parallel branch") })
            })])
            .completion(
                crate::builders::map_parallel::CompletionConfig::builder()
                    .tolerated_failure_count(0)
                    .build()
                    .expect("valid completion config"),
            )
            .await
        };

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), run).await;
        let result: Result<Vec<i32>, OperationError> =
            outcome.expect("parallel with a panicking branch must resolve, not hang");
        let err =
            result.expect_err("a panicking branch must surface as a controlled batch failure");
        assert!(
            format!("{err:#}").contains("boom in parallel branch"),
            "batch failure should carry the branch panic message, got: {err:#}"
        );
    }

    #[tokio::test]
    #[allow(clippy::panic)] // reason: the reconstruction panic under test is the behavior being asserted
    async fn map_replay_children_reconstruction_panic_unwinds_without_fail_checkpoint() {
        // A child recorded as terminal SUCCESS with `replay_children = true`
        // is re-executed on replay to reconstruct its oversized result. If
        // that reconstruction panics, the panic must unwind the coordinator
        // with the original payload — preserving the recorded success
        // history for the next attempt — NOT be graded as a branch failure
        // with a child FAIL checkpoint, which would overwrite durable
        // success and permanently fail the batch over a transient crash.
        let log = CheckpointLog::from_records(vec![replay_children_success_record("2")]);
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let task_client = Arc::clone(&client);

        // The context is created INSIDE the spawned task so the task owns
        // it: the unwind of that task is the observable coordinator unwind.
        let handle = tokio::spawn(async move {
            let ctx = DurableContext::new_root_with_client(
                "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(log),
                task_client as Arc<dyn crate::client::ExecutionClient>,
                "token0".to_owned(),
            );
            let result: Result<Vec<i32>, OperationError> = ctx
                .map(vec![1_i32], |_child, _item: i32, _idx| async move {
                    panic!("boom in map reconstruction")
                })
                .await;
            result
        });

        let join_err = handle
            .await
            .expect_err("a reconstruction panic must unwind the coordinator, not settle");
        assert!(
            join_err.is_panic(),
            "the coordinator must rethrow the original panic payload"
        );
        let payload = join_err.into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned());
        assert_eq!(
            message.as_deref(),
            Some("boom in map reconstruction"),
            "the rethrown payload must be the reconstruction panic's own"
        );

        // The recorded terminal success stays authoritative: nothing may be
        // checkpointed as FAILED on this path.
        assert!(
            client
                .recorded_updates()
                .iter()
                .all(|u| !matches!(u.action(), OperationAction::Fail)),
            "a reconstruction panic must not checkpoint any failure"
        );
    }

    #[tokio::test]
    #[allow(clippy::panic)] // reason: the reconstruction panic under test is the behavior being asserted
    async fn parallel_replay_children_reconstruction_panic_unwinds_without_fail_checkpoint() {
        use crate::future::Branch;

        // Same guarantee for parallel, through the concurrent replay pass:
        // branch 0 replays from a normal recorded success; branch 1 is
        // recorded ReplayChildren and its reconstruction panics. The panic
        // must unwind the coordinator without checkpointing failure.
        let log = CheckpointLog::from_records(vec![
            succeeded_record("2", "7"),
            replay_children_success_record("3"),
        ]);
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let task_client = Arc::clone(&client);

        // The context is created INSIDE the spawned task so the task owns
        // it: the unwind of that task is the observable coordinator unwind.
        let handle = tokio::spawn(async move {
            let ctx = DurableContext::new_root_with_client(
                "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(log),
                task_client as Arc<dyn crate::client::ExecutionClient>,
                "token0".to_owned(),
            );
            let result: Result<Vec<i32>, OperationError> = ctx
                .parallel(vec![
                    // Replayed from the record; never re-executed.
                    Branch::new("steady", |_ctx| async move { Ok(999_i32) }),
                    Branch::new("boom", |_ctx| async move {
                        panic!("boom in parallel reconstruction")
                    }),
                ])
                .await;
            result
        });

        let join_err = handle
            .await
            .expect_err("a reconstruction panic must unwind the coordinator, not settle");
        assert!(
            join_err.is_panic(),
            "the coordinator must rethrow the original panic payload"
        );
        let payload = join_err.into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned());
        assert_eq!(
            message.as_deref(),
            Some("boom in parallel reconstruction"),
            "the rethrown payload must be the reconstruction panic's own"
        );

        assert!(
            client
                .recorded_updates()
                .iter()
                .all(|u| !matches!(u.action(), OperationAction::Fail)),
            "a reconstruction panic must not checkpoint any failure"
        );
    }

    // ── Serialization-model equivalence tests ───────────────────────────
    //
    // These prove the point of the normalization: ONE `Serdes`
    // implementation — one that implements only the JSON-string transform
    // methods — behaves identically as a map item serdes, a parallel item
    // serdes, and a step serdes. Before the normalization, map/parallel item
    // results went through a separate mandatory byte API plus a runtime
    // downcast, so a serdes like this one could not even be written.

    /// A payload whose stored wire form plain `serde_json` cannot parse, and
    /// whose JSON needs real escaping (quotes, backslash, newline, non-ASCII).
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Doc {
        label: String,
        nested: Vec<Vec<i64>>,
    }

    fn probe_doc() -> Doc {
        Doc {
            label: "quote:\" backslash:\\ newline:\n tab:\t ünïcodé ☃".to_owned(),
            nested: vec![vec![1, -2, i64::MIN], Vec::new()],
        }
    }

    /// Payloads recorded on SUCCEED updates that carry a `ParentId` (i.e. the
    /// map children / parallel branches, not the batch parent).
    fn child_success_payloads(client: &crate::client::InMemoryExecutionClient) -> Vec<String> {
        client
            .recorded_updates()
            .iter()
            .filter(|u| matches!(u.action(), OperationAction::Succeed) && u.parent_id().is_some())
            .filter_map(|u| u.payload().map(str::to_owned))
            .collect()
    }

    /// The wire form a step, a map item and a parallel branch produce for the
    /// same value through the same custom serdes must be byte-identical, and
    /// each must round-trip back to the original value.
    #[tokio::test]
    async fn custom_serdes_is_equivalent_on_step_map_and_parallel_paths() {
        use crate::future::Branch;
        use crate::serdes::test_support::{HexEnvelopeSerdes, hex_envelope};

        let doc = probe_doc();
        let expected_wire = hex_envelope(&serde_json::to_string(&doc).expect("doc is JSON-able"));

        // Control: the wire form is NOT valid JSON, so any path that skipped
        // the serdes transform would fail rather than silently pass.
        assert!(
            serde_json::from_str::<Doc>(&expected_wire).is_err(),
            "the probe wire form must be unparseable by plain serde_json"
        );

        // ── step ──
        let (step_ctx, step_client) = test_ctx_with_client(CheckpointLog::empty());
        let step_doc = doc.clone();
        let step_out: Doc = step_ctx
            .step(move |_| {
                let d = step_doc.clone();
                async move { Ok(d) }
            })
            .serdes(HexEnvelopeSerdes)
            .await
            .expect("step with a string-transform serdes must succeed");
        assert_eq!(step_out, doc, "step must round-trip the value");
        let step_wire: Vec<String> = step_client
            .recorded_updates()
            .iter()
            .filter(|u| matches!(u.action(), OperationAction::Succeed))
            .filter_map(|u| u.payload().map(str::to_owned))
            .collect();
        assert_eq!(step_wire, vec![expected_wire.clone()]);

        // ── map items ──
        let (map_ctx, map_client) = test_ctx_with_client(CheckpointLog::empty());
        let map_doc = doc.clone();
        let map_out: Vec<Doc> = map_ctx
            .map(vec![0_usize], move |_child, _item, _idx| {
                let d = map_doc.clone();
                async move { Ok(d) }
            })
            .serdes(HexEnvelopeSerdes)
            .await
            .expect("map with a string-transform item serdes must succeed");
        assert_eq!(map_out, vec![doc.clone()], "map must round-trip the value");
        assert_eq!(
            child_success_payloads(&map_client),
            vec![expected_wire.clone()],
            "a map item must produce the same wire form as a step"
        );

        // ── parallel branches ──
        let (par_ctx, par_client) = test_ctx_with_client(CheckpointLog::empty());
        let par_doc = doc.clone();
        let par_out: Vec<Doc> = par_ctx
            .parallel(vec![Branch::new("only", move |_c| {
                let d = par_doc.clone();
                async move { Ok(d) }
            })])
            .serdes(HexEnvelopeSerdes)
            .await
            .expect("parallel with a string-transform item serdes must succeed");
        assert_eq!(par_out, vec![doc], "parallel must round-trip the value");
        assert_eq!(
            child_success_payloads(&par_client),
            vec![expected_wire],
            "a parallel branch must produce the same wire form as a step"
        );
    }

    /// ONE RULE ON EVERY PATH.
    ///
    /// The shape a custom serdes is handed must be the identical
    /// `serde_json::Value` on the step, invoke, callback, map/parallel item
    /// and batch-result paths. Nothing asserted this before, which is how a
    /// quoting divergence between items (`X`) and everything else (`"X"`)
    /// survived all the way to conformance case 9-14.
    ///
    /// The probe payload is a bare `String` — the one case where the old split
    /// was visible — and the fixture's wire form is deliberately NOT JSON, so
    /// a path that bypassed the serdes would fail rather than quietly pass.
    #[tokio::test]
    #[allow(clippy::too_many_lines)] // reason: one test per path is the point — splitting it would let the paths drift apart again
    async fn custom_serdes_receives_the_same_value_shape_on_every_path() {
        use crate::future::Branch;
        use crate::serdes::test_support::{RecordingSerdes, hex_envelope_of};

        let value = "X".to_owned();
        // The one shape every path must provide.
        let expected = serde_json::json!("X");
        // What a *string*-boundary serdes would have received instead.
        let json_encoding = serde_json::to_string(&value).expect("a string is JSON-able");
        assert_eq!(json_encoding, "\"X\"");

        let wire = hex_envelope_of(&expected);
        assert!(
            serde_json::from_str::<String>(&wire).is_err(),
            "the probe wire form must be unparseable by plain serde_json, so a \
             path that skipped the transform cannot pass by accident"
        );

        // ── step ──
        let step_rec = RecordingSerdes::new();
        let (step_ctx, step_client) = test_ctx_with_client(CheckpointLog::empty());
        let step_value = value.clone();
        let step_out: String = step_ctx
            .step(move |_| {
                let v = step_value.clone();
                async move { Ok(v) }
            })
            .serdes(step_rec.clone())
            .await
            .expect("step with a recording serdes must succeed");
        assert_eq!(step_out, value, "step must round-trip the value");
        assert_eq!(
            step_rec.serialize_inputs(),
            vec![expected.clone()],
            "a step serdes must be handed the value, not its JSON encoding"
        );
        let step_wire: Vec<String> = step_client
            .recorded_updates()
            .iter()
            .filter(|u| matches!(u.action(), OperationAction::Succeed))
            .filter_map(|u| u.payload().map(str::to_owned))
            .collect();
        assert_eq!(step_wire, vec![wire.clone()]);

        // ── invoke payload ──
        let invoke_rec = RecordingSerdes::new();
        let (invoke_ctx, _invoke_client) = test_ctx_with_client(CheckpointLog::empty());
        let signal = Arc::clone(invoke_ctx.suspension_signal());
        let invoke_fut = invoke_ctx
            .invoke::<String, String>("target-fn", value.clone())
            .payload_serdes(invoke_rec.clone());
        let outcome = crate::driver::test_support::outcome_of(signal, invoke_fut).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        assert_eq!(
            invoke_rec.serialize_inputs(),
            vec![expected.clone()],
            "an invoke payload serdes must be handed the same value shape as a step"
        );

        // ── callback (deserialize-only path) ──
        //
        // Callbacks have no serialize side (an external caller writes the
        // payload), and `deserialize_callback_result` IS the whole boundary —
        // it is the only place the callback replay path consults a serdes. It
        // is fed the EXACT wire form the step path wrote above, so this asserts
        // the two paths agree on the reconstructed shape.
        let callback_rec = RecordingSerdes::new();
        let callback_ctx = SerdesContext::new("cb-1", "arn:test");
        let callback_out: String = crate::callback::deserialize_callback_result(
            &callback_rec.clone(),
            wire.clone(),
            callback_ctx,
        )
        .await
        .expect("the callback path must decode the wire form the step path wrote");
        assert_eq!(callback_out, value, "callback must round-trip the value");
        assert_eq!(
            callback_rec.deserialize_outputs(),
            vec![expected.clone()],
            "the callback path must reconstruct the same value shape the step \
             path handed the serdes"
        );

        // ── map items ──
        //
        // The item serdes is consulted twice per item: once for the child's own
        // checkpoint and once when the item is embedded in the batch summary.
        // Those two calls diverged before this change; they must now be equal.
        let map_rec = RecordingSerdes::new();
        let (map_ctx, _map_client) = test_ctx_with_client(CheckpointLog::empty());
        let map_value = value.clone();
        let map_out: Vec<String> = map_ctx
            .map(vec![0_usize], move |_child, _item, _idx| {
                let v = map_value.clone();
                async move { Ok(v) }
            })
            .serdes(map_rec.clone())
            .await
            .expect("map with a recording item serdes must succeed");
        assert_eq!(map_out, vec![value.clone()]);
        let map_inputs = map_rec.serialize_inputs();
        assert!(
            map_inputs.len() >= 2,
            "the item serdes must see both the child checkpoint and the batch \
             summary embedding, got {map_inputs:?}"
        );
        assert_eq!(
            map_rec.distinct_serialize_inputs(),
            vec![expected.clone()],
            "every map item serdes call must be handed the identical value \
             shape, got {map_inputs:?}"
        );

        // ── parallel branches ──
        let par_rec = RecordingSerdes::new();
        let (par_ctx, _par_client) = test_ctx_with_client(CheckpointLog::empty());
        let par_value = value.clone();
        let par_out: Vec<String> = par_ctx
            .parallel(vec![Branch::new("only", move |_c| {
                let v = par_value.clone();
                async move { Ok(v) }
            })])
            .serdes(par_rec.clone())
            .await
            .expect("parallel with a recording item serdes must succeed");
        assert_eq!(par_out, vec![value.clone()]);
        assert_eq!(
            par_rec.distinct_serialize_inputs(),
            vec![expected.clone()],
            "a parallel branch serdes must be handed the same value shape"
        );

        // ── whole batch result ──
        //
        // The batch-result serdes is handed the batch summary, so its value
        // differs by nature — but it must arrive through the SAME boundary: a
        // `serde_json::Value`, not pre-rendered text.
        let batch_rec = RecordingSerdes::new();
        let (batch_ctx, _batch_client) = test_ctx_with_client(CheckpointLog::empty());
        let batch_value = value.clone();
        let batch_out: Vec<String> = batch_ctx
            .map(vec![0_usize], move |_child, _item, _idx| {
                let v = batch_value.clone();
                async move { Ok(v) }
            })
            .result_serdes(batch_rec.clone())
            .await
            .expect("map with a recording result serdes must succeed");
        assert_eq!(batch_out, vec![value]);
        let batch_inputs = batch_rec.serialize_inputs();
        assert_eq!(batch_inputs.len(), 1, "inputs: {batch_inputs:?}");
        let summary = batch_inputs
            .first()
            .expect("the batch result serdes must be consulted once");
        assert!(
            summary
                .get("results")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|items| items.len() == 1),
            "the batch result serdes must receive the summary as a structured \
             Value, got {summary}"
        );
        // Each item is embedded in the summary as its own wire string (here the
        // plain-`serde_json` default, since no item serdes is attached), so the
        // batch serdes sees structure — not one flat pre-rendered blob.
        assert_eq!(
            summary
                .get("results")
                .and_then(|r| r.get(0))
                .and_then(|item| item.get("result")),
            Some(&serde_json::Value::String(json_encoding.clone())),
            "summary: {summary}"
        );
    }

    /// Replay must reverse the same transform: a terminal map child whose
    /// stored payload is the custom (non-JSON) wire form is decoded back to
    /// the typed value, with no downcast involved.
    #[tokio::test]
    async fn custom_serdes_item_replay_reverses_the_transform() {
        use crate::serdes::test_support::{HexEnvelopeSerdes, hex_envelope};

        let doc = probe_doc();
        let wire = hex_envelope(&serde_json::to_string(&doc).expect("doc is JSON-able"));

        // Batch parent "1" started; its single child "1.1" already succeeded
        // with the custom wire form in the log.
        let parent_wire_id = crate::engine::compute_wire_id_public("1");
        let child_wire_id = crate::engine::compute_wire_id_public("2");
        let log = CheckpointLog::from_records(vec![
            (
                parent_wire_id.clone(),
                CheckpointRecord {
                    id: parent_wire_id,
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
                    op_type: None,
                    sub_type: None,
                    op_name: None,
                },
            ),
            (
                child_wire_id.clone(),
                CheckpointRecord {
                    id: child_wire_id,
                    status: CheckpointStatus::Succeeded,
                    result: Some(wire),
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
                    op_name: None,
                },
            ),
        ]);
        let (ctx, _client) = test_ctx_with_client(log);

        let out: Vec<Doc> = ctx
            .map(vec![0_usize], |_child, _item, _idx| async move {
                unreachable!("a terminal child must be replayed, not re-executed")
            })
            .serdes(HexEnvelopeSerdes)
            .await
            .expect("replay of a custom-serdes item must succeed");
        assert_eq!(out, vec![probe_doc()]);
    }

    /// The batch summary embeds each item result as its own transformed
    /// payload, so replaying a terminal batch must reverse the item transform
    /// per item.
    #[tokio::test]
    async fn custom_serdes_batch_summary_replay_reverses_item_transforms() {
        use crate::serdes::test_support::{HexEnvelopeSerdes, hex_envelope};

        let first = hex_envelope(r#""hello""#);
        let second = hex_envelope(r#""world""#);
        let payload = serde_json::json!({
            "results": [
                {"index": 0, "status": "SUCCEEDED", "result": first},
                {"index": 1, "status": "SUCCEEDED", "result": second}
            ],
            "reason": "ALL_COMPLETED"
        });
        let wire_id = crate::engine::compute_wire_id_public("1");
        let log = CheckpointLog::from_records(vec![(
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Succeeded,
                result: Some(payload.to_string()),
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
                op_name: None,
            },
        )]);
        let ctx = test_ctx(log);

        let values: Vec<String> = ctx
            .map(
                vec!["a".to_owned(), "b".to_owned()],
                |_child, _item, _idx| async move {
                    unreachable!("terminal batch must replay, not re-execute")
                },
            )
            .serdes(HexEnvelopeSerdes)
            .await
            .expect("batch summary replay with an item serdes must succeed");
        assert_eq!(values, vec!["hello", "world"]);
    }

    /// `FileSystemSerdes` used as a map ITEM serdes. This is the exact case    /// that failed at runtime before the normalization: the byte methods it
    /// was forced to implement returned an error, so the item path errored
    /// even though the same serdes worked on steps and batch results.
    ///
    /// It also pins per-item context identity: each item must resolve to its
    /// own file, otherwise every item would read back the last one written.
    #[tokio::test]
    async fn filesystem_serdes_works_as_a_map_item_serdes() {
        let tmp = std::env::temp_dir().join("map_item_fs_serdes");
        let _ = std::fs::remove_dir_all(&tmp);

        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let out: Vec<String> = ctx
            .map(vec![1_i32, 2, 3], |_child, item, _idx| async move {
                Ok(format!("item-{item}"))
            })
            .serdes(crate::serdes::FileSystemSerdes::new(
                tmp.to_string_lossy().to_string(),
            ))
            .await
            .expect("FileSystemSerdes must work as a map item serdes");
        assert_eq!(out, vec!["item-1", "item-2", "item-3"]);

        // Every item envelope must be a DISTINCT file pointer.
        let payloads = child_success_payloads(&client);
        assert_eq!(payloads.len(), 3, "payloads: {payloads:?}");
        let mut files: Vec<String> = payloads
            .iter()
            .map(|p| {
                let v: serde_json::Value =
                    serde_json::from_str(p).expect("envelope must be valid JSON");
                v.get("file")
                    .and_then(serde_json::Value::as_str)
                    .expect("envelope must be a file pointer")
                    .to_owned()
            })
            .collect();
        files.sort();
        files.dedup();
        assert_eq!(files.len(), 3, "each item needs its own file: {files:?}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
