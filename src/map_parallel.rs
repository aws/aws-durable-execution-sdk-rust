//! Map and parallel operation execution engine.
//!
//! Implements the coordinator loop for bounded-concurrency fan-out with
//! completion-threshold semantics. Both `map` and `parallel` share the
//! same underlying batch engine: they differ only in how items are
//! produced (from a `Vec<I>` + closure, or from named `Branch<O>` values).
//!
//! Wire shape:
//! - Parent: `OperationType::Context`, `SubType` `Map` or `Parallel`
//! - Children: `OperationType::Context`, `SubType` `MapIteration` or `ParallelBranch`
//! - Actions: Start → (Succeed | Fail)
//! - `ParentId` on children points to the parent's wire ID
//!
//! Invariants:
//! - A branch that waits on a durable operation stays pending in the
//!   coordinator's join set and RETAINS its concurrency slot (only terminal
//!   events decrement in-flight). It continues in the same invocation when
//!   its wait completes
//! - Never-started branches are OMITTED from results
//! - Completion decision uses the STARTED set
//! - Results collected positionally per branch index
//! - Each branch runs in its own child context
//! - Fail-fast when `min_successful` can no longer be met

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tracing::Instrument as _;

use crate::BoxError;
use crate::Serdes;
use crate::context::{ActiveBody, DurableContext, run_body};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, ChildFnError, OperationError, OperationErrorKind,
};
use crate::serdes::SerdesContext;
use crate::task::{NoRuntime, TaskOutcome, TrackedJoinSet};

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

/// The status of one item or branch in a batch operation.
///
/// Each item in a [`BatchResult`] has a status indicating whether it
/// completed successfully, failed, or had started but not yet reached a
/// terminal state when the batch completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchItemStatus {
    /// The item completed successfully.
    Succeeded,
    /// The item failed.
    Failed,
    /// The item was dispatched and was not terminal when the batch
    /// completed.
    ///
    /// It appears live and on replay. A completion rule that ends the batch
    /// early leaves running and parked items in this status. The SDK
    /// cancels a running item at its next `.await`, and none of its later
    /// checkpoints is written. Its child context can still show
    /// `Succeeded` or `Failed` in history when its outcome was recorded
    /// before the batch completed. The item carries no result and no
    /// error.
    Started,
}

/// The outcome of one item or branch in a batch operation.
///
/// Contains the index, optional display name, status, and either the
/// result value or the item's complete recorded failure record (type
/// identifier, message, opaque `error_data` payload, and stack trace). A
/// [`BatchItemStatus::Started`] item carries neither.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::{BatchItemStatus, BatchItem};
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
    /// Status: `Succeeded` or `Failed` for a terminal item, or `Started`
    /// for an item that was not terminal when the batch completed.
    pub status: BatchItemStatus,
    /// Result value (only meaningful when status is Succeeded).
    pub result: Option<O>,
    /// Error message (only meaningful when status is Failed).
    pub error_message: Option<String>,
    /// Error type identifier (only meaningful when status is Failed).
    ///
    /// Carries the `errType` recorded in the batch checkpoint payload: the
    /// recorded type of the error the item body returned (`"Error"` for an
    /// untyped error), so error identity
    /// survives checkpoint replay alongside the message. `None` when the
    /// payload recorded no type (a payload written before error typing).
    pub error_type: Option<String>,
    /// Opaque wire `ErrorData` payload recorded with the failure (only
    /// meaningful when status is Failed).
    ///
    /// Passed through verbatim from the failing item's recorded wire
    /// record, never interpreted by the SDK. `None` when the failure
    /// recorded no payload (including records written before this field
    /// existed).
    pub error_data: Option<String>,
    /// Recorded wire `StackTrace` frames from the failure (only
    /// meaningful when status is Failed).
    ///
    /// Store-and-expose only, exactly like
    /// [`WireError::stack_trace`](crate::WireError::stack_trace). Empty
    /// when the failure recorded no frames (including records written
    /// before this field existed).
    pub stack_trace: Vec<String>,
}

impl<O> BatchItem<O> {
    /// Creates a new batch item.
    ///
    /// The failure payload fields ([`error_data`](Self::error_data) and
    /// [`stack_trace`](Self::stack_trace)) start empty; set them directly
    /// when constructing a failed item that carries them.
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
            error_data: None,
            stack_trace: Vec::new(),
        }
    }

    /// Creates a failed item carrying the complete recorded wire failure
    /// record: all four wire fields survive into the item, so live and
    /// replayed batches expose the same failure identity.
    pub(crate) fn failed_from_wire(
        index: usize,
        name: String,
        wire: &crate::error::WireError,
    ) -> Self {
        Self {
            index,
            name,
            status: BatchItemStatus::Failed,
            result: None,
            error_message: wire.error_message().map(str::to_owned),
            error_type: wire.error_type().map(str::to_owned),
            error_data: wire.error_data().map(str::to_owned),
            stack_trace: wire.stack_trace().to_vec(),
        }
    }

    /// Rebuilds the item's recorded failure as a wire record (all four
    /// fields), for surfacing the failure as an operation error.
    pub(crate) fn failure_wire(&self) -> crate::error::WireError {
        crate::error::WireError::new(
            self.error_type.as_deref(),
            Some(
                self.error_message
                    .as_deref()
                    .unwrap_or("branch failed")
                    .to_owned(),
            ),
        )
        .with_error_data(self.error_data.as_deref())
        .with_stack_trace(self.stack_trace.clone())
    }
}

/// The overall status of a completed batch.
///
/// Returned by [`BatchResult::status`]. A custom completion decides the
/// status on its own: [`CompletionReason::CustomCompletionSucceeded`] gives
/// [`BatchStatus::Succeeded`] even when an item failed, and
/// [`CompletionReason::CustomCompletionFailed`] gives [`BatchStatus::Failed`]
/// even when no item failed. For every other reason the status is
/// [`BatchStatus::Failed`] exactly when a started item failed. The `Display`
/// implementation renders the wire strings `"SUCCEEDED"` and `"FAILED"`.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::BatchStatus;
///
/// assert_eq!(BatchStatus::Succeeded.as_str(), "SUCCEEDED");
/// assert_eq!(BatchStatus::Failed.to_string(), "FAILED");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BatchStatus {
    /// The predicate completed the batch as succeeded, or, for any other
    /// reason, no started item failed.
    Succeeded,
    /// The predicate completed the batch as failed, or, for any other
    /// reason, at least one started item failed.
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
/// use aws_durable_execution_sdk::builders::map_parallel::{BatchResult, BatchItem, BatchItemStatus, CompletionReason};
///
/// let result: BatchResult<i32> = BatchResult::new(
///     vec![BatchItem::new(
///         2,
///         "branch-c".to_owned(),
///         BatchItemStatus::Failed,
///         None,
///         Some("boom".into()),
///         Some("Error".into()),
///     )],
///     CompletionReason::AllCompleted,
/// );
/// let errors = result.errors();
/// assert_eq!(errors[0].index, 2);
/// assert_eq!(errors[0].name, "branch-c");
/// assert_eq!(errors[0].message, "boom");
/// assert_eq!(errors[0].error_type, Some("Error"));
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
    /// batch checkpoint payload: the recorded type of the error the item
    /// body returned (`"Error"` for an untyped error).
    /// `None` when the payload recorded no type (a payload written before
    /// error typing).
    pub error_type: Option<&'a str>,
}

/// Why the batch completed.
///
/// Records the reason a [`BatchResult`] finished: all items ran, the
/// success threshold was met, the failure tolerance was exceeded, or a
/// custom completion predicate completed the batch as a success or a
/// failure.
///
/// Earlier previews of this SDK wrote `PREDICATE_MATCHED` for a predicate
/// completion, a value JS and Python never write. This version does not
/// read that value: a small batch summary that recorded it replays as
/// `ALL_COMPLETED`, and the decision record of an oversized batch that
/// recorded it fails replay. So an execution that recorded it must finish,
/// or be restarted, before you upgrade.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::CompletionReason;
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
    /// The custom completion predicate returned
    /// `Complete(CompletionOutcome::Succeeded)`.
    ///
    /// The predicate is configured through
    /// [`crate::builders::map_parallel::CompletionConfig::with_completion_predicate`] or
    /// [`crate::builders::map_parallel::CompletionConfigBuilder::completion_predicate`]. Like
    /// [`CompletionReason::MinSuccessfulReached`], a batch completed this
    /// way is a successful early completion: item failures inside it are
    /// tolerated rather than propagated as errors.
    CustomCompletionSucceeded,
    /// The custom completion predicate returned
    /// `Complete(CompletionOutcome::Failed)`.
    ///
    /// [`BatchResult::status`] is [`BatchStatus::Failed`] even when no item
    /// failed, and awaiting the batch returns an error.
    CustomCompletionFailed,
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
            Self::CustomCompletionSucceeded => "CUSTOM_COMPLETION_SUCCEEDED",
            Self::CustomCompletionFailed => "CUSTOM_COMPLETION_FAILED",
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
            "CUSTOM_COMPLETION_SUCCEEDED" => Self::CustomCompletionSucceeded,
            "CUSTOM_COMPLETION_FAILED" => Self::CustomCompletionFailed,
            _ => Self::AllCompleted,
        }
    }

    /// Strictly parses the wire representation, returning `None` for a
    /// string this SDK never writes.
    ///
    /// The batch decision record uses this instead of [`Self::from_wire`]:
    /// a decision record is SDK-owned bookkeeping whose reason replay
    /// reports verbatim, so an unrecognized value there is corruption to
    /// reject, not a future variant to tolerate.
    pub(crate) fn try_from_wire(s: &str) -> Option<Self> {
        match s {
            "ALL_COMPLETED" => Some(Self::AllCompleted),
            "MIN_SUCCESSFUL_REACHED" => Some(Self::MinSuccessfulReached),
            "FAILURE_TOLERANCE_EXCEEDED" => Some(Self::FailureToleranceExceeded),
            "CUSTOM_COMPLETION_SUCCEEDED" => Some(Self::CustomCompletionSucceeded),
            "CUSTOM_COMPLETION_FAILED" => Some(Self::CustomCompletionFailed),
            _ => None,
        }
    }
}

/// What a completion predicate decides before an item starts or after an item settles.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::{
///     CompletionDecision, CompletionOutcome,
/// };
///
/// let decision = CompletionDecision::Complete(CompletionOutcome::Failed);
/// assert_ne!(decision, CompletionDecision::Continue);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionDecision {
    /// Keep starting and awaiting items.
    Continue,
    /// Complete the batch now with this outcome.
    Complete(CompletionOutcome),
}

/// Whether a custom completion counts as a success or a failure of the batch.
///
/// `Succeeded` records [`CompletionReason::CustomCompletionSucceeded`], and
/// `Failed` records [`CompletionReason::CustomCompletionFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionOutcome {
    /// The batch completed as a success.
    Succeeded,
    /// The batch completed as a failure.
    Failed,
}

/// One input item as a completion predicate sees it.
///
/// [`BatchStats::items`] holds one entry per input item, by index. The SDK
/// asks the predicate before each item starts and after each item settles,
/// so an entry can be not started, running, or settled. The live settle
/// order is not recorded, and on resume the recorded items settle first, in
/// index order; so a predicate must be pure and monotonic (see
/// [`CompletionConfigBuilder::completion_predicate`](crate::builders::map_parallel::CompletionConfigBuilder::completion_predicate)).
///
/// The public constructor exists so a completion predicate can be unit
/// tested against hand-built statistics without running a batch.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::{BatchItemStatus, CompletionItem};
///
/// let item = CompletionItem::new(3, Some("charge".to_owned()), Some(BatchItemStatus::Started));
/// assert_eq!(item.index, 3);
/// assert_eq!(item.name.as_deref(), Some("charge"));
/// assert_eq!(item.status, Some(BatchItemStatus::Started));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompletionItem {
    /// Zero-based position in the input.
    pub index: usize,
    /// The name the caller gave the item: the item namer's result, or the
    /// branch name. `None` when the caller gave no name. An empty string is
    /// `Some("")`. The recorded name is independent and falls back to
    /// `map-item-{index}` or `parallel-branch-{index}`.
    pub name: Option<String>,
    /// `None`: not started. `Some(Started)`: running or waiting.
    /// `Some(Succeeded)` or `Some(Failed)`: settled.
    pub status: Option<BatchItemStatus>,
}

impl CompletionItem {
    /// Builds an item (public so a predicate can be unit tested).
    #[must_use]
    pub fn new(index: usize, name: Option<String>, status: Option<BatchItemStatus>) -> Self {
        Self {
            index,
            name,
            status,
        }
    }
}

/// A snapshot of a map/parallel batch, passed to a custom completion
/// predicate.
///
/// The SDK builds a snapshot before each item starts and after each item
/// settles. The counts follow the settle order, and
/// [`items`](Self::items) holds every input item by index, with its name
/// and its status. The live settle order is not recorded, and on resume the
/// recorded items settle first, in index order. So the sequence of
/// snapshots can differ between runs, and a predicate must be a pure and
/// monotonic function of the snapshot: see
/// [`CompletionConfigBuilder::completion_predicate`](crate::builders::map_parallel::CompletionConfigBuilder::completion_predicate).
///
/// The public constructor exists so a completion predicate can be unit
/// tested against hand-built statistics without running a batch.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::{
///     BatchItemStatus, BatchStats, CompletionItem,
/// };
///
/// let items = [
///     CompletionItem::new(0, None, Some(BatchItemStatus::Failed)),
///     CompletionItem::new(1, None, Some(BatchItemStatus::Succeeded)),
///     CompletionItem::new(2, None, Some(BatchItemStatus::Started)),
///     CompletionItem::new(3, None, None),
/// ];
/// let stats = BatchStats::new(&items);
/// assert_eq!(stats.succeeded(), 1);
/// assert_eq!(stats.failed(), 1);
/// assert_eq!(stats.settled(), 2);
/// assert_eq!(stats.total_items(), 4);
/// assert_eq!(stats.items()[2].status, Some(BatchItemStatus::Started));
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct BatchStats<'a> {
    /// Count of items that have succeeded so far.
    succeeded: usize,
    /// Count of items that have failed so far.
    failed: usize,
    /// Every input item, by index.
    items: &'a [CompletionItem],
}

impl<'a> BatchStats<'a> {
    /// Builds a snapshot, counting the settled items (public for unit tests).
    #[must_use]
    pub fn new(items: &'a [CompletionItem]) -> Self {
        let count = |status| {
            items
                .iter()
                .filter(|item| item.status == Some(status))
                .count()
        };
        Self {
            succeeded: count(BatchItemStatus::Succeeded),
            failed: count(BatchItemStatus::Failed),
            items,
        }
    }

    /// Builds a snapshot from the coordinator's running counts.
    pub(crate) fn from_counts(
        succeeded: usize,
        failed: usize,
        items: &'a [CompletionItem],
    ) -> Self {
        Self {
            succeeded,
            failed,
            items,
        }
    }

    /// Items that succeeded so far.
    pub fn succeeded(&self) -> usize {
        self.succeeded
    }

    /// Items that failed so far.
    pub fn failed(&self) -> usize {
        self.failed
    }

    /// `succeeded() + failed()`.
    pub fn settled(&self) -> usize {
        self.succeeded + self.failed
    }

    /// `items().len()`, the number of input items.
    pub fn total_items(&self) -> usize {
        self.items.len()
    }

    /// Every input item, by index: `items()[i].index == i`.
    pub fn items(&self) -> &'a [CompletionItem] {
        self.items
    }
}

/// The collected outcome of a map/parallel operation.
///
/// Contains per-item outcomes and the reason the batch completed. Provides
/// accessor methods for building conformance projections.
///
/// Only items that were actually started are included: items that never
/// started due to early completion are omitted. An item that was still
/// running or parked when the completion rule ended the batch appears as
/// [`BatchItemStatus::Started`], live and on replay.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::{
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
///             Some("Error".into()),
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
    /// Per-item outcomes in input order (only started items, including
    /// [`BatchItemStatus::Started`] ones).
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
    /// Failed items, `Started` items, and items that never started are
    /// omitted.
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
    pub fn success_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == BatchItemStatus::Succeeded)
            .count()
    }

    /// Returns the count of failures.
    pub fn failure_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == BatchItemStatus::Failed)
            .count()
    }

    /// Reports whether any item failed.
    pub fn has_failure(&self) -> bool {
        self.items
            .iter()
            .any(|i| i.status == BatchItemStatus::Failed)
    }

    /// Returns the batch status.
    ///
    /// [`CompletionReason::CustomCompletionFailed`] gives
    /// [`BatchStatus::Failed`], and
    /// [`CompletionReason::CustomCompletionSucceeded`] gives
    /// [`BatchStatus::Succeeded`], whatever the items did. Any other reason
    /// gives [`BatchStatus::Failed`] exactly when an item failed.
    pub fn status(&self) -> BatchStatus {
        match self.reason {
            CompletionReason::CustomCompletionFailed => BatchStatus::Failed,
            CompletionReason::CustomCompletionSucceeded => BatchStatus::Succeeded,
            _ if self.has_failure() => BatchStatus::Failed,
            _ => BatchStatus::Succeeded,
        }
    }

    /// Returns the total number of items that were started.
    ///
    /// The count includes [`BatchItemStatus::Started`] items, which can
    /// appear live when the completion rule ends the batch early.
    pub fn total_count(&self) -> usize {
        self.items.len()
    }

    /// Returns the errors from failed items, in input order.
    ///
    /// Each [`BatchError`] carries the failed item's index, display name,
    /// error type identifier, and error message, so a caller can tell
    /// which input produced which failure and what kind of error it was.
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
/// use aws_durable_execution_sdk::builders::map_parallel::NestingMode;
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
    /// Items run in a virtual context under the parent, no per-item
    /// context events are emitted.
    Flat,
}

/// Internal state for a map execution passed from the builder.
///
/// Holds the user's map closure as `Arc<F>`: shared rather than boxed,
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
    /// [`ItemDispatch`] object, so [`execute_batch`], the checkpoint state
    /// machine, compiles once per result type while every item still runs
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
        // Mirrors map through the shared [`collect_successful`] projection:
        // failed branches within tolerance are skipped; the batch errors
        // when the failure tolerance was exceeded or the predicate
        // completed it as failed.
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
/// on a success trigger (`MinSuccessfulReached` or
/// `CustomCompletionSucceeded`), failed items are expected and NOT
/// propagated as errors: they are simply skipped. A batch that ended because
/// the failure tolerance was exceeded (including the default fail-fast case)
/// or because the predicate completed it as failed (`CustomCompletionFailed`)
/// becomes an `Err`: a `ChildFailed`-kinded error carrying the first failed
/// item's recorded wire record (`error_type`, message). A
/// `CustomCompletionFailed` batch with no failed item becomes a
/// `BatchCompletionFailed` error instead.
fn collect_successful<O>(batch_result: BatchResult<O>) -> Result<Vec<O>, OperationError> {
    let fails_batch = matches!(
        batch_result.reason,
        CompletionReason::FailureToleranceExceeded | CompletionReason::CustomCompletionFailed
    );
    let mut results = Vec::with_capacity(batch_result.items.len());
    for item in batch_result.items {
        match item.status {
            BatchItemStatus::Succeeded => {
                if let Some(value) = item.result {
                    results.push(value);
                }
            }
            BatchItemStatus::Failed => {
                if fails_batch {
                    // A user item body failed: classify as ChildFailed
                    // carrying the item's COMPLETE recorded wire record
                    // (type, message, `error_data`, `stack_trace`),
                    // reserving `Internal` for coordinator defects.
                    return Err(item_failure_error(item.failure_wire()));
                }
                // Within tolerance: skip this item in the Vec<O> output.
            }
            // Started-but-not-terminal at batch completion (live, or
            // replayed from the decision record): carries no value, so it is
            // skipped in the Vec<O> success view exactly like a tolerated
            // failure.
            BatchItemStatus::Started => {}
        }
    }
    // Check if the batch itself failed with no failed item.
    match batch_result.reason {
        CompletionReason::FailureToleranceExceeded => {
            Err(batch_error("failure tolerance exceeded"))
        }
        CompletionReason::CustomCompletionFailed => Err(batch_completion_failed_error()),
        _ => Ok(results),
    }
}

/// A pre-claimed child operation with its metadata.
struct PreClaimed {
    index: usize,
    op_id: OperationId,
    is_terminal: bool,
    /// Whether the checkpoint log holds ANY record for this child: a
    /// terminal record or a `Started` one from a prior invocation. Drives
    /// two decisions at dispatch: a child with a record never gets a second
    /// `Start` write, and a recorded child counts as "ever started" in the
    /// batch decision record.
    has_record: bool,
}

/// The payload a branch task delivers through the coordinator's join set:
/// the item index plus the item's terminal outcome. A branch that waits on
/// a durable operation stays pending in the join set and keeps its
/// concurrency slot until it finishes.
type ItemJoin<O> = (usize, Result<BatchItem<O>, OperationError>);

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
    /// The body must run; the child context is ready.
    Run {
        /// The child context the item body receives.
        child_ctx: DurableContext,
    },
}

/// Pre-closure half of one batch item: replay decode, child START
/// checkpoint, child-context creation. Generic only over the result type
/// `O`, no user closure reaches this code.
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
        let child_ctx = req.ctx.new_flat_child(&child_positional, &req.parent_wire);
        return Ok(ItemPrelude::Run { child_ctx });
    }

    // Create the child context. The child's `Start` was already
    // checkpointed by the coordinator at dispatch (the coordinator is the
    // sole writer of child starts; see `execute_batch`), or exists from a
    // prior invocation.
    Ok(ItemPrelude::Run {
        child_ctx: req.ctx.new_child(&child_positional),
    })
}

/// Post-closure half of one batch item: outcome checkpointing and
/// [`BatchItem`] assembly. Generic only over the result type `O`, no user
/// closure reaches this code.
#[expect(clippy::too_many_lines)] // reason: FLAT/NORMAL outcome checkpointing reads better in one flow
async fn item_after<O, IS>(
    req: &ItemRequest<IS>,
    outcome: Result<O, ChildFnError>,
) -> Result<BatchItem<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let child_wire = req.child_op_id.wire().to_owned();
    let serdes_ctx = SerdesContext::new(child_wire.clone(), req.ctx.execution_arn());
    let serdes = &req.serdes;
    // The item's own terminal update is written on the item's behalf: once
    // the batch abandoned the item at early completion, the write is
    // rejected, as every write of the item's body is.
    let writer = req.ctx.with_owner(req.child_op_id.positional());

    // FLAT nesting emits no child-context events: only the value transform
    // (round-trip for live == replay consistency) happens here.
    if req.nesting == NestingMode::Flat {
        return match outcome {
            Ok(value) => {
                // A result serialization failure is LOCAL and
                // deterministic: the item closure already ran, so the item
                // settles as a failed `BatchItem` instead of yielding a
                // catchable coordinator error with no record (issue #43).
                // FLAT items have no per-child record: the failure is
                // recorded inside the parent batch's summary payload when
                // the parent checkpoints, and replay reconstructs the same
                // failed item from it, so live and replayed batches agree.
                let round_trip = async {
                    let name = item_fault_name(&req.item_name);
                    let serialized =
                        serialize_value(&req.ctx, name, value, serdes, serdes_ctx.clone()).await?;
                    let deserialized: O =
                        deserialize_value(&req.ctx, name, serialized, serdes, serdes_ctx).await?;
                    Ok::<_, OperationError>(deserialized)
                };
                match round_trip.await {
                    Ok(deserialized) => Ok(BatchItem {
                        index: req.index,
                        name: req.item_name.clone(),
                        status: BatchItemStatus::Succeeded,
                        result: Some(deserialized),
                        error_message: None,
                        error_type: None,
                        error_data: None,
                        stack_trace: Vec::new(),
                    }),
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        Ok(BatchItem::failed_from_wire(
                            req.index,
                            req.item_name.clone(),
                            &wire,
                        ))
                    }
                }
            }
            Err(child_err) => {
                // The wire record is derived from the error the body
                // returned: its recorded type, and its flattened chain as
                // the message.
                let wire = child_err.failure_wire();
                Ok(BatchItem::failed_from_wire(
                    req.index,
                    req.item_name.clone(),
                    &wire,
                ))
            }
        };
    }

    match outcome {
        Ok(value) => {
            // Serialize and checkpoint success.
            //
            // A result serialization failure is LOCAL and deterministic:
            // the item closure already ran, so a terminal FAIL is
            // persisted for the child FIRST, and the item settles as a
            // failed `BatchItem` (issue #43). That is exactly the shape a
            // replay of the FAIL record produces (see
            // [`replay_terminal_child`]), so live and replayed batches
            // agree, and the closure never re-runs for this failure.
            let serialized = match serialize_value(
                &req.ctx,
                item_fault_name(&req.item_name),
                value,
                serdes,
                serdes_ctx.clone(),
            )
            .await
            {
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
                    if let Err(client_err) = writer.checkpoint_updates(vec![update]).await {
                        // Audit (#43): batch child FAIL (serialization):
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
                        return writer
                            .checkpoint_failure_unrecoverable(
                                &child_wire,
                                client_err,
                                Some(terminal),
                            )
                            .await;
                    }
                    return Ok(BatchItem::failed_from_wire(
                        req.index,
                        req.item_name.clone(),
                        &wire,
                    ));
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

            #[expect(clippy::expect_used)] // reason: all required fields are set above
            let update = builder
                .build()
                .expect("all required OperationUpdate fields set");

            if let Err(err) = writer.checkpoint_updates(vec![update]).await {
                // Audit (#43): batch child SUCCEED: the item closure
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
                return writer
                    .checkpoint_failure_unrecoverable(&child_wire, err, Some(terminal))
                    .await;
            }

            // Round-trip deserialize for live == replay consistency.
            let deserialized: O = deserialize_value(
                &req.ctx,
                item_fault_name(&req.item_name),
                serialized,
                serdes,
                serdes_ctx,
            )
            .await?;

            Ok(BatchItem {
                index: req.index,
                name: req.item_name.clone(),
                status: BatchItemStatus::Succeeded,
                result: Some(deserialized),
                error_message: None,
                error_type: None,
                error_data: None,
                stack_trace: Vec::new(),
            })
        }
        Err(child_err) => {
            // Checkpoint failure. The wire record is derived from the
            // error the body returned, so the type is the body error's
            // recorded type, the message is its flattened chain, and
            // `error_data`/`stack_trace` pass through the boundary.
            let wire = child_err.failure_wire();
            let builder = OperationUpdate::builder()
                .id(child_wire.clone())
                .r#type(OperationType::Context)
                .sub_type(req.child_sub_type.clone())
                .action(OperationAction::Fail)
                .parent_id(req.parent_wire.clone())
                .error(wire.to_error_object());

            #[expect(clippy::expect_used)] // reason: all required fields are set above
            let update = builder
                .build()
                .expect("all required OperationUpdate fields set");

            if let Err(client_err) = writer.checkpoint_updates(vec![update]).await {
                // Audit (#43): batch child FAIL: the item closure ran
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
                return writer
                    .checkpoint_failure_unrecoverable(&child_wire, client_err, Some(terminal))
                    .await;
            }

            Ok(BatchItem::failed_from_wire(
                req.index,
                req.item_name.clone(),
                &wire,
            ))
        }
    }
}

/// Thin generic wrapper around one map item: the ONLY code monomorphized
/// per map call site. Everything before and after the user closure is the
/// non-generic [`item_before`] / [`item_after`] pair; this wrapper just
/// polls the user's concrete future between them under the branch driver.
async fn run_single_item<O, F, Fut, IS>(
    req: ItemRequest<IS>,
    run_item: Arc<F>,
) -> Result<BatchItem<O>, OperationError>
where
    O: Send + 'static,
    F: Fn(DurableContext, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ChildFnError>> + Send + 'static,
    IS: Serdes<O>,
{
    match item_before(&req).await? {
        ItemPrelude::Done(item) => Ok(item),
        ItemPrelude::Run { child_ctx } => {
            // Instrument the branch body with the branch namespace's
            // replay-aware span: a resumed branch's pre-wait log lines are
            // suppressed while its own operations replay, independently of
            // the root handler span and of sibling branches.
            let span = child_ctx.replay_span();
            let index = req.index;
            // The active-body scope encloses the closure call, so code the
            // closure runs before it returns its future is checked too. A
            // wait inside the body keeps this task pending in the join set.
            let body = ActiveBody::Context(child_ctx.prefix().to_owned());
            let outcome = run_body(body, async move { run_item(child_ctx, index).await })
                .instrument(span)
                .await;
            item_after(&req, outcome).await
        }
    }
}

/// Non-generic branch runner for `parallel`: the branch body is already the
/// single erased future carried by [`crate::future::BranchBody`], so
/// nothing here monomorphizes per user call site.
async fn run_branch_item<O, IS>(
    req: ItemRequest<IS>,
    slots: Arc<Vec<std::sync::Mutex<Option<crate::future::BranchBody<O>>>>>,
) -> Result<BatchItem<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    match item_before(&req).await? {
        ItemPrelude::Done(item) => Ok(item),
        ItemPrelude::Run { child_ctx } => {
            let span = child_ctx.replay_span();
            let outcome = match take_branch_body(&slots, req.index) {
                Ok(body) => {
                    // `start` injects the child context and returns the
                    // branch's single erased future, awaited here exactly
                    // like a map item's body. The factory closure runs
                    // inside the erased future, so wrapping the future
                    // encloses the closure call.
                    let active = ActiveBody::Context(child_ctx.prefix().to_owned());
                    run_body(active, body.start(child_ctx))
                        .instrument(span)
                        .await
                }
                Err(err) => Err(err),
            };
            item_after(&req, outcome).await
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
/// [`TrackedJoinSet`] as a CONCRETE (unboxed) future.
///
/// This boundary is what keeps [`execute_batch`], the batch checkpoint
/// state machine, non-generic over the user's closure: only the dispatcher
/// and the thin wrapper it spawns ([`run_single_item`]) monomorphize per
/// call site, while the futures inside the join set stay unboxed.
trait ItemDispatch<O, IS>: Send + Sync {
    /// Spawns the item body for `req` and returns the task's ID, or `None`
    /// when the coordinator refused the spawn. The join set blesses the
    /// task before the body's first poll.
    fn spawn_item(
        &self,
        set: &mut TrackedJoinSet<ItemJoin<O>>,
        req: ItemRequest<IS>,
    ) -> Result<Option<tokio::task::Id>, NoRuntime>;
}

/// Map dispatcher: shares the user closure as `Arc<F>` and produces one
/// concrete future per item: the join set holds no per-item box.
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
        set: &mut TrackedJoinSet<ItemJoin<O>>,
        req: ItemRequest<IS>,
    ) -> Result<Option<tokio::task::Id>, NoRuntime> {
        let run_item = Arc::clone(&self.run_item);
        let ownership = Arc::clone(req.ctx.task_ownership());
        set.spawn(&ownership, async move {
            let index = req.index;
            (index, run_single_item(req, run_item).await)
        })
    }
}

/// Parallel dispatcher: hands each spawned task the shared take-once slots
/// holding the single-erasure branch bodies. Non-generic: the erased
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
        set: &mut TrackedJoinSet<ItemJoin<O>>,
        req: ItemRequest<IS>,
    ) -> Result<Option<tokio::task::Id>, NoRuntime> {
        let slots = Arc::clone(&self.slots);
        let ownership = Arc::clone(req.ctx.task_ownership());
        set.spawn(&ownership, async move {
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
    /// Which items were EVER admitted: dispatched live this invocation, or
    /// holding a checkpoint record from a prior one. Ordered admission makes
    /// this a contiguous prefix of the input, and its length is the
    /// `totalCount` the batch decision record carries: the number of items
    /// ever started, NOT the input length.
    started: Vec<bool>,
}

impl<O> BatchProgress<O> {
    /// Builds the bookkeeping for one batch. `names` holds the caller name
    /// of each input item (see [`CompletionItem::name`]); its length is the
    /// input length.
    fn new(names: Vec<Option<String>>) -> Self {
        let total_items = names.len();
        Self {
            tracker: CompletionTracker::new(names),
            results: (0..total_items).map(|_| None).collect(),
            started: vec![false; total_items],
        }
    }

    /// Marks one item as ever admitted (see [`Self::started`]).
    fn mark_started(&mut self, index: usize) {
        if let Some(slot) = self.started.get_mut(index) {
            *slot = true;
        }
    }

    /// Records a live dispatch of one item, for the decision record and for
    /// the completion predicate's view.
    fn mark_dispatched(&mut self, index: usize) {
        self.mark_started(index);
        self.tracker.mark_dispatched(index);
    }

    /// Records one terminal item outcome: it feeds the completion
    /// statistics and its result slot.
    fn settle_outcome(
        &mut self,
        completion_cfg: &crate::builders::map_parallel::CompletionConfig,
        total_items: usize,
        index: usize,
        item: BatchItem<O>,
    ) {
        self.tracker
            .settle(completion_cfg, total_items, index, item.status);
        if let Some(slot) = self.results.get_mut(index) {
            *slot = Some(item);
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
    /// Processes one joined branch task: its outcome feeds the progress
    /// accounting, and a coordinator-level error (e.g. a failed checkpoint
    /// call) is returned. A panic of the task resumes unwinding here, with
    /// its original payload, and nothing is recorded for it: a panic fails
    /// the invocation, not the item.
    ///
    /// Returns the index of the item.
    fn settle_joined(
        &self,
        joined: TaskOutcome<ItemJoin<O>>,
        progress: &mut BatchProgress<O>,
    ) -> Result<usize, OperationError> {
        let (index, outcome) = joined.into_value();
        // A coordinator-level failure surfaces; the caller's join set drop
        // aborts the remaining branches.
        let item = outcome?;
        progress.settle_outcome(self.completion_cfg, self.total_items, index, item);
        Ok(index)
    }

    /// Resolves a recorded-terminal child inline on the coordinator, in
    /// input order. The common case decodes the recorded outcome without
    /// touching the item body; a `ReplayChildren` child (result too large to
    /// checkpoint) is re-executed through the dispatcher and drained
    /// immediately, so input order is preserved and the join set still
    /// holds only concrete futures.
    ///
    /// A panic during that re-execution resumes unwinding (through
    /// [`Self::settle_joined`]) and checkpoints nothing, as every panic
    /// does: the child's durable record stays a terminal SUCCESS for the
    /// next attempt to reconstruct.
    async fn resolve_terminal_inline(
        &self,
        op_id: &OperationId,
        index: usize,
        item_name: &str,
        join_set: &mut TrackedJoinSet<ItemJoin<O>>,
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
                progress.settle_outcome(self.completion_cfg, self.total_items, index, item);
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
                    parent_wire: self.parent_wire.to_owned(),
                    child_sub_type: self.child_sub_type.to_owned(),
                    item_name: item_name.to_owned(),
                    nesting: self.nesting,
                    serdes: Arc::clone(self.serdes),
                };
                let reconstruction_id = self
                    .dispatch
                    .spawn_item(join_set, req)
                    .map_err(NoRuntime::operation_error)?;
                let Some((joined_id, joined)) = join_set.join_next_with_id().await else {
                    return Err(batch_error(
                        "re-executed replay child produced no join event",
                    ));
                };
                // Verify (rather than assume) the quiescence invariant: the
                // join event must belong to the reconstruction task spawned
                // above.
                if Some(joined_id) != reconstruction_id {
                    return Err(batch_error(
                        "re-executed replay child joined an unexpected task",
                    ));
                }
                self.settle_joined(joined, progress).map(drop)
            }
        }
    }
}

/// Core batch execution: schedule items with bounded concurrency and
/// completion checking.
///
/// Non-generic over the user closure: item bodies enter only through the
/// [`ItemDispatch`] object, so this coordinator, the batch's checkpoint
/// state machine, monomorphizes once per result type `O`.
#[expect(clippy::too_many_lines)]
// reason: batch coordination has distinct phases (claim, schedule, collect, checkpoint) that read better as one flow
#[expect(clippy::too_many_arguments)] // reason: batch execution requires all these parameters
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
    // Share the item serdes across items behind one `Arc`: the forwarding
    // `impl Serdes for Arc<S>` makes the handle itself a serdes, so no
    // `Clone` bound is required of the user's implementation.
    let serdes = Arc::new(serdes);

    // 1. Task-ownership check.
    ctx.enforce_task_ownership()?;

    // 1b. Validate max_concurrency BEFORE any checkpointing.
    // Zero is invalid: must be positive or unset (None = unlimited).
    if max_concurrency == Some(0) {
        return Err(batch_error(
            "max concurrency must be positive or unset for unlimited",
        ));
    }

    // The caller name of each input item, computed once, before any replay
    // or dispatch, and reused everywhere: the predicate view, replay
    // identity validation, the child checkpoints and the result. So the
    // namer runs exactly once per input item, and every consumer sees the
    // same name. `parallel` passes its branch names as the namer, so every
    // branch has `Some(name)`; a map item has `Some(..)` only when a namer
    // is set.
    let names: Vec<Option<String>> = (0..total_items)
        .map(|i| item_namer.as_ref().map(|namer| namer(i)))
        .collect();
    drop(item_namer);
    // The recorded item name: the caller name, or empty when none was given.
    let item_name_of = |i: usize| -> String { names.get(i).cloned().flatten().unwrap_or_default() };

    let parent_positional = parent_op_id.positional().to_owned();
    let parent_wire = parent_op_id.wire().to_owned();

    // 2. Check if the parent batch is already terminal in the checkpoint log.
    // Identity validation and the status read happen in one read-guard pass;
    // the terminal payload/error projection is cloned only when the batch is
    // actually terminal and must be replayed.
    // When the parent is terminal with `replay_children` and its payload
    // parses as a batch decision record, replay OBEYS the record instead of
    // re-deriving the outcome. `Some` constrains the re-execution below to
    // the recorded admission set and completion reason.
    let mut replay_plan: Option<ReplayPlan> = None;

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

            // A successful batch too large to checkpoint inline recorded a
            // decision record (`totalCount`, `completionReason`, index set)
            // instead of its payload. A record-less `replay_children`
            // payload, written by an older SDK, falls through to the
            // unconstrained re-execution, exactly as before; a non-empty
            // payload that fails to parse is corrupt and fails the replay
            // instead of falling back.
            let decision = if matches!(snapshot.status, CheckpointStatus::Succeeded)
                && snapshot.replay_children
            {
                parse_decision_record(snapshot.result.as_deref())?
            } else {
                None
            };

            if let Some(record) = decision {
                let plan = ReplayPlan::from_record(&record, total_items)?;
                let concurrency = max_concurrency.unwrap_or(total_items).max(1);
                // Fast path (Normal nesting): decode every completed item
                // straight from its child record (after validating each
                // child's replay identity), report items in the started
                // set as STARTED without consulting their child
                // checkpoints, and omit every index at or beyond
                // `totalCount`. Flat items have no
                // child records, so Flat always reconstructs through the
                // plan-constrained re-execution below.
                if nesting != NestingMode::Flat {
                    match replay_batch_from_record(&ctx, &plan, &names, &serdes, child_sub_type)
                        .await
                    {
                        Ok(result) => {
                            // Recorded terminal batch returned without
                            // re-running the batch.
                            ctx.emit_operation_replayed(
                                &parent_wire,
                                parent_name.as_deref(),
                                "Context",
                                Some(parent_sub_type),
                                view.attempt,
                            );
                            // Counter parity with the live run: concurrent
                            // pre-claims an ID per input item; sequential
                            // minted one per ADMITTED item, and ordered
                            // admission makes that exactly `totalCount`.
                            if concurrency == 1 {
                                ctx.advance_counter(plan.total_count);
                            } else {
                                ctx.advance_counter(total_items);
                            }
                            return Ok(result);
                        }
                        Err(e) => {
                            let is_replay_children =
                                crate::error::chain_string(&e).contains(REPLAY_CHILDREN_SENTINEL);
                            if !is_replay_children {
                                return Err(e);
                            }
                            // A completed CHILD whose own result was too
                            // large to checkpoint inline: reconstruct it by
                            // re-execution, still constrained by the plan.
                        }
                    }
                }
                replay_plan = Some(plan);
                // Fall through: re-execute ONLY the recorded admission set.
            } else {
                match replay_terminal_batch(
                    &ctx,
                    parent_name.as_deref(),
                    &snapshot,
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
                        // Advance the parent counter past the iteration IDs
                        // that were consumed during the original execution.
                        // Sequential (concurrency=1) only claims started
                        // items; concurrent claims all items upfront.
                        let concurrency = max_concurrency.unwrap_or(total_items).max(1);
                        if concurrency == 1 {
                            ctx.advance_counter(result.items.len());
                        } else {
                            ctx.advance_counter(total_items);
                        }
                        return Ok(result);
                    }
                    Err(e) => {
                        // ReplayChildren sentinel: fall through to
                        // re-execution. The children's operations are still
                        // in the checkpoint log, so re-executing the batch
                        // will replay each child from its terminal record.
                        let is_replay_children =
                            crate::error::chain_string(&e).contains(REPLAY_CHILDREN_SENTINEL);
                        if !is_replay_children {
                            // A recorded batch FAILURE replays as this
                            // error; an internal replay problem
                            // (missing/corrupt payload) is not a replayed
                            // outcome and emits nothing.
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
                        // Fall through: the batch parent is terminal but we
                        // need to re-execute children to reconstruct the
                        // result.
                    }
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
            // Audit (#43): batch parent START: no item closure has run,
            // so no terminal FAIL is needed; re-invocation reconverges
            // on the same write.
            return ctx
                .checkpoint_failure_unrecoverable(&parent_wire, err, None)
                .await;
        }
    }

    let completion_cfg = completion.unwrap_or_default();

    // 4. Empty collection: checkpoint success immediately. As in JS, a
    // zero-item batch asks the predicate once, on an empty view; without a
    // predicate the decision is `Continue`, which records `AllCompleted`.
    if total_items == 0 {
        let reason =
            custom_reason(completion_cfg.predicate_decision(&BatchStats::from_counts(0, 0, &[])))
                .unwrap_or(CompletionReason::AllCompleted);
        let result = BatchResult {
            items: Vec::new(),
            reason,
        };
        let (payload, result) =
            from_batch_result(&ctx, result, &serdes, &parent_wire, ctx.execution_arn()).await?;
        let serialized_payload = serialize_value(
            &ctx,
            parent_name.as_deref(),
            payload,
            &result_serdes,
            SerdesContext::new(&parent_wire, ctx.execution_arn()),
        )
        .await?;
        let decision_json =
            serialize_decision_record(&build_decision_record(&result.items, 0, result.reason))?;
        checkpoint_batch_success_serialized(
            &ctx,
            &parent_wire,
            parent_name.as_deref(),
            parent_sub_type,
            &serialized_payload,
            &decision_json,
        )
        .await?;
        return Ok(result);
    }

    // 5. Determine effective concurrency (needed before pre-claiming).
    let concurrency = max_concurrency.unwrap_or(total_items).max(1);

    // Validate completion config.
    if let Err(msg) = completion_cfg.validate() {
        return Err(batch_error(&msg));
    }

    // 6. Pre-claim child IDs.
    // For concurrency > 1 (concurrent path): claim ALL child IDs upfront
    // on the owning task (determinism rule 4: deterministic ID ordering
    // regardless of completion order). Claiming is LOCAL, no `Start` is
    // checkpointed here; the coordinator writes each child's `Start` at
    // dispatch, so a child is admitted (and recorded) only when a
    // concurrency slot is actually free for it.
    // For concurrency == 1 (sequential path): mint IDs lazily, one at a
    // time inside the loop, so only STARTED items consume IDs
    // and the service doesn't see dangling operations for never-started
    // items on early termination.
    let mut pre_claimed: Vec<PreClaimed> = Vec::with_capacity(total_items);
    if concurrency > 1 {
        for i in 0..total_items {
            let child_op_id = ctx.mint_id();
            // Under a replay plan, only indexes the record admits as
            // COMPLETED have their child checkpoints consulted. Started-set
            // entries are reconstructed from the record alone and
            // never-started entries are omitted, so reading (and
            // identity-validating) their views here would violate the
            // record's contract and could raise a spurious
            // non-determinism error for a record replay never touches.
            // The ID is still minted above for counter parity with the
            // live run, which claims one per input item.
            let consult_checkpoint = replay_plan.as_ref().is_none_or(|plan| plan.admits(i));
            let (has_record, is_terminal) = if consult_checkpoint {
                let child_positional = child_op_id.positional().to_owned();
                let child_wire = child_op_id.wire().to_owned();
                let child_name = names.get(i).cloned().flatten();
                // Non-determinism detection on child items happens inside
                // the validated view fetch; only the status is consumed
                // here.
                let view = ctx.checkpoint_view_validated(
                    &child_positional,
                    &child_wire,
                    "Context",
                    Some(child_sub_type),
                    child_name.as_deref(),
                )?;
                (view.is_some(), view.is_some_and(|v| v.status.is_terminal()))
            } else {
                (false, false)
            };
            pre_claimed.push(PreClaimed {
                index: i,
                op_id: child_op_id,
                is_terminal,
                has_record,
            });
        }
    }

    // 7. Execute items with bounded concurrency, branch-local suspension, and
    // slot-holding accounting. Admission is ordered and LAZY: the
    // coordinator, the sole writer of child starts, checkpoints each
    // child's `Start` at the moment it dispatches that child, in index
    // order, as concurrency slots free up. No child `Start` is written
    // before its item is about to run, so an early completion trigger or a
    // failure part-way leaves no starts recorded for children that never
    // ran.

    // Coordinator loop. In-flight items are bounded by `concurrency`. An
    // item that waits on a durable operation stays pending in the join set
    // and counts in `running` until its task ends, so it KEEPS its slot
    // (only terminal completion frees one: the slot-holding invariant). It
    // continues when its wait completes; when nothing else can make
    // progress, the invocation coordinator suspends the invocation.
    let child_sub_type_owned = child_sub_type.to_owned();
    // Coordinator-owned task set. Dropping it (on error return OR when the
    // driver drops this coordinator on invocation teardown) aborts every
    // still-running branch task, and the invocation driver waits for each
    // aborted task to unwind, so no branch task outlives the invocation.
    // Every task delivers its outcome through the set, including a panic,
    // which `settle_joined` resumes.
    let mut join_set: TrackedJoinSet<ItemJoin<O>> =
        TrackedJoinSet::new(Arc::clone(ctx.coordinator()));

    // The completion predicate sees each caller name as
    // `CompletionItem::name`.
    let mut progress = BatchProgress::<O>::new(names.clone());
    // A batch rebuilt under its decision record already completed. The
    // record holds its reason, so the rebuild evaluates no completion rule.
    if replay_plan.is_some() {
        progress.tracker.stop_evaluating();
    }
    let mut running: usize = 0;
    // The owner of each dispatched item that is not terminal, running or
    // parked: its index and its child wire ID. At early completion these
    // owners are abandoned (see `Admission`).
    let mut running_owners: BTreeMap<usize, String> = BTreeMap::new();
    let mut next_index: usize = 0;

    // A child holding ANY checkpoint record was admitted by a prior
    // invocation: it counts toward the "ever started" prefix the decision
    // record carries even if a completion trigger keeps this invocation
    // from re-dispatching it.
    for pre in &pre_claimed {
        if pre.has_record {
            progress.mark_started(pre.index);
        }
    }

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
    // any live work, never through the live `JoinSet` schedule. This is
    // what keeps completion triggers deterministic under replay: outcomes
    // already in the checkpoint log feed the statistics in canonical input
    // order before any live settlement joins, so identical recorded state
    // yields identical trigger decisions no matter how the scheduler orders
    // the join events of resumed live branches. Recorded terminals are
    // completed history, real work whose outcome the service persisted,
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
            // Under a replay plan only the recorded admission set
            // contributes to the result (a terminal record outside it
            // cannot exist for a well-formed record; skipping is
            // defensive).
            if let Some(plan) = &replay_plan
                && !plan.admits(i)
            {
                continue;
            }
            let item_name = item_name_of(i);
            env.resolve_terminal_inline(&pre.op_id, i, &item_name, &mut join_set, &mut progress)
                .await?;
        }
    }

    loop {
        // Dispatch while capacity remains and not-started eligible work exists.
        // A completion trigger (`tracker.stop_reason`) halts new dispatch,
        // and the join below abandons the running items. Under a replay
        // plan the trigger check
        // is bypassed: the recorded admission set IS the outcome being
        // reproduced, so every admitted item is dispatched regardless of
        // what the thresholds would re-derive, and the recorded reason is
        // taken verbatim below.
        while (replay_plan.is_some() || progress.tracker.stop_reason.is_none())
            && next_index < total_items
            && running < concurrency
        {
            let i = next_index;
            next_index += 1;

            // Under a replay plan, everything outside the recorded
            // admission set is resolved from the record alone: an index at
            // or beyond `totalCount` never started and is omitted; an
            // index in the started set was non-terminal when the batch
            // completed and is reported as STARTED: with no result and no
            // error, its child checkpoints never consulted. A started-set
            // item still consumed an ID on the live sequential path, so
            // its counter position is skipped rather than reused.
            if let Some(plan) = &replay_plan
                && !plan.admits(i)
            {
                if i < plan.total_count {
                    let item_name = item_name_of(i);
                    if let Some(slot) = progress.results.get_mut(i) {
                        *slot = Some(BatchItem {
                            index: i,
                            name: item_name,
                            status: BatchItemStatus::Started,
                            result: None,
                            error_message: None,
                            error_type: None,
                            error_data: None,
                            stack_trace: Vec::new(),
                        });
                    }
                    if concurrency == 1 {
                        ctx.advance_counter(1);
                    }
                }
                continue;
            }

            // Concurrent mode: recorded-terminal children were already
            // applied by the replay pass above: skip them here.
            if concurrency > 1 && pre_claimed.get(i).is_some_and(|pre| pre.is_terminal) {
                continue;
            }

            // Determine the PreClaimed item: for concurrent mode use the
            // pre-claimed vector, which holds only non-terminal items here;
            // for sequential mode, read the record of the ID the item WOULD
            // get, without minting it yet. The sequential path mints lazily
            // (only started items consume IDs), so it must know whether the
            // item is recorded-terminal before the dispatch gate runs.
            let pc = if concurrency > 1 {
                let pre = pre_claimed
                    .get(i)
                    .ok_or_else(|| batch_error("pre-claimed index out of range"))?;
                PreClaimed {
                    index: pre.index,
                    op_id: pre.op_id.clone(),
                    is_terminal: pre.is_terminal,
                    has_record: pre.has_record,
                }
            } else {
                let child_op_id = ctx.peek_id_at(0);
                let child_name = names.get(i).cloned().flatten();
                // Non-determinism detection on child items happens inside
                // the validated view fetch; only the status is consumed.
                let view = ctx.checkpoint_view_validated(
                    child_op_id.positional(),
                    child_op_id.wire(),
                    "Context",
                    Some(child_sub_type),
                    child_name.as_deref(),
                )?;
                let has_record = view.is_some();
                let is_terminal = view.is_some_and(|v| v.status.is_terminal());
                PreClaimed {
                    index: i,
                    op_id: child_op_id,
                    is_terminal,
                    has_record,
                }
            };

            // Recorded-terminal child (sequential path only: the
            // concurrent path applied its recorded terminals in the replay
            // pass above): resolve it inline on the coordinator, at the
            // dispatch cursor, in input order. This is the replay pass of
            // the sequential path, not a live dispatch, so the dispatch
            // gate does not run for it; its settle evaluates the
            // completion rule, except under a replay plan. The item consumed an ID on the run that
            // recorded it, so the ID is minted for counter parity.
            if pc.is_terminal {
                ctx.mint_id();
                // The item enters the "ever started" prefix the batch
                // decision record reports. It settles at once, with no
                // evaluation in between, so the predicate never sees its
                // `Started`.
                progress.mark_dispatched(i);
                let item_name = item_name_of(pc.index);
                env.resolve_terminal_inline(
                    &pc.op_id,
                    pc.index,
                    &item_name,
                    &mut join_set,
                    &mut progress,
                )
                .await?;
                continue;
            }

            // Dispatch gate: as JS does before each start, ask the
            // completion predicate on the current view. A `Complete`
            // decision sets the reason and stops dispatch; when nothing
            // runs, the loop below ends the batch at once, and otherwise
            // the next settle ends it through early completion. The gate
            // runs before the sequential path mints this item's ID, so an
            // item the gate stops consumes no ID, live or on replay. Under
            // a replay plan the recorded reason is reproduced instead.
            if replay_plan.is_none() && progress.tracker.dispatch_gate(&completion_cfg) {
                break;
            }

            // The sequential path claims the ID it peeked above.
            if concurrency == 1 {
                ctx.mint_id();
            }

            // The item is admitted: it enters the "ever started" prefix the
            // batch decision record reports, and the predicate's view reads
            // it as `Started`.
            progress.mark_dispatched(i);

            // Compute the item name in the coordinator: the start checkpoint
            // and the branch body both carry it.
            let item_name = item_name_of(pc.index);
            let branch_index = pc.index;
            let branch_wire = pc.op_id.wire().to_owned();

            // Ordered lazy admission: the coordinator, the sole writer of
            // child starts, checkpoints this child's `Start` NOW, at
            // dispatch, just before spawning its body. Writing on the
            // owning task (never inside the spawned task) keeps every
            // start committed before the child's own writes, and no start
            // exists for a child that never ran. A child holding a record
            // from a prior invocation resumes without a second `Start`.
            if nesting != NestingMode::Flat && !pc.has_record {
                let update = build_child_update(
                    &branch_wire,
                    &item_name,
                    &child_sub_type_owned,
                    &parent_wire,
                    OperationAction::Start,
                );
                if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
                    // Audit (#43): batch child START: the item closure has
                    // not run, so no terminal FAIL is needed; re-invocation
                    // reconverges on the same write. Routing unrecoverable
                    // (not as an item failure) matters doubly here: a
                    // tolerant completion config must not absorb a
                    // checkpoint failure.
                    return ctx
                        .checkpoint_failure_unrecoverable(&branch_wire, err, None)
                        .await;
                }
            }

            let req = ItemRequest {
                ctx: ctx.clone(),
                child_op_id: pc.op_id,
                index: branch_index,
                is_terminal: pc.is_terminal,
                parent_wire: parent_wire.clone(),
                child_sub_type: child_sub_type_owned.clone(),
                item_name: item_name.clone(),
                nesting,
                serdes: Arc::clone(&serdes),
            };
            dispatch
                .spawn_item(&mut join_set, req)
                .map_err(NoRuntime::operation_error)?;
            running += 1;
            running_owners.insert(branch_index, branch_wire);
        }

        if running == 0 {
            // Nothing in flight: either all dispatched work is terminal, or
            // a threshold stopped dispatch.
            break;
        }

        // Await the next branch task to terminate. A panic in user branch
        // code resumes in `settle_joined`.
        let Some((_, joined)) = join_set.join_next_with_id().await else {
            break; // set is empty: unreachable while running > 0
        };
        running -= 1;
        let settled = env.settle_joined(joined, &mut progress)?;
        running_owners.remove(&settled);

        // Early completion: the completion rule is met, so the batch ends
        // now, without waiting for its running items (as in JS). Their
        // owners are abandoned first, so a write they issue from now on is
        // rejected. Then their tasks are aborted and never joined: an
        // outcome of an abandoned item is never observed. Each aborted task
        // stays counted until it unwinds. Under a replay plan the recorded
        // outcome is reproduced instead, so nothing is abandoned.
        if replay_plan.is_none() && progress.tracker.stop_reason.is_some() {
            ctx.admission()
                .abandon(running_owners.values().map(String::as_str));
            join_set.abort_all();
            break;
        }
    }

    let BatchProgress {
        tracker,
        mut results,
        started,
    } = progress;

    // 9. Assemble results in input order. On the live path, every
    // dispatched item that is not terminal (running when the trigger fired,
    // parked, or finished but not joined) is reported as STARTED, as in JS.
    // Never-started items are omitted. Under a replay plan the record's
    // started set already appears as STARTED items.
    if replay_plan.is_none() {
        // `results` and `started` both have one entry per input item.
        for (i, (slot, started)) in results.iter_mut().zip(&started).enumerate() {
            if slot.is_none() && *started {
                *slot = Some(BatchItem {
                    index: i,
                    name: item_name_of(i),
                    status: BatchItemStatus::Started,
                    result: None,
                    error_message: None,
                    error_type: None,
                    error_data: None,
                    stack_trace: Vec::new(),
                });
            }
        }
    }
    let final_items: Vec<BatchItem<O>> = results.into_iter().flatten().collect();

    // 8b. Under a replay plan, the batch outcome is already durably
    // recorded: the reason is taken verbatim from the decision record and
    // NOTHING is re-checkpointed: the record being obeyed is the parent's
    // terminal state. `final_items` holds the recorded admission set's
    // reconstructed outcomes.
    if let Some(plan) = replay_plan {
        return Ok(BatchResult {
            items: final_items,
            reason: plan.reason,
        });
    }

    // 10. Determine completion reason: the first trigger to fire during the
    // run (recorded at the dispatch gate or the settle that fired it: first
    // trigger wins), or `AllCompleted` when no trigger fired. A non-empty
    // batch reaches this point only after a settle or after the dispatch
    // gate set the reason, and the zero-item batch returned at step 4.
    let reason = tracker
        .stop_reason
        .unwrap_or(CompletionReason::AllCompleted);

    // Ordered admission makes the ever-started set the contiguous prefix
    // `[0, ever_started)`: its length is the `totalCount` the decision
    // record carries: the number of items ever started, NOT the input
    // length.
    let ever_started = started.iter().rposition(|s| *s).map_or(0, |i| i + 1);

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
    let (payload, batch_result) = from_batch_result(
        &ctx,
        batch_result,
        &serdes,
        &parent_wire,
        ctx.execution_arn(),
    )
    .await?;
    let serialized_payload = serialize_value(
        &ctx,
        parent_name.as_deref(),
        payload,
        &result_serdes,
        SerdesContext::new(&parent_wire, ctx.execution_arn()),
    )
    .await?;

    // The decision record travels alongside the payload: when the payload
    // is too large to checkpoint inline, the record is written in its place
    // so replay can obey the recorded outcome instead of re-deriving it.
    let decision_json = serialize_decision_record(&build_decision_record(
        &batch_result.items,
        ever_started,
        batch_result.reason,
    ))?;

    // 12. Checkpoint the parent SUCCEED.
    checkpoint_batch_success_serialized(
        &ctx,
        &parent_wire,
        parent_name.as_deref(),
        parent_sub_type,
        &serialized_payload,
        &decision_json,
    )
    .await?;

    Ok(batch_result)
}

/// Replays a terminal batch (parent already succeeded/failed in the log).
///
/// Takes the targeted [`crate::engine::TerminalReplaySnapshot`] projection
/// rather than the full checkpoint record: the status, `replay_children`,
/// and payload/error strings are all this helper reads.
async fn replay_terminal_batch<O, IS, RS>(
    ctx: &DurableContext,
    name: Option<&str>,
    snapshot: &crate::engine::TerminalReplaySnapshot,
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
                // alone: the caller must fall through to re-execution.
                // Signal this by returning a sentinel error that the caller
                // catches to continue normal execution.
                return Err(batch_error(REPLAY_CHILDREN_SENTINEL));
            }
            // Deserialize the stored batch summary.
            let payload_str = snapshot
                .result
                .clone()
                .ok_or_else(|| batch_error("terminal batch has no result payload"))?;
            // Reverse the result serdes transform: through the same helper
            // every other path uses.
            let payload: BatchSummary =
                deserialize_value(ctx, name, payload_str, result_serdes, serdes_ctx.clone())
                    .await?;
            // The batch parent's serdes context carries the parent wire ID and
            // the execution ARN, which is exactly what the per-item contexts
            // are derived from.
            to_batch_result(
                ctx,
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
            // Shouldn't happen: we checked is_terminal() above.
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
            let value: O = deserialize_value(
                ctx,
                item_fault_name(item_name),
                payload,
                serdes,
                serdes_ctx.clone(),
            )
            .await?;
            Ok(BatchItem {
                index,
                name: item_name.to_owned(),
                status: BatchItemStatus::Succeeded,
                result: Some(value),
                error_message: None,
                error_type: None,
                error_data: None,
                stack_trace: Vec::new(),
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
                error_data: record.error_data.clone(),
                stack_trace: record.stack_trace.clone().unwrap_or_default(),
            })
        }
        _ => Err(batch_error(
            "unexpected non-terminal child status in replay",
        )),
    }
}

/// The SDK-owned decision record a batch checkpoints in place of a result
/// payload too large to store inline.
///
/// Mirrors the Python SDK's summary envelope: `totalCount` is the number of
/// items EVER STARTED, not the input length; ordered admission makes the
/// started set the contiguous prefix `[0, totalCount)`: `completionReason`
/// is the wire form of [`CompletionReason`], and one of `startedIndexes` /
/// `completedIndexes` (whichever is shorter) names the partition of that
/// prefix: `startedIndexes` lists items that began but were not terminal
/// when the batch completed, `completedIndexes` lists the terminal ones.
/// The read side accepts either form and reconstructs the other against the
/// same prefix, so the record stays compact at both early-exit extremes.
///
/// The record is plain JSON and deliberately BYPASSES `result_serdes`,
/// matching Python: it is the SDK's own bookkeeping, carries no user-typed
/// values, and must stay readable even when the configured serdes is the
/// reason the real payload outgrew the checkpoint limit. `replay_children`
/// on the parent record is the discriminator: a `replay_children` payload
/// is a decision record; a plain payload is the serdes-transformed batch
/// summary.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BatchDecisionRecord {
    /// Number of items ever started (the admitted prefix length).
    #[serde(rename = "totalCount")]
    total_count: usize,
    /// Wire form of the batch's [`CompletionReason`], taken verbatim by
    /// replay.
    #[serde(rename = "completionReason")]
    completion_reason: String,
    /// Indexes (within `[0, totalCount)`) of items that began but were not
    /// terminal when the batch completed. Written only when it is the
    /// shorter set.
    #[serde(
        rename = "startedIndexes",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    started_indexes: Option<Vec<usize>>,
    /// Indexes (within `[0, totalCount)`) of items that reached a terminal
    /// state. Written only when it is the shorter set.
    #[serde(
        rename = "completedIndexes",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    completed_indexes: Option<Vec<usize>>,
}

/// Builds the decision record for a completed batch: `ever_started` is the
/// admitted prefix length, `items` are the reported items, and the shorter
/// of the two index sets is the one written. Only a `Succeeded` or `Failed`
/// item counts as completed.
fn build_decision_record<O>(
    items: &[BatchItem<O>],
    ever_started: usize,
    reason: CompletionReason,
) -> BatchDecisionRecord {
    let mut completed_flags = vec![false; ever_started];
    for item in items {
        // A STARTED item was not terminal when the batch completed, so it
        // stays in the started set.
        if item.status == BatchItemStatus::Started {
            continue;
        }
        if let Some(slot) = completed_flags.get_mut(item.index) {
            *slot = true;
        }
    }
    let completed: Vec<usize> = completed_flags
        .iter()
        .enumerate()
        .filter_map(|(i, done)| done.then_some(i))
        .collect();
    let started_only: Vec<usize> = completed_flags
        .iter()
        .enumerate()
        .filter_map(|(i, done)| (!done).then_some(i))
        .collect();
    let (started_indexes, completed_indexes) = if started_only.len() <= completed.len() {
        (Some(started_only), None)
    } else {
        (None, Some(completed))
    };
    BatchDecisionRecord {
        total_count: ever_started,
        completion_reason: reason.as_str().to_owned(),
        started_indexes,
        completed_indexes,
    }
}

/// Serializes a decision record as plain JSON (never through
/// `result_serdes`: see [`BatchDecisionRecord`]).
fn serialize_decision_record(record: &BatchDecisionRecord) -> Result<String, OperationError> {
    serde_json::to_string(record)
        .map_err(|e| batch_error(&format!("serialize batch decision record: {e}")))
}

/// Parses a terminal batch's `replay_children` payload as a decision
/// record.
///
/// `Ok(None)`, an absent or empty payload, means the record predates
/// decision records (a record-less `replay_children` write from an older
/// SDK), and the caller falls back to the pre-record re-execution
/// behaviour. A NON-EMPTY payload on a `replay_children` batch parent can
/// only be a decision record this SDK wrote, so one that fails to parse is
/// corrupt bookkeeping: the safe response is an error, never a silent
/// fall-back to unconstrained re-execution that could resurrect items the
/// recorded batch never started.
fn parse_decision_record(
    payload: Option<&str>,
) -> Result<Option<BatchDecisionRecord>, OperationError> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    let payload = payload.trim();
    if payload.is_empty() {
        return Ok(None);
    }
    serde_json::from_str(payload)
        .map(Some)
        .map_err(|e| batch_error(&format!("malformed batch decision record: {e}")))
}

/// What a decision record instructs replay to do: which indexes were ever
/// started, which of those completed, and the reason the batch ended.
#[derive(Debug)]
struct ReplayPlan {
    /// Number of items ever started (`totalCount`, validated to be no
    /// larger than the input length).
    total_count: usize,
    /// Per-index completion flag over `[0, total_count)`.
    completed: Vec<bool>,
    /// The recorded completion reason, reported verbatim.
    reason: CompletionReason,
}

impl ReplayPlan {
    /// Derives the plan from a parsed decision record, rejecting any record
    /// this SDK's writer could not have produced. The writer emits exactly
    /// one index form (the shorter set), every index inside
    /// `[0, totalCount)`, a `totalCount` no larger than the input it ran
    /// over, and a known completion-reason wire value, so a record that
    /// violates any of those is corrupt bookkeeping. Silently repairing it
    /// (clamping, ignoring, defaulting) would convert recorded outcomes
    /// into different ones; failing keeps replay deterministic and safe.
    fn from_record(
        record: &BatchDecisionRecord,
        total_items: usize,
    ) -> Result<Self, OperationError> {
        if record.total_count > total_items {
            return Err(batch_error(&format!(
                "invalid batch decision record: totalCount {} exceeds the input length {}",
                record.total_count, total_items
            )));
        }
        let Some(reason) = CompletionReason::try_from_wire(&record.completion_reason) else {
            return Err(batch_error(&format!(
                "invalid batch decision record: unknown completionReason {:?}",
                record.completion_reason
            )));
        };
        let total_count = record.total_count;
        let (indexes, indexes_are_completed) =
            match (&record.completed_indexes, &record.started_indexes) {
                (Some(_), Some(_)) => {
                    return Err(batch_error(
                        "invalid batch decision record: both startedIndexes and \
                         completedIndexes are present",
                    ));
                }
                (None, None) => {
                    return Err(batch_error(
                        "invalid batch decision record: neither startedIndexes nor \
                         completedIndexes is present",
                    ));
                }
                (Some(completed), None) => (completed, true),
                (None, Some(started)) => (started, false),
            };
        let mut completed = vec![!indexes_are_completed; total_count];
        for &i in indexes {
            let Some(slot) = completed.get_mut(i) else {
                return Err(batch_error(&format!(
                    "invalid batch decision record: index {i} is outside [0, {total_count})",
                )));
            };
            *slot = indexes_are_completed;
        }
        Ok(Self {
            total_count,
            completed,
            reason,
        })
    }

    /// Whether replay admits item `index` for terminal reconstruction: it
    /// must be inside the started prefix AND recorded as completed.
    /// Started-but-not-terminal items are reported as
    /// [`BatchItemStatus::Started`] from the record alone, and
    /// never-started items are omitted entirely.
    fn admits(&self, index: usize) -> bool {
        index < self.total_count && self.completed.get(index).copied().unwrap_or(false)
    }
}

/// Fast replay of a large-result batch (Normal nesting): reconstructs the
/// batch result directly from the decision record and the child checkpoint
/// records, without re-running any item body.
///
/// Child IDs are PEEKED, never minted, so a fall-back to re-execution, an
/// admitted child whose own result was too large to checkpoint inline
/// surfaces as the `ReplayChildren` sentinel, leaves the parent counter
/// untouched for the re-execution path; the caller advances the counter
/// only after a successful reconstruction. Items in the started set are
/// reported as [`BatchItemStatus::Started`] without consulting their child
/// checkpoints, and indexes at or beyond `totalCount` are omitted entirely.
///
/// Every completed child's replay identity (type, sub-type, name) is
/// validated against its checkpoint record before its terminal payload is
/// read, so a handler change that renames or retypes a child raises
/// `NonDeterministicExecution` here exactly as it does on the re-execution
/// path. Started-set children are exempt: their records are never
/// consulted at all.
async fn replay_batch_from_record<O, IS>(
    ctx: &DurableContext,
    plan: &ReplayPlan,
    names: &[Option<String>],
    serdes: &Arc<IS>,
    child_sub_type: &str,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let mut items = Vec::new();
    for i in 0..plan.total_count {
        let child_name = names.get(i).cloned().flatten();
        if !plan.completed.get(i).copied().unwrap_or(false) {
            // STARTED at batch completion: reported from the record with
            // no result and no error; its child checkpoints are never
            // read.
            items.push(BatchItem {
                index: i,
                name: child_name.unwrap_or_default(),
                status: BatchItemStatus::Started,
                result: None,
                error_message: None,
                error_type: None,
                error_data: None,
                stack_trace: Vec::new(),
            });
            continue;
        }
        let op_id = ctx.peek_id_at(i);
        // Non-determinism detection: validate the claimed child identity
        // against the record before decoding its terminal payload.
        ctx.checkpoint_view_validated(
            op_id.positional(),
            op_id.wire(),
            "Context",
            Some(child_sub_type),
            child_name.as_deref(),
        )?;
        let item_name = child_name.unwrap_or_default();
        let serdes_ctx = SerdesContext::new(op_id.wire().to_owned(), ctx.execution_arn());
        let item =
            replay_terminal_child(ctx, op_id.positional(), i, &item_name, serdes, &serdes_ctx)
                .await?;
        items.push(item);
    }
    Ok(BatchResult {
        items,
        reason: plan.reason,
    })
}

/// Checkpoints the parent batch as SUCCEEDED with the pre-serialized result.
///
/// When the serialized payload exceeds [`CHECKPOINT_SIZE_LIMIT_BYTES`], the
/// record is written with `replay_children(true)` and `decision_json`, the
/// batch decision record, as its payload, so replay can obey the recorded
/// outcome (`totalCount`, `completionReason`, index set) instead of
/// re-deriving it by re-execution. A modern terminal batch is NEVER written
/// without its decision record: should the record itself exceed the limit
/// (a pathological interleaving across tens of thousands of items), the
/// checkpoint is refused with an error rather than written record-less,
/// because a record-less write would replay through the unconstrained
/// re-execution path and could resurrect items the live batch never
/// started. The refusal is deterministic, replay reaches the same state
/// and fails the same way, so no false outcome is ever recorded.
async fn checkpoint_batch_success_serialized(
    ctx: &DurableContext,
    parent_wire: &str,
    parent_name: Option<&str>,
    parent_sub_type: &str,
    serialized_payload: &str,
    decision_json: &str,
) -> Result<(), OperationError> {
    let mut builder = OperationUpdate::builder()
        .id(parent_wire.to_owned())
        .r#type(OperationType::Context)
        .sub_type(parent_sub_type.to_owned())
        .action(OperationAction::Succeed);

    if let Some(n) = parent_name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire_id) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire_id);
    }

    if serialized_payload.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
        if decision_json.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
            // Never write a modern terminal batch without its decision
            // record (see the function docs).
            return Err(batch_error(
                "batch decision record exceeds the checkpoint size limit; \
                 refusing to record the batch outcome without it",
            ));
        }
        builder = builder
            .context_options(
                aws_sdk_lambda::types::ContextOptions::builder()
                    .replay_children(true)
                    .build(),
            )
            .payload(decision_json.to_owned());
    } else {
        builder = builder.payload(serialized_payload.to_owned());
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43): batch parent SUCCEED: the batch's items ran, so
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
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
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

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds a child-level `FAIL` update carrying `wire` as its error: the
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
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds a parent-level `FAIL` update carrying `wire` as its error: the
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
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Running completion state of one batch coordinator: the settle-order
/// counts, the view of every input item that a completion predicate sees,
/// and the first completion trigger to fire.
///
/// Every settle, a recorded-terminal child applied inline or a live join,
/// flows through [`settle`](Self::settle), and every live dispatch first
/// passes [`dispatch_gate`](Self::dispatch_gate). So the completion rule is
/// evaluated in exactly these two places, as JS evaluates it in `isComplete`
/// and `shouldContinue`.
///
/// The fixed thresholds and the predicate cannot both be configured
/// ([`CompletionConfigBuilder::build`](crate::builders::map_parallel::CompletionConfigBuilder::build)
/// rejects the mix), so their order within one settle does not matter.
/// The live settle order is not recorded, and on resume the recorded items
/// settle first, in index order. The thresholds only grow with the counts,
/// so they reach the same decision in any order. A predicate reaches the
/// same decision only when it is monotonic, which its rustdoc requires.
struct CompletionTracker {
    /// Count of items that have succeeded so far (settle order).
    success_count: usize,
    /// Count of items that have failed so far (settle order).
    failure_count: usize,
    /// One entry per input item, by index: the predicate's view.
    items: Vec<CompletionItem>,
    /// The first completion trigger to fire (first trigger wins). `Some`
    /// halts new dispatch, and nothing changes it afterwards.
    stop_reason: Option<CompletionReason>,
    /// `false` while the coordinator rebuilds a batch that already
    /// completed: its reason is read from the decision record, so neither
    /// the thresholds nor the predicate run.
    evaluating: bool,
}

impl CompletionTracker {
    /// Builds one not-started [`CompletionItem`] per input item, carrying
    /// its caller name.
    fn new(names: Vec<Option<String>>) -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            items: names
                .into_iter()
                .enumerate()
                .map(|(index, name)| CompletionItem::new(index, name, None))
                .collect(),
            stop_reason: None,
            evaluating: true,
        }
    }

    /// Stops every later evaluation of the completion rule. The
    /// coordinator calls it when it rebuilds a batch that already
    /// completed: the recorded reason is the outcome, and a predicate
    /// called on the rebuilt view could complete or fail differently.
    fn stop_evaluating(&mut self) {
        self.evaluating = false;
    }

    /// The predicate's snapshot of the current view.
    fn stats(&self) -> BatchStats<'_> {
        BatchStats::from_counts(self.success_count, self.failure_count, &self.items)
    }

    /// Marks one item as dispatched live: its view entry reads `Started`.
    fn mark_dispatched(&mut self, index: usize) {
        if let Some(item) = self.items.get_mut(index) {
            item.status = Some(BatchItemStatus::Started);
        }
    }

    /// Evaluates the predicate before a live dispatch, while no trigger has
    /// fired. Returns `true` when a trigger has fired, so the item must not
    /// be dispatched.
    fn dispatch_gate(&mut self, cfg: &crate::builders::map_parallel::CompletionConfig) -> bool {
        if self.stop_reason.is_none() {
            self.stop_reason = custom_reason(cfg.predicate_decision(&self.stats()));
        }
        self.stop_reason.is_some()
    }

    /// Applies one settled outcome to the counts and the view and, while no
    /// trigger has fired yet, evaluates the thresholds, then the predicate
    /// (first trigger wins: once `stop_reason` is set it never changes).
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
            // Not a terminal outcome: never fed to the tracker (a Started
            // item exists only in replayed results, and the replay path
            // does not consult the tracker).
            BatchItemStatus::Started => return,
        }
        if let Some(item) = self.items.get_mut(index) {
            item.status = Some(status);
        }
        if !self.evaluating {
            return;
        }
        if self.stop_reason.is_none() {
            self.stop_reason =
                evaluate_thresholds(cfg, self.success_count, self.failure_count, total_items);
        }
        if self.stop_reason.is_none() {
            self.stop_reason = custom_reason(cfg.predicate_decision(&self.stats()));
        }
    }
}

/// Maps a predicate decision to the reason it records: `None` for
/// `Continue`.
fn custom_reason(decision: CompletionDecision) -> Option<CompletionReason> {
    match decision {
        CompletionDecision::Continue => None,
        CompletionDecision::Complete(CompletionOutcome::Succeeded) => {
            Some(CompletionReason::CustomCompletionSucceeded)
        }
        CompletionDecision::Complete(CompletionOutcome::Failed) => {
            Some(CompletionReason::CustomCompletionFailed)
        }
    }
}

/// Evaluates the fixed completion thresholds against the running batch
/// counts, returning the first that fires (or `None`).
///
/// Called after each settled item. The order follows JS: the failure
/// tolerance first, then "every item finished" (which leaves the reason
/// `AllCompleted`), then `min_successful`.
fn evaluate_thresholds(
    cfg: &crate::builders::map_parallel::CompletionConfig,
    success_count: usize,
    failure_count: usize,
    total_items: usize,
) -> Option<CompletionReason> {
    if should_stop_failure(cfg, failure_count, total_items) {
        return Some(CompletionReason::FailureToleranceExceeded);
    }
    if success_count + failure_count < total_items && should_stop_min(cfg, success_count) {
        return Some(CompletionReason::MinSuccessfulReached);
    }
    None
}

/// Checks if the `min_successful` threshold has been met. A minimum of `0`
/// is a configured minimum that any count meets, as in JS.
fn should_stop_min(
    cfg: &crate::builders::map_parallel::CompletionConfig,
    success_count: usize,
) -> bool {
    cfg.min_successful().is_some_and(|min| success_count >= min)
}

/// Checks if the failure tolerance has been exceeded.
fn should_stop_failure(
    cfg: &crate::builders::map_parallel::CompletionConfig,
    failure_count: usize,
    total_items: usize,
) -> bool {
    // Empty config: fail fast. When NO completion criterion is configured
    // (no thresholds and no predicate, which is also what an absent
    // config becomes), a single failure exceeds tolerance, matching the
    // Python and JS SDKs (issue #52). Configuring any criterion opts out
    // of this implicit fail-fast and hands the decision to that criterion.
    if !cfg.has_criteria() {
        return failure_count > 0;
    }

    // Count-based tolerance.
    if let Some(tolerated) = cfg.tolerated_failure_count()
        && failure_count > tolerated
    {
        return true;
    }

    // Percentage-based tolerance.
    // Uses cross-multiplication (failure_count * 100 > pct * total_items)
    // rather than JS's division: it is exact for integer percentages and
    // totals below 2^53, so 7 failures of 50 do not exceed 14.0, where JS's
    // `7 / 50 * 100` gives 14.000000000000002. A true failure rate of 33.3%
    // still exceeds a 33.0% threshold (100 > 99). When pct == 0.0, any
    // failure exceeds the threshold (fail-fast).
    #[expect(clippy::cast_precision_loss)]
    // reason: item counts stay far below 2^53, where the cast is exact
    let exceeds = |pct: f64| failure_count as f64 * 100.0 > pct * total_items as f64;
    if let Some(pct) = cfg.tolerated_failure_percentage()
        && total_items > 0
        && exceeds(pct)
    {
        return true;
    }

    false
}

/// Serializes a value through the configured serdes (ownership transfers;
/// the serdes decides where its work runs).
///
/// This is the same boundary every other operation (step, invoke, callback,
/// child, batch result) uses, and, since the item paths call this helper
/// too, the same one map/parallel ITEM results use. There is no separate
/// item rule.
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes a
/// batch error.
async fn serialize_value<T, S: Serdes<T>>(
    ctx: &DurableContext,
    name: Option<&str>,
    value: T,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx.hold_activity(serdes.serialize(value, serdes_ctx)).await {
        Ok(serialized) => Ok(serialized),
        Err(err) => {
            let e = ctx
                .route_serdes_error("serialize", &operation_id, name, err)
                .await;
            Err(batch_error(&format!("serialize result: {e}")))
        }
    }
}

/// Deserializes a value through the configured serdes.
///
/// Reverses [`serialize_value`]: the serdes turns the wire payload directly
/// back into the typed value, no intermediate representation and no
/// runtime downcast.
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes a
/// batch error.
async fn deserialize_value<T, S: Serdes<T>>(
    ctx: &DurableContext,
    name: Option<&str>,
    payload: String,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<T, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx
        .hold_activity(serdes.deserialize(payload, serdes_ctx))
        .await
    {
        Ok(value) => Ok(value),
        Err(err) => {
            let e = ctx
                .route_serdes_error("deserialize", &operation_id, name, err)
                .await;
            Err(batch_error(&format!("deserialize result: {e}")))
        }
    }
}

/// The name a serdes fault reports for a batch item: an empty item name
/// (a map without an item namer) is `None`.
fn item_fault_name(item_name: &str) -> Option<&str> {
    (!item_name.is_empty()).then_some(item_name)
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
#[non_exhaustive]
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
    /// Opaque per-item error payload, passed through verbatim.
    #[serde(default, rename = "errData", skip_serializing_if = "Option::is_none")]
    err_data: Option<String>,
    /// Recorded per-item stack trace frames.
    #[serde(default, rename = "stackTrace", skip_serializing_if = "Vec::is_empty")]
    stack_trace: Vec<String>,
}

/// Converts a live `BatchResult` into the checkpoint payload format,
/// round-tripping every successful item value through the item serdes.
///
/// The batch result is consumed: each successful item's value transfers by
/// ownership to `Serdes::serialize` (which lets the serdes move it into a
/// blocking task without requiring `O: Sync`), and the returned
/// `BatchResult` carries the values reconstructed from their wire form,
/// so live and replay observe identical values.
async fn from_batch_result<O, IS>(
    ctx: &DurableContext,
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
            error_data,
            stack_trace,
        } = item;
        let status_str = match status {
            BatchItemStatus::Succeeded => "SUCCEEDED",
            BatchItemStatus::Failed => "FAILED",
            BatchItemStatus::Started => "STARTED",
        };
        let (result_str, rebuilt_value) = match (status, result) {
            (BatchItemStatus::Succeeded, Some(value)) => {
                let item_ctx = item_summary_serdes_ctx(parent_wire, execution_arn, index);
                let item_name = item_fault_name(&name);
                let wire =
                    serialize_value(ctx, item_name, value, serdes.as_ref(), item_ctx.clone())
                        .await?;
                let back: O =
                    deserialize_value(ctx, item_name, wire.clone(), serdes.as_ref(), item_ctx)
                        .await?;
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
                error_type.clone().unwrap_or_else(|| "Error".to_owned())
            } else {
                String::new()
            },
            err_message: error_message.clone().unwrap_or_default(),
            err_data: if status == BatchItemStatus::Failed {
                error_data.clone()
            } else {
                None
            },
            stack_trace: if status == BatchItemStatus::Failed {
                stack_trace.clone()
            } else {
                Vec::new()
            },
        });
        rebuilt_items.push(BatchItem {
            index,
            name,
            status,
            result: rebuilt_value,
            error_message,
            error_type,
            error_data,
            stack_trace,
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
    ctx: &DurableContext,
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
            "STARTED" => BatchItemStatus::Started,
            other => return Err(batch_error(&format!("unknown item status: {other}"))),
        };
        let result = if status == BatchItemStatus::Succeeded && !cp.result.is_empty() {
            let item_ctx = item_summary_serdes_ctx(parent_wire, execution_arn, cp.index);
            Some(
                deserialize_value::<O, _>(
                    ctx,
                    item_fault_name(&cp.name),
                    cp.result,
                    serdes.as_ref(),
                    item_ctx,
                )
                .await?,
            )
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
            error_data: if status == BatchItemStatus::Failed {
                cp.err_data
            } else {
                None
            },
            stack_trace: if status == BatchItemStatus::Failed {
                cp.stack_trace
            } else {
                Vec::new()
            },
        });
    }

    Ok(BatchResult { items, reason })
}

/// Constructs a batch operation error for a coordinator defect: an
/// internal invariant breach or a malformed record, never a user item's
/// failure (those route through [`item_failure_error`], which classifies
/// as `ChildFailed`).
fn batch_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::Internal,
        Some(message.to_owned().into()),
    )))
}

/// Constructs the operation error a batch surfaces when a USER item body
/// failed (the failure tolerance was exceeded): classified as
/// `ChildFailed`, carrying the failing item's COMPLETE recorded wire
/// record so the error's identity (`error_type`, message, `error_data`,
/// `stack_trace`) stays programmatically recoverable and identical live
/// and on replay (both paths read the same recorded per-item fields).
fn item_failure_error(wire: crate::error::WireError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::ChildFailed,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )))
    .with_wire(wire)
}

/// Constructs the operation error a batch surfaces when its completion
/// predicate completed it as failed (`CUSTOM_COMPLETION_FAILED`) and no item
/// failed. It carries a wire record of type `BatchCompletionError`, the type
/// JS throws in the same case.
fn batch_completion_failed_error() -> OperationError {
    let wire = crate::error::WireError::new(
        Some("BatchCompletionError"),
        Some("batch completed as failed by the completion predicate (CUSTOM_COMPLETION_FAILED)"),
    );
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::BatchCompletionFailed,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )))
    .with_wire(wire)
}

/// Wraps owned map inputs in indexed take-once slots shareable across
/// branch tasks. Each item is MOVED out exactly once by [`take_item`],
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

#[cfg(test)]
#[expect(clippy::indexing_slicing)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
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

    /// A user item body's failure surfaces as `ChildFailed` carrying the
    /// item's COMPLETE recorded wire record — all four wire fields, not
    /// just type and message (finding: item-body failures and coordinator
    /// defects must not share a classification, and the recorded
    /// `error_data`/`stack_trace` must not be discarded).
    #[test]
    fn collect_successful_classifies_item_failure_as_child_failed() {
        let batch = BatchResult {
            items: vec![BatchItem::<i32> {
                index: 0,
                name: "item-0".to_owned(),
                status: BatchItemStatus::Failed,
                result: None,
                error_message: Some("boom-item".to_owned()),
                error_type: Some("MyItemError".to_owned()),
                error_data: Some(r#"{"detail":42}"#.to_owned()),
                stack_trace: vec!["frame-0".to_owned(), "frame-1".to_owned()],
            }],
            reason: CompletionReason::FailureToleranceExceeded,
        };
        let err = collect_successful(batch).expect_err("tolerance exceeded must error");
        let OperationErrorKind::ChildContext(child_err) = err.kind() else {
            unreachable!("expected a child-context error, got {:?}", err.kind());
        };
        assert!(
            matches!(child_err.kind(), ChildContextErrorKind::ChildFailed),
            "item failures classify as ChildFailed: {:?}",
            child_err.kind()
        );
        let wire = err.wire().expect("the item's wire record is attached");
        assert_eq!(wire.error_type(), Some("MyItemError"));
        assert_eq!(wire.error_message(), Some("boom-item"));
        assert_eq!(wire.error_data(), Some(r#"{"detail":42}"#));
        assert_eq!(
            wire.stack_trace(),
            ["frame-0".to_owned(), "frame-1".to_owned()]
        );
    }

    /// Coordinator defects keep the `Internal` classification.
    #[test]
    fn batch_error_keeps_internal_classification_for_coordinator_defects() {
        let err = batch_error("invariant breached");
        let OperationErrorKind::ChildContext(child_err) = err.kind() else {
            unreachable!("expected a child-context error, got {:?}", err.kind());
        };
        assert!(
            matches!(child_err.kind(), ChildContextErrorKind::Internal),
            "coordinator defects classify as Internal: {:?}",
            child_err.kind()
        );
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
    /// records: the public naming API must not be silently discarded.
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
                due_at: None,
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
                error_data: Some(r#"{"recorded":true}"#.to_owned()),
                stack_trace: Some(vec!["recorded-frame".to_owned()]),
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
                due_at: None,
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
        // Live execution records the body error's type: an untyped body
        // error records "Error", and the structured view exposes that
        // identity alongside the message.
        assert_eq!(errors[0].error_type, Some("Error"));
        assert_eq!(by_index(1).error_type.as_deref(), Some("Error"));
    }

    /// Partial replay must not discard branch names: when a branch reached a
    /// terminal state on a previous invocation (e.g. it failed while a
    /// sibling suspended), the next invocation reconstructs it from the
    /// checkpoint log via `replay_terminal_child`, and the resulting
    /// [`BatchItem`], and the structured error view, must still carry the
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
                // Would SUCCEED if re-executed: the failure below proves
                // the item was reconstructed from the checkpoint record.
                Branch::new("boom", |_ctx| async move { Ok(999_i32) }),
                // Would return 999 if re-executed: the 7 below proves
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
        // The replayed item carries the child record's COMPLETE wire
        // failure record, not just its type and message.
        assert_eq!(failed.error_type.as_deref(), Some("ChildFnError"));
        assert_eq!(failed.error_data.as_deref(), Some(r#"{"recorded":true}"#));
        assert_eq!(failed.stack_trace, ["recorded-frame".to_owned()]);

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
        assert_eq!(errors[0].error_type, Some("ChildFnError"));
    }

    /// The batch summary payload the parent recorded: the first recorded
    /// update whose payload decodes as a [`BatchSummary`].
    fn recorded_batch_summary(client: &crate::client::InMemoryExecutionClient) -> BatchSummary {
        client
            .recorded_updates()
            .iter()
            .filter(|u| u.parent_id().is_none())
            .filter_map(|u| u.payload())
            .find_map(|p| serde_json::from_str::<BatchSummary>(p).ok())
            .expect("the parent records a batch summary payload")
    }

    /// A failed map item records the type of the error its body returned
    /// (`"Error"` for an untyped error), and the four views of the failure
    /// agree: the child FAIL record, `BatchItem::error_type`,
    /// `BatchError::error_type` and the summary `errType`.
    #[tokio::test]
    async fn map_item_failure_records_body_error_type() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![1_i32], |_child, _item, _idx| async move {
                Err::<i32, BoxError>("plain boom".into())
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(0),
            )
            .await_batch()
            .await
            .expect("await_batch reports a failed item as a batch result");

        let fail = client
            .recorded_updates()
            .into_iter()
            .find(|u| {
                matches!(u.action(), OperationAction::Fail)
                    && matches!(u.r#type(), OperationType::Context)
            })
            .expect("the failed item records a child FAIL update");
        let error = fail
            .error()
            .expect("the FAIL update carries an error object");
        assert_eq!(error.error_type(), Some("Error"));
        assert_eq!(error.error_message(), Some("plain boom"));

        let item = batch
            .items
            .iter()
            .find(|i| i.index == 0)
            .expect("the batch reports the failed item");
        assert_eq!(item.status, BatchItemStatus::Failed);
        assert_eq!(item.error_type.as_deref(), Some("Error"));
        assert_eq!(batch.errors()[0].error_type, Some("Error"));

        let summary = recorded_batch_summary(&client);
        assert_eq!(
            summary.results.first().map(|r| r.err_type.as_str()),
            Some("Error")
        );
    }

    /// The same as [`map_item_failure_records_body_error_type`] for
    /// [`NestingMode::Flat`], which writes no child records.
    #[tokio::test]
    async fn flat_item_failure_records_body_error_type() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![1_i32], |_child, _item, _idx| async move {
                Err::<i32, BoxError>("plain boom".into())
            })
            .nesting(NestingMode::Flat)
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(0),
            )
            .await_batch()
            .await
            .expect("await_batch reports a failed item as a batch result");

        let item = batch
            .items
            .iter()
            .find(|i| i.index == 0)
            .expect("the batch reports the failed item");
        assert_eq!(item.status, BatchItemStatus::Failed);
        assert_eq!(item.error_type.as_deref(), Some("Error"));

        let summary = recorded_batch_summary(&client);
        assert_eq!(
            summary.results.first().map(|r| r.err_type.as_str()),
            Some("Error")
        );
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

    /// Issue #52: with NO completion config, the first item failure fails
    /// the batch (fail-fast), matching the Python and JS SDKs. Verified
    /// end-to-end on both `map` and `parallel` plain `.await`.
    #[tokio::test]
    async fn absent_completion_config_fails_fast_end_to_end() {
        use crate::future::Branch;

        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let err = ctx
            .parallel(vec![
                Branch::new("ok", |_ctx| async move { Ok(1_i32) }),
                Branch::new("boom", |_ctx| async move { Err("intentional".into()) }),
            ])
            .await
            .expect_err("a branch failure without a completion config must fail the batch");
        assert!(
            format!("{err:#}").contains("intentional"),
            "the batch error must carry the failing branch's message, got: {err:#}"
        );

        let (map_ctx, _c) = test_ctx_with_client(CheckpointLog::empty());
        let map_err = map_ctx
            .map(vec![1_i32, -1], |_child, item, _idx| async move {
                if item < 0 {
                    Err("intentional".into())
                } else {
                    Ok(item)
                }
            })
            .await
            .expect_err("an item failure without a completion config must fail the batch");
        assert!(
            format!("{map_err:#}").contains("intentional"),
            "the batch error must carry the failing item's message, got: {map_err:#}"
        );
    }

    /// Issue #52: an EMPTY `CompletionConfig` behaves exactly as no config,
    /// fail-fast, and `await_batch` reports the batch as ended by
    /// `FAILURE_TOLERANCE_EXCEEDED`, the same reason an explicit
    /// `with_tolerated_failure_count(0)` records.
    #[tokio::test]
    async fn empty_completion_config_fails_fast_end_to_end() {
        use crate::future::Branch;

        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .parallel(vec![Branch::new("boom", |_ctx| async move {
                Err::<i32, _>("intentional".into())
            })])
            .completion(crate::builders::map_parallel::CompletionConfig::default())
            .await_batch()
            .await
            .expect("await_batch must return the batch even when it fails fast");
        assert_eq!(
            batch.reason,
            CompletionReason::FailureToleranceExceeded,
            "an empty config must record the same reason as explicit fail-fast"
        );
        assert_eq!(batch.failure_count(), 1);

        // Plain `.await` on the same workload converts it into an error.
        let (ctx2, _c2) = test_ctx_with_client(CheckpointLog::empty());
        let err = ctx2
            .parallel(vec![Branch::new("boom", |_ctx| async move {
                Err::<i32, _>("intentional".into())
            })])
            .completion(crate::builders::map_parallel::CompletionConfig::default())
            .await
            .expect_err("a failure under an empty config must fail the batch");
        assert!(
            format!("{err:#}").contains("intentional"),
            "the batch error must carry the branch's message, got: {err:#}"
        );
    }

    /// Parity: `map` and `parallel` `await_batch` must agree on the batch
    /// shape for equivalent workloads: same completion reason, same
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
    // `async move` bodies using `?`, no manual `Box::pin`, no error
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
                    due_at: None,
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
            .tolerated_failure_percentage(20.0)
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
            .tolerated_failure_percentage(33.0)
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
            .tolerated_failure_percentage(34.0)
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
            .tolerated_failure_percentage(0.0)
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

    /// Issue #52: an EMPTY config (no threshold, no predicate, which is
    /// also what an absent config becomes) fails fast on the first failure,
    /// matching the Python and JS SDKs.
    #[tokio::test]
    async fn empty_completion_config_fails_fast() {
        let cfg = crate::builders::map_parallel::CompletionConfig::default();
        assert!(
            should_stop_failure(&cfg, 1, 10),
            "an empty config must fail fast on the first failure"
        );
        assert!(
            !should_stop_failure(&cfg, 0, 10),
            "an empty config with no failures must not stop"
        );
    }

    /// Issue #52: configuring ANY criterion opts out of the implicit
    /// fail-fast: a min_successful-only or predicate-only config tolerates
    /// failures and lets its own criterion drive completion.
    #[tokio::test]
    async fn any_configured_criterion_disables_implicit_fail_fast() {
        // min_successful only: failures are tolerated (Python parity:
        // has_criteria is true, and no failure tolerance is configured).
        let min_only = crate::builders::map_parallel::CompletionConfig::with_min_successful(2);
        assert!(
            !should_stop_failure(&min_only, 1, 10),
            "a min_successful-only config must tolerate failures"
        );

        // predicate only: the predicate owns the completion decision
        // (JS parity).
        let predicate_only =
            crate::builders::map_parallel::CompletionConfig::with_completion_predicate(|_stats| {
                CompletionDecision::Continue
            });
        assert!(
            !should_stop_failure(&predicate_only, 1, 10),
            "a predicate-only config must tolerate failures"
        );

        // pct=100: the tolerate-all recipe: the failure rate can never
        // strictly exceed 100%.
        let tolerate_all =
            crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_percentage(
                100.0,
            );
        assert!(
            !should_stop_failure(&tolerate_all, 10, 10),
            "pct=100 must tolerate even a fully failed batch"
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
        // (Go/JS allow it: first threshold fires). No error.
        let cfg = crate::builders::map_parallel::CompletionConfig::builder()
            .min_successful(2)
            .tolerated_failure_count(1)
            .build()
            .expect("valid completion config");
        assert!(cfg.validate().is_ok());
    }

    /// The completion types the tests below use. `CompletionConfig` gets the
    /// short alias `Config`, because the older tests in this module spell out
    /// its full path.
    use crate::builders::map_parallel::{
        CompletionConfig as Config, CompletionDecision, CompletionItem, CompletionOutcome,
    };

    /// `Complete(Succeeded)` when `done`, else `Continue`.
    fn succeed_when(done: bool) -> CompletionDecision {
        if done {
            CompletionDecision::Complete(CompletionOutcome::Succeeded)
        } else {
            CompletionDecision::Continue
        }
    }

    /// `Complete(Failed)` when `done`, else `Continue`.
    fn fail_when(done: bool) -> CompletionDecision {
        if done {
            CompletionDecision::Complete(CompletionOutcome::Failed)
        } else {
            CompletionDecision::Continue
        }
    }

    /// Reports whether item `index` of the view has the given status.
    fn item_is(view: &BatchStats<'_>, index: usize, status: BatchItemStatus) -> bool {
        view.items()
            .get(index)
            .is_some_and(|item| item.status == Some(status))
    }

    /// The payload of the batch parent's `SUCCEED` update.
    fn parent_payload(client: &crate::client::InMemoryExecutionClient) -> String {
        parent_succeed_update(client)
            .payload()
            .map(str::to_owned)
            .expect("the batch parent SUCCEED carries a payload")
    }

    /// Runs `map(items)` twice on fresh contexts: once through
    /// `await_batch()` and once through `.await`. An item `Some(v)` returns
    /// `v`, and `None` returns `Err("boom")`. Returns the batch, the parent
    /// `SUCCEED` payload of the `await_batch()` run, and the `.await` result.
    async fn run_map(
        items: Vec<Option<u32>>,
        completion: Config,
        max_concurrency: usize,
    ) -> (BatchResult<u32>, String, Result<Vec<u32>, OperationError>) {
        let body = |_child: DurableContext, item: Option<u32>, _idx: usize| async move {
            item.ok_or_else(|| BoxError::from("boom"))
        };
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(items.clone(), body)
            .max_concurrency(max_concurrency)
            .completion(completion.clone())
            .await_batch()
            .await
            .expect("await_batch never fails for a completed batch");
        let payload = parent_payload(&client);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let awaited = ctx
            .map(items, body)
            .max_concurrency(max_concurrency)
            .completion(completion)
            .await;
        (batch, payload, awaited)
    }

    /// Asserts the reason, the recorded reason string and the status of one
    /// row of the batch table.
    fn assert_row(
        batch: &BatchResult<u32>,
        payload: &str,
        reason: CompletionReason,
        status: BatchStatus,
    ) {
        assert_eq!(batch.reason, reason);
        assert!(
            payload.contains(reason.as_str()),
            "the parent SUCCEED payload must carry {}, got: {payload}",
            reason.as_str()
        );
        assert_eq!(batch.status(), status);
    }

    /// The message of an error's attached wire record.
    fn wire_message(err: &OperationError) -> Option<String> {
        err.wire()
            .and_then(crate::error::WireError::error_message)
            .map(str::to_owned)
    }

    /// The child-context kind of an error.
    fn child_kind(err: &OperationError) -> &ChildContextErrorKind {
        let OperationErrorKind::ChildContext(child) = err.kind() else {
            unreachable!("expected a child-context error, got {:?}", err.kind());
        };
        child.kind()
    }

    /// JS checks "every item finished" before `minSuccessful`, so a
    /// minimum equal to the item count records `ALL_COMPLETED`.
    #[tokio::test]
    async fn min_successful_equal_to_total_records_all_completed() {
        let (batch, payload, awaited) =
            run_map(vec![Some(1), Some(2)], Config::with_min_successful(2), 1).await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::AllCompleted,
            BatchStatus::Succeeded,
        );
        assert_eq!(awaited.expect("Ok"), vec![1, 2]);
    }

    #[tokio::test]
    async fn min_successful_before_last_item_records_min_successful() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), Some(2), Some(3)],
            Config::with_min_successful(1),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::MinSuccessfulReached,
            BatchStatus::Succeeded,
        );
        assert_eq!(batch.items.len(), 1);
        assert_eq!(awaited.expect("Ok"), vec![1]);
    }

    /// JS checks the failure tolerance before "every item finished".
    #[tokio::test]
    async fn tolerance_exceeded_on_last_item_records_failure() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), None],
            Config::with_tolerated_failure_count(0),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::FailureToleranceExceeded,
            BatchStatus::Failed,
        );
        let err = awaited.expect_err("the tolerance is exceeded");
        assert!(matches!(
            child_kind(&err),
            ChildContextErrorKind::ChildFailed
        ));
        assert_eq!(wire_message(&err).as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn min_not_reached_records_all_completed() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), None, Some(3)],
            Config::with_min_successful(3),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::AllCompleted,
            BatchStatus::Failed,
        );
        assert_eq!(awaited.expect("Ok"), vec![1, 3]);
    }

    #[tokio::test]
    async fn predicate_complete_succeeded() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), Some(2), Some(3)],
            Config::with_completion_predicate(|s| succeed_when(s.succeeded() >= 1)),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::CustomCompletionSucceeded,
            BatchStatus::Succeeded,
        );
        assert_eq!(batch.items.len(), 1);
        assert_eq!(awaited.expect("Ok"), vec![1]);
    }

    /// As JS's `throwIfError`, a batch the predicate completed as failed
    /// with no failed item raises a `BatchCompletionError`.
    #[tokio::test]
    async fn predicate_complete_failed_without_failed_item() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), Some(2)],
            Config::with_completion_predicate(|s| fail_when(s.settled() >= 1)),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::CustomCompletionFailed,
            BatchStatus::Failed,
        );
        assert_eq!(batch.items.len(), 1);
        let err = awaited.expect_err("the predicate failed the batch");
        assert!(matches!(
            child_kind(&err),
            ChildContextErrorKind::BatchCompletionFailed
        ));
        let wire = err.wire().expect("a wire record is attached");
        assert_eq!(wire.error_type(), Some("BatchCompletionError"));
        assert_eq!(
            wire.error_message(),
            Some(
                "batch completed as failed by the completion predicate (CUSTOM_COMPLETION_FAILED)"
            )
        );
        assert_eq!(child_kind(&err).to_string(), "batch completed as failed");
    }

    /// As JS's `throwIfError`, a batch the predicate completed as failed
    /// raises the first failed item's error.
    #[tokio::test]
    async fn predicate_complete_failed_with_failed_item() {
        let (batch, payload, awaited) = run_map(
            vec![None, Some(2)],
            Config::with_completion_predicate(|s| fail_when(s.failed() >= 1)),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::CustomCompletionFailed,
            BatchStatus::Failed,
        );
        assert_eq!(batch.items.len(), 1);
        let err = awaited.expect_err("the predicate failed the batch");
        assert!(matches!(
            child_kind(&err),
            ChildContextErrorKind::ChildFailed
        ));
        assert_eq!(wire_message(&err).as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn predicate_continue_records_all_completed() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), None],
            Config::with_completion_predicate(|_| CompletionDecision::Continue),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::AllCompleted,
            BatchStatus::Failed,
        );
        assert_eq!(awaited.expect("Ok"), vec![1]);
    }

    /// Item 0 fails and exceeds the tolerance while item 1 still runs. The
    /// batch records that first trigger; item 1's later success would meet
    /// `min_successful(1)`, but it never replaces the reason.
    #[tokio::test(start_paused = true)]
    async fn first_trigger_reason_wins_while_a_later_item_runs() {
        let body = |_child: DurableContext, item: Option<u32>, _idx: usize| async move {
            match item {
                Some(v) => {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    Ok(v)
                }
                None => Err(BoxError::from("boom")),
            }
        };
        let completion = Config::builder()
            .min_successful(1)
            .tolerated_failure_count(0)
            .build()
            .expect("valid completion config");
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![None, Some(2)], body)
            .max_concurrency(2)
            .completion(completion.clone())
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_row(
            &batch,
            &parent_payload(&client),
            CompletionReason::FailureToleranceExceeded,
            BatchStatus::Failed,
        );
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let err = ctx
            .map(vec![None, Some(2)], body)
            .max_concurrency(2)
            .completion(completion)
            .await
            .expect_err("the tolerance is exceeded");
        assert!(matches!(
            child_kind(&err),
            ChildContextErrorKind::ChildFailed
        ));
        assert_eq!(wire_message(&err).as_deref(), Some("boom"));
    }

    /// JS checks the predicate before "every item finished", so a predicate
    /// that completes at the last settle records the custom reason.
    #[tokio::test]
    async fn predicate_complete_on_last_settle_records_custom() {
        let (batch, payload, awaited) = run_map(
            vec![Some(1), Some(2)],
            Config::with_completion_predicate(|s| succeed_when(s.settled() == s.total_items())),
            1,
        )
        .await;
        assert_row(
            &batch,
            &payload,
            CompletionReason::CustomCompletionSucceeded,
            BatchStatus::Succeeded,
        );
        assert_eq!(awaited.expect("Ok"), vec![1, 2]);
    }

    /// Counts how many bodies of `map([0, 1, ..])` ran, per item.
    fn counting_body(
        counters: Arc<Vec<AtomicUsize>>,
    ) -> impl Fn(DurableContext, usize, usize) -> std::future::Ready<Result<usize, BoxError>>
    + Send
    + Sync
    + 'static {
        move |_child, item, _idx| {
            if let Some(counter) = counters.get(item) {
                counter.fetch_add(1, Ordering::SeqCst);
            }
            std::future::ready(Ok(item))
        }
    }

    fn counters(n: usize) -> Arc<Vec<AtomicUsize>> {
        Arc::new((0..n).map(|_| AtomicUsize::new(0)).collect())
    }

    fn counts(counters: &[AtomicUsize]) -> Vec<usize> {
        counters.iter().map(|c| c.load(Ordering::SeqCst)).collect()
    }

    /// As JS, a predicate that completes before any item starts resolves the
    /// batch at once with no item.
    #[tokio::test]
    async fn predicate_complete_before_first_dispatch_runs_no_body() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let ran = counters(2);
        let batch = ctx
            .map(vec![0_usize, 1], counting_body(Arc::clone(&ran)))
            .completion(Config::with_completion_predicate(|_| succeed_when(true)))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(counts(&ran), [0, 0]);
        assert!(batch.items.is_empty());
        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
        let parent_succeeds = client
            .recorded_updates()
            .iter()
            .filter(|u| u.parent_id().is_none() && matches!(u.action(), OperationAction::Succeed))
            .count();
        assert_eq!(parent_succeeds, 1);
    }

    #[tokio::test]
    async fn predicate_complete_at_dispatch_gate_stops_later_starts() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let ran = counters(3);
        let batch = ctx
            .map(vec![0_usize, 1, 2], counting_body(Arc::clone(&ran)))
            .max_concurrency(1)
            .completion(Config::with_completion_predicate(|s| {
                succeed_when(item_is(s, 0, BatchItemStatus::Succeeded))
            }))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(counts(&ran), [1, 0, 0]);
        assert_eq!(statuses(&batch), [(0, BatchItemStatus::Succeeded)]);
        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
    }

    /// A zero-item batch asks the predicate once (as JS does), and a
    /// `Complete(Failed)` decision fails `.await`.
    #[tokio::test]
    async fn empty_batch_predicate_failed_records_custom_failed() {
        let completion = Config::with_completion_predicate(|_| fail_when(true));
        let body = |_child: DurableContext, item: u32, _idx: usize| async move { Ok(item) };
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(Vec::<u32>::new(), body)
            .completion(completion.clone())
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(batch.reason, CompletionReason::CustomCompletionFailed);
        assert_eq!(batch.status(), BatchStatus::Failed);
        assert!(parent_payload(&client).contains("CUSTOM_COMPLETION_FAILED"));
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let err = ctx
            .map(Vec::<u32>::new(), body)
            .completion(completion)
            .await
            .expect_err("the predicate failed the empty batch");
        assert!(matches!(
            child_kind(&err),
            ChildContextErrorKind::BatchCompletionFailed
        ));
    }

    #[tokio::test]
    async fn empty_batch_predicate_continue_records_all_completed() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let batch = ctx
            .map(
                Vec::<u32>::new(),
                |_child, item, _idx| async move { Ok(item) },
            )
            .completion(Config::with_completion_predicate(move |s| {
                seen.fetch_add(1, Ordering::SeqCst);
                assert_eq!(s.total_items(), 0);
                CompletionDecision::Continue
            }))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(batch.reason, CompletionReason::AllCompleted);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "asked exactly once");
    }

    #[tokio::test]
    async fn min_successful_zero_one_item_records_all_completed() {
        let (batch, _payload, awaited) =
            run_map(vec![None], Config::with_min_successful(0), 1).await;
        assert_eq!(batch.reason, CompletionReason::AllCompleted);
        assert_eq!(batch.status(), BatchStatus::Failed);
        assert_eq!(awaited.expect("Ok"), Vec::<u32>::new());
    }

    #[tokio::test]
    async fn min_successful_zero_completes_at_first_settle() {
        let ran = Arc::new(AtomicUsize::new(0));
        let item_one_ran = Arc::clone(&ran);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![None, Some(1_u32)], move |_child, item, _idx| {
                let item_one_ran = Arc::clone(&item_one_ran);
                async move {
                    match item {
                        Some(v) => {
                            item_one_ran.fetch_add(1, Ordering::SeqCst);
                            Ok(v)
                        }
                        None => Err(BoxError::from("boom")),
                    }
                }
            })
            .max_concurrency(1)
            .completion(Config::with_min_successful(0))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        assert_eq!(statuses(&batch), [(0, BatchItemStatus::Failed)]);
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    /// Runs 50-or-fewer items of which the first `failures` fail, under a
    /// failure percentage, and returns the reason.
    async fn percentage_reason(total: usize, failures: usize, pct: f64) -> CompletionReason {
        let items: Vec<Option<u32>> = (0..total)
            .map(|i| if i < failures { None } else { Some(1) })
            .collect();
        let (batch, _payload, _awaited) =
            run_map(items, Config::with_tolerated_failure_percentage(pct), 1).await;
        batch.reason
    }

    /// Cross-multiplication keeps 7 failures of 50 at 14.0 within the
    /// tolerance, where JS's division exceeds it.
    #[tokio::test]
    async fn percentage_boundary_is_exact() {
        assert_ne!(
            percentage_reason(50, 7, 14.0).await,
            CompletionReason::FailureToleranceExceeded
        );
        assert_eq!(
            percentage_reason(50, 8, 14.0).await,
            CompletionReason::FailureToleranceExceeded
        );
    }

    #[tokio::test]
    async fn fractional_percentage() {
        assert_ne!(
            percentage_reason(8, 1, 12.5).await,
            CompletionReason::FailureToleranceExceeded
        );
        assert_eq!(
            percentage_reason(8, 2, 12.5).await,
            CompletionReason::FailureToleranceExceeded
        );
    }

    #[test]
    fn percentage_out_of_range_is_rejected() {
        for pct in [100.5, -1.0, f64::NAN] {
            let built = Config::builder()
                .tolerated_failure_percentage(pct)
                .build()
                .expect_err("out of range");
            assert!(
                built
                    .to_string()
                    .contains("tolerated_failure_percentage must be 0-100"),
                "{pct}: {built}"
            );
            let validated = Config::with_tolerated_failure_percentage(pct)
                .validate()
                .expect_err("out of range");
            assert!(
                validated.contains("tolerated_failure_percentage must be 0-100"),
                "{pct}: {validated}"
            );
        }
        // Infinity is not finite, and the bounds themselves are valid.
        assert!(
            Config::with_tolerated_failure_percentage(f64::INFINITY)
                .validate()
                .is_err()
        );
        for pct in [0.0, 100.0] {
            assert!(
                Config::with_tolerated_failure_percentage(pct)
                    .validate()
                    .is_ok()
            );
        }
    }

    /// The statuses of a predicate's view, as `(name, status)` pairs.
    type ViewLog = Arc<std::sync::Mutex<Vec<Vec<(Option<String>, Option<BatchItemStatus>)>>>>;

    fn view_of(stats: &BatchStats<'_>) -> Vec<(Option<String>, Option<BatchItemStatus>)> {
        stats
            .items()
            .iter()
            .map(|i| (i.name.clone(), i.status))
            .collect()
    }

    fn named(
        entries: &[(&str, Option<BatchItemStatus>)],
    ) -> Vec<(Option<String>, Option<BatchItemStatus>)> {
        entries
            .iter()
            .map(|(name, status)| (Some((*name).to_owned()), *status))
            .collect()
    }

    /// A branch that settles at once with `value`.
    fn ok_branch(name: &str, value: u32) -> crate::future::Branch<u32> {
        crate::future::Branch::new(name, move |_ctx| async move { Ok(value) })
    }

    /// A branch whose step never returns, so the branch stays running.
    fn pending_branch(name: &str) -> crate::future::Branch<u32> {
        crate::future::Branch::new(name, |ctx| async move {
            Ok(ctx
                .step(|_| std::future::pending::<Result<u32, BoxError>>())
                .name("forever")
                .await?)
        })
    }

    /// As JS, a predicate can complete on items 1 and 2 without waiting for
    /// a slower item 0.
    #[tokio::test]
    async fn predicate_completes_on_later_items_before_a_slow_earlier_item() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.parallel(vec![
                pending_branch("slow"),
                ok_branch("b", 2),
                ok_branch("c", 3),
            ])
            .completion(Config::with_completion_predicate(|s| {
                succeed_when(
                    item_is(s, 1, BatchItemStatus::Succeeded)
                        && item_is(s, 2, BatchItemStatus::Succeeded),
                )
            }))
            .await_batch(),
        )
        .await
        .expect("the predicate does not wait for item 0")
        .expect("await_batch returns the batch");
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Started),
                (1, BatchItemStatus::Succeeded),
                (2, BatchItemStatus::Succeeded)
            ]
        );
        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
    }

    /// The predicate sees every item by index with its caller name, and is
    /// asked at each dispatch gate and after each settle.
    #[tokio::test]
    async fn predicate_sees_names_and_not_started_items() {
        let log: ViewLog = Arc::default();
        let writer = Arc::clone(&log);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.parallel(vec![
            ok_branch("a", 1),
            ok_branch("b", 2),
            ok_branch("c", 3),
        ])
        .max_concurrency(1)
        .completion(Config::with_completion_predicate(move |s| {
            writer.lock().expect("no poisoned lock").push(view_of(s));
            CompletionDecision::Continue
        }))
        .await_batch()
        .await
        .expect("await_batch returns the batch");
        let ok = Some(BatchItemStatus::Succeeded);
        let first = named(&[("a", None), ("b", None), ("c", None)]);
        let after_a = named(&[("a", ok), ("b", None), ("c", None)]);
        let after_b = named(&[("a", ok), ("b", ok), ("c", None)]);
        let after_c = named(&[("a", ok), ("b", ok), ("c", ok)]);
        assert_eq!(
            *log.lock().expect("no poisoned lock"),
            vec![
                first,
                after_a.clone(),
                after_a,
                after_b.clone(),
                after_b,
                after_c
            ]
        );
    }

    #[tokio::test]
    async fn predicate_sees_a_running_item_as_started() {
        let log: ViewLog = Arc::default();
        let writer = Arc::clone(&log);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.parallel(vec![ok_branch("a", 1), pending_branch("b")])
                .completion(Config::with_completion_predicate(move |s| {
                    writer.lock().expect("no poisoned lock").push(view_of(s));
                    succeed_when(s.succeeded() >= 1)
                }))
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for item b")
        .expect("await_batch returns the batch");
        assert_eq!(
            log.lock().expect("no poisoned lock").last().cloned(),
            Some(named(&[
                ("a", Some(BatchItemStatus::Succeeded)),
                ("b", Some(BatchItemStatus::Started))
            ]))
        );
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started)
            ]
        );
    }

    #[tokio::test]
    async fn predicate_counts_follow_settle_order() {
        type CountLog = Arc<std::sync::Mutex<Vec<(usize, usize, usize)>>>;
        let log: CountLog = Arc::default();
        let writer = Arc::clone(&log);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        tokio::time::timeout(
            EARLY_BOUND,
            ctx.parallel(vec![
                pending_branch("slow"),
                ok_branch("b", 2),
                ok_branch("c", 3),
            ])
            .completion(Config::with_completion_predicate(move |s| {
                if s.settled() > 0 {
                    writer.lock().expect("no poisoned lock").push((
                        s.succeeded(),
                        s.failed(),
                        s.settled(),
                    ));
                }
                succeed_when(s.settled() == 2)
            }))
            .await_batch(),
        )
        .await
        .expect("the counts do not wait for item 0")
        .expect("await_batch returns the batch");
        assert_eq!(
            *log.lock().expect("no poisoned lock"),
            vec![(1, 0, 1), (2, 0, 2)]
        );
    }

    /// As JS on resume, recorded items settle first, in index order, and
    /// every recorded outcome is kept even after the predicate completed.
    #[tokio::test]
    async fn resumed_batch_keeps_recorded_outcomes() {
        let log = CheckpointLog::from_records(vec![
            succeeded_record("2", "1"),
            succeeded_record("3", "2"),
            failed_record("4", "recorded failure"),
        ]);
        let (ctx, _client) = test_ctx_with_client(log);
        let ran = Arc::new(AtomicUsize::new(0));
        let body_ran = Arc::clone(&ran);
        let batch = ctx
            .map(vec![0_u32, 1, 2], move |_child, _item, _idx| {
                body_ran.fetch_add(1, Ordering::SeqCst);
                async move { Ok(999_u32) }
            })
            .max_concurrency(2)
            .completion(Config::with_completion_predicate(|s| {
                succeed_when(s.succeeded() >= 1)
            }))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Succeeded),
                (2, BatchItemStatus::Failed)
            ]
        );
        assert_eq!(batch.results(), [&1, &2]);
        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
        assert_eq!(ran.load(Ordering::SeqCst), 0, "no body ran");
    }

    /// Once a threshold fired, a later settle that meets another threshold
    /// leaves the reason unchanged.
    #[test]
    fn first_threshold_trigger_wins_across_settles() {
        let cfg = Config::builder()
            .min_successful(1)
            .tolerated_failure_count(0)
            .build()
            .expect("valid completion config");

        let mut tracker = CompletionTracker::new(vec![None; 3]);
        tracker.settle(&cfg, 3, 0, BatchItemStatus::Failed);
        assert_eq!(
            tracker.stop_reason,
            Some(CompletionReason::FailureToleranceExceeded)
        );
        tracker.settle(&cfg, 3, 1, BatchItemStatus::Succeeded);
        assert_eq!(
            tracker.stop_reason,
            Some(CompletionReason::FailureToleranceExceeded)
        );

        let mut tracker = CompletionTracker::new(vec![None; 3]);
        tracker.settle(&cfg, 3, 0, BatchItemStatus::Succeeded);
        assert_eq!(
            tracker.stop_reason,
            Some(CompletionReason::MinSuccessfulReached)
        );
        tracker.settle(&cfg, 3, 1, BatchItemStatus::Failed);
        assert_eq!(
            tracker.stop_reason,
            Some(CompletionReason::MinSuccessfulReached)
        );
    }

    /// A tracker that stopped evaluating, as on a rebuild of a completed
    /// batch, updates its counts and view but runs no threshold and never
    /// calls the predicate.
    #[test]
    fn stopped_tracker_settles_without_evaluating() {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let predicate = Config::with_completion_predicate(move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            CompletionDecision::Complete(CompletionOutcome::Failed)
        });
        let threshold = Config::with_tolerated_failure_count(0);

        for cfg in [&predicate, &threshold] {
            let mut tracker = CompletionTracker::new(vec![None; 2]);
            tracker.stop_evaluating();
            tracker.settle(cfg, 2, 0, BatchItemStatus::Failed);
            tracker.settle(cfg, 2, 1, BatchItemStatus::Succeeded);
            assert_eq!(tracker.stop_reason, None);
            assert_eq!((tracker.success_count, tracker.failure_count), (1, 1));
            assert_eq!(
                tracker
                    .items
                    .iter()
                    .map(|item| item.status)
                    .collect::<Vec<_>>(),
                [
                    Some(BatchItemStatus::Failed),
                    Some(BatchItemStatus::Succeeded)
                ]
            );
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the predicate is never called"
        );
    }

    /// Once the predicate completed the batch, the SDK does not ask it
    /// again, so a later decision never replaces the first.
    #[test]
    fn first_predicate_decision_wins_across_settles() {
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let cfg = Config::with_completion_predicate(move |s| {
            seen.fetch_add(1, Ordering::SeqCst);
            match s.settled() {
                1 => CompletionDecision::Complete(CompletionOutcome::Succeeded),
                2 => CompletionDecision::Complete(CompletionOutcome::Failed),
                _ => CompletionDecision::Continue,
            }
        });
        let mut tracker = CompletionTracker::new(vec![None; 3]);
        tracker.settle(&cfg, 3, 0, BatchItemStatus::Succeeded);
        tracker.settle(&cfg, 3, 1, BatchItemStatus::Succeeded);
        assert_eq!(
            tracker.stop_reason,
            Some(CompletionReason::CustomCompletionSucceeded)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the predicate ran once");
        // A dispatch gate after the trigger asks nothing either.
        assert!(tracker.dispatch_gate(&cfg));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn batch_stats_new_counts_items() {
        let items = [
            CompletionItem::new(0, None, Some(BatchItemStatus::Succeeded)),
            CompletionItem::new(1, None, Some(BatchItemStatus::Failed)),
            CompletionItem::new(2, None, Some(BatchItemStatus::Started)),
            CompletionItem::new(3, None, None),
        ];
        let stats = BatchStats::new(&items);
        assert_eq!(stats.succeeded(), 1);
        assert_eq!(stats.failed(), 1);
        assert_eq!(stats.settled(), 2);
        assert_eq!(stats.total_items(), 4);
        assert_eq!(stats.items().len(), 4);
    }

    /// All six orders of settling items 0, 1 and 2.
    const SETTLE_ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    fn abc() -> Vec<Option<String>> {
        vec![Some("a".into()), Some("b".into()), Some("c".into())]
    }

    /// A monotonic predicate records the same reason in every settle order.
    #[test]
    fn monotonic_predicate_records_the_same_outcome_in_every_settle_order() {
        for order in SETTLE_ORDERS {
            let completes = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&completes);
            let cfg = Config::with_completion_predicate(move |s| {
                let decision = succeed_when(
                    item_is(s, 1, BatchItemStatus::Succeeded)
                        && item_is(s, 2, BatchItemStatus::Succeeded),
                );
                if decision != CompletionDecision::Continue {
                    seen.fetch_add(1, Ordering::SeqCst);
                }
                decision
            });
            let mut tracker = CompletionTracker::new(abc());
            for index in order {
                tracker.settle(&cfg, 3, index, BatchItemStatus::Succeeded);
            }
            assert_eq!(
                tracker.stop_reason,
                Some(CompletionReason::CustomCompletionSucceeded),
                "order {order:?}"
            );
            assert_eq!(completes.load(Ordering::SeqCst), 1, "order {order:?}");
        }
    }

    /// The JS counterexample: a predicate that is not monotonic can complete
    /// in one settle order and never in another. The SDK does not reject it.
    #[test]
    fn non_monotonic_predicate_depends_on_settle_order() {
        let cfg = Config::with_completion_predicate(|s| {
            succeed_when(s.succeeded() == 1 && s.failed() == 0)
        });

        let mut order_a = CompletionTracker::new(abc());
        order_a.settle(&cfg, 3, 0, BatchItemStatus::Succeeded);
        order_a.settle(&cfg, 3, 1, BatchItemStatus::Failed);
        assert_eq!(
            order_a.stop_reason,
            Some(CompletionReason::CustomCompletionSucceeded)
        );

        let mut order_b = CompletionTracker::new(abc());
        order_b.settle(&cfg, 3, 1, BatchItemStatus::Failed);
        order_b.settle(&cfg, 3, 0, BatchItemStatus::Succeeded);
        assert_eq!(order_b.stop_reason, None);
    }

    #[test]
    fn tracker_marks_dispatched_items_started() {
        let cfg = Config::default();
        let mut tracker = CompletionTracker::new(vec![Some("a".into()), Some("b".into())]);
        let view = |tracker: &CompletionTracker| -> Vec<Option<BatchItemStatus>> {
            tracker.items.iter().map(|i| i.status).collect()
        };
        assert_eq!(view(&tracker), [None, None]);
        tracker.mark_dispatched(0);
        assert_eq!(view(&tracker), [Some(BatchItemStatus::Started), None]);
        tracker.settle(&cfg, 2, 0, BatchItemStatus::Succeeded);
        assert_eq!(view(&tracker), [Some(BatchItemStatus::Succeeded), None]);
        tracker.mark_dispatched(1);
        assert_eq!(
            view(&tracker),
            [
                Some(BatchItemStatus::Succeeded),
                Some(BatchItemStatus::Started)
            ]
        );
        assert_eq!(
            tracker.items[1].name.as_deref(),
            Some("b"),
            "the view keeps the caller name"
        );
    }

    /// A `map` item has a caller name only when a namer is set, and an
    /// empty name is `Some("")`.
    #[tokio::test]
    async fn predicate_sees_map_namer_names() {
        let log: ViewLog = Arc::default();
        let writer = Arc::clone(&log);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.map(vec![0_u32, 1], |_child, item, _idx| async move { Ok(item) })
            .item_namer(|i| {
                if i == 0 {
                    String::new()
                } else {
                    format!("n{i}")
                }
            })
            .completion(Config::with_completion_predicate(move |s| {
                writer.lock().expect("no poisoned lock").push(view_of(s));
                CompletionDecision::Continue
            }))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        let names: Vec<Option<String>> = log.lock().expect("no poisoned lock")[0]
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        assert_eq!(names, [Some(String::new()), Some("n1".to_owned())]);

        let log: ViewLog = Arc::default();
        let writer = Arc::clone(&log);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.map(vec![0_u32], |_child, item, _idx| async move { Ok(item) })
            .completion(Config::with_completion_predicate(move |s| {
                writer.lock().expect("no poisoned lock").push(view_of(s));
                CompletionDecision::Continue
            }))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        assert_eq!(log.lock().expect("no poisoned lock")[0], [(None, None)]);
    }

    /// The item namer runs once per input item, on the sequential and the
    /// concurrent path. A stateful namer gives a different name on each
    /// call, so the predicate view, the child START records and the result
    /// agree only when every consumer reads the one computed name.
    #[tokio::test]
    async fn item_namer_runs_once_per_item_and_names_agree() {
        for concurrency in [1, 3] {
            let calls = Arc::new(AtomicUsize::new(0));
            let counter = Arc::clone(&calls);
            let log: ViewLog = Arc::default();
            let writer = Arc::clone(&log);
            let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
            let batch = ctx
                .map(
                    vec![0_u32, 1, 2],
                    |_child, item, _idx| async move { Ok(item) },
                )
                .item_namer(move |i| {
                    let call = counter.fetch_add(1, Ordering::SeqCst);
                    format!("n{i}-call{call}")
                })
                .max_concurrency(concurrency)
                .completion(Config::with_completion_predicate(move |s| {
                    writer.lock().expect("no poisoned lock").push(view_of(s));
                    CompletionDecision::Continue
                }))
                .await_batch()
                .await
                .expect("await_batch returns the batch");

            assert_eq!(
                calls.load(Ordering::SeqCst),
                3,
                "the namer runs once per item (concurrency {concurrency})"
            );
            let result_names: Vec<Option<String>> =
                batch.items.iter().map(|i| Some(i.name.clone())).collect();
            let view_names: Vec<Option<String>> = log
                .lock()
                .expect("no poisoned lock")
                .last()
                .expect("the predicate ran")
                .iter()
                .map(|(name, _)| name.clone())
                .collect();
            let mut start_names = child_start_names(&client);
            start_names.sort();
            assert_eq!(view_names, result_names, "concurrency {concurrency}");
            assert_eq!(start_names, result_names, "concurrency {concurrency}");
        }
    }

    /// On the sequential path, a resumed batch replays its recorded-terminal
    /// item without a dispatch gate: the replay pass is not a live dispatch.
    /// So the predicate is first asked after item 0's recorded settle, and
    /// the live items still get the IDs the first run would give them.
    #[tokio::test]
    async fn sequential_replay_asks_no_gate_for_a_recorded_item() {
        let log = CheckpointLog::from_records(vec![succeeded_record("2", "1")]);
        let (ctx, client) = test_ctx_with_client(log);
        let views: ViewLog = Arc::default();
        let writer = Arc::clone(&views);
        let ran = Arc::new(AtomicUsize::new(0));
        let body_ran = Arc::clone(&ran);
        let batch = ctx
            .map(vec![0_u32, 1, 2], move |_child, item, _idx| {
                body_ran.fetch_add(1, Ordering::SeqCst);
                async move { Ok(item + 10) }
            })
            .max_concurrency(1)
            .completion(Config::with_completion_predicate(move |s| {
                writer.lock().expect("no poisoned lock").push(view_of(s));
                CompletionDecision::Continue
            }))
            .await_batch()
            .await
            .expect("await_batch returns the batch");

        let ok = Some(BatchItemStatus::Succeeded);
        let after_0 = vec![(None, ok), (None, None), (None, None)];
        let after_1 = vec![(None, ok), (None, ok), (None, None)];
        let after_2 = vec![(None, ok), (None, ok), (None, ok)];
        assert_eq!(
            *views.lock().expect("no poisoned lock"),
            vec![after_0.clone(), after_0, after_1.clone(), after_1, after_2],
            "settle of item 0, gate and settle of item 1, gate and settle of item 2"
        );
        assert_eq!(batch.results(), [&1, &11, &12]);
        assert_eq!(ran.load(Ordering::SeqCst), 2, "only the live items ran");

        let started: Vec<String> = client
            .recorded_updates()
            .iter()
            .filter(|u| matches!(u.action(), OperationAction::Start) && u.parent_id().is_some())
            .map(|u| u.id().to_owned())
            .collect();
        assert_eq!(
            started,
            [
                crate::engine::compute_wire_id_public("3"),
                crate::engine::compute_wire_id_public("4")
            ],
            "the live items keep the IDs the first run gives them"
        );
    }

    /// A custom completion predicate ends a live map batch early, dispatch
    /// stops, and the recorded batch payload carries the
    /// `CUSTOM_COMPLETION_SUCCEEDED` reason.
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
                    .completion_predicate(|stats| succeed_when(stats.settled() >= 2))
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("a predicate-completed batch must not become an operation error");

        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
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
            parent_success_payload.contains("CUSTOM_COMPLETION_SUCCEEDED"),
            "recorded payload must carry the predicate reason, got: {parent_success_payload}"
        );
    }

    /// The predicate receives every item's status, so it can key off WHICH
    /// item settled and how, not just the counts.
    #[tokio::test]
    async fn completion_predicate_observes_settled_outcomes() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        // Stop as soon as item index 1 is observed to have failed.
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
                    .completion_predicate(|stats| {
                        succeed_when(item_is(stats, 1, BatchItemStatus::Failed))
                    })
                    .build()
                    .expect("valid completion config"),
            )
            .await_batch()
            .await
            .expect("a predicate-completed batch must not become an operation error");

        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
        assert_eq!(
            batch.items.len(),
            2,
            "items 2 and 3 must never start once the predicate fires"
        );
        assert_eq!(batch.failure_count(), 1);
    }

    /// A zero-item batch asks the predicate once, on an empty view, as JS
    /// does: a `Complete(Succeeded)` decision records
    /// `CUSTOM_COMPLETION_SUCCEEDED`.
    #[tokio::test]
    async fn empty_batch_predicate_succeeded_records_custom_succeeded() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .map(
                Vec::<u32>::new(),
                |_child, item, _idx| async move { Ok(item) },
            )
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_completion_predicate(|_| {
                    CompletionDecision::Complete(CompletionOutcome::Succeeded)
                }),
            )
            .await_batch()
            .await
            .expect("an empty batch completes immediately");

        assert_eq!(batch.reason, CompletionReason::CustomCompletionSucceeded);
        assert!(batch.items.is_empty());
    }

    /// Both custom reasons survive the checkpoint round-trip, through the
    /// tolerant and the strict parser.
    #[test]
    fn custom_reasons_round_trip_wire() {
        for reason in [
            CompletionReason::CustomCompletionSucceeded,
            CompletionReason::CustomCompletionFailed,
        ] {
            assert_eq!(CompletionReason::from_wire(reason.as_str()), reason);
            assert_eq!(
                CompletionReason::try_from_wire(reason.as_str()),
                Some(reason)
            );
        }
    }

    /// The value earlier previews wrote is not parsed.
    #[test]
    fn predicate_matched_is_not_parsed() {
        assert_eq!(CompletionReason::try_from_wire("PREDICATE_MATCHED"), None);
    }

    /// `status()` follows a custom outcome whatever the items did.
    #[test]
    fn status_follows_custom_outcome() {
        let failed_item = || {
            vec![BatchItem::<u32>::new(
                0,
                String::new(),
                BatchItemStatus::Failed,
                None,
                None,
                None,
            )]
        };
        let ok_item = || {
            vec![BatchItem::<u32>::new(
                0,
                String::new(),
                BatchItemStatus::Succeeded,
                Some(1),
                None,
                None,
            )]
        };
        assert_eq!(
            BatchResult::new(failed_item(), CompletionReason::CustomCompletionSucceeded).status(),
            BatchStatus::Succeeded
        );
        assert_eq!(
            BatchResult::new(ok_item(), CompletionReason::CustomCompletionFailed).status(),
            BatchStatus::Failed
        );
        assert_eq!(
            BatchResult::new(ok_item(), CompletionReason::MinSuccessfulReached).status(),
            BatchStatus::Succeeded
        );
        assert_eq!(
            BatchResult::new(failed_item(), CompletionReason::AllCompleted).status(),
            BatchStatus::Failed
        );
    }

    // Gap 1: BatchResult/BatchItem public API tests

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
                    Some("Error".into()),
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
        assert_eq!(errors[0].error_type, Some("Error"));
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
        assert_eq!(result.errors(), []);
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
            CompletionReason::CustomCompletionSucceeded,
            CompletionReason::CustomCompletionFailed,
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
                    Some("Error".into()),
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
        assert_eq!(errors[0].error_type, Some("Error"));
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

    /// A failed item's COMPLETE wire failure record — type, message,
    /// `error_data`, and `stack_trace` — must survive the checkpoint
    /// payload round-trip (`from_batch_result` → `to_batch_result`), so
    /// error identity is preserved across suspension and replay rather
    /// than being discarded on the way back in.
    #[tokio::test]
    async fn batch_checkpoint_round_trips_error_type_and_message() {
        let mut failed_item = BatchItem::new(
            1,
            "boom".to_owned(),
            BatchItemStatus::Failed,
            None,
            Some("it broke".to_owned()),
            Some("CustomFailure".to_owned()),
        );
        failed_item.error_data = Some(r#"{"attempt":3}"#.to_owned());
        failed_item.stack_trace = vec!["frame-a".to_owned(), "frame-b".to_owned()];
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
                failed_item,
            ],
            CompletionReason::FailureToleranceExceeded,
        );

        let json = Arc::new(crate::serdes::JsonSerdes);
        let (payload, _rebuilt) = from_batch_result(
            &test_ctx(CheckpointLog::empty()),
            original,
            &json,
            "parent-wire",
            "arn:test",
        )
        .await
        .expect("serializing a batch result must succeed");
        let replayed: BatchResult<i32> = to_batch_result(
            &test_ctx(CheckpointLog::empty()),
            payload,
            &json,
            "parent-wire",
            "arn:test",
        )
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
        assert_eq!(failed.error_data.as_deref(), Some(r#"{"attempt":3}"#));
        assert_eq!(
            failed.stack_trace,
            ["frame-a".to_owned(), "frame-b".to_owned()]
        );
        let errors = replayed.errors();
        assert_eq!(errors[0].error_type, Some("CustomFailure"));
        assert_eq!(errors[0].message, "it broke");
        let ok = &replayed.items[0];
        assert_eq!(ok.result, Some(7));
        assert!(ok.error_type.is_none());
        assert!(ok.error_data.is_none());
        assert_eq!(ok.stack_trace, Vec::<String>::new());
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
        let json = Arc::new(crate::serdes::JsonSerdes);
        let (payload, _rebuilt) = from_batch_result(
            &test_ctx(CheckpointLog::empty()),
            original,
            &json,
            "parent-wire",
            "arn:test",
        )
        .await
        .expect("serializing a batch result must succeed");
        assert_eq!(
            payload.results.first().map(|r| r.err_type.as_str()),
            Some("Error")
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
        let json = Arc::new(crate::serdes::JsonSerdes);
        let replayed: BatchResult<i32> = to_batch_result(
            &test_ctx(CheckpointLog::empty()),
            payload,
            &json,
            "parent-wire",
            "arn:test",
        )
        .await
        .expect("deserializing the payload must succeed");
        let item = replayed.items.first().expect("one item");
        assert_eq!(item.error_message.as_deref(), Some("boom"));
        assert_eq!(item.error_type, None);
        // A payload written before `errData`/`stackTrace` existed replays
        // with an empty failure payload rather than a fabricated one.
        assert_eq!(item.error_data, None);
        assert_eq!(item.stack_trace, Vec::<String>::new());
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
                    due_at: None,
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
        assert_eq!(errors[0].error_type, Some("ChildFnError"));
    }

    // Gap 3: NestingMode tests

    #[test]
    fn nesting_mode_default_is_normal() {
        assert_eq!(NestingMode::default(), NestingMode::Normal);
    }

    #[test]
    fn nesting_mode_equality() {
        assert_ne!(NestingMode::Flat, NestingMode::Normal);
    }

    // Gap 5: Completion reason wire strings

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
        assert_eq!(
            CompletionReason::CustomCompletionSucceeded.as_str(),
            "CUSTOM_COMPLETION_SUCCEEDED"
        );
        assert_eq!(
            CompletionReason::CustomCompletionFailed.as_str(),
            "CUSTOM_COMPLETION_FAILED"
        );
    }

    // Bug 1: Early-stopping race: stopped re-check after acquire

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

    // Bug 3: FLAT nesting ParentId points to batch parent

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
        let flat_child = ctx.new_flat_child(&child_positional, &batch_wire);
        // Operations inside the flat child should report the batch parent
        // as their parent wire ID.
        assert_eq!(
            flat_child.parent_wire_id(),
            Some(batch_wire.as_str()),
            "flat child must report the batch parent's wire ID"
        );
    }

    // Bug 5: max_concurrency=0 validation error

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

    // Checkpoint serialization tests

    /// A token-validating execution client that rejects checkpoint calls
    /// with stale tokens: simulating the service's actual behavior.
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
                            crate::client::ExecutionClientError,
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
                    // Stale token: the real service would return an error.
                    let mut stale = self
                        .stale_token_count
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *stale += 1;
                    return Err(crate::client::ExecutionClientError::non_retryable(format!(
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
                    checkpoint_token: Some(new_token),
                    updated_operations: Vec::new(),
                    next_marker: None,
                })
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::client::GetStateOutput,
                            crate::client::ExecutionClientError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(crate::client::GetStateOutput::new(Vec::new(), None)) })
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
        // latest token: ensuring it succeeds.
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

        // The public API propagates failures as errors: that's correct
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

    /// Drives `run` through the invocation driver over a fresh recording
    /// context, asserts that the driver returned a panic, and returns its
    /// `&'static str` payload and whether any recorded update has action `FAIL`.
    async fn panic_of<F, Fut>(run: F) -> (Option<&'static str>, bool)
    where
        F: FnOnce(DurableContext) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let handler = run(ctx);
        let outcome = crate::driver::drive_invocation(
            async move {
                handler.await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            unreachable!("the item panic must reach the invocation driver, got {outcome:?}");
        };
        let wrote_fail = client
            .recorded_updates()
            .iter()
            .any(|u| matches!(u.action(), OperationAction::Fail));
        (
            crate::task::test_support::str_payload(payload.as_ref()),
            wrote_fail,
        )
    }

    /// A panic in a map item fails the invocation with its original
    /// payload, and the batch records no failure for it.
    #[tokio::test]
    async fn map_item_panic_propagates_payload() {
        let (message, wrote_fail) = panic_of(|ctx| async move {
            let _ = ctx
                .map(vec![0_i32, 1], |_child, _item: i32, idx| async move {
                    assert!(idx != 1, "boom in map branch");
                    Ok(idx)
                })
                .await;
        })
        .await;
        assert_eq!(message, Some("boom in map branch"));
        assert!(!wrote_fail, "a panic records no FAIL");
    }

    /// A panic in a parallel branch fails the invocation with its original
    /// payload, and the batch records no failure for it.
    #[tokio::test]
    #[expect(clippy::panic)] // reason: the branch panic under test is the behavior being asserted
    async fn parallel_branch_panic_propagates_payload() {
        use crate::future::Branch;

        let (message, wrote_fail) = panic_of(|ctx| async move {
            let _ = ctx
                .parallel(vec![
                    Branch::new("steady", |_ctx| async move { Ok(1_i32) }),
                    Branch::new(
                        "boom",
                        |_ctx| async move { panic!("boom in parallel branch") },
                    ),
                ])
                .await;
        })
        .await;
        assert_eq!(message, Some("boom in parallel branch"));
        assert!(!wrote_fail, "a panic records no FAIL");
    }

    /// A child `FAIL` that an earlier release wrote for a panicked map item
    /// still replays as a failed item, without running the item body.
    #[tokio::test]
    async fn map_item_panic_record_from_an_earlier_release_still_replays() {
        fn panic_record(positional_id: &str) -> (String, CheckpointRecord) {
            let (wire_id, mut record) = failed_record(positional_id, "boom in map branch");
            record.error_data = None;
            record.stack_trace = None;
            (wire_id, record)
        }
        let log = || {
            let (parent_wire, mut parent) = succeeded_record("1", "null");
            parent.status = CheckpointStatus::Started;
            parent.result = None;
            CheckpointLog::from_records(vec![
                (parent_wire, parent),
                succeeded_record("2", "1"),
                panic_record("3"),
            ])
        };
        let ran = Arc::new(AtomicBool::new(false));
        let body = |ran: Arc<AtomicBool>| {
            move |_child: DurableContext, item: i32, _idx: usize| {
                let ran = Arc::clone(&ran);
                async move {
                    ran.store(true, Ordering::SeqCst);
                    Ok(item)
                }
            }
        };

        let (ctx, _client) = test_ctx_with_client(log());
        let err = ctx
            .map(vec![0_i32, 1], body(Arc::clone(&ran)))
            .await
            .expect_err("the replayed failed item fails the batch");
        let OperationErrorKind::ChildContext(child_err) = err.kind() else {
            unreachable!("expected a child-context error, got {:?}", err.kind());
        };
        assert!(
            matches!(child_err.kind(), ChildContextErrorKind::ChildFailed),
            "expected ChildFailed, got {:?}",
            child_err.kind()
        );
        assert!(
            format!("{err:#}").contains("boom in map branch"),
            "unexpected error: {err:#}"
        );

        let (ctx, _client) = test_ctx_with_client(log());
        let batch = ctx
            .map(vec![0_i32, 1], body(Arc::clone(&ran)))
            .await_batch()
            .await
            .expect("await_batch reports the replayed items");
        let failed = batch
            .items
            .iter()
            .find(|i| i.index == 1)
            .expect("item 1 is in the batch");
        assert_eq!(failed.status, BatchItemStatus::Failed);
        assert_eq!(failed.error_type.as_deref(), Some("ChildFnError"));
        assert!(
            !ran.load(Ordering::SeqCst),
            "no item body may run on replay"
        );
    }

    /// Polls `run` once on a thread with no Tokio runtime and asserts that
    /// the batch resolved to the no-runtime error, instead of panicking.
    fn assert_batch_no_runtime_error<F, Fut>(log: CheckpointLog, run: F)
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Vec<i32>, OperationError>>,
    {
        let (ctx, _client) = test_ctx_with_client(log);
        let polled = crate::task::test_support::poll_once_without_runtime(move || run(ctx));
        let std::task::Poll::Ready(Err(err)) = polled else {
            unreachable!("expected an immediate error, got {polled:?}");
        };
        assert!(
            format!("{err:#}").contains(crate::task::NO_RUNTIME_MESSAGE),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn map_without_runtime_returns_error() {
        assert_batch_no_runtime_error(CheckpointLog::empty(), |ctx| {
            ctx.map(
                vec![0_i32],
                |_child, item: i32, _idx| async move { Ok(item) },
            )
            .into_future()
        });
    }

    #[test]
    fn parallel_without_runtime_returns_error() {
        use crate::future::Branch;

        assert_batch_no_runtime_error(CheckpointLog::empty(), |ctx| {
            ctx.parallel(vec![Branch::new("one", |_ctx| async move { Ok(1_i32) })])
                .into_future()
        });
    }

    /// The reconstruction of a `ReplayChildren` item needs a task too.
    #[test]
    fn map_reconstruction_without_runtime_returns_error() {
        let log = CheckpointLog::from_records(vec![replay_children_success_record("2")]);
        assert_batch_no_runtime_error(log, |ctx| {
            ctx.map(
                vec![1_i32],
                |_child, item: i32, _idx| async move { Ok(item) },
            )
            .into_future()
        });
    }

    #[tokio::test]
    #[expect(clippy::panic)] // reason: the reconstruction panic under test is the behavior being asserted
    async fn map_replay_children_reconstruction_panic_unwinds_without_fail_checkpoint() {
        // A child recorded as terminal SUCCESS with `replay_children = true`
        // is re-executed on replay to reconstruct its oversized result. If
        // that reconstruction panics, the panic must unwind the coordinator
        // with the original payload, preserving the recorded success
        // history for the next attempt, NOT be graded as a branch failure
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
    #[expect(clippy::panic)] // reason: the reconstruction panic under test is the behavior being asserted
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

    // Serialization-model equivalence tests
    //
    // These prove the point of the normalization: ONE `Serdes`
    // implementation, one that implements only the JSON-string transform
    // methods, behaves identically as a map item serdes, a parallel item
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

        // step
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

        // map items
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

        // parallel branches
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
    /// The probe payload is a bare `String`, the one case where the old split
    /// was visible, and the fixture's wire form is deliberately NOT JSON, so
    /// a path that bypassed the serdes would fail rather than quietly pass.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // reason: one test per path is the point; splitting it would let the paths drift apart again
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

        // step
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

        // invoke payload
        let invoke_rec = RecordingSerdes::new();
        let (invoke_ctx, _invoke_client) = test_ctx_with_client(CheckpointLog::empty());
        let signal = Arc::clone(invoke_ctx.coordinator());
        let invoke_fut = invoke_ctx
            .invoke::<String, String>("target-fn", value.clone())
            .payload_serdes(invoke_rec.clone());
        let outcome = crate::driver::test_support::outcome_of(
            invoke_ctx.clone(),
            Arc::clone(invoke_ctx.task_ownership()),
            signal,
            invoke_fut,
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        assert_eq!(
            invoke_rec.serialize_inputs(),
            vec![expected.clone()],
            "an invoke payload serdes must be handed the same value shape as a step"
        );

        // callback (deserialize-only path)
        //
        // Callbacks have no serialize side (an external caller writes the
        // payload), and `deserialize_callback_result` IS the whole boundary:
        // it is the only place the callback replay path consults a serdes. It
        // is fed the EXACT wire form the step path wrote above, so this asserts
        // the two paths agree on the reconstructed shape.
        let callback_rec = RecordingSerdes::new();
        let callback_ctx = SerdesContext::new("cb-1", "arn:test");
        let callback_out: String = crate::callback::deserialize_callback_result(
            &step_ctx,
            None,
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

        // map items
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

        // parallel branches
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

        // whole batch result
        //
        // The batch-result serdes is handed the batch summary, so its value
        // differs by nature, but it must arrive through the SAME boundary: a
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
        // batch serdes sees structure, not one flat pre-rendered blob.
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
                    due_at: None,
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
                    due_at: None,
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
                due_at: None,
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

    // Issue #36: ordered lazy admission + batch decision record

    /// Splits the recorded updates into child starts seen BEFORE the first
    /// child terminal write and the total child start count. A child update
    /// is one carrying a `ParentId`.
    fn child_starts_before_first_terminal(
        client: &crate::client::InMemoryExecutionClient,
    ) -> (usize, usize) {
        let updates = client.recorded_updates();
        let mut starts_before_first_terminal = 0;
        let mut total_starts = 0;
        let mut saw_terminal = false;
        for u in &updates {
            if u.parent_id().is_none() {
                continue;
            }
            match u.action() {
                OperationAction::Start => {
                    total_starts += 1;
                    if !saw_terminal {
                        starts_before_first_terminal += 1;
                    }
                }
                _ => saw_terminal = true,
            }
        }
        (starts_before_first_terminal, total_starts)
    }

    /// Acceptance (issue #36): a batch larger than the concurrency limit
    /// issues no child `Start` checkpoint before its first item runs.
    /// Ordered lazy admission writes each `Start` at dispatch, so at most
    /// `concurrency` starts can exist before the first item outcome: never
    /// the whole batch.
    #[tokio::test]
    async fn batch_larger_than_concurrency_issues_no_start_before_first_item_runs() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let result = ctx
            .map(
                vec![1_i32, 2, 3, 4, 5, 6],
                |_child, item, _idx| async move { Ok(item * 10) },
            )
            .max_concurrency(2)
            .await
            .expect("map must succeed");
        assert_eq!(result, vec![10, 20, 30, 40, 50, 60]);

        let (starts_before, total_starts) = child_starts_before_first_terminal(&client);
        assert_eq!(
            total_starts, 6,
            "every admitted item gets exactly one Start"
        );
        assert!(
            starts_before <= 2,
            "at most `concurrency` child Starts may precede the first item \
             outcome (lazy admission), found {starts_before}"
        );
    }

    /// Finds the batch parent's SUCCEED update (the terminal update with no
    /// `ParentId` on a root-level batch).
    fn parent_succeed_update(client: &crate::client::InMemoryExecutionClient) -> OperationUpdate {
        client
            .recorded_updates()
            .into_iter()
            .find(|u| u.parent_id().is_none() && matches!(u.action(), OperationAction::Succeed))
            .expect("batch parent SUCCEED update must be recorded")
    }

    /// A batch whose serialized result exceeds the checkpoint size limit
    /// records the decision record, `totalCount`, `completionReason`, and
    /// the shorter index set, as the `replay_children` payload, instead of
    /// no payload at all.
    #[tokio::test]
    async fn large_batch_result_checkpoints_decision_record() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let big = "x".repeat(200 * 1024);
        let result = ctx
            .map(
                vec![big.clone(), big],
                |_child, item: String, _idx| async move { Ok(item) },
            )
            .await
            .expect("map must succeed");
        assert_eq!(result.len(), 2);

        let parent = parent_succeed_update(&client);
        assert_eq!(
            parent.context_options().and_then(|c| c.replay_children),
            Some(true),
            "oversized batch payload must set replay_children"
        );
        let record: BatchDecisionRecord = serde_json::from_str(
            parent
                .payload()
                .expect("large-result parent must carry the decision record"),
        )
        .expect("payload must parse as a plain-JSON decision record");
        assert_eq!(record.total_count, 2);
        assert_eq!(record.completion_reason, "ALL_COMPLETED");
        // Everything completed: the empty started set is the shorter form.
        assert_eq!(record.started_indexes, Some(Vec::new()));
        assert_eq!(record.completed_indexes, None);
    }

    /// Acceptance (issue #36): `totalCount` is the number of items EVER
    /// STARTED, not the input length. A sequential batch of 6 that stops on
    /// `min_successful(1)` after its first item records `totalCount = 1`.
    #[tokio::test]
    async fn decision_record_total_count_is_ever_started_not_input_length() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .map(
                vec![0_usize, 1, 2, 3, 4, 5],
                |_child, _item, idx| async move { Ok("y".repeat(300 * 1024) + &idx.to_string()) },
            )
            .max_concurrency(1)
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
            .await_batch()
            .await
            .expect("map must succeed");
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        assert_eq!(batch.items.len(), 1);

        let parent = parent_succeed_update(&client);
        let record: BatchDecisionRecord = serde_json::from_str(
            parent
                .payload()
                .expect("large-result parent must carry the decision record"),
        )
        .expect("payload must parse as a plain-JSON decision record");
        assert_eq!(
            record.total_count, 1,
            "totalCount counts items ever started, not the 6 input items"
        );
        assert_eq!(record.completion_reason, "MIN_SUCCESSFUL_REACHED");
        assert_eq!(record.started_indexes, Some(Vec::new()));
    }

    /// The decision record must be PLAIN JSON even when a custom
    /// `result_serdes` transforms the batch summary: it is the SDK's own
    /// bookkeeping and must stay readable when the configured serdes is the
    /// reason the payload outgrew the limit.
    #[tokio::test]
    async fn decision_record_bypasses_result_serdes() {
        use crate::serdes::test_support::HexEnvelopeSerdes;

        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let big = "z".repeat(200 * 1024);
        let result = ctx
            .map(
                vec![big.clone(), big],
                |_child, item: String, _idx| async move { Ok(item) },
            )
            .result_serdes(HexEnvelopeSerdes)
            .await
            .expect("map must succeed");
        assert_eq!(result.len(), 2);

        let parent = parent_succeed_update(&client);
        let record: BatchDecisionRecord = serde_json::from_str(
            parent
                .payload()
                .expect("large-result parent must carry the decision record"),
        )
        .expect("the decision record must parse as plain JSON, not the hex wire form");
        assert_eq!(record.total_count, 2);
    }

    /// Builds a terminal batch-parent record carrying a decision record as
    /// its `replay_children` payload.
    fn decision_record_parent(positional_id: &str, decision: &str) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Succeeded,
                result: Some(decision.to_owned()),
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
                due_at: None,
                op_name: None,
            },
        )
    }

    /// Acceptance (issue #36): a batch that stopped on a completion
    /// threshold, replayed on the large-result path, reproduces the live
    /// item statuses and reason from the decision record without re-running
    /// unstarted items (or any item body at all: completed items decode
    /// from their child records).
    #[tokio::test]
    async fn replay_of_threshold_stopped_large_batch_obeys_decision_record() {
        // Live history: 6 input items, min_successful stopped the batch
        // after items 0 and 1 completed; items 2..5 never started. The
        // parent's payload was too large to store, so the decision record
        // was written in its place. Children claimed IDs "2" and "3".
        let decision =
            r#"{"totalCount":2,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            succeeded_record("2", "\"hello\""),
            succeeded_record("3", "\"world\""),
        ]);
        let ctx = test_ctx(log);

        let batch: BatchResult<String> = ctx
            .map(
                vec![0_usize, 1, 2, 3, 4, 5],
                |_child, _item, _idx| async move {
                    unreachable!("replay must not re-run any item body")
                },
            )
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(2))
            .await_batch()
            .await
            .expect("replay must succeed");

        assert_eq!(
            batch.reason,
            CompletionReason::MinSuccessfulReached,
            "the recorded reason is taken verbatim"
        );
        assert_eq!(batch.items.len(), 2, "unstarted items stay omitted");
        assert_eq!(batch.items[0].index, 0);
        assert_eq!(batch.items[0].result.as_deref(), Some("hello"));
        assert_eq!(batch.items[1].index, 1);
        assert_eq!(batch.items[1].result.as_deref(), Some("world"));
    }

    /// Acceptance (issue #36): an index in the decision record's STARTED
    /// set is reported as [`BatchItemStatus::Started`] without consulting
    /// its child checkpoints: the child here holds only a non-terminal
    /// `Started` record, which would error if replay tried to decode it as
    /// terminal. Never-started indexes (at or beyond `totalCount`) stay
    /// omitted.
    #[tokio::test]
    async fn replay_reports_started_set_as_started_without_reading_child_checkpoints() {
        let started_record = |positional_id: &str| {
            let wire_id = crate::engine::compute_wire_id_public(positional_id);
            (
                wire_id.clone(),
                CheckpointRecord {
                    id: wire_id,
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
                    due_at: None,
                    op_name: None,
                },
            )
        };
        // Items 0 and 2 completed; item 1 started but was still running
        // when the min-successful trigger completed the batch.
        let decision =
            r#"{"totalCount":3,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[1]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            succeeded_record("2", "\"a\""),
            started_record("3"),
            succeeded_record("4", "\"c\""),
        ]);
        let ctx = test_ctx(log);

        let batch: BatchResult<String> = ctx
            .map(vec![0_usize, 1, 2, 3], |_child, _item, _idx| async move {
                unreachable!("replay must not re-run any item body")
            })
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(2))
            .await_batch()
            .await
            .expect("replay must succeed");

        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        let statuses: Vec<(usize, BatchItemStatus)> =
            batch.items.iter().map(|i| (i.index, i.status)).collect();
        assert_eq!(
            statuses,
            vec![
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started),
                (2, BatchItemStatus::Succeeded),
            ],
            "started-set item is reported STARTED; the never-started item is omitted"
        );
        assert!(
            batch.items[1].result.is_none() && batch.items[1].error_message.is_none(),
            "a STARTED item carries no result and no error"
        );
        // The Vec<O> success view skips the STARTED item.
        assert_eq!(batch.results(), vec!["a", "c"]);
    }

    /// The plan-constrained FALLBACK (fast replay hit a completed child
    /// recorded `replay_children` and fell through to re-execution) must
    /// consult child checkpoints only for indexes the plan admits as
    /// COMPLETED. Started-set and never-started indexes here hold records
    /// with a hostile identity (`op_type: Step`) that would raise
    /// `NonDeterministicExecution` if the concurrent pre-claim loop read
    /// them: as it did before this fix.
    #[tokio::test]
    async fn plan_constrained_fallback_reads_only_admitted_child_checkpoints() {
        // A record whose stored identity can never match a batch child's
        // claim ("Context"): any validated read of it fails.
        let poisoned_record = |positional_id: &str| {
            let wire_id = crate::engine::compute_wire_id_public(positional_id);
            (
                wire_id.clone(),
                CheckpointRecord {
                    id: wire_id,
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
                    op_type: Some("Step".to_owned()),
                    sub_type: None,
                    due_at: None,
                    op_name: None,
                },
            )
        };
        // Live history: items 0 and 2 completed, item 1 started-not-
        // terminal, item 3 never started. Item 0's own result was too
        // large to store inline (`replay_children`), so fast replay falls
        // through and the plan constrains the re-execution.
        let decision =
            r#"{"totalCount":3,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[1]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            replay_children_success_record("2"),
            poisoned_record("3"),
            succeeded_record("4", "\"c\""),
            poisoned_record("5"),
        ]);
        // The fallback re-executes item 0's body to reconstruct its
        // oversized result, so a live client is needed.
        let (ctx, _client) = test_ctx_with_client(log);

        let batch: BatchResult<String> = ctx
            .map(vec![0_usize, 1, 2, 3], |_child, _item, idx| async move {
                assert_eq!(idx, 0, "only the replay_children item re-executes");
                Ok("a".to_owned())
            })
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(2))
            .await_batch()
            .await
            .expect("plan-constrained fallback must not read non-admitted child checkpoints");

        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        let statuses: Vec<(usize, BatchItemStatus)> =
            batch.items.iter().map(|i| (i.index, i.status)).collect();
        assert_eq!(
            statuses,
            vec![
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started),
                (2, BatchItemStatus::Succeeded),
            ],
            "started-set item reported STARTED from the record; never-started omitted"
        );
        assert_eq!(batch.results(), vec!["a", "c"]);
    }

    /// A record-less `replay_children` payload, written before decision
    /// records existed, falls back to today's behaviour: re-execution,
    /// which replays each recorded-terminal child from its own record.
    #[tokio::test]
    async fn record_less_replay_children_payload_falls_back_to_reexecution() {
        let parent_wire = crate::engine::compute_wire_id_public("1");
        let log = CheckpointLog::from_records(vec![
            (
                parent_wire.clone(),
                CheckpointRecord {
                    id: parent_wire,
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
                    due_at: None,
                    op_name: None,
                },
            ),
            succeeded_record("2", "\"hello\""),
            succeeded_record("3", "\"world\""),
        ]);
        // Re-execution re-checkpoints the parent, so a live client is
        // needed, exactly as before decision records existed.
        let (ctx, _client) = test_ctx_with_client(log);

        let result: Vec<String> = ctx
            .map(
                vec!["a".to_owned(), "b".to_owned()],
                |_child, _item: String, _idx| async move {
                    unreachable!("recorded-terminal children replay from their records")
                },
            )
            .await
            .expect("fall-back re-execution must succeed");
        assert_eq!(result, vec!["hello", "world"]);
    }

    /// Acceptance (issue #36): `NestingMode::Flat` reports its started set
    /// from the record. Flat items have no child checkpoints, so the
    /// decision record is the only source: replay re-runs exactly the
    /// completed set (reconstructing values), reports the started set as
    /// [`BatchItemStatus::Started`] without running those bodies, skips
    /// the never-started items, and takes the reason verbatim.
    #[tokio::test]
    async fn flat_batch_replay_reports_started_set_from_record() {
        // Items 0 and 2 completed; item 1 started but was still running
        // when the trigger completed the batch; items 3..5 never started.
        let decision =
            r#"{"totalCount":3,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[1]}"#;
        let log = CheckpointLog::from_records(vec![decision_record_parent("1", decision)]);
        let ctx = test_ctx(log);

        let invoked = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let invoked_probe = Arc::clone(&invoked);

        let batch = ctx
            .map(vec![0_usize, 1, 2, 3, 4, 5], move |_child, _item, idx| {
                let invoked = Arc::clone(&invoked_probe);
                async move {
                    invoked
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(idx);
                    Ok(format!("flat-{idx}"))
                }
            })
            .nesting(NestingMode::Flat)
            .max_concurrency(2)
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(2))
            .await_batch()
            .await
            .expect("flat replay must succeed");

        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        let statuses: Vec<(usize, BatchItemStatus)> =
            batch.items.iter().map(|i| (i.index, i.status)).collect();
        assert_eq!(
            statuses,
            vec![
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started),
                (2, BatchItemStatus::Succeeded),
            ],
            "the started set is reported STARTED from the record; unstarted items are omitted"
        );
        let mut ran = invoked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        ran.sort_unstable();
        assert_eq!(
            ran,
            vec![0, 2],
            "flat replay re-runs the completed set only — never started-set or unstarted items"
        );
    }

    /// Under a replay plan, a completion rule that fires again on the
    /// reconstructed outcomes abandons and aborts nothing. The record says
    /// items 0 and 1 both completed, and the current configuration needs
    /// only one success. Item 0 settles first and meets that rule while
    /// item 1 is still running. The batch still waits for item 1 and
    /// reports both recorded completions, because the record, not the
    /// re-derived trigger, decides the outcome.
    #[tokio::test]
    async fn replay_plan_trigger_keeps_recorded_completed_items() {
        let decision =
            r#"{"totalCount":2,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[]}"#;
        let log = CheckpointLog::from_records(vec![decision_record_parent("1", decision)]);
        let ctx = test_ctx(log);
        let finished = Arc::new(AtomicBool::new(false));
        let finished_probe = Arc::clone(&finished);

        let batch = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ctx.map(vec![0_usize, 1], move |_child, _item, idx| {
                let finished = Arc::clone(&finished_probe);
                async move {
                    if idx == 1 {
                        // Item 0 settles while item 1 is still running.
                        for _ in 0..20 {
                            tokio::task::yield_now().await;
                        }
                        finished.store(true, Ordering::SeqCst);
                    }
                    Ok(format!("flat-{idx}"))
                }
            })
            .nesting(NestingMode::Flat)
            .max_concurrency(2)
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
            .await_batch(),
        )
        .await
        .expect("the replay resolves")
        .expect("the replay succeeds");

        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        let items: Vec<(usize, BatchItemStatus, Option<String>)> = batch
            .items
            .iter()
            .map(|i| (i.index, i.status, i.result.clone()))
            .collect();
        assert_eq!(
            items,
            vec![
                (0, BatchItemStatus::Succeeded, Some("flat-0".to_owned())),
                (1, BatchItemStatus::Succeeded, Some("flat-1".to_owned())),
            ],
            "both recorded completions are reconstructed"
        );
        assert!(
            finished.load(Ordering::SeqCst),
            "item 1 ran to its end and was not aborted"
        );
    }
    #[test]
    fn decision_record_writes_the_shorter_index_set() {
        let items = vec![
            BatchItem::<u32>::new(
                0,
                String::new(),
                BatchItemStatus::Succeeded,
                Some(1),
                None,
                None,
            ),
            BatchItem::<u32>::new(
                1,
                String::new(),
                BatchItemStatus::Succeeded,
                Some(2),
                None,
                None,
            ),
            BatchItem::<u32>::new(2, String::new(), BatchItemStatus::Failed, None, None, None),
        ];
        // 3 completed of 4 started: the 1-element started set is shorter.
        let record = build_decision_record(&items, 4, CompletionReason::CustomCompletionSucceeded);
        assert_eq!(record.total_count, 4);
        assert_eq!(record.completion_reason, "CUSTOM_COMPLETION_SUCCEEDED");
        assert_eq!(record.started_indexes, Some(vec![3]));
        assert_eq!(record.completed_indexes, None);

        // 1 completed of 4 started: the 1-element completed set is shorter.
        let one = vec![BatchItem::<u32>::new(
            2,
            String::new(),
            BatchItemStatus::Succeeded,
            Some(1),
            None,
            None,
        )];
        let record = build_decision_record(&one, 4, CompletionReason::AllCompleted);
        assert_eq!(record.started_indexes, None);
        assert_eq!(record.completed_indexes, Some(vec![2]));
    }

    #[test]
    fn replay_plan_reconstructs_either_index_form() {
        // From the started form.
        let record = BatchDecisionRecord {
            total_count: 4,
            completion_reason: "ALL_COMPLETED".to_owned(),
            started_indexes: Some(vec![1, 3]),
            completed_indexes: None,
        };
        let plan = ReplayPlan::from_record(&record, 10).expect("well-formed record");
        assert_eq!(plan.total_count, 4);
        assert!(plan.admits(0) && plan.admits(2));
        assert!(!plan.admits(1) && !plan.admits(3));
        assert!(!plan.admits(4), "beyond totalCount is never admitted");

        // From the completed form.
        let record = BatchDecisionRecord {
            total_count: 3,
            completion_reason: "MIN_SUCCESSFUL_REACHED".to_owned(),
            started_indexes: None,
            completed_indexes: Some(vec![2]),
        };
        let plan = ReplayPlan::from_record(&record, 10).expect("well-formed record");
        assert!(!plan.admits(0) && !plan.admits(1) && plan.admits(2));
        assert_eq!(plan.reason, CompletionReason::MinSuccessfulReached);
    }

    /// A syntactically valid decision record that the SDK's writer could
    /// not have produced is rejected outright: silently repairing it
    /// (clamping `totalCount`, ignoring out-of-range indexes, defaulting a
    /// missing index form or an unknown reason) would convert recorded
    /// outcomes into different ones.
    #[test]
    fn replay_plan_rejects_semantically_invalid_records() {
        let base = || BatchDecisionRecord {
            total_count: 3,
            completion_reason: "ALL_COMPLETED".to_owned(),
            started_indexes: Some(vec![1]),
            completed_indexes: None,
        };

        // totalCount larger than the current input length.
        let record = base();
        let err = ReplayPlan::from_record(&record, 2).expect_err("totalCount > input");
        assert!(
            crate::error::chain_string(&err).contains("exceeds the input length"),
            "unexpected error: {err}"
        );

        // Both index forms present.
        let mut record = base();
        record.completed_indexes = Some(vec![0, 2]);
        let err = ReplayPlan::from_record(&record, 10).expect_err("both index forms");
        assert!(crate::error::chain_string(&err).contains("both startedIndexes"));

        // Neither index form present.
        let mut record = base();
        record.started_indexes = None;
        let err = ReplayPlan::from_record(&record, 10).expect_err("neither index form");
        assert!(crate::error::chain_string(&err).contains("neither startedIndexes"));

        // Index outside [0, totalCount): in either form.
        let mut record = base();
        record.started_indexes = Some(vec![3]);
        let err = ReplayPlan::from_record(&record, 10).expect_err("started index out of range");
        assert!(crate::error::chain_string(&err).contains("outside [0, 3)"));
        let mut record = base();
        record.started_indexes = None;
        record.completed_indexes = Some(vec![50]);
        let err = ReplayPlan::from_record(&record, 10).expect_err("completed index out of range");
        assert!(crate::error::chain_string(&err).contains("outside [0, 3)"));

        // Unknown completion-reason wire value.
        let mut record = base();
        record.completion_reason = "SOME_FUTURE_REASON".to_owned();
        let err = ReplayPlan::from_record(&record, 10).expect_err("unknown reason");
        assert!(crate::error::chain_string(&err).contains("unknown completionReason"));
    }

    #[test]
    fn parse_decision_record_legacy_fallback_only_for_absent_or_empty() {
        // Absent or empty payloads are legacy record-less writes: fall back.
        assert!(matches!(parse_decision_record(None), Ok(None)));
        assert!(matches!(parse_decision_record(Some("")), Ok(None)));
        assert!(matches!(parse_decision_record(Some("   ")), Ok(None)));
        // A NON-EMPTY payload that is not a decision record is corrupt
        // bookkeeping: an error, never a silent fall-back.
        assert!(parse_decision_record(Some("not json")).is_err());
        assert!(parse_decision_record(Some(r#"{"unrelated":true}"#)).is_err());
        let record = parse_decision_record(Some(
            r#"{"totalCount":2,"completionReason":"ALL_COMPLETED","startedIndexes":[]}"#,
        ))
        .expect("parse must not error")
        .expect("well-formed record must parse");
        assert_eq!(record.total_count, 2);
    }

    /// A malformed (non-empty, unparseable) decision record fails the
    /// replay instead of falling back to unconstrained re-execution, which
    /// could resurrect items the recorded batch never started.
    #[tokio::test]
    async fn malformed_decision_record_fails_replay_instead_of_reexecuting() {
        let log = CheckpointLog::from_records(vec![decision_record_parent(
            "1",
            r#"{"totalCount":"#, // truncated JSON: corrupt, not legacy
        )]);
        let ctx = test_ctx(log);

        let result: Result<Vec<String>, _> = ctx
            .map(vec![0_usize, 1, 2], |_child, _item, _idx| async move {
                unreachable!("a corrupt decision record must not re-execute any item")
            })
            .await;

        let Err(err) = result else {
            unreachable!("replay of a corrupt decision record must fail")
        };
        let display = crate::error::chain_string(&err);
        assert!(
            display.contains("malformed batch decision record"),
            "expected the malformed-record error, got: {display}"
        );
    }

    /// The oversized-record write path refuses to checkpoint rather than
    /// writing a modern terminal batch without its decision record (a
    /// record-less write would replay through unconstrained re-execution).
    #[tokio::test]
    async fn oversized_decision_record_refuses_checkpoint_instead_of_dropping_record() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let oversized_payload = "p".repeat(CHECKPOINT_SIZE_LIMIT_BYTES + 1);
        let oversized_decision = "d".repeat(CHECKPOINT_SIZE_LIMIT_BYTES + 1);
        let parent_wire = crate::engine::compute_wire_id_public("1");

        let result = checkpoint_batch_success_serialized(
            &ctx,
            &parent_wire,
            None,
            MAP_SUB_TYPE,
            &oversized_payload,
            &oversized_decision,
        )
        .await;

        let Err(err) = result else {
            unreachable!("an oversized decision record must refuse the checkpoint")
        };
        let display = crate::error::chain_string(&err);
        assert!(
            display.contains("decision record exceeds the checkpoint size limit"),
            "expected the oversized-record error, got: {display}"
        );
        assert!(
            client
                .recorded_updates()
                .iter()
                .all(|u| !matches!(u.action(), OperationAction::Succeed)),
            "no terminal SUCCEED may be written without its decision record"
        );
    }

    /// Non-determinism regression (fast replay path): a completed child's
    /// replay identity is validated before its terminal payload is read, so
    /// a changed child sub-type raises `NonDeterministicExecution` instead
    /// of replaying the stale record successfully.
    #[tokio::test]
    async fn fast_replay_validates_completed_child_identity() {
        // Child "2" (item 0) carries a recorded identity that does NOT
        // match a map iteration: same type but a different sub-type.
        let mismatched_child = {
            let wire_id = crate::engine::compute_wire_id_public("2");
            (
                wire_id.clone(),
                CheckpointRecord {
                    id: wire_id,
                    status: CheckpointStatus::Succeeded,
                    result: Some("\"a\"".to_owned()),
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
                    sub_type: Some("ParallelBranch".to_owned()),
                    due_at: None,
                    op_name: None,
                },
            )
        };
        let decision = r#"{"totalCount":2,"completionReason":"ALL_COMPLETED","startedIndexes":[]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            mismatched_child,
            succeeded_record("3", "\"b\""),
        ]);
        let ctx = test_ctx(log);

        let result: Result<BatchResult<String>, _> = ctx
            .map(vec![0_usize, 1], |_child, _item, _idx| async move {
                unreachable!("identity mismatch must fail before any re-execution")
            })
            .await_batch()
            .await;

        let Err(err) = result else {
            unreachable!("a mismatched child identity must fail the replay")
        };
        assert!(
            matches!(err.kind(), OperationErrorKind::NonDeterministicExecution(_)),
            "expected NonDeterministicExecution, got: {err:#}"
        );
    }

    // Early completion: running items are abandoned, not drained.

    /// The bound on every batch await in the early-completion tests.
    const EARLY_BOUND: std::time::Duration = std::time::Duration::from_secs(5);

    /// The wire IDs of the two items of a root map at position 1: the
    /// batch claims its items' IDs next, in the root namespace.
    fn item_wires() -> [String; 2] {
        [
            crate::engine::compute_wire_id_public("2"),
            crate::engine::compute_wire_id_public("3"),
        ]
    }

    /// The number of recorded updates with `action` for item `index`'s
    /// child context.
    fn item_action_count(
        client: &crate::client::InMemoryExecutionClient,
        index: usize,
        action: &OperationAction,
    ) -> usize {
        let wire = item_wires()[index].clone();
        client
            .recorded_updates()
            .iter()
            .filter(|u| u.id == wire && u.action() == action)
            .count()
    }

    /// The statuses of a batch's items, in input order.
    fn statuses<O>(batch: &BatchResult<O>) -> Vec<(usize, BatchItemStatus)> {
        batch.items.iter().map(|i| (i.index, i.status)).collect()
    }

    /// A boxed item body future.
    type ItemFut = Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>;

    /// The body of the early-completion maps over `[0, 1]`. Item 0 waits
    /// until item 1 runs, then runs a step that returns at once, or returns
    /// `Err("boom")` from its body when `fail` is set. Item 1 holds `guard`
    /// until its task is dropped, and runs a step whose body never returns.
    /// So item 1 is running when item 0 settles.
    fn early_body(
        fail: bool,
        guard: Arc<AtomicBool>,
    ) -> impl Fn(DurableContext, i32, usize) -> ItemFut + Send + Sync + 'static {
        let running = Arc::new(tokio::sync::Notify::new());
        move |child, item, _idx| {
            let guard = Arc::clone(&guard);
            let running = Arc::clone(&running);
            Box::pin(async move {
                if item == 0 {
                    running.notified().await;
                    if fail {
                        return Err("boom".into());
                    }
                    return Ok(child.step(|_| async { Ok(0_i32) }).name("fast").await?);
                }
                let _guard = SetOnDrop(guard);
                running.notify_one();
                Ok(child
                    .step(|_| std::future::pending::<Result<i32, BoxError>>())
                    .name("forever")
                    .await?)
            })
        }
    }

    /// Sets its flag when dropped.
    struct SetOnDrop(Arc<AtomicBool>);

    impl Drop for SetOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn no_guard() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[tokio::test]
    async fn early_completion_does_not_wait_for_running_item() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], early_body(false, no_guard()))
                .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for its running item")
        .expect("the batch succeeds");
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started)
            ]
        );
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        assert_eq!(batch.total_count(), 2);
        assert_started_defaults(&batch.items[1], "");
        assert_eq!(item_action_count(&client, 0, &OperationAction::Succeed), 1);
        assert_eq!(item_action_count(&client, 1, &OperationAction::Succeed), 0);
        // The batch abandoned only the item that was still running.
        let [settled, running] = item_wires();
        assert!(
            ctx.admission().admits(&[settled]),
            "item 0 is not abandoned"
        );
        assert!(!ctx.admission().admits(&[running]), "item 1 is abandoned");
    }

    /// A client that, at the batch parent's `SUCCEED`, yields until
    /// `guard` is set (at most 1,000 times) and records whether it was.
    #[derive(Debug)]
    struct GuardAtParentWrite {
        guard: Arc<AtomicBool>,
        seen: Arc<AtomicBool>,
    }

    impl crate::client::ExecutionClient for GuardAtParentWrite {
        fn checkpoint(
            &self,
            _arn: &str,
            _token: &str,
            updates: Vec<OperationUpdate>,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::client::CheckpointOutput,
                            crate::client::ExecutionClientError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let parent = crate::engine::compute_wire_id_public("1");
            let at_parent = updates
                .iter()
                .any(|u| u.id == parent && u.action() == &OperationAction::Succeed);
            Box::pin(async move {
                if at_parent {
                    for _ in 0..1_000 {
                        if self.guard.load(Ordering::SeqCst) {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    self.seen
                        .store(self.guard.load(Ordering::SeqCst), Ordering::SeqCst);
                }
                Ok(crate::client::CheckpointOutput {
                    checkpoint_token: Some("token".to_owned()),
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
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::client::GetStateOutput,
                            crate::client::ExecutionClientError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::client::GetStateOutput {
                    operations: Vec::new(),
                    next_marker: None,
                })
            })
        }
    }

    /// The batch aborts its running items at the trigger, before it writes
    /// its own `SUCCEED`. So item 1's task unwinds while that write is in
    /// flight. Without the abort, the task would be dropped only with the
    /// join set, after the write.
    #[tokio::test]
    async fn early_completion_aborts_running_item_before_parent_write() {
        let guard = no_guard();
        let seen = no_guard();
        let client = Arc::new(GuardAtParentWrite {
            guard: Arc::clone(&guard),
            seen: Arc::clone(&seen),
        });
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], early_body(false, Arc::clone(&guard)))
                .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for its running item")
        .expect("the batch succeeds");
        assert!(
            seen.load(Ordering::SeqCst),
            "item 1's task unwound before the batch parent's SUCCEED was written"
        );
    }

    /// A `Started` item of a `parallel` carries its branch name.
    #[tokio::test]
    async fn started_branch_keeps_its_name() {
        use crate::future::Branch;

        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.parallel(vec![
                Branch::new("fast", |_child| async { Ok(0_i32) }),
                Branch::new("slow", |_child| std::future::pending()),
            ])
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
            .await_batch(),
        )
        .await
        .expect("the batch does not wait for its running branch")
        .expect("the batch succeeds");
        let started = batch
            .items
            .iter()
            .find(|item| item.status == BatchItemStatus::Started)
            .expect("the running branch is reported as Started");
        assert_started_defaults(started, "slow");
        assert_eq!(started.index, 1);
    }

    /// A live `Started` item carries `name`, no result, no error fields and
    /// an empty stack trace.
    fn assert_started_defaults<O: std::fmt::Debug>(item: &BatchItem<O>, name: &str) {
        assert_eq!(item.status, BatchItemStatus::Started);
        assert_eq!(item.name, name);
        assert!(item.result.is_none(), "no result: {item:?}");
        assert_eq!(item.error_message, None);
        assert_eq!(item.error_type, None);
        assert_eq!(item.error_data, None);
        assert!(item.stack_trace.is_empty(), "no stack trace: {item:?}");
    }

    /// Yields until `client` recorded the `START` of the operation named
    /// `name`, at most 1,000 times. Panics at the bound.
    async fn wait_for_start(client: &crate::client::InMemoryExecutionClient, name: &str) {
        for _ in 0..1_000 {
            if has_start(client, name) {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            has_start(client, name),
            "the START of {name} was never recorded"
        );
    }

    /// Whether `client` recorded the `START` of the operation named `name`.
    fn has_start(client: &crate::client::InMemoryExecutionClient, name: &str) -> bool {
        client
            .recorded_updates()
            .iter()
            .any(|u| u.name() == Some(name) && u.action() == &OperationAction::Start)
    }

    #[tokio::test(start_paused = true)]
    async fn early_completion_abandons_running_item() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let observer = Arc::clone(&client);
        let start = tokio::time::Instant::now();
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(
                vec![0_i32, 1],
                move |child: DurableContext, item: i32, _idx| {
                    let observer = Arc::clone(&observer);
                    async move {
                        if item == 0 {
                            // Item 0 settles only once item 1 runs step `a`:
                            // its START is recorded.
                            wait_for_start(&observer, "a").await;
                            return Ok(child.step(|_| async { Ok(0_i32) }).name("fast").await?);
                        }
                        child
                            .step(|_| async {
                                tokio::time::sleep(std::time::Duration::from_millis(800)).await;
                                Ok(1_i32)
                            })
                            .name("a")
                            .await?;
                        Ok(child.step(|_| async { Ok(2_i32) }).name("b").await?)
                    }
                },
            )
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
            .await_batch(),
        )
        .await
        .expect("the batch returns")
        .expect("the batch succeeds");
        assert!(
            start.elapsed() < std::time::Duration::from_millis(800),
            "the batch returned after {:?}, so it waited for item 1",
            start.elapsed()
        );
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started)
            ]
        );
        assert!(has_start(&client, "a"), "item 1 entered step a");
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        assert!(
            !client
                .recorded_updates()
                .iter()
                .any(|u| u.name() == Some("b")),
            "the abandoned item started no later operation"
        );
        assert_eq!(item_action_count(&client, 1, &OperationAction::Succeed), 0);
    }

    #[tokio::test]
    async fn fail_fast_does_not_wait_for_running_sibling() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], early_body(true, no_guard()))
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for its running item")
        .expect("await_batch reports item failures as items");
        assert_eq!(
            statuses(&batch),
            [(0, BatchItemStatus::Failed), (1, BatchItemStatus::Started)]
        );
        assert_eq!(batch.reason, CompletionReason::FailureToleranceExceeded);
        assert_eq!(item_action_count(&client, 0, &OperationAction::Fail), 1);
        assert_eq!(item_action_count(&client, 1, &OperationAction::Fail), 0);

        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let err = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], early_body(true, no_guard()))
                .into_future(),
        )
        .await
        .expect("the batch does not wait for its running item")
        .expect_err("fail-fast ends the batch with an error");
        assert!(
            matches!(
                err.kind(),
                OperationErrorKind::ChildContext(e) if matches!(e.kind(), ChildContextErrorKind::ChildFailed)
            ),
            "unexpected error kind: {:?}",
            err.kind()
        );
        assert_eq!(item_action_count(&client, 0, &OperationAction::Fail), 1);
        assert_eq!(item_action_count(&client, 1, &OperationAction::Fail), 0);
    }

    #[tokio::test]
    async fn predicate_completion_does_not_wait_for_running_item() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], early_body(false, no_guard()))
                .completion(
                    crate::builders::map_parallel::CompletionConfig::with_completion_predicate(
                        |s| succeed_when(s.succeeded() >= 1),
                    ),
                )
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for its running item")
        .expect("the batch succeeds");
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started)
            ]
        );
    }

    #[test]
    fn decision_record_keeps_started_items_out_of_completed() {
        let item =
            |index, status| BatchItem::<i32>::new(index, String::new(), status, None, None, None);
        let record = build_decision_record(
            &[
                item(0, BatchItemStatus::Succeeded),
                item(1, BatchItemStatus::Succeeded),
                item(2, BatchItemStatus::Started),
            ],
            3,
            CompletionReason::MinSuccessfulReached,
        );
        assert_eq!(record.started_indexes, Some(vec![2]));
        assert_eq!(record.completed_indexes, None);
    }

    #[tokio::test]
    async fn abandoned_item_leaves_no_task_after_invocation() {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let guard = no_guard();
        let handler = {
            let ctx = ctx.clone();
            let guard = Arc::clone(&guard);
            async move {
                ctx.map(vec![0_i32, 1], early_body(false, guard))
                    .completion(
                        crate::builders::map_parallel::CompletionConfig::with_min_successful(1),
                    )
                    .await_batch()
                    .await
                    .expect("the batch succeeds");
                Ok::<_, crate::error::WireError>("done".to_owned())
            }
        };
        let outcome = tokio::time::timeout(
            EARLY_BOUND,
            crate::driver::drive_invocation(handler, ownership, signal, root, None),
        )
        .await
        .expect("the invocation does not wait for the abandoned item");
        assert!(
            matches!(outcome, Ok(crate::driver::InvocationOutcome::Complete(_))),
            "unexpected outcome: {outcome:?}"
        );
        assert!(
            guard.load(Ordering::SeqCst),
            "the abandoned item's task unwound before the invocation returned"
        );
        assert_eq!(ctx.coordinator().tasks().live(), 0);
    }

    /// Plain `.await` of an early-completing map or parallel returns only
    /// the succeeded values: the running item, reported `Started`, adds
    /// none.
    #[tokio::test]
    async fn early_completion_await_returns_only_succeeded_values() {
        use crate::future::Branch;

        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let values = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], early_body(false, no_guard()))
                .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
                .into_future(),
        )
        .await
        .expect("the map does not wait for its running item")
        .expect("the map succeeds");
        assert_eq!(values, [0]);

        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let values = tokio::time::timeout(
            EARLY_BOUND,
            ctx.parallel(vec![
                Branch::new("fast", |_child| async { Ok(0_i32) }),
                Branch::new("slow", |_child| std::future::pending()),
            ])
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
            .into_future(),
        )
        .await
        .expect("the parallel does not wait for its running branch")
        .expect("the parallel succeeds");
        assert_eq!(values, [0]);
    }

    /// How the late item of `queued_outcome_batch` finishes.
    #[derive(Debug, Clone, Copy)]
    enum Late {
        Value,
        Error,
        Panic,
    }

    /// Runs a FLAT map over `[0, 1]` with `min_successful(1)`, so that
    /// both items have finished, item 0 first, before the coordinator joins
    /// either outcome. Item 1 finishes as `late` says. The coordinator then
    /// joins item 0, the trigger fires, and item 1's queued outcome is
    /// never joined.
    #[expect(clippy::panic)] // reason: the late item's panic is a case under test
    async fn queued_outcome_batch(late: Late) -> BatchResult<i32> {
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let tasks = Arc::clone(ctx.coordinator().tasks());
        let entered = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new([tokio::sync::Notify::new(), tokio::sync::Notify::new()]);
        let body = {
            let entered = Arc::clone(&entered);
            let ready = Arc::clone(&ready);
            move |_child: DurableContext, item: i32, idx: usize| {
                let entered = Arc::clone(&entered);
                let ready = Arc::clone(&ready);
                async move {
                    entered.fetch_add(1, Ordering::SeqCst);
                    ready[idx].notified().await;
                    if item == 0 {
                        return Ok(0_i32);
                    }
                    match late {
                        Late::Value => Ok(1_i32),
                        Late::Error => Err("late failure".into()),
                        Late::Panic => panic!("late panic"),
                    }
                }
            }
        };
        let mut batch = std::pin::pin!(
            ctx.map(vec![0_i32, 1], body)
                .nesting(NestingMode::Flat)
                .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
                .await_batch()
        );

        // 1. Drive the batch until both items wait.
        let early = std::future::poll_fn(|cx| {
            if entered.load(Ordering::SeqCst) == 2 {
                return std::task::Poll::Ready(None);
            }
            if let std::task::Poll::Ready(done) = batch.as_mut().poll(cx) {
                return std::task::Poll::Ready(Some(done));
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        });
        let early = tokio::time::timeout(EARLY_BOUND, early)
            .await
            .expect("both items start");
        assert!(early.is_none(), "the batch completed before its items ran");

        // 2. Without polling the batch, let item 0 finish, then item 1.
        //    Each finished task releases its count once its outcome is
        //    queued.
        let wait_live = |n: usize| {
            let tasks = Arc::clone(&tasks);
            async move {
                for _ in 0..10_000 {
                    if tasks.live() == n {
                        return;
                    }
                    tokio::task::yield_now().await;
                }
                assert_eq!(tasks.live(), n, "the live task count never reached {n}");
            }
        };
        ready[0].notify_one();
        wait_live(1).await;
        ready[1].notify_one();
        wait_live(0).await;

        // 3. The coordinator joins item 0 first, and the trigger fires.
        tokio::time::timeout(EARLY_BOUND, batch)
            .await
            .expect("the batch completes")
            .expect("the late outcome is not observed")
    }

    /// An item that finished but whose outcome the coordinator had not
    /// joined when the trigger fired is reported `Started`: its value, its
    /// error and its panic are never observed.
    #[tokio::test]
    async fn finished_unjoined_item_is_started() {
        for late in [Late::Value, Late::Error, Late::Panic] {
            let batch = queued_outcome_batch(late).await;
            assert_eq!(
                statuses(&batch),
                [
                    (0, BatchItemStatus::Succeeded),
                    (1, BatchItemStatus::Started)
                ],
                "late {late:?}"
            );
            assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
            assert_started_defaults(&batch.items[1], "");
        }
    }

    /// The payload of a terminal map or parallel parent with two
    /// succeeded items.
    fn terminal_batch_payload() -> String {
        serde_json::json!({
            "results": [
                {"index": 0, "status": "SUCCEEDED", "result": "\"hello\""},
                {"index": 1, "status": "SUCCEEDED", "result": "\"world\""}
            ],
            "reason": "ALL_COMPLETED"
        })
        .to_string()
    }

    /// An abandoned owner does not replay a recorded map or parallel from
    /// `await_batch`. The terminal parent replays without a write, so only
    /// the admission check at the terminal can hold it back.
    #[tokio::test]
    async fn abandoned_owner_replays_no_batch() {
        use crate::future::Branch;

        let bound = std::time::Duration::from_secs(1);
        for parallel in [false, true] {
            let log = CheckpointLog::from_records(vec![succeeded_record(
                "1-1",
                &terminal_batch_payload(),
            )]);
            let (root, client) = test_ctx_with_client(log);
            let replay = |ctx: DurableContext| async move {
                if parallel {
                    ctx.parallel(vec![
                        Branch::new("a", |_c| async { Ok(String::new()) }),
                        Branch::new("b", |_c| async { Ok(String::new()) }),
                    ])
                    .await_batch()
                    .await
                } else {
                    ctx.map(vec![0_i32, 1], |_c, _item, _idx| async {
                        Ok(String::new())
                    })
                    .await_batch()
                    .await
                }
            };
            // The control: an owner that is not abandoned replays the batch.
            let control = root.new_child("1");
            let replayed = tokio::time::timeout(bound, replay(control))
                .await
                .expect("the recorded batch replays")
                .expect("the replay succeeds");
            assert_eq!(
                replayed.results(),
                [&"hello".to_owned(), &"world".to_owned()]
            );

            let item = root.new_child("1");
            root.admission()
                .abandon(item.owners().iter().map(String::as_str));
            let replayed = tokio::time::timeout(bound, replay(item)).await;
            assert!(
                replayed.is_err(),
                "parallel {parallel}: the recorded batch is not replayed for an abandoned owner"
            );
            assert!(client.recorded_updates().is_empty(), "nothing is sent");
        }
    }

    /// An item that is abandoned while its body runs on another worker
    /// thread does not write its terminal update afterwards. Item 1 blocks
    /// its worker thread inside its body, so the abort cannot land, until
    /// the batch has completed. Then item 1 returns, and its `SUCCEED` is
    /// rejected because the item is abandoned.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abandoned_item_terminal_write_is_rejected() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let tasks = Arc::clone(ctx.coordinator().tasks());
        let entered = Arc::new(tokio::sync::Notify::new());
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Arc::new(std::sync::Mutex::new(Some(released)));
        let body = {
            let entered = Arc::clone(&entered);
            move |_child: DurableContext, item: i32, _idx: usize| {
                let entered = Arc::clone(&entered);
                let released = Arc::clone(&released);
                async move {
                    if item == 0 {
                        entered.notified().await;
                        return Ok(0_i32);
                    }
                    let released = released.lock().unwrap().take().expect("item 1 runs once");
                    entered.notify_one();
                    // Blocks inside the poll, so the abort of this task
                    // cannot take effect. `block_in_place` hands this
                    // worker's queued tasks to another thread, so item 0
                    // still runs.
                    tokio::task::block_in_place(|| {
                        released.recv().expect("the test releases item 1");
                    });
                    Ok(1_i32)
                }
            }
        };
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.map(vec![0_i32, 1], body)
                .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for its running item")
        .expect("the batch succeeds");
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started)
            ]
        );
        assert_eq!(item_action_count(&client, 0, &OperationAction::Succeed), 1);

        // Release item 1, and wait until its task and any write it started
        // have ended.
        release.send(()).expect("item 1 waits");
        let mut ended = false;
        for _ in 0..500 {
            if tasks.live() == 0 {
                ended = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(ended, "item 1's task ended");
        assert_eq!(
            item_action_count(&client, 1, &OperationAction::Succeed),
            0,
            "the abandoned item wrote no terminal update"
        );
    }

    /// A waiting item is not terminal, so an early completion abandons it
    /// too. An operation it created but did not await then never starts.
    /// Item 1 creates a step future and waits. Only then does item 0
    /// succeed and fire the trigger: paused time advances item 0's timer
    /// only once every other task is idle.
    #[tokio::test(start_paused = true)]
    async fn early_completion_abandons_parked_item() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let slot: Arc<std::sync::Mutex<Option<crate::DurableFuture<i32>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let ran = no_guard();
        let body = {
            let slot = Arc::clone(&slot);
            let ran = Arc::clone(&ran);
            move |child: DurableContext, item: i32, _idx: usize| {
                let slot = Arc::clone(&slot);
                let ran = Arc::clone(&ran);
                async move {
                    if item == 0 {
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        return Ok(0_i32);
                    }
                    let escaped = child
                        .step(move |_| async move {
                            ran.store(true, Ordering::SeqCst);
                            Ok(5_i32)
                        })
                        .name("escaped")
                        .future();
                    *slot.lock().unwrap() = Some(escaped);
                    child
                        .wait(std::time::Duration::from_secs(5))
                        .name("park")
                        .await?;
                    Ok(1_i32)
                }
            }
        };
        // The test awaits the batch on its own root future, which the SDK
        // does not poll through a permit-aware task. The held activity
        // stands in for the handler task's permit.
        let batch = tokio::time::timeout(
            EARLY_BOUND,
            ctx.hold_activity(
                ctx.map(vec![0_i32, 1], body)
                    .completion(
                        crate::builders::map_parallel::CompletionConfig::with_min_successful(1),
                    )
                    .await_batch(),
            ),
        )
        .await
        .expect("the batch completes")
        .expect("the batch succeeds");
        assert_eq!(
            statuses(&batch),
            [
                (0, BatchItemStatus::Succeeded),
                (1, BatchItemStatus::Started)
            ]
        );
        assert!(has_start(&client, "park"), "item 1 parked on its wait");
        let [_, parked] = item_wires();
        assert!(
            !ctx.admission().admits(&[parked]),
            "the parked item is abandoned"
        );

        let escaped = slot
            .lock()
            .unwrap()
            .take()
            .expect("item 1 stored its step future");
        let awaited = tokio::time::timeout(std::time::Duration::from_secs(1), escaped).await;
        assert!(awaited.is_err(), "the escaped step never resolves");
        assert!(!ran.load(Ordering::SeqCst), "the escaped step never ran");
        assert!(
            !has_start(&client, "escaped"),
            "the escaped step wrote nothing"
        );
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: an item panics on purpose, and test assertions
mod coordinator_tests {
    use crate::DurableContext;
    use crate::builders::map_parallel::CompletionConfig;
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{
        ScriptedClient, drive_handler, named, operation_like, scripted_ctx, wire_err,
    };
    use aws_sdk_lambda::types::{OperationAction, OperationStatus};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Counts the items whose body is running and keeps the peak.
    #[derive(Default)]
    struct InFlight {
        now: AtomicUsize,
        peak: AtomicUsize,
        events: Mutex<Vec<String>>,
    }

    impl InFlight {
        fn enter(&self, idx: usize) {
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            self.events
                .lock()
                .expect("lock")
                .push(format!("start{idx}"));
        }

        fn leave(&self, idx: usize) {
            self.events.lock().expect("lock").push(format!("end{idx}"));
            self.now.fetch_sub(1, Ordering::SeqCst);
        }

        fn position(&self, event: &str) -> usize {
            self.events
                .lock()
                .expect("lock")
                .iter()
                .position(|e| e == event)
                .unwrap_or_else(|| panic!("no event {event}"))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn waiting_item_keeps_slot_and_continues() {
        let client = Arc::new(ScriptedClient::new(|updates, history| {
            if named(updates, "s1", &OperationAction::Succeed).is_none() {
                return Vec::new();
            }
            named(history, "w0", &OperationAction::Start)
                .map(|start| {
                    operation_like(start, OperationStatus::Succeeded)
                        .build()
                        .expect("all required fields set")
                })
                .into_iter()
                .collect()
        }));
        let ctx = scripted_ctx(&client);
        let flight = Arc::new(InFlight::default());
        let body_flight = Arc::clone(&flight);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let batch = ctx
                .map(
                    vec![0_u32, 1, 2],
                    move |child: DurableContext, _item, idx| {
                        let flight = Arc::clone(&body_flight);
                        let client = Arc::clone(&body_client);
                        async move {
                            flight.enter(idx);
                            match idx {
                                0 => {
                                    child.wait(Duration::from_secs(1)).name("w0").await?;
                                    child.step(|_| async { Ok(0_u32) }).name("after0").await?;
                                }
                                1 => {
                                    child
                                        .step(move |_| async move {
                                            // The response that completes w0
                                            // comes after w0 registered.
                                            for _ in 0..1000 {
                                                if client.received("w0", &OperationAction::Start) {
                                                    return Ok(1_u32);
                                                }
                                                tokio::time::sleep(Duration::from_millis(1)).await;
                                            }
                                            Err("w0 never started".into())
                                        })
                                        .name("s1")
                                        .await?;
                                }
                                _ => {
                                    child.step(|_| async { Ok(2_u32) }).name("s2").await?;
                                }
                            }
                            flight.leave(idx);
                            Ok(idx)
                        }
                    },
                )
                .max_concurrency(2)
                .await_batch()
                .await
                .map_err(wire_err)?;
            let statuses: Vec<String> = batch
                .items
                .iter()
                .map(|item| format!("{:?}", item.status))
                .collect();
            Ok(statuses.join(","))
        })
        .await
        .expect("no panic");
        assert_eq!(
            outcome,
            InvocationOutcome::Complete("Succeeded,Succeeded,Succeeded".to_owned()),
            "the batch completes in one invocation"
        );
        assert_eq!(flight.peak.load(Ordering::SeqCst), 2);
        assert!(
            flight.position("end1") < flight.position("start2"),
            "item 2 starts after item 1 finished"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn item_panic_while_others_wait_propagates() {
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ = ctx
                .map(
                    vec![0_u32, 1, 2],
                    |child: DurableContext, _item, idx| async move {
                        if idx == 0 {
                            child
                                .step(|_| async {
                                    tokio::time::sleep(Duration::from_millis(10)).await;
                                    Ok(0_u32)
                                })
                                .await?;
                            panic!("boom while others wait");
                        }
                        child.wait(Duration::from_secs(3600)).await?;
                        Ok(idx)
                    },
                )
                .max_concurrency(3)
                .await;
            Ok("done".to_owned())
        })
        .await;
        let Err(payload) = outcome else {
            panic!("expected the item panic, got {outcome:?}");
        };
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"boom while others wait")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn early_completion_cancels_running_item_while_others_wait() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let batch = ctx
                .map(
                    vec![0_u32, 1, 2, 3],
                    |child: DurableContext, _item, idx| async move {
                        match idx {
                            0 => {}
                            1 | 2 => child.wait(Duration::from_secs(3600)).await?,
                            _ => {
                                child
                                    .step(|_| async {
                                        tokio::time::sleep(Duration::from_millis(800)).await;
                                        Ok(())
                                    })
                                    .name("a")
                                    .await?;
                                child.step(|_| async { Ok(()) }).name("late").await?;
                            }
                        }
                        Ok(idx)
                    },
                )
                .max_concurrency(4)
                .completion(CompletionConfig::with_min_successful(1))
                .await_batch()
                .await
                .map_err(wire_err)?;
            ctx.hold_activity(ctx.coordinator().tasks().wait_idle())
                .await;
            let waits = ctx.coordinator().live_waits();
            let statuses: Vec<String> = batch
                .items
                .iter()
                .map(|item| format!("{:?}", item.status))
                .collect();
            Ok(format!("{}:{waits}", statuses.join(",")))
        })
        .await
        .expect("no panic");
        assert_eq!(
            outcome,
            InvocationOutcome::Complete("Succeeded,Started,Started,Started:0".to_owned())
        );
        assert!(
            client.updates().iter().all(|u| u.name() != Some("late")),
            "the cancelled item wrote no later update"
        );
        assert_eq!(coordinator.tasks().live(), 0);
    }
}
