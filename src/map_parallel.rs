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

use std::future::Future;
use std::sync::Arc;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tracing::Instrument as _;

use crate::BoxError;
use crate::Serdes;
use crate::checkpoint_coalescer::ScopeClose;
use crate::child::CHECKPOINT_SIZE_LIMIT_BYTES;
use crate::context::{DurableContext, SubmittedWrite, run_body};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, ChildFnError, OperationError, OperationErrorKind,
};
use crate::plugin::UserBody;
use crate::serdes::SerdesContext;
use crate::start::StartStep;
use crate::task::{NoRuntime, TaskOutcome, TrackedJoinSet};
use serde_json::value::RawValue;

/// Wire sub-type for map operations.
pub(crate) const MAP_SUB_TYPE: &str = "Map";
/// Wire sub-type for map iteration children.
pub(crate) const MAP_ITERATION_SUB_TYPE: &str = "MapIteration";
/// Wire sub-type for parallel operations.
pub(crate) const PARALLEL_SUB_TYPE: &str = "Parallel";
/// Wire sub-type for parallel branch children.
pub(crate) const PARALLEL_BRANCH_SUB_TYPE: &str = "ParallelBranch";

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
    /// The name history records for the item: the map item namer's result
    /// or the branch name, or `map-item-{index}` or
    /// `parallel-branch-{index}` when that name is missing or empty.
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
    /// Carries the type recorded in the batch checkpoint payload (the
    /// item error's `Cause.ErrorType`): the recorded type of the error the
    /// item body returned (`"Error"` for an untyped error), so error
    /// identity survives checkpoint replay alongside the message. `None`
    /// when the payload recorded no type.
    pub error_type: Option<String>,
    /// Opaque wire `ErrorData` payload recorded with the failure (only
    /// meaningful when status is Failed).
    ///
    /// Passed through verbatim from the failing item's recorded wire
    /// record, never interpreted by the SDK. `None` when the failure
    /// recorded no payload.
    pub error_data: Option<String>,
    /// Recorded wire `StackTrace` frames from the failure (only
    /// meaningful when status is Failed).
    ///
    /// Store-and-expose only, exactly like
    /// [`WireError::stack_trace`](crate::WireError::stack_trace). Empty
    /// when the failure recorded no frames.
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
    /// The recorded name of the failed item: the map item namer's result or
    /// the branch name, or `map-item-{index}` or `parallel-branch-{index}`
    /// when that name is missing or empty.
    pub name: &'a str,
    /// The error message the item failed with.
    pub message: &'a str,
    /// The error type identifier the item failed with, as recorded in the
    /// batch checkpoint payload: the recorded type of the error the item
    /// body returned (`"Error"` for an untyped error).
    /// `None` when the payload recorded no type.
    pub error_type: Option<&'a str>,
}

/// Why the batch completed.
///
/// Records the reason a [`BatchResult`] finished: all items ran, the
/// success threshold was met, the failure tolerance was exceeded, or a
/// custom completion predicate completed the batch as a success or a
/// failure.
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

    /// Returns the items with status [`BatchItemStatus::Succeeded`], in
    /// input order.
    pub fn succeeded(&self) -> impl Iterator<Item = &BatchItem<O>> + '_ {
        self.with_status(BatchItemStatus::Succeeded)
    }

    /// Returns the items with status [`BatchItemStatus::Failed`], in input
    /// order.
    pub fn failed(&self) -> impl Iterator<Item = &BatchItem<O>> + '_ {
        self.with_status(BatchItemStatus::Failed)
    }

    /// Returns the items with status [`BatchItemStatus::Started`], in input
    /// order: the items that were still running when the completion rule
    /// ended the batch.
    pub fn started(&self) -> impl Iterator<Item = &BatchItem<O>> + '_ {
        self.with_status(BatchItemStatus::Started)
    }

    /// Returns the number of [`BatchItemStatus::Started`] items.
    pub fn started_count(&self) -> usize {
        self.started().count()
    }

    /// Returns `Ok(self)` unless an item failed or the batch status is
    /// [`BatchStatus::Failed`].
    ///
    /// This is JS `throwIfError()`. It returns an error for a failure
    /// within the configured tolerance too. `.await` on the builder differs:
    /// it returns an error only when the batch failed.
    ///
    /// # Errors
    ///
    /// The first failed item's error, in input order: the error `.await`
    /// returns for that item, of kind
    /// [`ChildContextErrorKind::ChildFailed`](crate::ChildContextErrorKind::ChildFailed)
    /// with the item's recorded failure. When no item failed and the
    /// status is [`BatchStatus::Failed`], the batch completion error, of
    /// kind
    /// [`ChildContextErrorKind::BatchCompletionFailed`](crate::ChildContextErrorKind::BatchCompletionFailed).
    pub fn into_result(self) -> Result<Self, OperationError> {
        if let Some(item) = self.failed().next() {
            return Err(item_failure_error(item.failure_wire()));
        }
        if self.status() == BatchStatus::Failed {
            return Err(batch_completion_failed_error());
        }
        Ok(self)
    }

    fn with_status(&self, status: BatchItemStatus) -> impl Iterator<Item = &BatchItem<O>> + '_ {
        self.items.iter().filter(move |item| item.status == status)
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
/// In both modes, item `i` of the batch with ID `b` has the ID `b-{i+1}`,
/// in the batch's own namespace, and the operations inside it have the IDs
/// `b-{i+1}-{n}`. A Normal item records a child context at its ID, with the
/// batch as its parent. A Flat item records nothing at its ID, and its
/// operations name the batch as their parent.
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

/// A map item namer: receives the item and its index, and returns the name
/// the caller gives the item.
pub(crate) type ItemNamer<I> = Box<dyn Fn(&I, usize) -> String + Send + Sync>;

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
    pub(crate) encoding: ItemEncoding,
    pub(crate) result_serdes: RS,
    pub(crate) nesting: NestingMode,
    pub(crate) item_namer: Option<ItemNamer<I>>,
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
    RS: Serdes<String>,
{
    /// Executes the map operation and returns the full `BatchResult`.
    ///
    /// The user closure enters the batch engine only through the
    /// [`ItemDispatch`] object, so [`execute_batch`], the checkpoint state
    /// machine, compiles once per result type while every item still runs
    /// as a concrete, unboxed future. The `Vec<O>` view of `.await` is
    /// projected by [`collect_successful`].
    ///
    /// Runs the batch's start step now, on the caller's task, and returns
    /// the future of the rest of the operation (see [`crate::start`]).
    pub(crate) fn execute_batch_result_started(
        self,
    ) -> impl Future<Output = Result<BatchResult<O>, OperationError>> + Send + 'static {
        let started = self.start();
        self.run(started)
    }

    /// The start step of the batch parent (see [`batch_start`]).
    fn start(&self) -> Result<StartStep, OperationError> {
        batch_start(
            &self.ctx,
            &self.op_id,
            self.name.as_deref(),
            MAP_SUB_TYPE,
            self.max_concurrency,
        )
    }

    /// The rest of the map after its start step `started`.
    async fn run(
        self,
        started: Result<StartStep, OperationError>,
    ) -> Result<BatchResult<O>, OperationError> {
        let started = started?;
        let total_items = self.items.len();
        // Every name is computed here, once per input item, while the items
        // are still readable: after `into_item_slots` they sit behind
        // take-once slots. So the namer runs exactly once per input item,
        // including items that never start, and sees the item itself.
        let user_names: Vec<Option<String>> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| self.item_namer.as_ref().map(|namer| namer(item, i)))
            .collect();
        drop(self.item_namer);
        let record_names = record_names_of(&user_names, "map-item");
        let items = into_item_slots(self.items);
        let closure = self.closure;
        let items_ref = Arc::clone(&items);
        let item_ids = ItemIds::new(&self.op_id, total_items);

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
            self.encoding,
            self.result_serdes,
            self.nesting,
            user_names,
            record_names,
            total_items,
            MAP_SUB_TYPE,
            MAP_ITERATION_SUB_TYPE,
            &dispatch,
            &item_ids,
            started,
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
    pub(crate) encoding: ItemEncoding,
    pub(crate) result_serdes: RS,
    pub(crate) nesting: NestingMode,
    pub(crate) branches: Vec<(Option<String>, crate::future::BranchBody<O>)>,
    /// The IDs of the branches (see [`ItemIds`]). `parallel_mixed` builds
    /// its item serdes from the same table.
    pub(crate) item_ids: ItemIds,
}

impl<O, IS, RS> ParallelExecution<O, IS, RS>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<String>,
{
    /// Executes the parallel operation and returns the full `BatchResult`.
    ///
    /// Branch bodies are heterogeneous, so each carries exactly one erased
    /// future ([`crate::future::BranchBody`], built at `Branch::new`); the
    /// batch engine and its per-item state machine are shared, non-generic
    /// code. The `Vec<O>` view of `.await` is projected by
    /// [`collect_successful`], as for map: failed branches within
    /// tolerance are skipped, and the batch errors when the failure
    /// tolerance was exceeded or the predicate completed it as failed.
    ///
    /// Runs the batch's start step now, on the caller's task, and returns
    /// the future of the rest of the operation (see [`crate::start`]).
    pub(crate) fn execute_batch_result_started(
        self,
    ) -> impl Future<Output = Result<BatchResult<O>, OperationError>> + Send + 'static {
        let started = self.start();
        self.run(started)
    }

    /// The start step of the batch parent (see [`batch_start`]).
    fn start(&self) -> Result<StartStep, OperationError> {
        batch_start(
            &self.ctx,
            &self.op_id,
            self.name.as_deref(),
            PARALLEL_SUB_TYPE,
            self.max_concurrency,
        )
    }

    /// The rest of the parallel after its start step `started`.
    async fn run(
        self,
        started: Result<StartStep, OperationError>,
    ) -> Result<BatchResult<O>, OperationError> {
        let started = started?;
        let total = self.branches.len();
        // Split each branch into the name the caller gave (`None` for
        // `Branch::unnamed`) and its erased body (kept in a take-once slot
        // since it runs at most once). The recorded names default a
        // missing or empty name.
        let mut user_names: Vec<Option<String>> = Vec::with_capacity(total);
        let mut slots: Vec<std::sync::Mutex<Option<crate::future::BranchBody<O>>>> =
            Vec::with_capacity(total);
        for (name, body) in self.branches {
            user_names.push(name);
            slots.push(std::sync::Mutex::new(Some(body)));
        }
        let record_names = record_names_of(&user_names, "parallel-branch");

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
            self.encoding,
            self.result_serdes,
            self.nesting,
            user_names,
            record_names,
            total,
            PARALLEL_SUB_TYPE,
            PARALLEL_BRANCH_SUB_TYPE,
            &dispatch,
            &self.item_ids,
            started,
        )
        .await
    }
}

/// The name history records for each batch item: the caller's name when it
/// is present and not empty, else `{prefix}-{index}`, as JS records
/// `item.name || item.id`.
fn record_names_of(user_names: &[Option<String>], prefix: &str) -> Vec<String> {
    user_names
        .iter()
        .enumerate()
        .map(|(i, name)| match name.as_deref() {
            Some(name) if !name.is_empty() => name.to_owned(),
            _ => format!("{prefix}-{i}"),
        })
        .collect()
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
pub(crate) fn collect_successful<O>(
    batch_result: BatchResult<O>,
) -> Result<Vec<O>, OperationError> {
    if let Some(err) = await_error(&batch_result) {
        return Err(err);
    }
    // Failed items within tolerance and items still started at batch
    // completion carry no value, so the `Vec<O>` view skips them.
    Ok(batch_result
        .items
        .into_iter()
        .filter(|item| item.status == BatchItemStatus::Succeeded)
        .filter_map(|item| item.result)
        .collect())
}

/// The error the `.await` of a map or parallel returns for `batch`, or
/// `None` when it returns the values.
///
/// A batch that ended because the failure tolerance was exceeded
/// (including the default fail-fast case) or because the predicate
/// completed it as failed returns the first failed item's error: a
/// `ChildFailed` error that carries the item's recorded wire record (type,
/// message, `error_data`, `stack_trace`), reserving `Internal` for
/// coordinator defects. When no item failed, the first case returns a
/// batch error and the second a `BatchCompletionFailed` error. Every other
/// batch returns `None`. `collect_successful` and `publish_outcome` both
/// call it, so `parallel` and `parallel_mixed` fail on the same batches
/// with the same error.
fn await_error<O>(batch: &BatchResult<O>) -> Option<OperationError> {
    if !matches!(
        batch.reason,
        CompletionReason::FailureToleranceExceeded | CompletionReason::CustomCompletionFailed
    ) {
        return None;
    }
    if let Some(item) = batch
        .items
        .iter()
        .find(|item| item.status == BatchItemStatus::Failed)
    {
        return Some(item_failure_error(item.failure_wire()));
    }
    Some(match batch.reason {
        CompletionReason::FailureToleranceExceeded => batch_error("failure tolerance exceeded"),
        _ => batch_completion_failed_error(),
    })
}

/// The operation IDs of a batch's items, in the batch's own namespace.
///
/// Item `i` of the batch at the positional ID `b` has the positional ID
/// `b-{i+1}`, as in JS. The table is built from the batch's ID and the item
/// count before anything runs. So an item's ID depends only on its index,
/// not on the polling order, the concurrency, the admission order, an early
/// completion or a replay. The live run, sequential admission, every replay
/// path and the `parallel_mixed` item serdes read item IDs from it.
#[derive(Clone)]
pub(crate) struct ItemIds(Arc<[OperationId]>);

impl ItemIds {
    /// Builds the table of the `count` items of the batch `batch`.
    pub(crate) fn new(batch: &OperationId, count: usize) -> Self {
        Self(
            (1..=count)
                .map(|n| OperationId::new(format!("{}-{n}", batch.positional())))
                .collect(),
        )
    }

    /// The ID of item `index`.
    ///
    /// Every caller passes an index below the item count: the batch engine
    /// iterates `0..total_items`, and the table has `total_items` entries,
    /// and the record readers check each index against `total_items`
    /// before they look it up. So the lookup cannot miss.
    #[expect(clippy::expect_used)]
    // reason: every caller passes an index below the item count the table was built with
    fn id(&self, index: usize) -> &OperationId {
        self.0
            .get(index)
            .expect("every caller passes an index below the item count")
    }
}

/// The item serdes of `parallel_mixed`: each branch's serdes, found from
/// the operation ID of the item that is being encoded or decoded.
///
/// `Serdes::deserialize` receives no item index, but its `SerdesContext`
/// names the item. So `by_op_id` maps the wire ID of each item of the
/// batch's [`ItemIds`] table to the branch index. It is built before the
/// batch runs and never changes. A value encodes itself: it carries its
/// own branch's serdes.
pub(crate) struct MixedItemSerdes {
    by_op_id: std::collections::HashMap<String, usize>,
    decoders: Arc<Vec<Arc<dyn crate::future::BranchDecoder>>>,
}

impl MixedItemSerdes {
    /// Builds the item serdes of the batch whose items have the IDs
    /// `item_ids`, one per decoder.
    pub(crate) fn new(
        item_ids: &ItemIds,
        decoders: Arc<Vec<Arc<dyn crate::future::BranchDecoder>>>,
    ) -> Self {
        let mut by_op_id = std::collections::HashMap::new();
        for index in 0..decoders.len() {
            by_op_id.insert(item_ids.id(index).wire().to_owned(), index);
        }
        Self { by_op_id, decoders }
    }
}

impl Serdes<crate::future::MixedValue> for MixedItemSerdes {
    async fn serialize(
        &self,
        value: crate::future::MixedValue,
        context: SerdesContext,
    ) -> Result<String, BoxError> {
        value.0.encode(context).await
    }

    async fn deserialize(
        &self,
        wire: String,
        context: SerdesContext,
    ) -> Result<crate::future::MixedValue, BoxError> {
        let decoder = self
            .by_op_id
            .get(context.operation_id())
            .and_then(|index| self.decoders.get(*index))
            .ok_or_else(|| {
                format!(
                    "parallel_mixed: no branch has operation ID \"{}\"",
                    context.operation_id()
                )
            })?;
        decoder.decode(wire, context).await
    }
}

/// Publishes each settled branch of a `parallel_mixed` batch to its handle,
/// and returns what the `.await` of `parallel_mixed` returns.
///
/// An `Err` from the batch publishes nothing. For an `Ok` batch, every
/// settled item is published, whatever [`await_error`] returns: a
/// succeeded item with a value fills its handle with the value, and a
/// failed item fills its handle with the error `parallel(...).await`
/// returns for it. Every other item publishes nothing, so a handle has a
/// result exactly when its branch settled. A failed item whose index names
/// no branch is skipped, because the item indexes of a recorded batch are
/// not checked against the branch count.
pub(crate) fn publish_outcome(
    result: Result<BatchResult<crate::future::MixedValue>, OperationError>,
    decoders: &[Arc<dyn crate::future::BranchDecoder>],
) -> Result<CompletionReason, OperationError> {
    let batch = result?;
    let error = await_error(&batch);
    let reason = batch.reason;
    for item in batch.items {
        match item.status {
            BatchItemStatus::Succeeded => {
                if let Some(value) = item.result {
                    value.0.publish();
                }
            }
            BatchItemStatus::Failed => {
                if let Some(decoder) = decoders.get(item.index) {
                    decoder.publish_error(item_failure_error(item.failure_wire()));
                }
            }
            BatchItemStatus::Started => {}
        }
    }
    error.map_or(Ok(reason), Err)
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
type ItemJoin<O> = (usize, Result<ItemOutcome<O>, OperationError>);

/// One item's terminal outcome, plus the wire string its child produced
/// when it succeeded: the string the child serialized for its record or its
/// size check, or the string replay read from its record. The batch record
/// takes the item value from it, so it needs no second serdes call.
type ItemOutcome<O> = (BatchItem<O>, Option<String>);

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
    /// The info the item's `on_operation_start` received; a live body's
    /// user-function hooks receive it with the item context's replay flag
    /// (`user_function_info`).
    start_info: Arc<crate::plugin::OperationInfo>,
    /// The item serdes, shared across items behind one `Arc` (the
    /// forwarding `impl Serdes for Arc<S>` makes the handle itself a
    /// serdes).
    serdes: Arc<IS>,
    /// The item's `START` write, queued at dispatch and still in flight.
    /// The outcome write takes it and waits for it. `None` for a flat
    /// item, and for an item whose record already existed. A mutex keeps
    /// the request `Sync`, so the item future can hold a reference to it.
    start: std::sync::Mutex<Option<SubmittedWrite>>,
}

impl<IS> ItemRequest<IS> {
    /// Waits for the item's retained `START` write before the item writes
    /// its outcome `action` (see [`DurableContext::await_retained_start`]).
    async fn await_start(&self, action: OperationAction) -> Result<(), OperationError> {
        let start = self
            .start
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        self.ctx
            .await_retained_start(self.child_op_id.wire(), start, action)
            .await
    }
}

/// What [`item_before`] decided: the item is already resolved from the
/// checkpoint log, or the body must run in the prepared child context.
enum ItemPrelude<O> {
    /// Recorded terminal outcome decoded from the checkpoint log.
    Done(ItemOutcome<O>),
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
        let child_ctx =
            req.ctx
                .new_flat_child(&child_positional, &req.parent_wire, Some(&req.item_name));
        return Ok(ItemPrelude::Run { child_ctx });
    }

    // Create the child context. The child's `Start` was already
    // checkpointed by the coordinator at dispatch (the coordinator is the
    // sole writer of child starts; see `execute_batch`), or exists from a
    // prior invocation.
    Ok(ItemPrelude::Run {
        child_ctx: req.ctx.new_child(&child_positional, Some(&req.item_name)),
    })
}

/// Post-closure half of one batch item: outcome checkpointing and
/// [`BatchItem`] assembly. Generic only over the result type `O`, no user
/// closure reaches this code.
#[expect(clippy::too_many_lines)] // reason: FLAT/NORMAL outcome checkpointing reads better in one flow
async fn item_after<O, IS>(
    req: &ItemRequest<IS>,
    outcome: ItemBody<O>,
    replaying: bool,
) -> Result<ItemOutcome<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    // A live run's failure keeps the one wire its end hooks saw.
    let outcome = match outcome {
        ItemBody::Ran(ran) => ran.into_parts(),
        ItemBody::Unhooked(result) => result.map_err(|error| {
            let wire = error.failure_wire();
            (error, wire)
        }),
    };
    let child_wire = req.child_op_id.wire().to_owned();
    let serdes_ctx = SerdesContext::new(child_wire.clone(), req.ctx.execution_arn());
    let serdes = &req.serdes;
    // The item's own terminal update is written through the batch's
    // context, as a child context's is written through its parent. The
    // item's scope already closed when its body returned, so the item's
    // owners would reject it. The batch scope closes at early completion,
    // so a running item's terminal update is rejected from then on.
    let writer = &req.ctx;

    // FLAT nesting emits no child-context events: only the value transform
    // (round-trip for live == replay consistency) happens here. A terminal
    // item re-runs only to rebuild a `ReplayChildren` result (see
    // `item_before`). Its record is already closed, and a checkpoint backend
    // rejects a second close of a CONTEXT. So it takes the same path and
    // writes nothing.
    if req.nesting == NestingMode::Flat || req.is_terminal {
        let settled = match outcome {
            Ok(value) => {
                // A result serialization failure is LOCAL and
                // deterministic: the item closure already ran, so the item
                // settles as a failed `BatchItem` instead of yielding a
                // catchable coordinator error with no record (issue #43).
                // FLAT items have no per-child record: the failure is
                // recorded inside the parent batch's record when
                // the parent checkpoints, and replay reconstructs the same
                // failed item from it, so live and replayed batches agree.
                let round_trip = async {
                    let name = item_fault_name(&req.item_name);
                    let serialized =
                        serialize_value(&req.ctx, name, value, serdes, serdes_ctx.clone()).await?;
                    let deserialized: O =
                        deserialize_value(&req.ctx, name, serialized.clone(), serdes, serdes_ctx)
                            .await?;
                    Ok::<_, OperationError>((deserialized, serialized))
                };
                match round_trip.await {
                    Ok((deserialized, serialized)) => Ok((
                        BatchItem {
                            index: req.index,
                            name: req.item_name.clone(),
                            status: BatchItemStatus::Succeeded,
                            result: Some(deserialized),
                            error_message: None,
                            error_type: None,
                            error_data: None,
                            stack_trace: Vec::new(),
                        },
                        Some(serialized),
                    )),
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        Ok((
                            BatchItem::failed_from_wire(req.index, req.item_name.clone(), &wire),
                            None,
                        ))
                    }
                }
            }
            Err((_error, wire)) => {
                // The wire record is derived from the error the body
                // returned: its recorded type, and its flattened chain as
                // the message.
                Ok((
                    BatchItem::failed_from_wire(req.index, req.item_name.clone(), &wire),
                    None,
                ))
            }
        };
        // A flat item reports its end as a virtual child. A terminal
        // Normal item that re-ran to rebuild its result fires nothing.
        if req.nesting == NestingMode::Flat
            && let Ok((item, _)) = &settled
        {
            // A settled flat item either succeeded or failed.
            if matches!(item.status, BatchItemStatus::Succeeded) {
                item_ended(
                    req,
                    crate::OperationStatus::Succeeded,
                    None,
                    None,
                    replaying,
                );
            } else {
                let wire = item.failure_wire();
                item_ended(
                    req,
                    crate::OperationStatus::Failed,
                    None,
                    Some(&wire),
                    replaying,
                );
            }
        }
        return settled;
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
                    req.await_start(OperationAction::Fail).await?;
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
                    item_ended(
                        req,
                        crate::OperationStatus::Failed,
                        None,
                        Some(&wire),
                        replaying,
                    );
                    return Ok((
                        BatchItem::failed_from_wire(req.index, req.item_name.clone(), &wire),
                        None,
                    ));
                }
            };
            let mut builder = OperationUpdate::builder()
                .id(child_wire.clone())
                .r#type(OperationType::Context)
                .sub_type(req.child_sub_type.clone())
                .action(OperationAction::Succeed)
                .parent_id(req.parent_wire.clone())
                .name(req.item_name.clone());

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
            // The end hook reports the payload the update carries: none for
            // a `ReplayChildren` result.
            let result = update.payload().map(str::to_owned);

            req.await_start(OperationAction::Succeed).await?;
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
            item_ended(
                req,
                crate::OperationStatus::Succeeded,
                result,
                None,
                replaying,
            );

            // Round-trip deserialize for live == replay consistency.
            let deserialized: O = deserialize_value(
                &req.ctx,
                item_fault_name(&req.item_name),
                serialized.clone(),
                serdes,
                serdes_ctx,
            )
            .await?;

            Ok((
                BatchItem {
                    index: req.index,
                    name: req.item_name.clone(),
                    status: BatchItemStatus::Succeeded,
                    result: Some(deserialized),
                    error_message: None,
                    error_type: None,
                    error_data: None,
                    stack_trace: Vec::new(),
                },
                Some(serialized),
            ))
        }
        Err((_error, wire)) => {
            // Checkpoint failure. The wire record is derived from the
            // error the body returned, so the type is the body error's
            // recorded type, the message is its flattened chain, and
            // `error_data`/`stack_trace` pass through the boundary.
            let builder = OperationUpdate::builder()
                .id(child_wire.clone())
                .r#type(OperationType::Context)
                .sub_type(req.child_sub_type.clone())
                .action(OperationAction::Fail)
                .parent_id(req.parent_wire.clone())
                .name(req.item_name.clone())
                .error(wire.to_error_object());

            #[expect(clippy::expect_used)] // reason: all required fields are set above
            let update = builder
                .build()
                .expect("all required OperationUpdate fields set");

            req.await_start(OperationAction::Fail).await?;
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
            item_ended(
                req,
                crate::OperationStatus::Failed,
                None,
                Some(&wire),
                replaying,
            );

            Ok((
                BatchItem::failed_from_wire(req.index, req.item_name.clone(), &wire),
                None,
            ))
        }
    }
}

/// The body result [`item_after`] writes: a live run through the
/// user-function runner, or a result that ran no hook (a `ReplayChildren`
/// rebuild, or a branch body that could not be taken).
enum ItemBody<O> {
    Ran(crate::plugin::Ran<O, ChildFnError>),
    Unhooked(Result<O, ChildFnError>),
}

/// Thin generic wrapper around one map item: the ONLY code monomorphized
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
            let body = UserBody::Context(child_ctx.prefix().to_owned());
            let item_ctx = child_ctx.clone();
            // The item's scope closes when the body returns, and also when
            // this future is dropped before that (see `item_scope`).
            let scope = item_scope(&req);
            let outcome = if req.is_terminal {
                // A terminal item re-runs only to rebuild a `ReplayChildren`
                // result: no frame and no user-function hook.
                let outcome =
                    run_body(body.into(), async move { run_item(child_ctx, index).await })
                        .instrument(span)
                        .await;
                drop(scope);
                ItemBody::Unhooked(outcome)
            } else {
                let info = user_function_info(&req, &item_ctx);
                ItemBody::Ran(
                    // Only the body carries the span: the hooks and wraps
                    // run outside it, the body inside them. The scope
                    // closes inside the body future, so it is closed
                    // before the end hooks, as for a child context.
                    crate::plugin::run_user_function(
                        item_ctx.shared_plugins(),
                        info,
                        body,
                        span,
                        move || async move {
                            let outcome = run_item(child_ctx, index).await;
                            drop(scope);
                            outcome
                        },
                        ChildFnError::failure_wire,
                    )
                    .await,
                )
            };
            item_after(&req, outcome, item_ctx.is_replaying()).await
        }
    }
}

/// The info a map item's or parallel branch's user-function hooks receive:
/// the value `on_operation_start` received, with the replay flag set to the
/// item context's `is_replaying()` when the body starts, as JS sets
/// `isReplay: childReplayMode !== ExecutionMode`. The start hook keeps its
/// record-based flag.
fn user_function_info<IS>(
    req: &ItemRequest<IS>,
    item_ctx: &DurableContext,
) -> Arc<crate::plugin::OperationInfo> {
    let is_replay = item_ctx.is_replaying();
    if req.start_info.is_replay() == is_replay {
        Arc::clone(&req.start_info)
    } else {
        Arc::new((*req.start_info).clone().with_replay(is_replay))
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
        ItemPrelude::Done(item) => Ok(item),
        ItemPrelude::Run { child_ctx } => {
            let span = child_ctx.replay_span();
            let item_ctx = child_ctx.clone();
            let scope = item_scope(&req);
            let outcome = match take_branch_body(&slots, req.index) {
                Ok(body) => {
                    // `start` injects the child context and returns the
                    // branch's single erased future, awaited here exactly
                    // like a map item's body. The factory closure runs
                    // inside the erased future, so wrapping the future
                    // encloses the closure call.
                    let active = UserBody::Context(child_ctx.prefix().to_owned());
                    if req.is_terminal {
                        // Rebuilding a `ReplayChildren` result: no frame
                        // and no user-function hook.
                        let outcome = run_body(active.into(), body.start(child_ctx))
                            .instrument(span)
                            .await;
                        drop(scope);
                        ItemBody::Unhooked(outcome)
                    } else {
                        let info = user_function_info(&req, &item_ctx);
                        ItemBody::Ran(
                            // Only the body carries the span, as for a
                            // map item. The scope closes inside the body
                            // future, so it is closed before the end
                            // hooks, as for a map item.
                            crate::plugin::run_user_function(
                                item_ctx.shared_plugins(),
                                info,
                                active,
                                span,
                                move || async move {
                                    let outcome = body.start(child_ctx).await;
                                    drop(scope);
                                    outcome
                                },
                                ChildFnError::failure_wire,
                            )
                            .await,
                        )
                    }
                }
                Err(err) => {
                    drop(scope);
                    ItemBody::Unhooked(Err(err))
                }
            };
            item_after(&req, outcome, item_ctx.is_replaying()).await
        }
    }
}

/// The scope of one item, normal or flat: it closes when the item's body
/// returns, and when the item's future is dropped before that because the
/// batch aborted the item (see `Admission`). An operation the body started
/// and did not await then writes no further checkpoint, and its task
/// stops. The item's own terminal update is written through the batch's
/// context, so it is admitted.
fn item_scope<IS>(req: &ItemRequest<IS>) -> ScopeClose {
    ScopeClose::new(req.ctx.admission(), req.child_op_id.wire())
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
        // The item runs inside the batch context's frames.
        set.spawn(
            crate::task::TaskContext::current(),
            &ownership,
            async move {
                let index = req.index;
                (index, run_single_item(req, run_item).await)
            },
        )
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
        // The branch runs inside the batch context's frames.
        set.spawn(
            crate::task::TaskContext::current(),
            &ownership,
            async move {
                let index = req.index;
                (index, run_branch_item(req, slots).await)
            },
        )
    }
}

/// Mutable coordinator bookkeeping shared by the join-drain paths.
struct BatchProgress<O> {
    /// Running completion statistics plus the first trigger to fire.
    tracker: CompletionTracker,
    /// Terminal outcomes, positional by item index.
    results: Vec<Option<BatchItem<O>>>,
    /// The wire string each succeeded item's child produced, positional by
    /// item index. The batch record takes each item value from it.
    wires: Vec<Option<String>>,
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
            wires: vec![None; total_items],
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
    /// statistics, its result slot and its wire slot.
    fn settle_outcome(
        &mut self,
        completion_cfg: &crate::builders::map_parallel::CompletionConfig,
        total_items: usize,
        index: usize,
        (item, wire): ItemOutcome<O>,
    ) {
        self.tracker
            .settle(completion_cfg, total_items, index, item.status);
        if let Some(slot) = self.results.get_mut(index) {
            *slot = Some(item);
        }
        if let Some(slot) = self.wires.get_mut(index) {
            *slot = wire;
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
    fn settle_joined(
        &self,
        joined: TaskOutcome<ItemJoin<O>>,
        progress: &mut BatchProgress<O>,
    ) -> Result<(), OperationError> {
        let (index, outcome) = joined.into_value();
        // A coordinator-level failure surfaces; the caller's join set drop
        // aborts the remaining branches.
        let item = outcome?;
        progress.settle_outcome(self.completion_cfg, self.total_items, index, item);
        Ok(())
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
                    // A reconstruction fires no hook, so nothing reads it.
                    start_info: Arc::new(item_info(
                        self.ctx,
                        op_id.wire(),
                        self.child_sub_type,
                        item_name,
                    )),
                    serdes: Arc::clone(self.serdes),
                    start: std::sync::Mutex::new(None),
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
                self.settle_joined(joined, progress)
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
    encoding: ItemEncoding,
    result_serdes: RS,
    nesting: NestingMode,
    user_names: Vec<Option<String>>,
    record_names: Vec<String>,
    total_items: usize,
    parent_sub_type: &str,
    child_sub_type: &str,
    dispatch: &dyn ItemDispatch<O, IS>,
    item_ids: &ItemIds,
    started: StartStep,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<String>,
{
    // Share the item serdes across items behind one `Arc`: the forwarding
    // `impl Serdes for Arc<S>` makes the handle itself a serdes, so no
    // `Clone` bound is required of the user's implementation.
    let serdes = Arc::new(serdes);

    // 1. The start step ran (see `batch_start`): admission, task
    // ownership, the `max_concurrency` check, the replay identity check,
    // and the parent START with its start hook. The items do not wait for
    // a queued START: the parent's outcome write does.
    let (recorded, parent_start) = match started {
        StartStep::Refused => return std::future::pending().await,
        StartStep::Recorded(view) => (Some(view), None),
        StartStep::Queued(write) => (None, Some(write)),
    };
    // A fresh `START` is a live start; a record resumes the batch.
    let resumed = recorded.is_some();

    // The caller computed both name lists once per input item, before the
    // items moved. `user_names` holds the name the caller gave, which only
    // the completion predicate sees. `record_names` holds the name history
    // records, which replay identity validation, the child checkpoints and
    // the result use.
    //
    // `record_names` has one entry per input item, and every caller of
    // this closure passes an index below `total_items`. So the lookup
    // cannot miss.
    #[expect(clippy::expect_used)]
    // reason: `record_names` has one entry per input item and every index is below `total_items`
    let item_name_of = |i: usize| -> String {
        record_names
            .get(i)
            .expect("record_names has one entry per input item")
            .clone()
    };

    let parent_positional = parent_op_id.positional().to_owned();
    let parent_wire = parent_op_id.wire().to_owned();

    // The batch's own namespace. The batch's START, SUCCEED and FAIL writes
    // and the whole-batch serdes stay on the caller's context `ctx`,
    // because they name the caller's parent. Item replay, item serdes
    // faults and the item contexts use `batch_ctx`, and item IDs come from
    // `item_ids`, so the batch never claims an ID from the caller.
    let batch_ctx = ctx.new_child(&parent_positional, parent_name.as_deref());

    // 2. Check if the parent batch is already terminal in the checkpoint log.
    // Identity validation and the status read happen in one read-guard pass;
    // the terminal payload/error projection is cloned only when the batch is
    // actually terminal and must be replayed.
    // When the parent is terminal with `replay_children` and its payload
    // parses as a batch decision record, replay OBEYS the record instead of
    // re-deriving the outcome. `Some` constrains the re-execution below to
    // the recorded admission set and completion reason.
    let mut replay_plan: Option<ReplayPlan> = None;

    // A non-terminal record (Started/Pending) continues from where it left
    // off. The start step fired its replay start hook.
    if let Some(view) = recorded
        && view.status.is_terminal()
    {
        let snapshot = ctx
            .checkpoint_terminal_replay(&parent_positional)
            .ok_or_else(|| batch_error("terminal batch has no checkpoint record"))?;
        let serdes_ctx = SerdesContext::new(&parent_wire, ctx.execution_arn());

        // A successful batch too large to checkpoint inline recorded a
        // decision record (`totalCount`, `completionReason`, index set)
        // instead of its payload. A payload that fails to parse is
        // corrupt and fails the replay.
        let decision =
            if matches!(snapshot.status, CheckpointStatus::Succeeded) && snapshot.replay_children {
                Some(parse_replay_children_payload(snapshot.result.as_deref())?)
            } else {
                None
            };

        if let Some(record) = decision {
            let plan = ReplayPlan::from_record(&record, total_items)?;
            // Fast path (Normal nesting): decode every completed item
            // straight from its child record (after validating each
            // child's replay identity), report items in the started
            // set as STARTED without consulting their child
            // checkpoints, and omit every index at or beyond
            // `totalCount`. Flat items have no
            // child records, so Flat always reconstructs through the
            // plan-constrained re-execution below.
            if nesting != NestingMode::Flat {
                match replay_batch_from_record(
                    &batch_ctx,
                    &plan,
                    &record_names,
                    &serdes,
                    child_sub_type,
                    item_ids,
                )
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
                encoding.clone(),
                total_items,
                &result_serdes,
                &serdes_ctx,
                item_ids,
            )
            .await
            {
                Ok(result) => {
                    // Recorded terminal batch record returned without
                    // re-running the batch (see `crate::observability`).
                    ctx.emit_operation_replayed(
                        &parent_wire,
                        parent_name.as_deref(),
                        "Context",
                        Some(parent_sub_type),
                        view.attempt,
                    );
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

    let completion_cfg = completion.unwrap_or_default();

    // Validate completion config. A zero-item batch reads no threshold, so
    // it succeeds whatever the config holds.
    if total_items > 0
        && let Err(msg) = completion_cfg.validate()
    {
        return Err(batch_error(&msg));
    }

    // 4 to 10: the batch-producing section, from the zero-item result or
    // the first record read to the batch and its completion reason. On a
    // live or resumed batch it is the container's user function: its hooks
    // and wraps run through the runner, under the batch child context's
    // body and span, and its end hooks always see `Ok`, because every item
    // failure is in the batch. A rebuild under a decision record polls it
    // with no frame and no user-function hook.
    let rebuild = replay_plan.is_some();
    let active = UserBody::Context(batch_ctx.prefix().to_owned());
    let span = batch_ctx.replay_span();
    let produce = || {
        produce_batch(BatchSection {
            ctx: &ctx,
            batch_ctx: &batch_ctx,
            max_concurrency,
            completion_cfg: &completion_cfg,
            serdes: &serdes,
            nesting,
            user_names,
            item_name_of: &item_name_of,
            total_items,
            child_sub_type,
            dispatch,
            item_ids,
            parent_wire: &parent_wire,
            replay_plan,
        })
    };
    let produced = if rebuild {
        run_body(active.into(), produce()).instrument(span).await
    } else {
        // The info `on_operation_start` received: a fresh `START` is live,
        // a resumed record a replay. The user-function hooks see the
        // batch child context's replay state when the body starts.
        let info = container_info(&ctx, &parent_wire, parent_sub_type, parent_name.as_deref());
        let info = if resumed {
            info
        } else {
            info.with_status(crate::OperationStatus::Started)
        };
        let info = Arc::new(info.with_replay(batch_ctx.is_replaying()));
        match crate::plugin::run_user_function(
            ctx.shared_plugins(),
            info,
            active,
            span,
            move || async move { Ok::<_, std::convert::Infallible>(produce().await) },
            crate::plugin::never_fails,
        )
        .await
        .into_parts()
        {
            Ok(produced) => produced,
            Err((never, _)) => match never {},
        }
    };
    let Produced {
        result: batch_result,
        wires: final_wires,
    } = produced?;
    // 8b. Under a replay plan, the batch outcome is already durably
    // recorded: NOTHING is re-checkpointed.
    if rebuild {
        return Ok(batch_result);
    }

    // 11. Build the batch record text from the item wire strings (no
    // serdes call), then pass it through the SAME `serialize_value` helper
    // as every other path. `result_serdes` receives the record's JSON text
    // and returns the payload to store.
    let payload = from_batch_result(&batch_result, &final_wires, &encoding);
    // The serialization runs after the container's body ended `Ok`, so its
    // failure keeps its own wire (#126 rule 7.6): the returned error
    // carries `serialization_failure_wire`, as a step's or a child's does.
    // The container writes no outcome record for it.
    let serialized_payload = serialize_value(
        &ctx,
        parent_name.as_deref(),
        payload,
        &result_serdes,
        SerdesContext::new(&parent_wire, ctx.execution_arn()),
    )
    .await
    .map_err(|err| {
        let wire = crate::error::serialization_failure_wire(&err);
        err.with_wire(wire)
    })?;

    // The JS summary travels alongside the payload: when the payload is
    // too large to checkpoint inline, the summary is written in its place
    // so replay can obey the recorded outcome instead of re-deriving it.
    let summary_json = oversized_batch_summary(&batch_result, parent_sub_type);

    // 12. Checkpoint the parent SUCCEED.
    checkpoint_batch_success_serialized(
        &ctx,
        &parent_wire,
        parent_name.as_deref(),
        parent_sub_type,
        &serialized_payload,
        &summary_json,
        parent_start,
    )
    .await?;

    Ok(batch_result)
}

/// The output of a batch's producing section: the batch, and the wire
/// string each reported item's child produced, in the same order, for the
/// batch record.
struct Produced<O> {
    result: BatchResult<O>,
    wires: Vec<Option<String>>,
}

/// The inputs of [`produce_batch`].
struct BatchSection<'a, O, IS> {
    ctx: &'a DurableContext,
    batch_ctx: &'a DurableContext,
    max_concurrency: Option<usize>,
    completion_cfg: &'a crate::builders::map_parallel::CompletionConfig,
    serdes: &'a Arc<IS>,
    nesting: NestingMode,
    user_names: Vec<Option<String>>,
    item_name_of: &'a (dyn Fn(usize) -> String + Send + Sync),
    total_items: usize,
    child_sub_type: &'a str,
    dispatch: &'a dyn ItemDispatch<O, IS>,
    item_ids: &'a ItemIds,
    parent_wire: &'a str,
    replay_plan: Option<ReplayPlan>,
}

/// The batch-producing section of [`execute_batch`], steps 4 to 10: the
/// zero-item result, or every item's record read, the items' dispatch and
/// settlement, and the batch with its completion reason. It returns `Ok`
/// with every item failure in the batch. The batch's scope closes inside
/// it, when dispatch ends.
#[expect(clippy::too_many_lines)]
// reason: batch coordination has distinct phases (read, schedule, collect) that read better as one flow
async fn produce_batch<O, IS>(
    section: BatchSection<'_, O, IS>,
) -> Result<Produced<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let BatchSection {
        ctx,
        batch_ctx,
        max_concurrency,
        completion_cfg,
        serdes,
        nesting,
        user_names,
        item_name_of,
        total_items,
        child_sub_type,
        dispatch,
        item_ids,
        parent_wire,
        replay_plan,
    } = section;

    // 4. Empty collection: the batch succeeds at once. As in JS, a
    // zero-item batch asks the predicate once, on an empty view; without a
    // predicate the decision is `Continue`, which records `AllCompleted`.
    if total_items == 0 {
        let reason =
            custom_reason(completion_cfg.predicate_decision(&BatchStats::from_counts(0, 0, &[])))
                .unwrap_or(CompletionReason::AllCompleted);
        return Ok(Produced {
            result: BatchResult {
                items: Vec::new(),
                reason,
            },
            wires: Vec::new(),
        });
    }

    // 5. Determine effective concurrency.
    let concurrency = max_concurrency.unwrap_or(total_items).max(1);

    // 6. Read every item's record up front (concurrent path). Each item's
    // ID comes from `item_ids`, so it depends on its index alone. Reading
    // is LOCAL, no `Start` is checkpointed here; the coordinator writes
    // each child's `Start` at dispatch, so a child is admitted (and
    // recorded) only when a concurrency slot is actually free for it.
    // The sequential path (`concurrency == 1`) reads each item's record at
    // the dispatch cursor instead.
    let mut pre_claimed: Vec<PreClaimed> = Vec::with_capacity(total_items);
    if concurrency > 1 {
        for i in 0..total_items {
            let child_op_id = item_ids.id(i).clone();
            // Under a replay plan, only indexes the record admits as
            // COMPLETED have their child checkpoints consulted. Started-set
            // entries are reconstructed from the record alone and
            // never-started entries are omitted, so reading (and
            // identity-validating) their views here would violate the
            // record's contract and could raise a spurious
            // non-determinism error for a record replay never touches.
            let consult_checkpoint = replay_plan.as_ref().is_none_or(|plan| plan.admits(i));
            let (has_record, is_terminal) = if consult_checkpoint {
                let child_positional = child_op_id.positional().to_owned();
                let child_wire = child_op_id.wire().to_owned();
                // Non-determinism detection on child items happens inside
                // the validated view fetch; only the status is consumed
                // here.
                let view = ctx.checkpoint_view_validated(
                    &child_positional,
                    &child_wire,
                    "Context",
                    Some(child_sub_type),
                    Some(&item_name_of(i)),
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
    let mut progress = BatchProgress::<O>::new(user_names);
    // A batch rebuilt under its decision record already completed. The
    // record holds its reason, so the rebuild evaluates no completion rule.
    if replay_plan.is_some() {
        progress.tracker.stop_evaluating();
    }
    let mut running: usize = 0;
    // The batch's scope: every item context runs inside it. It closes at
    // early completion, before the running items are aborted, and when
    // this future ends or is dropped (see `Admission`).
    let mut batch_scope = Some(ScopeClose::new(ctx.admission(), parent_wire));
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
        ctx: batch_ctx,
        dispatch,
        completion_cfg,
        total_items,
        nesting,
        parent_wire,
        child_sub_type: &child_sub_type_owned,
        serdes,
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
            // error, its child checkpoints never consulted.
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
            // for sequential mode, read the record of the item's ID now, so
            // the cursor knows whether the item is recorded-terminal before
            // the dispatch gate runs.
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
                let child_op_id = item_ids.id(i).clone();
                // Non-determinism detection on child items happens inside
                // the validated view fetch; only the status is consumed.
                let view = ctx.checkpoint_view_validated(
                    child_op_id.positional(),
                    child_op_id.wire(),
                    "Context",
                    Some(child_sub_type),
                    Some(&item_name_of(i)),
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
            // completion rule, except under a replay plan.
            if pc.is_terminal {
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
            // the next settle ends it through early completion. Under a
            // replay plan the recorded reason is reproduced instead.
            if replay_plan.is_none() && progress.tracker.dispatch_gate(completion_cfg) {
                break;
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
            //
            // As in JS, the START is queued, then `on_operation_start`
            // fires, then the START is released, and the body starts
            // without waiting for its response. The item's outcome write
            // waits for it (see `ItemRequest::await_start`). A failed
            // START ends the invocation there, rather than as an item
            // failure, so a tolerant completion config cannot absorb a
            // checkpoint failure.
            let reserved = (nesting != NestingMode::Flat && !pc.has_record).then(|| {
                ctx.reserve_checkpoint_updates(vec![build_child_update(
                    &branch_wire,
                    &item_name,
                    &child_sub_type_owned,
                    parent_wire,
                    OperationAction::Start,
                )])
            });
            let info = item_info(batch_ctx, &branch_wire, &child_sub_type_owned, &item_name);
            let info = if nesting == NestingMode::Flat {
                // A flat item writes no record: it reports as a virtual
                // child, with the replay state its context will start
                // with.
                info.with_virtual(true)
                    .with_replay(batch_ctx.namespace_is_replaying(pc.op_id.positional()))
            } else if pc.has_record {
                info.with_replay(true)
            } else {
                info.with_status(crate::OperationStatus::Started)
                    .with_replay(false)
            };
            if let Some(plugins) = ctx.plugins() {
                crate::plugin::operation_started(plugins, &info);
            }
            let start = reserved.map(crate::context::ReservedWrite::release);

            let req = ItemRequest {
                ctx: batch_ctx.clone(),
                child_op_id: pc.op_id,
                index: branch_index,
                is_terminal: pc.is_terminal,
                parent_wire: parent_wire.to_owned(),
                child_sub_type: child_sub_type_owned.clone(),
                item_name: item_name.clone(),
                nesting,
                start_info: Arc::new(info),
                serdes: Arc::clone(serdes),
                start: std::sync::Mutex::new(start),
            };
            dispatch
                .spawn_item(&mut join_set, req)
                .map_err(NoRuntime::operation_error)?;
            running += 1;
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
        env.settle_joined(joined, &mut progress)?;

        // Early completion: the completion rule is met, so the batch ends
        // now, without waiting for its running items (as in JS). The batch
        // scope closes first, so a write a running item issues from now on
        // is rejected, and every task started inside the batch, by a
        // running or a settled item, is aborted. Then the item tasks are
        // aborted and never joined: an outcome of an abandoned item is
        // never observed. Each aborted task stays counted until it unwinds.
        // Under a replay plan the recorded outcome is reproduced instead,
        // so nothing is abandoned.
        if replay_plan.is_none() && progress.tracker.stop_reason.is_some() {
            drop(batch_scope.take());
            join_set.abort_all();
            break;
        }
    }

    // Dispatch ended: the batch's scope closes here, inside the body, so it
    // is closed before the end hooks. At early completion it closed
    // before the running items were stopped.
    drop(batch_scope);
    drop(join_set);

    let BatchProgress {
        tracker,
        mut results,
        wires,
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
    // Each reported item keeps the wire string its child produced, in the
    // same order, for the batch record.
    let (final_items, final_wires): (Vec<BatchItem<O>>, Vec<Option<String>>) = results
        .into_iter()
        .zip(wires)
        .filter_map(|(slot, wire)| slot.map(|item| (item, wire)))
        .unzip();

    // 8b. Under a replay plan, the batch outcome is already durably
    // recorded: the reason is taken verbatim from the decision record and
    // NOTHING is re-checkpointed: the record being obeyed is the parent's
    // terminal state. `final_items` holds the recorded admission set's
    // reconstructed outcomes.
    if let Some(plan) = replay_plan {
        return Ok(Produced {
            result: BatchResult {
                items: final_items,
                reason: plan.reason,
            },
            wires: final_wires,
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

    // Ordered admission makes the ever-started set a contiguous prefix of
    // the input, and every started item is reported (a non-terminal one as
    // STARTED). So `batch_result.items` is that prefix, and its length is
    // the `totalCount` the oversized summary carries: the number of items
    // ever started, NOT the input length.
    Ok(Produced {
        result: BatchResult {
            items: final_items,
            reason,
        },
        wires: final_wires,
    })
}

/// The start step of a batch parent (see [`crate::start`]).
///
/// It returns [`StartStep::Refused`] when admission refuses the batch. It
/// runs the task-ownership check. A `max_concurrency` of zero is a
/// configuration mistake that fails the same way on every retry, and a
/// catchable error would let the handler record a different outcome. So,
/// as in JS, it fails the execution, queues nothing, and returns
/// [`StartStep::Refused`], whose future never resolves. Then it runs the
/// start step: without a record, the parent `START` is queued, then
/// `on_operation_start` fires, then the `START` is released, as JS orders
/// them. A non-terminal record fires the replay start hook.
fn batch_start(
    ctx: &DurableContext,
    op_id: &OperationId,
    name: Option<&str>,
    sub_type: &'static str,
    max_concurrency: Option<usize>,
) -> Result<StartStep, OperationError> {
    if !ctx.admission().admits(ctx.owners()) {
        return Ok(StartStep::Refused);
    }
    ctx.enforce_task_ownership()?;
    if max_concurrency == Some(0) {
        ctx.coordinator()
            .record_fatal("Error".to_owned(), MAX_CONCURRENCY_ZERO_MESSAGE.to_owned());
        return Ok(StartStep::Refused);
    }
    let wire_id = op_id.wire();
    crate::start::start_step(
        ctx,
        &crate::start::StartSpec {
            op_id,
            operation_type: "Context",
            sub_type,
            name,
            order: crate::start::StartOrder::QueueThenHook,
        },
        |view| {
            if view.status.is_terminal() {
                crate::start::Cycle::Record
            } else {
                crate::start::Cycle::Resume
            }
        },
        || build_parent_update(wire_id, name, sub_type, OperationAction::Start, ctx),
        |plugins, is_replay| {
            let info = container_info(ctx, wire_id, sub_type, name);
            let info = if is_replay {
                info.with_replay(true)
            } else {
                info.with_status(crate::OperationStatus::Started)
                    .with_replay(false)
            };
            crate::plugin::operation_started(plugins, &info);
        },
    )
}

/// Replays a terminal batch (parent already succeeded/failed in the log).
///
/// Takes the targeted [`crate::engine::TerminalReplaySnapshot`] projection
/// rather than the full checkpoint record: the status, `replay_children`,
/// and payload/error strings are all this helper reads.
#[expect(clippy::too_many_arguments)] // reason: the record check needs the input length beside the serdes inputs
async fn replay_terminal_batch<O, IS, RS>(
    ctx: &DurableContext,
    name: Option<&str>,
    snapshot: &crate::engine::TerminalReplaySnapshot,
    serdes: &Arc<IS>,
    encoding: ItemEncoding,
    total_items: usize,
    result_serdes: &RS,
    serdes_ctx: &SerdesContext,
    item_ids: &ItemIds,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<String>,
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
            // Read the stored batch record.
            let payload_str = snapshot
                .result
                .clone()
                .ok_or_else(|| batch_error("terminal batch has no result payload"))?;
            // Reverse the result serdes transform: through the same helper
            // every other path uses.
            let record_text: String =
                deserialize_value(ctx, name, payload_str, result_serdes, serdes_ctx.clone())
                    .await?;
            to_batch_result(
                ctx,
                &record_text,
                serdes,
                encoding,
                total_items,
                serdes_ctx.durable_execution_arn(),
                item_ids,
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
) -> Result<ItemOutcome<O>, OperationError>
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
                payload.clone(),
                serdes,
                serdes_ctx.clone(),
            )
            .await?;
            Ok((
                BatchItem {
                    index,
                    name: item_name.to_owned(),
                    status: BatchItemStatus::Succeeded,
                    result: Some(value),
                    error_message: None,
                    error_type: None,
                    error_data: None,
                    stack_trace: Vec::new(),
                },
                Some(payload),
            ))
        }
        CheckpointStatus::Failed => {
            let msg = record.error_message.as_deref().unwrap_or("child failed");
            Ok((
                BatchItem {
                    index,
                    name: item_name.to_owned(),
                    status: BatchItemStatus::Failed,
                    result: None,
                    error_message: Some(msg.to_owned()),
                    error_type: record.error_type.clone(),
                    error_data: record.error_data.clone(),
                    stack_trace: record.stack_trace.clone().unwrap_or_default(),
                },
                None,
            ))
        }
        _ => Err(batch_error(
            "unexpected non-terminal child status in replay",
        )),
    }
}

/// The decision record the Python SDK checkpoints in place of a batch
/// result too large to store inline. This SDK writes the JS summary
/// instead (see [`oversized_batch_summary`]), and reads both.
///
/// `totalCount` is the number of items EVER STARTED, not the input length;
/// ordered admission makes the started set the contiguous prefix
/// `[0, totalCount)`. `completionReason` is the wire form of
/// [`CompletionReason`], and one of `startedIndexes` / `completedIndexes`
/// names the partition of that prefix: `startedIndexes` lists items that
/// began but were not terminal when the batch completed, `completedIndexes`
/// lists the terminal ones. The read side accepts either form and
/// reconstructs the other against the same prefix.
///
/// The JS summary is read into this same shape (see
/// [`parse_replay_children_payload`]), so one [`ReplayPlan`] serves both.
#[derive(Debug, serde::Deserialize)]
struct BatchDecisionRecord {
    /// Number of items ever started (the admitted prefix length).
    #[serde(rename = "totalCount")]
    total_count: usize,
    /// Wire form of the batch's [`CompletionReason`], taken verbatim by
    /// replay.
    #[serde(rename = "completionReason")]
    completion_reason: String,
    /// Indexes (within `[0, totalCount)`) of items that began but were not
    /// terminal when the batch completed.
    #[serde(rename = "startedIndexes", default)]
    started_indexes: Option<Vec<usize>>,
    /// Indexes (within `[0, totalCount)`) of items that reached a terminal
    /// state.
    #[serde(rename = "completedIndexes", default)]
    completed_indexes: Option<Vec<usize>>,
}

/// Reads the payload of a terminal `replay_children` batch parent. An
/// absent payload reads as the empty string. Then, in this order:
///
/// 1. A JSON object with the key `itemStatuses` is the JS summary. Its
///    `itemStatuses` must be a non-empty string of `S`, `F` and `-`; the
///    started prefix is its length, the terminal indexes are its `S` and
///    `F` positions, and the reason is `completionReason`. As in JS,
///    `itemStatuses` is the only per-item field replay reads.
/// 2. A JSON object with `totalCount` and `startedIndexes` or
///    `completedIndexes` is the decision record the Python SDK writes
///    (see [`parse_decision_record`]).
/// 3. Any other payload, including an absent or empty one, is corrupt
///    bookkeeping, and replay fails rather than re-executing items the
///    recorded batch may never have started.
fn parse_replay_children_payload(
    payload: Option<&str>,
) -> Result<BatchDecisionRecord, OperationError> {
    let payload = payload.unwrap_or_default().trim();
    let value: serde_json::Value = serde_json::from_str(payload)
        .map_err(|e| batch_error(&format!("malformed batch decision record: {e}")))?;
    if let Some(statuses) = value.get("itemStatuses") {
        let statuses = statuses
            .as_str()
            .filter(|s| !s.is_empty() && s.chars().all(|c| matches!(c, 'S' | 'F' | '-')))
            .ok_or_else(|| {
                batch_error(
                    "malformed batch decision record: itemStatuses must be a non-empty \
                     string of S, F and -",
                )
            })?;
        let completed = statuses
            .chars()
            .enumerate()
            .filter_map(|(i, c)| (c != '-').then_some(i))
            .collect();
        return Ok(BatchDecisionRecord {
            total_count: statuses.chars().count(),
            completion_reason: value
                .get("completionReason")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            started_indexes: None,
            completed_indexes: Some(completed),
        });
    }
    if value.get("totalCount").is_some()
        && (value.get("startedIndexes").is_some() || value.get("completedIndexes").is_some())
    {
        return parse_decision_record(payload);
    }
    Err(batch_error(
        "malformed batch decision record: the payload is neither a batch summary \
         nor a decision record",
    ))
}

/// Parses a terminal batch's `replay_children` payload as the decision
/// record the Python SDK writes.
///
/// A payload that fails to parse is corrupt bookkeeping: the safe
/// response is an error, never a silent fall-back to unconstrained
/// re-execution that could resurrect items the recorded batch never
/// started.
fn parse_decision_record(payload: &str) -> Result<BatchDecisionRecord, OperationError> {
    serde_json::from_str(payload.trim())
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
/// Each child's ID comes from the batch's [`ItemIds`] table. An admitted
/// child whose own result was too large to checkpoint inline surfaces as
/// the `ReplayChildren` sentinel, and the caller falls back to
/// re-execution, which reads the same table. Items in the started set are
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
    record_names: &[String],
    serdes: &Arc<IS>,
    child_sub_type: &str,
    item_ids: &ItemIds,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let mut items = Vec::new();
    for i in 0..plan.total_count {
        // `ReplayPlan::from_record` rejects a `totalCount` above the input
        // length, and `record_names` has one entry per input item. So the
        // lookup cannot miss.
        #[expect(clippy::expect_used)]
        // reason: `totalCount` is at most the input length, which is the length of `record_names`
        let item_name = record_names
            .get(i)
            .expect("record_names has one entry per input item")
            .clone();
        if !plan.completed.get(i).copied().unwrap_or(false) {
            // STARTED at batch completion: reported from the record with
            // no result and no error; its child checkpoints are never
            // read.
            items.push(BatchItem {
                index: i,
                name: item_name,
                status: BatchItemStatus::Started,
                result: None,
                error_message: None,
                error_type: None,
                error_data: None,
                stack_trace: Vec::new(),
            });
            continue;
        }
        let op_id = item_ids.id(i);
        // Non-determinism detection: validate the claimed child identity
        // against the record before decoding its terminal payload.
        ctx.checkpoint_view_validated(
            op_id.positional(),
            op_id.wire(),
            "Context",
            Some(child_sub_type),
            Some(&item_name),
        )?;
        let serdes_ctx = SerdesContext::new(op_id.wire().to_owned(), ctx.execution_arn());
        let (item, _wire) =
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
/// record is written with `replay_children(true)` and `summary_json`, the
/// JS summary of [`oversized_batch_summary`], as its payload, so replay can
/// obey the recorded outcome (started prefix, terminal items,
/// `completionReason`) instead of re-deriving it by re-execution. A
/// terminal batch is never written without its summary. If the summary
/// exceeds the limit, the checkpoint is refused because replay requires it
/// to reconstruct the recorded outcome.
async fn checkpoint_batch_success_serialized(
    ctx: &DurableContext,
    parent_wire: &str,
    parent_name: Option<&str>,
    parent_sub_type: &str,
    serialized_payload: &str,
    summary_json: &str,
    start: Option<SubmittedWrite>,
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
        if summary_json.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
            // Never write a terminal batch without its summary (see the
            // function docs).
            return Err(batch_error(
                "batch summary exceeds the checkpoint size limit; \
                 refusing to record the batch outcome without it",
            ));
        }
        builder = builder
            .context_options(
                aws_sdk_lambda::types::ContextOptions::builder()
                    .replay_children(true)
                    .build(),
            )
            .payload(summary_json.to_owned());
    } else {
        builder = builder.payload(serialized_payload.to_owned());
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");
    // The end hook reports the payload the update carries.
    let result = update.payload().map(str::to_owned);

    // The outcome waits for the parent's START write, still in flight
    // when this invocation queued it.
    ctx.await_retained_start(parent_wire, start, OperationAction::Succeed)
        .await?;
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
    if let Some(plugins) = ctx.plugins() {
        let info = container_info(ctx, parent_wire, parent_sub_type, parent_name)
            .with_status(crate::OperationStatus::Succeeded)
            .with_result(result)
            .with_replay(false);
        crate::plugin::operation_ended(plugins, &info);
    }

    Ok(())
}

/// The plugin info of the `map` or `parallel` container `wire_id`, created
/// in the caller's context `ctx`.
fn container_info(
    ctx: &DurableContext,
    wire_id: &str,
    sub_type: &str,
    name: Option<&str>,
) -> crate::plugin::OperationInfo {
    crate::plugin::OperationInfo::for_operation(
        ctx,
        wire_id,
        crate::OperationKind::Context,
        sub_type,
        name,
    )
}

/// The plugin info of the map item or parallel branch `wire_id`, created
/// in the batch's own context `batch_ctx`. That context's parent is the
/// container, so the info names the container as its parent.
fn item_info(
    batch_ctx: &DurableContext,
    wire_id: &str,
    sub_type: &str,
    name: &str,
) -> crate::plugin::OperationInfo {
    crate::plugin::OperationInfo::for_operation(
        batch_ctx,
        wire_id,
        crate::OperationKind::Context,
        sub_type,
        Some(name),
    )
}

/// Fires `on_operation_end` for the item of `req` once its outcome is
/// known: `status` with `result` and `error`. A flat item reports as a
/// virtual child with the replay state `replaying` of its context.
fn item_ended(
    req: &ItemRequest<impl Sized>,
    status: crate::OperationStatus,
    result: Option<String>,
    error: Option<&crate::error::WireError>,
    replaying: bool,
) {
    if let Some(plugins) = req.ctx.plugins() {
        let info = item_info(
            &req.ctx,
            req.child_op_id.wire(),
            &req.child_sub_type,
            &req.item_name,
        )
        .with_status(status)
        .with_result(result)
        .with_error(error.cloned());
        let info = if req.nesting == NestingMode::Flat {
            info.with_virtual(true).with_replay(replaying)
        } else {
            info.with_replay(false)
        };
        crate::plugin::operation_ended(plugins, &info);
    }
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
    // The recorded name of the map item or parallel branch, so it appears
    // on the child's history records. It is never empty: a missing or empty
    // caller name defaults to `map-item-{i}` or `parallel-branch-{i}`.
    let builder = OperationUpdate::builder()
        .id(child_wire.to_owned())
        .r#type(OperationType::Context)
        .sub_type(child_sub_type.to_owned())
        .action(action)
        .parent_id(parent_wire.to_owned())
        .name(child_name.to_owned());

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
    // The recorded name is never empty: a missing or empty caller name
    // defaults to `map-item-{i}` or `parallel-branch-{i}`.
    let builder = OperationUpdate::builder()
        .id(child_wire.to_owned())
        .r#type(OperationType::Context)
        .sub_type(child_sub_type.to_owned())
        .action(OperationAction::Fail)
        .parent_id(parent_wire.to_owned())
        .error(wire.to_error_object())
        .name(child_name.to_owned());
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

/// How the batch record stores the value of a succeeded item.
///
/// JS stores each item value as a JSON value. A custom item serdes writes a
/// string that is not JSON in general, so the record stores that string
/// as it is. The builder knows which serdes it holds: it starts with the
/// default JSON serdes, and `.serdes(..)` replaces it. The record itself
/// has no field for the encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ItemEncoding {
    /// Every item uses the default JSON serdes. The record embeds each
    /// value as the JSON value of its child wire string.
    AllJson,
    /// Every item uses a custom serdes. The record stores each value as its
    /// child wire string, in a JSON string.
    AllSerdes,
    /// Each item has its own serdes, as in `parallel_mixed`. Entry `i` is
    /// `true` when item `i` uses the default JSON serdes, and `false` when
    /// it uses a custom serdes. There is one entry per item.
    PerItem(Arc<[bool]>),
}

impl ItemEncoding {
    /// Whether the record embeds the value of item `index` as JSON, rather
    /// than as its wire string in a JSON string.
    fn is_json(&self, index: usize) -> bool {
        match self {
            Self::AllJson => true,
            Self::AllSerdes => false,
            // `parallel_mixed` builds one flag per branch, and every item
            // index of a batch is below its item count: the live batch
            // numbers its items, and a recorded batch with more items than
            // the input fails before any item is decoded. So the lookup
            // cannot miss.
            #[expect(clippy::expect_used)]
            // reason: there is one flag per item, and every item index is below the item count
            Self::PerItem(flags) => *flags
                .get(index)
                .expect("parallel_mixed has one encoding flag per branch"),
        }
    }
}

// The batch record a map or parallel operation checkpoints.
//
// It serializes in the shape of the JS SDK's batch result:
//
// ```json
// {"all": [item, ...], "completionReason": "ALL_COMPLETED"}
// ```
//
// `all` holds one item per started index, in index order. Each item has
// `index`, `status` (`"SUCCEEDED"`, `"FAILED"` or `"STARTED"`) and `name`,
// the recorded item name. JS has no `name` field; this record keeps it so
// that per-item errors name the item after replay. A succeeded item has
// `result`: the exact JSON text the default item serdes wrote, or the
// custom item serdes output as a JSON string. A failed item has `error`, a
// one-cause form of JS's recursive error object:
//
// ```json
// {"ErrorType": "ChildContextError", "ErrorMessage": "...",
//  "Cause": {"ErrorType": "<recorded type>", "ErrorMessage": "..."}}
// ```
//
// `ErrorData` and `StackTrace` appear at both levels when the item
// recorded them. A started item has neither `result` nor `error`.
//
// The SDK writes and reads this record with serde_json only. A JSON item
// value is a `RawValue`, which keeps its text only through serde_json's own
// serializer and deserializer. So the result serdes receives the finished
// record text as a `String`, never the record itself (see
// [`from_batch_result`] and [`parse_batch_record`]).

/// The JS-shaped batch record the SDK writes and replays.
#[derive(Debug, serde::Serialize)]
struct JsBatchRecord {
    all: Vec<JsBatchItem>,
    #[serde(rename = "completionReason")]
    completion_reason: String,
}

/// One item of a [`JsBatchRecord`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct JsBatchItem {
    index: usize,
    status: String,
    /// Always written. A record without it fails replay, because the
    /// per-item errors name the item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// Present exactly when the item succeeded. The value is the exact JSON
    /// text of the item value. A present `null` is a value (the JSON
    /// encoding of `()`), so it reads as `Some`.
    #[serde(
        default,
        deserialize_with = "present_raw_value",
        skip_serializing_if = "Option::is_none"
    )]
    result: Option<Box<RawValue>>,
    /// Present exactly when the item failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<JsErrorObject>,
}

/// The error object of a failed [`JsBatchItem`], and its `Cause`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "PascalCase")]
struct JsErrorObject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error_data: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stack_trace: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cause: Option<Box<JsErrorObject>>,
}

/// Reads a field that is present, including a present `null`, as `Some`,
/// and keeps its exact JSON text. With `#[serde(default)]`, an absent
/// field reads as `None`.
fn present_raw_value<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Box<RawValue>>, D::Error> {
    <Box<RawValue> as serde::Deserialize>::deserialize(deserializer).map(Some)
}

/// The top-level fields of a batch record text, read in one pass.
///
/// Each field is optional, and a field whose value is `null` reads as
/// absent. [`parse_batch_record`] then requires both fields. Other fields
/// are ignored.
#[derive(Debug, Default)]
struct BatchRecordFields {
    all: Option<Vec<JsBatchItem>>,
    completion_reason: Option<String>,
}

impl<'de> serde::Deserialize<'de> for BatchRecordFields {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(BatchRecordFieldsVisitor)
    }
}

/// Visits the top-level map of a batch record. It accepts only a map, so a
/// top-level array or scalar fails before the field pairs are checked.
struct BatchRecordFieldsVisitor;

impl<'de> serde::de::Visitor<'de> for BatchRecordFieldsVisitor {
    type Value = BatchRecordFields;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a batch record object")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut fields = BatchRecordFields::default();
        // A field counts as seen even when its value is `null`, so a
        // duplicate after a `null` still fails.
        let mut seen = [false; 2];
        let [seen_all, seen_completion_reason] = &mut seen;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "all" => read_field_once(&mut map, seen_all, "all", &mut fields.all)?,
                "completionReason" => read_field_once(
                    &mut map,
                    seen_completion_reason,
                    "completionReason",
                    &mut fields.completion_reason,
                )?,
                _ => {
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
        }
        Ok(fields)
    }
}

/// Reads the value of a recognized batch record field into `slot`. A
/// `null` value leaves the field absent. A second occurrence of the field
/// fails, also when the first value was `null`.
fn read_field_once<'de, A, T>(
    map: &mut A,
    seen: &mut bool,
    name: &'static str,
    slot: &mut Option<T>,
) -> Result<(), A::Error>
where
    A: serde::de::MapAccess<'de>,
    T: serde::Deserialize<'de>,
{
    if *seen {
        return Err(<A::Error as serde::de::Error>::duplicate_field(name));
    }
    *seen = true;
    *slot = map.next_value()?;
    Ok(())
}

/// Parses a batch record text in one pass with `serde_json`.
///
/// Both `all` and `completionReason` must be present. A record without
/// one of them fails with an error that names the missing field.
fn parse_batch_record(text: &str) -> Result<JsBatchRecord, String> {
    let fields: BatchRecordFields =
        serde_json::from_str(text).map_err(|err| format!("batch record is not valid: {err}"))?;
    match fields {
        BatchRecordFields {
            all: Some(all),
            completion_reason: Some(completion_reason),
        } => Ok(JsBatchRecord {
            all,
            completion_reason,
        }),
        BatchRecordFields { all: Some(_), .. } => {
            Err("batch record has `all` without `completionReason`".to_owned())
        }
        BatchRecordFields {
            completion_reason: Some(_),
            ..
        } => Err("batch record has `completionReason` without `all`".to_owned()),
        BatchRecordFields { .. } => {
            Err("batch record has neither `all` nor `completionReason`".to_owned())
        }
    }
}

/// The wire form of an item status.
fn item_status_wire(status: BatchItemStatus) -> &'static str {
    match status {
        BatchItemStatus::Succeeded => "SUCCEEDED",
        BatchItemStatus::Failed => "FAILED",
        BatchItemStatus::Started => "STARTED",
    }
}

/// Reads the wire form of an item status.
fn parse_item_status(status: &str) -> Result<BatchItemStatus, OperationError> {
    match status {
        "SUCCEEDED" => Ok(BatchItemStatus::Succeeded),
        "FAILED" => Ok(BatchItemStatus::Failed),
        "STARTED" => Ok(BatchItemStatus::Started),
        other => Err(batch_error(&format!("unknown item status: {other}"))),
    }
}

/// The error object a failed item records: JS's `ChildContextError` with
/// one cause, which carries the item's own recorded type.
fn js_item_error<O>(item: &BatchItem<O>) -> JsErrorObject {
    let message = item.error_message.clone().unwrap_or_default();
    let cause = JsErrorObject {
        error_type: Some(
            item.error_type
                .clone()
                .unwrap_or_else(|| "Error".to_owned()),
        ),
        error_message: Some(message.clone()),
        error_data: item.error_data.clone(),
        stack_trace: item.stack_trace.clone(),
        cause: None,
    };
    JsErrorObject {
        error_type: Some("ChildContextError".to_owned()),
        error_message: Some(message),
        error_data: item.error_data.clone(),
        stack_trace: item.stack_trace.clone(),
        cause: Some(Box::new(cause)),
    }
}

/// Builds the batch record text of a live `BatchResult`.
///
/// `wires` holds, for each item of `result` in the same order, the wire
/// string the item's child produced: the string it serialized for its
/// record or its size check, or the string replay read from its record.
/// Each value comes from that string, so building the record calls no
/// serdes. A JSON item's value is its wire string, unchanged. A
/// custom-serdes item's value is the JSON string encoding of its wire
/// string.
fn from_batch_result<O>(
    result: &BatchResult<O>,
    wires: &[Option<String>],
    encoding: &ItemEncoding,
) -> String {
    let all = result
        .items
        .iter()
        .zip(wires)
        .map(|(item, wire)| {
            // Only a succeeded item has a wire string.
            let value = wire.as_ref().map(|wire| {
                let text = if encoding.is_json(item.index) {
                    wire.clone()
                } else {
                    #[expect(clippy::expect_used)]
                    // reason: serializing a `String` to JSON cannot fail
                    serde_json::to_string(wire).expect("a string serializes to JSON")
                };
                // The default JSON serdes wrote `wire` with
                // `serde_json::to_string`, or read it back with
                // `serde_json::from_str`, so it is valid JSON. A JSON string
                // encoding is valid JSON too.
                #[expect(clippy::expect_used)]
                // reason: the default JSON serdes and a JSON string encoding only produce valid JSON
                RawValue::from_string(text).expect("the item value is valid JSON")
            });
            JsBatchItem {
                index: item.index,
                status: item_status_wire(item.status).to_owned(),
                name: Some(item.name.clone()),
                result: value,
                error: (item.status == BatchItemStatus::Failed).then(|| js_item_error(item)),
            }
        })
        .collect();
    let record = JsBatchRecord {
        all,
        completion_reason: result.reason.as_str().to_owned(),
    };
    #[expect(clippy::expect_used)]
    // reason: the record holds only strings, integers, valid raw JSON and nested records of those, all of which serialize to JSON
    serde_json::to_string(&record).expect("the batch record serializes to JSON")
}

/// The JS summary an oversized batch checkpoints in place of its record.
///
/// A map writes `{"type":"MapResult", totalCount, successCount,
/// failureCount, completionReason, status, itemStatuses}`; a parallel
/// writes `"ParallelResult"` and adds `startedCount`. `totalCount` counts
/// the started items, and `itemStatuses` has one character per started
/// index: `S` succeeded, `F` failed, `-` started and not terminal. The
/// summary is plain JSON: it never goes through `result_serdes`, because
/// it must stay readable even when the result serdes output is what
/// outgrew the limit.
fn oversized_batch_summary<O>(result: &BatchResult<O>, parent_sub_type: &str) -> String {
    let total = result.items.len();
    let succeeded = result.success_count();
    let failed = result.failure_count();
    let item_statuses: String = result
        .items
        .iter()
        .map(|item| match item.status {
            BatchItemStatus::Succeeded => 'S',
            BatchItemStatus::Failed => 'F',
            BatchItemStatus::Started => '-',
        })
        .collect();
    let status = match result.status() {
        BatchStatus::Failed => "FAILED",
        BatchStatus::Succeeded => "SUCCEEDED",
    };
    let parallel = parent_sub_type == PARALLEL_SUB_TYPE;
    let mut summary = serde_json::Map::new();
    let type_name = if parallel {
        "ParallelResult"
    } else {
        "MapResult"
    };
    summary.insert("type".to_owned(), type_name.into());
    summary.insert("totalCount".to_owned(), total.into());
    summary.insert("successCount".to_owned(), succeeded.into());
    summary.insert("failureCount".to_owned(), failed.into());
    if parallel {
        summary.insert(
            "startedCount".to_owned(),
            (total - succeeded - failed).into(),
        );
    }
    summary.insert("completionReason".to_owned(), result.reason.as_str().into());
    summary.insert("status".to_owned(), status.into());
    summary.insert("itemStatuses".to_owned(), item_statuses.into());
    serde_json::Value::Object(summary).to_string()
}

/// Parses a batch record text and converts it back into a `BatchResult`.
///
/// Each item value is decoded with the item serdes and the child's serdes
/// context. `total_items` is the length of the current input.
async fn to_batch_result<O, IS>(
    ctx: &DurableContext,
    record_text: &str,
    serdes: &Arc<IS>,
    encoding: ItemEncoding,
    total_items: usize,
    execution_arn: &str,
    item_ids: &ItemIds,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    let record = parse_batch_record(record_text).map_err(|message| batch_error(&message))?;
    js_to_batch_result(
        ctx,
        record,
        serdes,
        encoding,
        total_items,
        execution_arn,
        item_ids,
    )
    .await
}

/// Converts a JS-form batch record back into a `BatchResult`.
///
/// The child of item `i` has the ID of item `i` in the batch's
/// [`ItemIds`] table, so its serdes context is derived from that ID. A failed item's type, message and data come from `Cause` when
/// it has them, else from the top level; its frames come from `Cause` when
/// it is present.
///
/// The record is checked before any item is decoded. Its `all` must hold
/// the started prefix of the input in index order: item `k` has index `k`,
/// and there are at most `total_items` items. The SDK writes records in
/// that form, but a custom `result_serdes` or a corrupt checkpoint can
/// produce another one. A sparse, duplicate or reordered index would
/// derive a child serdes context that is not the child's own ID, and an
/// index past the input would name no item of the table. So such a record
/// fails the replay with an error.
async fn js_to_batch_result<O, IS>(
    ctx: &DurableContext,
    record: JsBatchRecord,
    serdes: &Arc<IS>,
    encoding: ItemEncoding,
    total_items: usize,
    execution_arn: &str,
    item_ids: &ItemIds,
) -> Result<BatchResult<O>, OperationError>
where
    O: Send + 'static,
    IS: Serdes<O>,
{
    if record.all.len() > total_items {
        return Err(batch_error(&format!(
            "batch record has {} items but the input has {total_items}",
            record.all.len()
        )));
    }
    for (position, item) in record.all.iter().enumerate() {
        if item.index != position {
            return Err(batch_error(&format!(
                "batch record item at position {position} has index {}",
                item.index
            )));
        }
    }
    let reason = CompletionReason::from_wire(&record.completion_reason);
    let mut items = Vec::with_capacity(record.all.len());
    for item in record.all {
        let index = item.index;
        let status = parse_item_status(&item.status)?;
        let name = item
            .name
            .ok_or_else(|| batch_error(&format!("batch record item {index}: missing name")))?;
        let result = match (status, item.result) {
            (BatchItemStatus::Succeeded, Some(value)) => {
                let wire = if encoding.is_json(index) {
                    // The item serdes reads the exact JSON text it wrote.
                    value.get().to_owned()
                } else {
                    serde_json::from_str::<String>(value.get()).map_err(|_| {
                        batch_error(&format!(
                            "batch record item {index}: expected a string result"
                        ))
                    })?
                };
                let child_op_id = item_ids.id(index);
                let item_ctx = SerdesContext::new(child_op_id.wire().to_owned(), execution_arn);
                Some(
                    deserialize_value::<O, _>(
                        ctx,
                        item_fault_name(&name),
                        wire,
                        serdes.as_ref(),
                        item_ctx,
                    )
                    .await?,
                )
            }
            _ => None,
        };
        let error = item.error.map(|top| match top.cause {
            Some(cause) => (
                cause.error_type.or(top.error_type),
                cause.error_message.or(top.error_message),
                cause.error_data.or(top.error_data),
                cause.stack_trace,
            ),
            None => (
                top.error_type,
                top.error_message,
                top.error_data,
                top.stack_trace,
            ),
        });
        let (error_type, error_message, error_data, stack_trace) =
            error.unwrap_or((None, None, None, Vec::new()));
        items.push(BatchItem {
            index,
            name,
            status,
            result,
            error_message,
            error_type,
            error_data,
            stack_trace,
        });
    }
    Ok(BatchResult { items, reason })
}

/// The execution error `max_concurrency(0)` records: JS's wording, with the
/// Rust setter's name.
const MAX_CONCURRENCY_ZERO_MESSAGE: &str =
    "Invalid max_concurrency: 0. Must be a positive number or unset for unlimited concurrency.";

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

    /// The in-memory client's checkpoint responses return no operations, so
    /// no item record exists when the item hooks fire. The item's infos
    /// still name the map as their parent, because the item is created in
    /// the batch's own context.
    #[tokio::test]
    async fn item_hooks_report_container_without_record() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
            None,
            Some(hooks.plugins()),
        );
        let values = ctx
            .map(vec![1_u32], |_child, item, _idx| async move { Ok(item) })
            .name("m")
            .await
            .expect("the map succeeds");
        assert_eq!(values, [1]);
        let map_wire = crate::engine::compute_wire_id_public("1");
        let item = hooks.of("map-item-0");
        let shape: Vec<_> = item
            .iter()
            .map(|r| (r.hook, r.parent_id.as_deref()))
            .collect();
        assert_eq!(
            shape,
            [
                ("start", Some(map_wire.as_str())),
                ("end", Some(map_wire.as_str()))
            ]
        );
    }

    /// A record of the operation at `positional_id` with `status` and no
    /// payload.
    fn record_at(positional_id: &str, status: CheckpointStatus) -> (String, CheckpointRecord) {
        let (wire_id, mut record) = succeeded_record(positional_id, "null");
        record.status = status;
        record.result = None;
        (wire_id, record)
    }

    /// Rule 7.3: a STARTED `Normal` item with no inner record reports a
    /// replay to `on_operation_start` and none to its user-function hooks
    /// and wraps. So does a resumed container whose batch child context
    /// has no inner record.
    #[tokio::test]
    async fn normal_item_without_inner_record_replay_flags() {
        use crate::plugin::recorder::{UfHooks, context};
        let map = |ctx: &DurableContext| {
            ctx.map(vec![1_u32], |_child, item, _idx| async move { Ok(item) })
                .name("m")
        };

        // Run 1: the map and its item 0 are STARTED.
        let hooks = UfHooks::default();
        let (ctx, _client) = context(
            &hooks,
            vec![
                record_at("1", CheckpointStatus::Started),
                record_at("1-1", CheckpointStatus::Started),
            ],
        );
        assert_eq!(map(&ctx).await.unwrap(), [1]);
        assert!(hooks.find("map-item-0", "start").replay);
        let item = hooks.user_function_of("map-item-0", true);
        assert!(item.iter().all(|r| !r.replay), "{item:?}");

        // Run 2: only the map is STARTED.
        let hooks = UfHooks::default();
        let (ctx, _client) = context(&hooks, vec![record_at("1", CheckpointStatus::Started)]);
        assert_eq!(map(&ctx).await.unwrap(), [1]);
        assert!(hooks.find("m", "start").replay);
        let container = hooks.user_function_of("m", true);
        assert!(container.iter().all(|r| !r.replay), "{container:?}");
    }

    /// Rule 11: each `parallel_mixed` branch and the container are user
    /// functions, in both modes. A branch has the sub-type `ParallelBranch`
    /// and the container as its parent; a flat branch is virtual.
    #[tokio::test]
    async fn parallel_mixed_branches_and_container_both_modes() {
        use crate::plugin::recorder::{UfHooks, context};
        for mode in [NestingMode::Normal, NestingMode::Flat] {
            let hooks = UfHooks::default();
            let (ctx, _client) = context(&hooks, Vec::new());
            let mut set = crate::BranchSet::new();
            let text = set.add("s", |_child| async { Ok("text".to_owned()) });
            let number = set.add("n", |_child| async { Ok(7_u32) });
            ctx.parallel_mixed(set)
                .name("pm")
                .nesting(mode)
                .await
                .unwrap();
            assert_eq!(text.into_result().unwrap(), "text", "{mode:?}");
            assert_eq!(number.into_result().unwrap(), 7, "{mode:?}");

            let container_wire = crate::engine::compute_wire_id_public("1");
            let container = hooks.user_function_of("pm", true);
            assert!(
                container
                    .iter()
                    .all(|r| { r.sub_type.as_deref() == Some(PARALLEL_SUB_TYPE) && !r.is_virtual }),
                "{mode:?}: {container:?}"
            );
            for name in ["s", "n"] {
                let branch = hooks.user_function_of(name, true);
                assert!(
                    branch.iter().all(|r| {
                        r.sub_type.as_deref() == Some(PARALLEL_BRANCH_SUB_TYPE)
                            && r.parent_id.as_deref() == Some(container_wire.as_str())
                            && r.is_virtual == (mode == NestingMode::Flat)
                    }),
                    "{mode:?} {name}: {branch:?}"
                );
                assert_eq!(hooks.find(name, "uf_end").ok, Some(true), "{mode:?}");
            }
            assert_eq!(hooks.find("pm", "uf_end").ok, Some(true), "{mode:?}");
        }
    }

    /// The batch operation [`failing_branches_report_stored_wire`] runs.
    #[derive(Clone, Copy, Debug)]
    enum FailingBatch {
        Map,
        Parallel,
        ParallelMixed,
    }

    /// Rule 11's item and container rows: in a `map`, a `parallel` and a
    /// `parallel_mixed` that tolerate one failure, the failing item's end
    /// hook gets the item's failure wire. A `Normal` item stores it in its
    /// FAIL, a `Flat` item in its `BatchItem`. The container ends `Ok` and
    /// records its SUCCEED.
    #[tokio::test]
    async fn failing_branches_report_stored_wire() {
        use crate::builders::map_parallel::CompletionConfig;
        use crate::plugin::recorder::{
            UfHooks, context, type_and_message, update_error, update_of,
        };
        let tolerate = || CompletionConfig::with_tolerated_failure_count(1);
        let fail = || Err::<u32, BoxError>("boom".into());
        for kind in [
            FailingBatch::Map,
            FailingBatch::Parallel,
            FailingBatch::ParallelMixed,
        ] {
            for mode in [NestingMode::Normal, NestingMode::Flat] {
                let case = format!("{kind:?} {mode:?}");
                let hooks = UfHooks::default();
                let (ctx, client) = context(&hooks, Vec::new());
                // The failing item's name, and the wire the batch keeps
                // for it.
                let (failing, kept) = match kind {
                    FailingBatch::Map => {
                        let batch = ctx
                            .map(vec![0_u32, 1], move |_child, item, _idx| async move {
                                if item == 1 { fail() } else { Ok(item) }
                            })
                            .name("c")
                            .nesting(mode)
                            .completion(tolerate())
                            .await_batch()
                            .await
                            .unwrap();
                        let errors = batch.errors();
                        let error = errors.first().expect("item 1 failed");
                        (
                            "map-item-1",
                            (
                                error.error_type.map(str::to_owned),
                                Some(error.message.to_owned()),
                            ),
                        )
                    }
                    FailingBatch::Parallel => {
                        let batch = ctx
                            .parallel(vec![
                                crate::Branch::new("ok", |_child| async { Ok(0_u32) }),
                                crate::Branch::new("bad", move |_child| async move { fail() }),
                            ])
                            .name("c")
                            .nesting(mode)
                            .completion(tolerate())
                            .await_batch()
                            .await
                            .unwrap();
                        let errors = batch.errors();
                        let error = errors.first().expect("branch 1 failed");
                        (
                            "bad",
                            (
                                error.error_type.map(str::to_owned),
                                Some(error.message.to_owned()),
                            ),
                        )
                    }
                    FailingBatch::ParallelMixed => {
                        let mut set = crate::BranchSet::new();
                        let _ok = set.add("ok", |_child| async { Ok(0_u32) });
                        let bad = set.add("bad", move |_child| async move { fail() });
                        ctx.parallel_mixed(set)
                            .name("c")
                            .nesting(mode)
                            .completion(tolerate())
                            .await
                            .unwrap();
                        let err = bad.into_result().expect_err("branch 1 failed");
                        ("bad", type_and_message(err.wire().expect("a wire record")))
                    }
                };
                let end = hooks.find(failing, "uf_end");
                assert_eq!(end.ok, Some(false), "{case}");
                let end = type_and_message(end.wire.as_ref().expect("an Err outcome"));
                assert_eq!(end.1.as_deref(), Some("boom"), "{case}");
                assert_eq!(end, kept, "{case}: the batch keeps the end hook's wire");
                let updates = client.recorded_updates();
                let item_fail = update_of(&updates, failing, &OperationAction::Fail);
                match mode {
                    NestingMode::Normal => assert_eq!(
                        update_error(&item_fail.expect("the item's FAIL")),
                        end,
                        "{case}: the FAIL stores the end hook's wire"
                    ),
                    NestingMode::Flat => assert!(item_fail.is_none(), "{case}"),
                }
                assert_eq!(hooks.find("c", "uf_end").ok, Some(true), "{case}");
                assert!(
                    update_of(&updates, "c", &OperationAction::Succeed).is_some(),
                    "{case}: the container's SUCCEED"
                );
            }
        }
    }

    /// A result serdes that fails every serialization with a permanent
    /// error.
    struct FailingResultSerdes;

    impl Serdes<String> for FailingResultSerdes {
        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            _value: String,
            _context: SerdesContext,
        ) -> Result<String, BoxError> {
            Err("cannot encode".into())
        }

        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            _wire: String,
            _context: SerdesContext,
        ) -> Result<String, BoxError> {
            Err("cannot decode".into())
        }
    }

    /// An in-memory client that permanently rejects the `SUCCEED` of the
    /// root's first operation.
    #[derive(Debug)]
    struct RejectsParentSucceed(Arc<crate::client::InMemoryExecutionClient>);

    impl crate::client::ExecutionClient for RejectsParentSucceed {
        fn checkpoint(
            &self,
            arn: &str,
            token: &str,
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
            if updates
                .iter()
                .any(|u| u.id == parent && u.action() == &OperationAction::Succeed)
            {
                return Box::pin(async {
                    Err(crate::client::ExecutionClientError::new_non_retryable(
                        "rejected",
                    ))
                });
            }
            self.0.checkpoint(arn, token, updates)
        }

        fn get_state(
            &self,
            arn: &str,
            token: &str,
            marker: Option<&str>,
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
            self.0.get_state(arn, token, marker)
        }
    }

    /// Rule 7.6: a container's body ends `Ok` when its batch is produced. A
    /// failure after it keeps its own wire: a result serdes failure is the
    /// handler's error with `serialization_failure_wire`, and a rejected
    /// `SUCCEED` records a terminal FAIL with `checkpoint_failure_wire`.
    #[tokio::test]
    async fn container_post_body_failures_keep_own_wire() {
        use crate::plugin::recorder::{UfHooks, context, context_over, update_error, update_of};

        // Run 1: the result serdes fails.
        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let err = ctx
            .map(vec![1_u32], |_child, item, _idx| async move { Ok(item) })
            .name("m")
            .result_serdes(FailingResultSerdes)
            .await
            .expect_err("the batch record cannot be serialized");
        let end = hooks.find("m", "uf_end");
        assert_eq!((end.ok, end.wire), (Some(true), None));
        // The serialization failure is the handler's error, and it keeps
        // its own wire, `serialization_failure_wire`. As on main, a
        // container writes no outcome record for it: the next invocation
        // produces the batch again from its items' records.
        assert_eq!(
            err.wire(),
            Some(&crate::error::WireError::new(
                Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE),
                Some(
                    "operation error: child_context: internal error: serialize result: cannot encode"
                ),
            ))
        );
        let updates = client.recorded_updates();
        assert!(
            update_of(&updates, "m", &OperationAction::Succeed).is_none()
                && update_of(&updates, "m", &OperationAction::Fail).is_none(),
            "{updates:?}"
        );

        // Run 2: the client rejects the container's SUCCEED.
        let hooks = UfHooks::default();
        let inner = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = context_over(
            &hooks,
            Vec::new(),
            Arc::new(RejectsParentSucceed(Arc::clone(&inner))),
        );
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            crate::driver::test_support::drive_handler(ctx, None, |ctx| async move {
                ctx.map(vec![1_u32], |_child, item, _idx| async move { Ok(item) })
                    .name("m")
                    .await
                    .map_err(crate::driver::test_support::wire_err)?;
                Ok("done".to_owned())
            }),
        )
        .await
        .expect("the invocation ends within 5 s")
        .expect("no panic");
        assert!(
            matches!(outcome, crate::driver::InvocationOutcome::Failed { .. }),
            "{outcome:?}"
        );
        let end = hooks.find("m", "uf_end");
        assert_eq!((end.ok, end.wire), (Some(true), None));
        let fail = update_of(&inner.recorded_updates(), "m", &OperationAction::Fail)
            .expect("the container's terminal FAIL");
        let expected = crate::error::checkpoint_failure_wire(
            &crate::client::ExecutionClientError::new_non_retryable("rejected"),
        );
        assert_eq!(
            update_error(&fail),
            (
                expected.error_type().map(str::to_owned),
                expected.error_message().map(str::to_owned)
            )
        );
    }

    /// Rule 3.4: a branch the batch abandons at its early completion, while
    /// its body runs, gets a start hook and no end hook. The branch that
    /// completed the batch gets both.
    #[tokio::test]
    async fn abandoned_branch_gets_no_end_hook() {
        use crate::builders::map_parallel::CompletionConfig;
        use crate::plugin::recorder::{UfHooks, context};
        let hooks = UfHooks::default();
        let (ctx, _client) = context(&hooks, Vec::new());
        let started = Arc::new(tokio::sync::Notify::new());
        let waits = Arc::clone(&started);
        let batch = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ctx.parallel(vec![
                crate::Branch::new("done", move |_child| async move {
                    waits.notified().await;
                    Ok(0_u32)
                }),
                crate::Branch::new("abandoned", move |_child| async move {
                    started.notify_one();
                    std::future::pending::<()>().await;
                    Ok(1_u32)
                }),
            ])
            .name("p")
            .completion(CompletionConfig::with_min_successful(1))
            .await_batch(),
        )
        .await
        .expect("the batch completes within 5 s")
        .unwrap();
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        hooks.user_function_of("abandoned", false);
        hooks.user_function_of("done", true);
    }

    /// Builds the request of item 0 of a map named `m` on a root context
    /// that carries the plugin `hooks`, without running the map.
    fn item_request(
        hooks: &crate::plugin::test_support::OpHooks,
        nesting: NestingMode,
    ) -> ItemRequest<crate::serdes::JsonSerdes> {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
            None,
            Some(hooks.plugins()),
        );
        ItemRequest {
            ctx,
            child_op_id: OperationId::new("1-1".to_owned()),
            index: 0,
            is_terminal: false,
            parent_wire: crate::engine::compute_wire_id_public("1"),
            child_sub_type: "MapIteration".to_owned(),
            item_name: "map-item-0".to_owned(),
            nesting,
            start_info: Arc::new(crate::plugin::OperationInfo::for_test("1-1")),
            serdes: Arc::new(crate::serdes::JsonSerdes),
            start: std::sync::Mutex::new(None),
        }
    }

    /// The end hook of a flat item reports the `is_replaying()` value that
    /// the caller read from the item's own context, for a success and for
    /// a failure. The test passes both values directly, so it needs no
    /// history that the item body does not follow. A Normal item writes
    /// its own record and ends live, so it reports `false` either way.
    #[tokio::test]
    async fn flat_item_end_reports_item_context_replay_state() {
        for replaying in [true, false] {
            let hooks = crate::plugin::test_support::OpHooks::default();
            let req = item_request(&hooks, NestingMode::Flat);
            item_after(&req, ItemBody::Unhooked(Ok(1_u32)), replaying)
                .await
                .expect("a flat success settles");
            let failed: Result<u32, ChildFnError> = Err(ChildFnError::from(BoxError::from("bad")));
            item_after(&req, ItemBody::Unhooked(failed), replaying)
                .await
                .expect("a flat failure settles");
            let ends: Vec<_> = hooks
                .of("map-item-0")
                .iter()
                .map(|r| (r.hook, r.replay, r.is_virtual, r.status))
                .collect();
            assert_eq!(
                ends,
                [
                    (
                        "end",
                        replaying,
                        true,
                        Some(crate::OperationStatus::Succeeded)
                    ),
                    ("end", replaying, true, Some(crate::OperationStatus::Failed)),
                ],
                "replaying = {replaying}"
            );
        }
        let hooks = crate::plugin::test_support::OpHooks::default();
        let req = item_request(&hooks, NestingMode::Normal);
        item_ended(&req, crate::OperationStatus::Succeeded, None, None, true);
        let normal: Vec<_> = hooks
            .of("map-item-0")
            .iter()
            .map(|r| (r.hook, r.replay, r.is_virtual))
            .collect();
        assert_eq!(normal, [("end", false, false)]);
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
            .item_namer(|_, i| format!("item-{i}"))
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

    /// Collects `(action, name)` of every recorded child update, in write
    /// order. A child update carries a `ParentId`; the batch parent's does
    /// not.
    fn child_update_names(
        client: &crate::client::InMemoryExecutionClient,
    ) -> Vec<(OperationAction, Option<String>)> {
        client
            .recorded_updates()
            .iter()
            .filter(|u| u.parent_id().is_some())
            .map(|u| (u.action().clone(), u.name().map(str::to_owned)))
            .collect()
    }

    /// A child's normal terminal `SUCCEED` and its body-error `FAIL` must
    /// carry the recorded name, as its `START` does. Item 0 has no namer
    /// result, so it records the default `map-item-0`.
    #[tokio::test]
    async fn child_terminal_updates_carry_the_recorded_name() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());

        let batch = ctx
            .map(vec![0_i32, 1], |_child, item, _idx| async move {
                if item == 1 {
                    Err::<i32, BoxError>("intentional".into())
                } else {
                    Ok(item)
                }
            })
            .item_namer(|_, i| {
                if i == 0 {
                    String::new()
                } else {
                    "named".to_owned()
                }
            })
            .max_concurrency(1)
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("a tolerated item failure must not become an operation error");
        assert_eq!(batch.failure_count(), 1);

        let updates = child_update_names(&client);
        let name_of = |action: OperationAction| -> Vec<Option<String>> {
            updates
                .iter()
                .filter(|(a, _)| *a == action)
                .map(|(_, n)| n.clone())
                .collect()
        };
        assert_eq!(
            name_of(OperationAction::Succeed),
            vec![Some("map-item-0".to_owned())],
            "a child SUCCEED must carry the recorded name"
        );
        assert_eq!(
            name_of(OperationAction::Fail),
            vec![Some("named".to_owned())],
            "a child body-error FAIL must carry the recorded name"
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
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
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
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
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
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
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
        // yet terminal, child 0 (positional "1-1") failed, and child 1
        // (positional "1-2") succeeded with 7.
        let log = CheckpointLog::from_records(vec![
            failed_record("1-1", "recorded failure"),
            succeeded_record("1-2", "7"),
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

    /// The JSON structure of a batch record text, for assertions on its
    /// shape. Exact-text assertions read the text itself instead.
    fn record_json(record_text: &str) -> serde_json::Value {
        serde_json::from_str(record_text).expect("a batch record is JSON")
    }

    /// The batch record text the parent recorded: the first recorded
    /// update payload that parses as a batch record.
    fn recorded_batch_record(client: &crate::client::InMemoryExecutionClient) -> String {
        client
            .recorded_updates()
            .iter()
            .filter(|u| u.parent_id().is_none())
            .filter_map(|u| u.payload())
            .find(|p| parse_batch_record(p).is_ok())
            .map(str::to_owned)
            .expect("the parent records a batch record payload")
    }

    /// A failed map item records the type of the error its body returned
    /// (`"Error"` for an untyped error), and the four views of the failure
    /// agree: the child FAIL record, `BatchItem::error_type`,
    /// `BatchError::error_type` and the summary's `Cause.ErrorType`.
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

        let record = recorded_batch_record(&client);
        assert_eq!(
            record_json(&record).pointer("/all/0/error/Cause/ErrorType"),
            Some(&serde_json::json!("Error"))
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

        let record = recorded_batch_record(&client);
        assert_eq!(
            record_json(&record).pointer("/all/0/error/Cause/ErrorType"),
            Some(&serde_json::json!("Error"))
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
            "all": [
                {"index": 0, "name": "map-item-0", "status": "SUCCEEDED", "result": "hello"},
                {"index": 1, "name": "map-item-1", "status": "SUCCEEDED", "result": "world"}
            ],
            "completionReason": "ALL_COMPLETED"
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
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

    /// An invalid percentage that skipped the builder's validation fails a
    /// non-empty batch before any item runs, and a zero-item batch, which
    /// reads no threshold, succeeds.
    #[tokio::test]
    async fn invalid_percentage_fails_only_a_non_empty_batch() {
        let body = |_child: DurableContext, item: u32, _idx: usize| async move { Ok(item) };
        let invalid = || Config::with_tolerated_failure_percentage(150.0);
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let err = ctx
            .map(vec![1_u32], body)
            .completion(invalid())
            .await
            .expect_err("an invalid percentage fails the batch");
        assert!(
            crate::error::chain_string(&err).contains("tolerated_failure_percentage must be 0-100"),
            "{err}"
        );
        let (ctx, _client) = test_ctx_with_client(CheckpointLog::empty());
        let empty = ctx
            .map(Vec::<u32>::new(), body)
            .completion(invalid())
            .await
            .expect("a zero-item batch succeeds");
        assert_eq!(empty, Vec::<u32>::new());
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
            succeeded_record("1-1", "1"),
            succeeded_record("1-2", "2"),
            failed_record("1-3", "recorded failure"),
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
            .item_namer(|_, i| {
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
                .item_namer(move |_, i| {
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

    /// The names one batch exposes: the child START names in record order,
    /// the result's item names, and the caller names of the predicate's
    /// first view, in index order.
    struct BatchNames {
        starts: Vec<Option<String>>,
        results: Vec<String>,
        predicate: Vec<Option<String>>,
    }

    impl BatchNames {
        fn of<O>(
            client: &crate::client::InMemoryExecutionClient,
            batch: &BatchResult<O>,
            log: &ViewLog,
        ) -> Self {
            Self {
                starts: child_start_names(client),
                results: batch.items.iter().map(|i| i.name.clone()).collect(),
                predicate: log.lock().expect("no poisoned lock")[0]
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect(),
            }
        }
    }

    /// A completion predicate that records every view it sees and never
    /// completes the batch.
    fn logging_predicate(log: &ViewLog) -> Config {
        let writer = Arc::clone(log);
        Config::with_completion_predicate(move |s| {
            writer.lock().expect("no poisoned lock").push(view_of(s));
            CompletionDecision::Continue
        })
    }

    fn some(names: &[&str]) -> Vec<Option<String>> {
        names.iter().map(|n| Some((*n).to_owned())).collect()
    }

    fn owned(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    /// The namer reads the item itself, so an item type that is not `Clone`
    /// can be named after its content.
    #[tokio::test]
    async fn map_item_namer_reads_the_item() {
        struct Order {
            id: String,
        }
        let log: ViewLog = Arc::default();
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let orders = vec![
            Order {
                id: "o-1".to_owned(),
            },
            Order {
                id: "o-2".to_owned(),
            },
        ];
        let batch = ctx
            .map(orders, |_child, order: Order, _idx| async move {
                Ok(order.id.len())
            })
            .item_namer(|o, _| o.id.clone())
            .max_concurrency(1)
            .completion(logging_predicate(&log))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        let names = BatchNames::of(&client, &batch, &log);
        assert_eq!(names.starts, some(&["o-1", "o-2"]));
        assert_eq!(names.results, owned(&["o-1", "o-2"]));
        assert_eq!(names.predicate, some(&["o-1", "o-2"]));
    }

    /// A map item without a namer records `map-item-{index}`, and the
    /// predicate sees `None`.
    #[tokio::test]
    async fn unnamed_map_items_record_default_names() {
        let log: ViewLog = Arc::default();
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![1_u32, 2], |_child, item, _idx| async move { Ok(item) })
            .max_concurrency(1)
            .completion(logging_predicate(&log))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        let names = BatchNames::of(&client, &batch, &log);
        assert_eq!(names.starts, some(&["map-item-0", "map-item-1"]));
        assert_eq!(names.results, owned(&["map-item-0", "map-item-1"]));
        assert_eq!(names.predicate, [None, None]);
    }

    /// An empty namer result records `map-item-{index}`, and the predicate
    /// sees `Some("")`.
    #[tokio::test]
    async fn empty_namer_result_records_default_name() {
        let log: ViewLog = Arc::default();
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .map(vec![1_u32, 2], |_child, item, _idx| async move { Ok(item) })
            .item_namer(|_, i| {
                if i == 0 {
                    String::new()
                } else {
                    "named".into()
                }
            })
            .max_concurrency(1)
            .completion(logging_predicate(&log))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        let names = BatchNames::of(&client, &batch, &log);
        assert_eq!(names.starts, some(&["map-item-0", "named"]));
        assert_eq!(names.results, owned(&["map-item-0", "named"]));
        assert_eq!(names.predicate, some(&["", "named"]));
    }

    /// An empty branch name records `parallel-branch-{index}`, and the
    /// predicate sees `Some("")`.
    #[tokio::test]
    async fn empty_branch_name_records_default_name() {
        use crate::future::Branch;
        let log: ViewLog = Arc::default();
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .parallel([
                Branch::new("", |_c| async move { Ok(1_u32) }),
                Branch::new("x", |_c| async move { Ok(2_u32) }),
            ])
            .max_concurrency(1)
            .completion(logging_predicate(&log))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        let names = BatchNames::of(&client, &batch, &log);
        assert_eq!(names.starts, some(&["parallel-branch-0", "x"]));
        assert_eq!(names.results, owned(&["parallel-branch-0", "x"]));
        assert_eq!(names.predicate, some(&["", "x"]));
    }

    /// A `Branch::unnamed` branch records `parallel-branch-{index}`, and the
    /// predicate sees `None`.
    #[tokio::test]
    async fn unnamed_branch_records_default_name() {
        use crate::future::Branch;
        let log: ViewLog = Arc::default();
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let batch = ctx
            .parallel([
                Branch::unnamed(|_c| async move { Ok(1_u32) }),
                Branch::new("x", |_c| async move { Ok(2_u32) }),
            ])
            .max_concurrency(1)
            .completion(logging_predicate(&log))
            .await_batch()
            .await
            .expect("await_batch returns the batch");
        let names = BatchNames::of(&client, &batch, &log);
        assert_eq!(names.starts, some(&["parallel-branch-0", "x"]));
        assert_eq!(names.results, owned(&["parallel-branch-0", "x"]));
        assert_eq!(names.predicate, [None, Some("x".to_owned())]);
    }

    /// The namer runs once for every input item when the map starts,
    /// including the two items that never start because the batch completes
    /// after the first success.
    #[tokio::test]
    async fn namer_runs_once_per_item() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.map(
            vec![1_u32, 2, 3],
            |_child, item, _idx| async move { Ok(item) },
        )
        .item_namer(move |_, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            String::new()
        })
        .max_concurrency(1)
        .completion(Config::with_min_successful(1))
        .await_batch()
        .await
        .expect("await_batch returns the batch");
        assert_eq!(child_start_names(&client), some(&["map-item-0"]));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// On the sequential path, a resumed batch replays its recorded-terminal
    /// item without a dispatch gate: the replay pass is not a live dispatch.
    /// So the predicate is first asked after item 0's recorded settle, and
    /// the live items still get the IDs the first run would give them.
    #[tokio::test]
    async fn sequential_replay_asks_no_gate_for_a_recorded_item() {
        let log = CheckpointLog::from_records(vec![succeeded_record("1-1", "1")]);
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
                crate::engine::compute_wire_id_public("1-2"),
                crate::engine::compute_wire_id_public("1-3")
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

    /// An unknown completion reason is not parsed.
    #[test]
    fn unknown_completion_reason_is_not_parsed() {
        assert_eq!(CompletionReason::try_from_wire("NOT_A_REASON"), None);
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
        let payload = from_batch_result(
            &original,
            &[Some("7".to_owned()), None],
            &ItemEncoding::AllJson,
        );
        let replayed: BatchResult<i32> = to_batch_result(
            &test_ctx(CheckpointLog::empty()),
            &payload,
            &json,
            ItemEncoding::AllJson,
            2,
            "arn:test",
            &ItemIds::new(&OperationId::new("1".to_owned()), 2),
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

    /// A failed item that recorded no error type writes the default
    /// `"Error"` as its `Cause.ErrorType`, and an item that recorded no
    /// message writes an empty `ErrorMessage`.
    #[test]
    fn from_batch_result_defaults_missing_error_type() {
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
        let payload = record_json(&from_batch_result(
            &original,
            &[None],
            &ItemEncoding::AllJson,
        ));
        assert_eq!(
            payload.pointer("/all/0/error/Cause/ErrorType"),
            Some(&serde_json::json!("Error"))
        );
        assert_eq!(
            payload.pointer("/all/0/error/ErrorMessage"),
            Some(&serde_json::json!("boom"))
        );
        let untyped: BatchResult<i32> = BatchResult::new(
            vec![BatchItem::new(
                0,
                String::new(),
                BatchItemStatus::Failed,
                None,
                None,
                None,
            )],
            CompletionReason::AllCompleted,
        );
        let payload = record_json(&from_batch_result(
            &untyped,
            &[None],
            &ItemEncoding::AllJson,
        ));
        assert_eq!(
            payload.pointer("/all/0/error/Cause/ErrorMessage"),
            Some(&serde_json::json!(""))
        );
    }

    /// A checkpoint payload whose failed item carries no `ErrorType`
    /// replays with `error_type: None` rather than a fabricated type.
    #[tokio::test]
    async fn to_batch_result_missing_err_type_is_none() {
        let payload = r#"{"all":[{"index":0,"name":"map-item-0","status":"FAILED","error":{"ErrorMessage":"boom"}}],"completionReason":"ALL_COMPLETED"}"#;
        let json = Arc::new(crate::serdes::JsonSerdes);
        let replayed: BatchResult<i32> = to_batch_result(
            &test_ctx(CheckpointLog::empty()),
            payload,
            &json,
            ItemEncoding::AllJson,
            2,
            "arn:test",
            &ItemIds::new(&OperationId::new("1".to_owned()), 2),
        )
        .await
        .expect("deserializing the payload must succeed");
        let item = replayed.items.first().expect("one item");
        assert_eq!(item.error_message.as_deref(), Some("boom"));
        assert_eq!(item.error_type, None);
        // A failure without `ErrorData` and `StackTrace` replays with an
        // empty failure payload rather than a fabricated one.
        assert_eq!(item.error_data, None);
        assert_eq!(item.stack_trace, Vec::<String>::new());
        assert_eq!(replayed.errors()[0].error_type, None);
    }

    /// Replaying a frozen terminal batch whose payload carries a per-item
    /// error type exposes that type through `errors()` without re-executing
    /// any item body.
    #[tokio::test]
    async fn replay_frozen_batch_preserves_error_type() {
        let payload = serde_json::json!({
            "all": [
                {"index": 0, "name": "map-item-0", "status": "SUCCEEDED", "result": "hello"},
                {
                    "index": 1,
                    "name": "map-item-1",
                    "status": "FAILED",
                    "error": {
                        "ErrorType": "ChildContextError",
                        "Cause": {"ErrorType": "ChildFnError", "ErrorMessage": "boom"}
                    }
                }
            ],
            "completionReason": "ALL_COMPLETED"
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
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
        let flat_child = ctx.new_flat_child(&child_positional, &batch_wire, None);
        // Operations inside the flat child should report the batch parent
        // as their parent wire ID.
        assert_eq!(
            flat_child.parent_wire_id(),
            Some(batch_wire.as_str()),
            "flat child must report the batch parent's wire ID"
        );
    }

    // BatchResult views and into_result, over hand-built results.

    fn item(index: usize, status: BatchItemStatus, result: Option<i32>) -> BatchItem<i32> {
        BatchItem::new(index, format!("item-{index}"), status, result, None, None)
    }

    fn failed_item(index: usize, message: &str, error_type: Option<&str>) -> BatchItem<i32> {
        BatchItem::new(
            index,
            format!("item-{index}"),
            BatchItemStatus::Failed,
            None,
            Some(message.to_owned()),
            error_type.map(str::to_owned),
        )
    }

    fn indexes<'a>(items: impl Iterator<Item = &'a BatchItem<i32>>) -> Vec<usize> {
        items.map(|item| item.index).collect()
    }

    /// A `parallel_mixed` item serdes with one branch, on a fresh context.
    fn one_branch_mixed_serdes() -> (MixedItemSerdes, crate::BranchHandle<u32>) {
        let mut set = crate::BranchSet::new();
        let handle = set.add("only", |_ctx| async { Ok(1_u32) });
        let (_bodies, decoders, _json) = set.into_parts();
        let batch = OperationId::new("1".to_owned());
        let item_ids = ItemIds::new(&batch, decoders.len());
        (MixedItemSerdes::new(&item_ids, decoders), handle)
    }

    #[tokio::test]
    async fn mixed_item_serdes_rejects_unknown_operation_id() {
        let (serdes, _handle) = one_branch_mixed_serdes();
        let err = serdes
            .deserialize("1".to_owned(), SerdesContext::new("other", "arn:test"))
            .await
            .err()
            .expect("an unknown operation ID is an error");
        assert_eq!(
            err.to_string(),
            r#"parallel_mixed: no branch has operation ID "other""#
        );
    }

    #[tokio::test]
    async fn mixed_item_serdes_uses_item_id_table() {
        let mut set = crate::BranchSet::new();
        let _first = set.add("first", |_ctx| async { Ok(1_u32) });
        let second = set.add("second", |_ctx| async { Ok(2_u32) });
        let (_bodies, decoders, _json) = set.into_parts();
        let batch = OperationId::new("1".to_owned());
        let item_ids = ItemIds::new(&batch, decoders.len());
        let serdes = MixedItemSerdes::new(&item_ids, decoders);
        // With no registration, the wire ID of `1-2` names branch 1: the
        // serdes reads it from the table, and publishing the decoded value
        // fills branch 1's handle.
        let child = SerdesContext::new(crate::engine::compute_wire_id_public("1-2"), "arn:test");
        let value = serdes
            .deserialize("7".to_owned(), child)
            .await
            .expect("the ID 1-2 names branch 1");
        value.0.publish();
        assert_eq!(second.into_result().expect("branch 1 holds a value"), 7);
    }

    #[test]
    fn publish_outcome_publishes_nothing_for_an_error_or_a_valueless_item() {
        let mut set = crate::BranchSet::new();
        let handle = set.add("only", |_ctx| async { Ok(1_u32) });
        let (_bodies, decoders, _json) = set.into_parts();
        let err = publish_outcome(Err(batch_error("checkpoint failed")), &decoders)
            .expect_err("the batch error is returned");
        assert_eq!(
            format!("{err:#}"),
            format!("{:#}", batch_error("checkpoint failed"))
        );
        // A succeeded item without a value publishes nothing either.
        let item = BatchItem::<crate::future::MixedValue>::new(
            0,
            "only".to_owned(),
            BatchItemStatus::Succeeded,
            None,
            None,
            None,
        );
        let reason = publish_outcome(
            Ok(BatchResult::new(vec![item], CompletionReason::AllCompleted)),
            &decoders,
        )
        .expect("an all-completed batch succeeds");
        assert_eq!(reason, CompletionReason::AllCompleted);
        let err = handle.into_result().expect_err("the handle has no result");
        assert!(matches!(
            err.kind(),
            OperationErrorKind::ChildContext(child)
                if matches!(child.kind(), ChildContextErrorKind::NotCompleted)
        ));
    }

    #[test]
    fn parallel_mixed_builder_debug_shows_the_name() {
        let ctx = test_ctx(CheckpointLog::empty());
        let builder = ctx.parallel_mixed(crate::BranchSet::new()).name("pm");
        assert_eq!(
            format!("{builder:?}"),
            r#"ParallelMixedBuilder { name: Some("pm"), .. }"#
        );
    }

    #[test]
    fn publish_outcome_skips_item_without_branch() {
        let mut set = crate::BranchSet::new();
        let handle = set.add("only", |_ctx| async { Ok(1_u32) });
        let (_bodies, decoders, _json) = set.into_parts();
        let failed = |index: usize| {
            BatchItem::<crate::future::MixedValue>::new(
                index,
                format!("item-{index}"),
                BatchItemStatus::Failed,
                None,
                Some(format!("boom-{index}")),
                Some(format!("Type{index}")),
            )
        };
        let batch = BatchResult::new(
            vec![failed(0), failed(1)],
            CompletionReason::FailureToleranceExceeded,
        );
        let err = publish_outcome(Ok(batch), &decoders)
            .expect_err("a batch over its failure tolerance fails");
        let wire = err.wire().expect("the error carries item 0's record");
        assert_eq!(wire.error_type(), Some("Type0"));
        assert_eq!(wire.error_message(), Some("boom-0"));
        let held = handle
            .into_result()
            .expect_err("handle 0 holds item 0's error");
        let held = held.wire().expect("the handle's error carries a record");
        assert_eq!(held.error_type(), Some("Type0"));
        assert_eq!(held.error_message(), Some("boom-0"));
    }

    /// A batch whose reason is `FailureToleranceExceeded` but that holds no
    /// failed item, as only a corrupt record can, fails with the internal
    /// error `failure tolerance exceeded`. `parallel(...).await` and
    /// `parallel_mixed(...).await` return that same error.
    #[test]
    fn tolerance_exceeded_without_failed_item_is_an_internal_error() {
        let batch = || {
            BatchResult::new(
                vec![item(0, BatchItemStatus::Succeeded, Some(1))],
                CompletionReason::FailureToleranceExceeded,
            )
        };
        let is_tolerance_error = |err: &OperationError| {
            matches!(
                err.kind(),
                OperationErrorKind::ChildContext(child)
                    if matches!(child.kind(), ChildContextErrorKind::Internal)
            ) && crate::error::chain_string(err).contains("failure tolerance exceeded")
        };
        let err = await_error(&batch()).expect("the reason is a failing reason");
        assert!(is_tolerance_error(&err), "{err:?}");
        let err = collect_successful(batch()).expect_err("parallel fails the batch");
        assert!(is_tolerance_error(&err), "{err:?}");
        let mixed = BatchResult::<crate::future::MixedValue>::new(
            Vec::new(),
            CompletionReason::FailureToleranceExceeded,
        );
        let err = publish_outcome(Ok(mixed), &[]).expect_err("parallel_mixed fails the batch");
        assert!(is_tolerance_error(&err), "{err:?}");
    }

    #[test]
    fn batch_result_views_split_by_status() {
        let batch = BatchResult::new(
            vec![
                item(0, BatchItemStatus::Succeeded, Some(1)),
                failed_item(1, "boom", None),
                item(2, BatchItemStatus::Started, None),
                item(3, BatchItemStatus::Succeeded, Some(4)),
            ],
            CompletionReason::MinSuccessfulReached,
        );
        assert_eq!(indexes(batch.succeeded()), [0, 3]);
        assert_eq!(indexes(batch.failed()), [1]);
        assert_eq!(indexes(batch.started()), [2]);
        assert_eq!(batch.started_count(), 1);
    }

    #[test]
    fn into_result_returns_first_failure() {
        let batch = BatchResult::new(
            vec![
                item(0, BatchItemStatus::Succeeded, Some(1)),
                failed_item(1, "a", Some("AError")),
                failed_item(2, "b", None),
            ],
            CompletionReason::AllCompleted,
        );
        let err = batch.into_result().expect_err("an item failed");
        let OperationErrorKind::ChildContext(child_err) = err.kind() else {
            unreachable!("expected a child-context error, got {:?}", err.kind());
        };
        assert!(matches!(
            child_err.kind(),
            ChildContextErrorKind::ChildFailed
        ));
        let wire = err.wire().expect("the item's wire record is attached");
        assert_eq!(wire.error_type(), Some("AError"));
        assert_eq!(wire.error_message(), Some("a"));
    }

    #[test]
    fn into_result_returns_completion_error_without_failed_item() {
        let batch = BatchResult::new(
            vec![item(0, BatchItemStatus::Succeeded, Some(1))],
            CompletionReason::CustomCompletionFailed,
        );
        let err = batch.into_result().expect_err("the batch status is Failed");
        let OperationErrorKind::ChildContext(child_err) = err.kind() else {
            unreachable!("expected a child-context error, got {:?}", err.kind());
        };
        assert!(matches!(
            child_err.kind(),
            ChildContextErrorKind::BatchCompletionFailed
        ));
        let wire = err.wire().expect("the completion wire record is attached");
        assert_eq!(wire.error_type(), Some("BatchCompletionError"));
    }

    #[test]
    fn into_result_keeps_a_successful_batch() {
        let batch = BatchResult::new(
            vec![
                item(0, BatchItemStatus::Succeeded, Some(1)),
                item(1, BatchItemStatus::Started, None),
            ],
            CompletionReason::MinSuccessfulReached,
        );
        let batch = batch.into_result().expect("no item failed");
        let items: Vec<_> = batch
            .items
            .iter()
            .map(|item| (item.index, item.status, item.result))
            .collect();
        assert_eq!(
            items,
            [
                (0, BatchItemStatus::Succeeded, Some(1)),
                (1, BatchItemStatus::Started, None),
            ]
        );
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
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
        let outcome = crate::driver::drive_outcome(
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

    /// Polls `run` once on a thread with no Tokio runtime and asserts that
    /// the batch resolved to the no-runtime error, instead of panicking.
    fn assert_batch_no_runtime_error<F, Fut>(log: CheckpointLog, run: F)
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Vec<i32>, OperationError>>,
    {
        // The context is built on the polling thread, so that thread owns it.
        let polled = crate::task::test_support::poll_once_without_runtime(move || {
            run(test_ctx_with_client(log).0)
        });
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

    /// Every path hands a custom serdes its own typed value.
    ///
    /// The step, invoke, callback and map/parallel item serdes are handed
    /// the identical typed value. A quoting divergence between items (`X`)
    /// and everything else (`"X"`) once survived all the way to
    /// conformance case 9-14, so this test asserts every path. The batch
    /// result serdes is a `Serdes<String>`: it is handed a `String` that
    /// holds the batch record text.
    ///
    /// The probe payload is a bare `String`, the one case where the old split
    /// was visible, and the fixture's wire form is deliberately NOT JSON, so
    /// a path that bypassed the serdes would fail rather than quietly pass.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // reason: one test per path is the point; splitting it would let the paths drift apart again
    async fn custom_serdes_receives_its_typed_value_on_every_path() {
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
        // The item serdes is consulted once per item, for the child's own
        // checkpoint. The batch summary takes the item value from the
        // child's wire string, so it calls no serdes.
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
        assert_eq!(
            map_inputs.len(),
            1,
            "the item serdes serializes the item once, for the child \
             checkpoint, got {map_inputs:?}"
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
        // The batch result serdes is a `Serdes<String>`. It is handed one
        // `String`: the batch record text. With the default item serdes,
        // the item's JSON text is embedded unchanged, as JS embeds the
        // value.
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
        assert_eq!(
            batch_inputs,
            vec![serde_json::Value::String(format!(
                r#"{{"all":[{{"index":0,"status":"SUCCEEDED","name":"map-item-0","result":{json_encoding}}}],"completionReason":"ALL_COMPLETED"}}"#
            ))],
            "the batch result serdes must be handed the record text once"
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

        // Batch parent "1" started; its single child "1-1" already succeeded
        // with the custom wire form in the log.
        let parent_wire_id = crate::engine::compute_wire_id_public("1");
        let child_wire_id = crate::engine::compute_wire_id_public("1-1");
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
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
    /// records the JS `MapResult` summary as the `replay_children` payload,
    /// instead of no payload at all.
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
        let summary: serde_json::Value = serde_json::from_str(
            parent
                .payload()
                .expect("large-result parent must carry the summary"),
        )
        .expect("payload must parse as a plain-JSON summary");
        assert_eq!(summary["totalCount"], 2);
        assert_eq!(summary["completionReason"], "ALL_COMPLETED");
        // Everything completed: every started item is terminal.
        assert_eq!(summary["itemStatuses"], "SS");
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
        let summary: serde_json::Value = serde_json::from_str(
            parent
                .payload()
                .expect("large-result parent must carry the summary"),
        )
        .expect("payload must parse as a plain-JSON summary");
        assert_eq!(
            summary["totalCount"], 1,
            "totalCount counts items ever started, not the 6 input items"
        );
        assert_eq!(summary["completionReason"], "MIN_SUCCESSFUL_REACHED");
        assert_eq!(summary["itemStatuses"], "S");
    }

    /// The oversized summary must be PLAIN JSON even when a custom
    /// `result_serdes` transforms the batch record: it is the SDK's own
    /// bookkeeping and must stay readable when the configured serdes is the
    /// reason the payload outgrew the limit.
    #[tokio::test]
    async fn oversized_summary_bypasses_result_serdes() {
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
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
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
        // was written in its place. Children have the IDs "1-1" and "1-2".
        let decision =
            r#"{"totalCount":2,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            succeeded_record("1-1", "\"hello\""),
            succeeded_record("1-2", "\"world\""),
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
                },
            )
        };
        // Items 0 and 2 completed; item 1 started but was still running
        // when the min-successful trigger completed the batch.
        let decision =
            r#"{"totalCount":3,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[1]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            succeeded_record("1-1", "\"a\""),
            started_record("1-2"),
            succeeded_record("1-3", "\"c\""),
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
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
            replay_children_success_record("1-1"),
            poisoned_record("1-2"),
            succeeded_record("1-3", "\"c\""),
            poisoned_record("1-4"),
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

    /// A succeeded `replay_children` batch parent must carry its summary.
    /// An absent payload and a blank payload each fail replay with a
    /// malformed decision record error, and no item body runs.
    #[tokio::test]
    async fn empty_replay_children_payload_fails_replay() {
        for payload in [None, Some("  ")] {
            let (parent_wire, mut parent) = decision_record_parent("1", "");
            parent.result = payload.map(str::to_owned);
            let log = CheckpointLog::from_records(vec![
                (parent_wire, parent),
                succeeded_record("1-1", "\"hello\""),
                succeeded_record("1-2", "\"world\""),
            ]);
            let (ctx, _client) = test_ctx_with_client(log);
            let ran = Arc::new(AtomicBool::new(false));
            let ran_probe = Arc::clone(&ran);

            let err = ctx
                .map(
                    vec!["a".to_owned(), "b".to_owned()],
                    move |_child, item: String, _idx| {
                        let ran = Arc::clone(&ran_probe);
                        async move {
                            ran.store(true, Ordering::SeqCst);
                            Ok(item)
                        }
                    },
                )
                .await
                .expect_err("a batch parent without its summary fails replay");
            assert!(
                format!("{err:#}").contains("malformed batch decision record"),
                "{payload:?}: unexpected error: {err:#}"
            );
            assert!(
                !ran.load(Ordering::SeqCst),
                "{payload:?}: no item body may run"
            );
        }
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
    fn parse_decision_record_rejects_what_is_not_a_record() {
        // A payload that is not a decision record is corrupt bookkeeping:
        // an error, never a silent fall-back.
        assert!(parse_decision_record("not json").is_err());
        assert!(parse_decision_record(r#"{"unrelated":true}"#).is_err());
        let record = parse_decision_record(
            r#"{"totalCount":2,"completionReason":"ALL_COMPLETED","startedIndexes":[]}"#,
        )
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
            r#"{"totalCount":"#, // truncated JSON: corrupt
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

    /// The oversized-record path refuses to checkpoint rather than writing
    /// a terminal batch without the decision record replay requires.
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
            None,
        )
        .await;

        let Err(err) = result else {
            unreachable!("an oversized decision record must refuse the checkpoint")
        };
        let display = crate::error::chain_string(&err);
        assert!(
            display.contains("batch summary exceeds the checkpoint size limit"),
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
        // Child "1-1" (item 0) carries a recorded identity that does NOT
        // match a map iteration: same type but a different sub-type.
        let mismatched_child = {
            let wire_id = crate::engine::compute_wire_id_public("1-1");
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
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
                },
            )
        };
        let decision = r#"{"totalCount":2,"completionReason":"ALL_COMPLETED","startedIndexes":[]}"#;
        let log = CheckpointLog::from_records(vec![
            decision_record_parent("1", decision),
            mismatched_child,
            succeeded_record("1-2", "\"b\""),
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
    /// items are numbered inside the batch.
    fn item_wires() -> [String; 2] {
        [
            crate::engine::compute_wire_id_public("1-1"),
            crate::engine::compute_wire_id_public("1-2"),
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
        assert_started_defaults(&batch.items[1], "map-item-1");
        assert_eq!(item_action_count(&client, 0, &OperationAction::Succeed), 1);
        assert_eq!(item_action_count(&client, 1, &OperationAction::Succeed), 0);
        // The whole batch scope is closed: the settled item and the running
        // item admit no later checkpoint.
        let [settled, running] = item_wires();
        assert!(!ctx.admission().admits(&[settled]), "item 0 is closed");
        assert!(!ctx.admission().admits(&[running]), "item 1 is closed");
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
            crate::driver::drive_outcome(handler, ownership, signal, root, None),
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
            assert_started_defaults(&batch.items[1], "map-item-1");
        }
    }

    /// The payload of a terminal map or parallel parent with two
    /// succeeded items.
    fn terminal_batch_payload() -> String {
        serde_json::json!({
            "all": [
                {"index": 0, "name": "a", "status": "SUCCEEDED", "result": "hello"},
                {"index": 1, "name": "b", "status": "SUCCEEDED", "result": "world"}
            ],
            "completionReason": "ALL_COMPLETED"
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
            let control = root.new_child("1", None);
            let replayed = tokio::time::timeout(bound, replay(control))
                .await
                .expect("the recorded batch replays")
                .expect("the replay succeeds");
            assert_eq!(
                replayed.results(),
                [&"hello".to_owned(), &"world".to_owned()]
            );

            let item = root.new_child("1", None);
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

    // The batch record in JS's shape (issue: write the batch record and the
    // oversized batch summary in JS's shape, and keep empty item payloads
    // across replay).

    /// A serdes that logs each call as `(method, operation id)` and encodes
    /// values as JSON.
    #[derive(Clone, Default)]
    struct CallLog(Arc<std::sync::Mutex<Vec<(&'static str, String)>>>);

    impl CallLog {
        fn calls(&self) -> Vec<(&'static str, String)> {
            let mut calls = self.0.lock().expect("lock").clone();
            calls.sort();
            calls
        }
    }

    impl<T> Serdes<T> for CallLog
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    {
        fn serialize(
            &self,
            value: T,
            context: SerdesContext,
        ) -> impl Future<Output = Result<String, BoxError>> + Send {
            self.0
                .lock()
                .expect("lock")
                .push(("serialize", context.operation_id().to_owned()));
            std::future::ready(serde_json::to_string(&value).map_err(Into::into))
        }

        fn deserialize(
            &self,
            wire: String,
            context: SerdesContext,
        ) -> impl Future<Output = Result<T, BoxError>> + Send {
            self.0
                .lock()
                .expect("lock")
                .push(("deserialize", context.operation_id().to_owned()));
            std::future::ready(serde_json::from_str(&wire).map_err(Into::into))
        }
    }

    /// A struct a map item returns.
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Order {
        id: u32,
        sku: String,
    }

    /// A serdes for [`Order`] that writes its JSON prefixed with `order:`.
    struct OrderSerdes;

    impl Serdes<Order> for OrderSerdes {
        fn serialize(
            &self,
            value: Order,
            _context: SerdesContext,
        ) -> impl Future<Output = Result<String, BoxError>> + Send {
            std::future::ready(
                serde_json::to_string(&value)
                    .map(|json| format!("order:{json}"))
                    .map_err(Into::into),
            )
        }

        fn deserialize(
            &self,
            wire: String,
            _context: SerdesContext,
        ) -> impl Future<Output = Result<Order, BoxError>> + Send {
            std::future::ready(
                wire.strip_prefix("order:")
                    .ok_or_else(|| BoxError::from("missing order: prefix"))
                    .and_then(|json| serde_json::from_str(json).map_err(Into::into)),
            )
        }
    }

    fn order(id: u32) -> Order {
        Order {
            id,
            sku: format!("sku-{id}"),
        }
    }

    /// The wire ID of the child at positional ID `positional`.
    fn wire_of(positional: &str) -> String {
        crate::engine::compute_wire_id_public(positional)
    }

    /// The parent payload of a batch as JSON.
    fn parent_payload_json(client: &crate::client::InMemoryExecutionClient) -> serde_json::Value {
        serde_json::from_str(&parent_payload(client)).expect("the batch payload is JSON")
    }

    /// The JSON structure of the batch record that an explicit
    /// `.result_serdes(JsonSerdes)` stored: the payload is a quoted JSON
    /// string, and its content is the record text.
    fn json_serdes_record_json(
        client: &crate::client::InMemoryExecutionClient,
    ) -> serde_json::Value {
        let payload = parent_payload(client);
        let record_text: String =
            serde_json::from_str(&payload).expect("JsonSerdes stores a quoted JSON string");
        record_json(&record_text)
    }

    /// A checkpoint log holding one terminal batch parent at positional ID
    /// `1` whose payload is `payload`.
    fn terminal_batch_log(payload: &str) -> CheckpointLog {
        CheckpointLog::from_records(vec![succeeded_record("1", payload)])
    }

    /// The JS-form record a batch record text holds.
    fn js_record(record_text: &str) -> JsBatchRecord {
        parse_batch_record(record_text).expect("the record parses")
    }

    #[test]
    fn batch_checkpoint_item_round_trips_empty_result() {
        let result: BatchResult<()> = BatchResult::new(
            vec![BatchItem::new(
                0,
                "map-item-0".to_owned(),
                BatchItemStatus::Succeeded,
                Some(()),
                None,
                None,
            )],
            CompletionReason::AllCompleted,
        );
        let json = from_batch_result(&result, &[Some(String::new())], &ItemEncoding::AllSerdes);
        assert!(json.contains(r#""result":"""#), "json: {json}");
        let back = js_record(&json);
        assert_eq!(
            back.all[0].result.as_deref().map(RawValue::get),
            Some(r#""""#)
        );
    }

    /// A started item is written with neither `result` nor `error`.
    #[test]
    fn batch_checkpoint_started_item_has_no_result_or_error() {
        let started: BatchResult<u32> = BatchResult::new(
            vec![BatchItem::new(
                0,
                "map-item-0".to_owned(),
                BatchItemStatus::Started,
                None,
                None,
                None,
            )],
            CompletionReason::MinSuccessfulReached,
        );
        let json = record_json(&from_batch_result(
            &started,
            &[None],
            &ItemEncoding::AllJson,
        ));
        let item = json.pointer("/all/0").expect("one item");
        assert!(item.get("result").is_none(), "item: {item}");
        assert!(item.get("error").is_none(), "item: {item}");
        assert_eq!(
            json,
            serde_json::json!({
                "all": [{"index": 0, "status": "STARTED", "name": "map-item-0"}],
                "completionReason": "MIN_SUCCESSFUL_REACHED",
            })
        );
    }

    #[tokio::test]
    async fn map_struct_values_are_embedded_as_json() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let out = ctx
            .map(vec![1_u32, 2], |_child, item, _idx| async move {
                Ok(order(item))
            })
            .await
            .expect("map succeeds");
        assert_eq!(out, vec![order(1), order(2)]);
        let payload = parent_payload_json(&client);
        for (i, id) in [1_u32, 2].into_iter().enumerate() {
            assert_eq!(
                payload["all"][i]["result"],
                serde_json::to_value(order(id)).expect("JSON"),
                "payload: {payload}"
            );
        }
    }

    #[tokio::test]
    async fn map_custom_serdes_values_are_strings() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let out = ctx
            .map(vec![1_u32, 2], |_child, item, _idx| async move {
                Ok(order(item))
            })
            .serdes(OrderSerdes)
            .await
            .expect("map succeeds");
        assert_eq!(out, vec![order(1), order(2)]);
        let mut child_payloads = child_success_payloads(&client);
        child_payloads.sort();
        let payload = parent_payload_json(&client);
        let mut results: Vec<String> = payload["all"]
            .as_array()
            .expect("all is an array")
            .iter()
            .map(|item| item["result"].as_str().expect("a string result").to_owned())
            .collect();
        results.sort();
        assert_eq!(results, child_payloads);
        assert!(results.iter().all(|r| r.starts_with("order:")));
    }

    /// `.result_serdes(..)` after `.serdes(..)` keeps the custom item
    /// encoding.
    #[tokio::test]
    async fn result_serdes_keeps_the_custom_item_encoding() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.map(
            vec![1_u32],
            |_child, item, _idx| async move { Ok(order(item)) },
        )
        .serdes(OrderSerdes)
        .result_serdes(crate::serdes::JsonSerdes)
        .await
        .expect("map succeeds");
        assert_eq!(
            json_serdes_record_json(&client)["all"][0]["result"],
            serde_json::json!(r#"order:{"id":1,"sku":"sku-1"}"#)
        );

        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.parallel(vec![crate::future::Branch::new("a", |_c| async {
            Ok(order(1))
        })])
        .serdes(OrderSerdes)
        .result_serdes(crate::serdes::JsonSerdes)
        .await
        .expect("parallel succeeds");
        assert_eq!(
            json_serdes_record_json(&client)["all"][0]["result"],
            serde_json::json!(r#"order:{"id":1,"sku":"sku-1"}"#)
        );
    }

    #[tokio::test]
    async fn failed_item_error_has_js_shape() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let live = ctx
            .map(vec![1_u32], |_child, _item, _idx| async move {
                Err::<u32, BoxError>(Box::new(
                    crate::TypedError::with_type("PaymentError", "declined").with_error_data("d"),
                ))
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("a tolerated failure is a batch result");
        let message = live.items[0]
            .error_message
            .clone()
            .expect("the failed item has a message");
        let payload = parent_payload(&client);
        let json: serde_json::Value = serde_json::from_str(&payload).expect("JSON");
        assert_eq!(
            json["all"][0]["error"],
            serde_json::json!({
                "ErrorType": "ChildContextError",
                "ErrorMessage": message,
                "ErrorData": "d",
                "Cause": {"ErrorType": "PaymentError", "ErrorMessage": message, "ErrorData": "d"},
            })
        );

        let replayed: BatchResult<u32> = test_ctx(terminal_batch_log(&payload))
            .map(vec![1_u32], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("replay succeeds");
        assert_eq!(
            replayed.items[0].error_type.as_deref(),
            Some("PaymentError")
        );
        assert_eq!(replayed.items[0].error_data.as_deref(), Some("d"));
        assert_eq!(
            replayed.items[0].error_message.as_deref(),
            Some(message.as_str())
        );
    }

    #[tokio::test]
    async fn failed_item_without_recorded_type_falls_back_to_error() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.map(vec![1_u32], |_child, _item, _idx| async move {
            Err::<u32, BoxError>("boom".into())
        })
        .completion(
            crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
        )
        .await_batch()
        .await
        .expect("a tolerated failure is a batch result");
        let payload = parent_payload(&client);
        let json: serde_json::Value = serde_json::from_str(&payload).expect("JSON");
        assert_eq!(json["all"][0]["error"]["Cause"]["ErrorType"], "Error");

        let replayed: BatchResult<u32> = test_ctx(terminal_batch_log(&payload))
            .map(vec![1_u32], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("replay succeeds");
        assert_eq!(replayed.items[0].error_type.as_deref(), Some("Error"));
    }

    /// A JS error object without `Cause` reads its fields from the top
    /// level.
    #[tokio::test]
    async fn js_error_without_cause_reads_top_level() {
        let payload = serde_json::json!({
            "all": [{
                "index": 0,
                "status": "FAILED",
                "name": "map-item-0",
                "error": {
                    "ErrorType": "TopError",
                    "ErrorMessage": "top",
                    "ErrorData": "top-data",
                    "StackTrace": ["top-frame"],
                },
                "result": 5,
            }],
            "completionReason": "ALL_COMPLETED",
        });
        let replayed: BatchResult<u32> = test_ctx(terminal_batch_log(&payload.to_string()))
            .map(vec![1_u32], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
            )
            .await_batch()
            .await
            .expect("replay succeeds");
        let item = &replayed.items[0];
        assert_eq!(item.error_type.as_deref(), Some("TopError"));
        assert_eq!(item.error_message.as_deref(), Some("top"));
        assert_eq!(item.error_data.as_deref(), Some("top-data"));
        assert_eq!(item.stack_trace, ["top-frame".to_owned()]);
        // Only a succeeded item decodes a value.
        assert_eq!(item.result, None);
    }

    /// A terminal map parent whose payload is the `{results, reason}` form
    /// that no SDK writes fails replay: the record has neither `all` nor
    /// `completionReason`.
    #[tokio::test]
    async fn results_reason_record_fails_replay() {
        let payload = r#"{"results":[{"index":0,"name":"a","status":"SUCCEEDED","result":"\"hello\""}],"reason":"ALL_COMPLETED"}"#;
        let err = test_ctx(terminal_batch_log(payload))
            .map(vec![0_u32], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .item_namer(|_, _| "a".to_owned())
            .await_batch()
            .await
            .map(|batch: BatchResult<String>| batch.items.len())
            .expect_err("the record fails replay");
        assert!(
            format!("{err:#}").contains("batch record has neither `all` nor `completionReason`"),
            "unexpected error: {err:#}"
        );
    }

    /// The batch record text of `wires`, one succeeded `map` item per wire.
    fn record_of_wires(wires: &[&str], encoding: &ItemEncoding) -> String {
        let items = (0..wires.len())
            .map(|index| {
                BatchItem::new(
                    index,
                    format!("map-item-{index}"),
                    BatchItemStatus::Succeeded,
                    Some(()),
                    None,
                    None,
                )
            })
            .collect();
        let result: BatchResult<()> = BatchResult::new(items, CompletionReason::AllCompleted);
        let wires: Vec<Option<String>> = wires.iter().map(|w| Some((*w).to_owned())).collect();
        from_batch_result(&result, &wires, encoding)
    }

    /// The exact `result` texts of a JS-form record text.
    fn result_texts(record_text: &str) -> Vec<Option<String>> {
        js_record(record_text)
            .all
            .iter()
            .map(|item| item.result.as_deref().map(|raw| raw.get().to_owned()))
            .collect()
    }

    /// A `Serdes<String>` item serdes that returns each wire unchanged and
    /// logs the serdes context it was handed.
    #[derive(Clone, Default)]
    struct WireLog(Arc<std::sync::Mutex<Vec<(String, String)>>>);

    impl WireLog {
        fn calls(&self) -> Vec<(String, String)> {
            self.0.lock().expect("lock").clone()
        }
    }

    impl Serdes<String> for WireLog {
        fn serialize(
            &self,
            value: String,
            _context: SerdesContext,
        ) -> impl Future<Output = Result<String, BoxError>> + Send {
            std::future::ready(Ok(value))
        }

        fn deserialize(
            &self,
            wire: String,
            context: SerdesContext,
        ) -> impl Future<Output = Result<String, BoxError>> + Send {
            self.0
                .lock()
                .expect("lock")
                .push((context.operation_id().to_owned(), wire.clone()));
            std::future::ready(Ok(wire))
        }
    }

    /// A JSON item value keeps its exact text through the record and back:
    /// the record embeds the wire string unchanged, and replay hands the
    /// item serdes that same text.
    #[tokio::test]
    async fn record_round_trip_keeps_exact_json_item_text() {
        let wires = [
            "null",
            r#""a string""#,
            "[1,2,3]",
            r#"{"n":340282366920938463463374607431768211455,"m":{"k":-170141183460469231731687303715884105728}}"#,
            "1e2",
            "1.0",
            "-0",
        ];
        let text = record_of_wires(&wires, &ItemEncoding::AllJson);
        for wire in wires {
            assert!(
                text.contains(&format!(r#""result":{wire}}}"#)),
                "the record embeds {wire} unchanged: {text}"
            );
        }
        assert_eq!(
            result_texts(&text),
            wires
                .iter()
                .map(|w| Some((*w).to_owned()))
                .collect::<Vec<_>>()
        );

        let log = WireLog::default();
        let ctx = test_ctx(CheckpointLog::empty());
        let replayed: BatchResult<String> = to_batch_result(
            &ctx,
            &text,
            &Arc::new(log.clone()),
            ItemEncoding::AllJson,
            wires.len(),
            "arn:test",
            &ItemIds::new(&OperationId::new("1".to_owned()), wires.len()),
        )
        .await
        .expect("the record replays");
        let values: Vec<String> = replayed
            .items
            .into_iter()
            .map(|item| item.result.expect("a value"))
            .collect();
        assert_eq!(values, wires);
        let handed: Vec<String> = log.calls().into_iter().map(|(_, wire)| wire).collect();
        assert_eq!(handed, wires);
    }

    /// A custom-serdes item records its wire string as a JSON string, also
    /// when the wire is empty or looks like JSON, and replay hands the item
    /// serdes the original wire.
    #[tokio::test]
    async fn custom_serdes_item_records_a_json_string() {
        let wires = [
            "",
            r#"{"a":1}"#,
            "340282366920938463463374607431768211455",
            "plain",
        ];
        let text = record_of_wires(&wires, &ItemEncoding::AllSerdes);
        assert_eq!(
            result_texts(&text),
            vec![
                Some(r#""""#.to_owned()),
                Some(r#""{\"a\":1}""#.to_owned()),
                Some(r#""340282366920938463463374607431768211455""#.to_owned()),
                Some(r#""plain""#.to_owned()),
            ]
        );
        let replayed: BatchResult<String> = to_batch_result(
            &test_ctx(CheckpointLog::empty()),
            &text,
            &Arc::new(WireLog::default()),
            ItemEncoding::AllSerdes,
            wires.len(),
            "arn:test",
            &ItemIds::new(&OperationId::new("1".to_owned()), wires.len()),
        )
        .await
        .expect("the record replays");
        let values: Vec<String> = replayed
            .items
            .into_iter()
            .map(|item| item.result.expect("a value"))
            .collect();
        assert_eq!(values, wires);
    }

    /// The error `parse_batch_record` returns for `text`.
    fn record_error(text: &str) -> String {
        parse_batch_record(text).expect_err("the record must be rejected")
    }

    #[test]
    fn record_parser_reads_the_js_form() {
        let js = r#"{"all":[{"index":0,"status":"SUCCEEDED","name":"a","result":7}],"completionReason":"ALL_COMPLETED"}"#;
        let record = parse_batch_record(js).expect("the JS form parses");
        assert_eq!(record.completion_reason, "ALL_COMPLETED");
        assert_eq!(
            record
                .all
                .first()
                .and_then(|i| i.result.as_deref())
                .map(RawValue::get),
            Some("7")
        );
    }

    #[test]
    fn record_parser_rejects_non_records() {
        assert!(record_error(r#"{"all":"#).contains("batch record is not valid"));
        let trailing = record_error(r#"{"all":[],"completionReason":"ALL_COMPLETED"} {}"#);
        assert!(trailing.contains("trailing characters"), "{trailing}");
        for text in [
            "[]",
            r#"[{"all":[],"completionReason":"ALL_COMPLETED"}]"#,
            "5",
            r#""s""#,
            "null",
            "true",
        ] {
            let err = record_error(text);
            assert!(
                err.contains("expected a batch record object"),
                "{text}: {err}"
            );
        }
    }

    #[test]
    fn record_parser_requires_both_fields() {
        let neither = "batch record has neither `all` nor `completionReason`";
        let cases = [
            ("{}", neither),
            (r#"{"results":[],"reason":"ALL_COMPLETED"}"#, neither),
            (
                r#"{"all":[]}"#,
                "batch record has `all` without `completionReason`",
            ),
            (
                r#"{"completionReason":"ALL_COMPLETED"}"#,
                "batch record has `completionReason` without `all`",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(record_error(text), expected, "{text}");
        }
    }

    /// A recognized field whose value is `null` counts as absent.
    #[test]
    fn record_parser_reads_null_fields_as_absent() {
        let cases = [
            (
                r#"{"all":null,"completionReason":"ALL_COMPLETED"}"#,
                "batch record has `completionReason` without `all`",
            ),
            (
                r#"{"all":[],"completionReason":null}"#,
                "batch record has `all` without `completionReason`",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(record_error(text), expected, "{text}");
        }
    }

    #[test]
    fn record_parser_rejects_wrong_field_types() {
        for (text, field_type) in [
            (
                r#"{"all":5,"completionReason":"ALL_COMPLETED"}"#,
                "invalid type: integer `5`, expected a sequence",
            ),
            (
                r#"{"all":[],"completionReason":5}"#,
                "invalid type: integer `5`, expected a string",
            ),
        ] {
            let err = record_error(text);
            assert!(err.contains(field_type), "{text}: {err}");
        }
    }

    #[test]
    fn record_parser_rejects_duplicate_fields() {
        for (text, field) in [
            (
                r#"{"all":[],"all":[],"completionReason":"ALL_COMPLETED"}"#,
                "all",
            ),
            (
                r#"{"all":null,"all":[],"completionReason":"ALL_COMPLETED"}"#,
                "all",
            ),
            (
                r#"{"all":[],"completionReason":"ALL_COMPLETED","completionReason":"ALL_COMPLETED"}"#,
                "completionReason",
            ),
            (
                r#"{"all":[],"completionReason":null,"completionReason":"ALL_COMPLETED"}"#,
                "completionReason",
            ),
        ] {
            let err = record_error(text);
            assert!(
                err.contains(&format!("duplicate field `{field}`")),
                "{text}: {err}"
            );
        }
    }

    #[test]
    fn record_parser_ignores_unknown_fields() {
        for text in [
            r#"{"extra":{"nested":[1,{"a":null}]},"all":[],"completionReason":"ALL_COMPLETED","type":"MapResult"}"#,
            r#"{"all":[],"completionReason":"ALL_COMPLETED","results":"x","reason":5}"#,
        ] {
            assert!(parse_batch_record(text).is_ok(), "{text}");
        }
    }

    /// Replays a JS-form batch record `payload` through a map over one
    /// `u32` item, with a custom item serdes when `custom` is set, and
    /// returns the error text.
    async fn js_replay_error(payload: &serde_json::Value, custom: bool) -> String {
        async fn never_runs(
            _child: DurableContext,
            _item: u32,
            _idx: usize,
        ) -> Result<u32, BoxError> {
            unreachable!("a terminal batch replays")
        }
        let ctx = test_ctx(terminal_batch_log(&payload.to_string()));
        let map = ctx.map(vec![1_u32], never_runs);
        let err = if custom {
            map.serdes(CallLog::default()).await_batch().await
        } else {
            map.await_batch().await
        }
        .expect_err("the record fails replay");
        crate::error::chain_string(&err)
    }

    /// A stored record without both `all` and `completionReason` fails
    /// replay with the parser's error.
    #[tokio::test]
    async fn malformed_stored_record_fails_replay() {
        let payload = serde_json::json!({"completionReason": "ALL_COMPLETED"});
        let err = js_replay_error(&payload, false).await;
        assert!(
            err.contains("batch record has `completionReason` without `all`"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn serdes_item_with_non_string_result_fails_replay() {
        let payload = serde_json::json!({
            "all": [{"index": 0, "status": "SUCCEEDED", "name": "map-item-0", "result": 1}],
            "completionReason": "ALL_COMPLETED",
        });
        let err = js_replay_error(&payload, true).await;
        assert!(
            err.contains("batch record item 0: expected a string result"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn js_item_without_name_fails_replay() {
        let payload = serde_json::json!({
            "all": [{"index": 0, "status": "SUCCEEDED", "result": 1}],
            "completionReason": "ALL_COMPLETED",
        });
        let err = js_replay_error(&payload, false).await;
        assert!(err.contains("batch record item 0: missing name"), "{err}");
    }

    #[tokio::test]
    async fn js_item_with_unknown_status_fails_replay() {
        let payload = serde_json::json!({
            "all": [{"index": 0, "status": "WEIRD", "name": "map-item-0"}],
            "completionReason": "ALL_COMPLETED",
        });
        let err = js_replay_error(&payload, false).await;
        assert!(err.contains("unknown item status: WEIRD"), "{err}");
    }

    #[tokio::test]
    async fn oversized_map_writes_js_summary() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let big = "x".repeat(101 * 1024);
        ctx.map(vec![0_u32, 1, 2], move |_child, _item, _idx| {
            let big = big.clone();
            async move { Ok(big) }
        })
        .await
        .expect("map succeeds");
        let parent = parent_succeed_update(&client);
        assert_eq!(
            parent.context_options().and_then(|c| c.replay_children),
            Some(true)
        );
        assert_eq!(
            parent_payload_json(&client),
            serde_json::json!({
                "type": "MapResult",
                "totalCount": 3,
                "successCount": 3,
                "failureCount": 0,
                "completionReason": "ALL_COMPLETED",
                "status": "SUCCEEDED",
                "itemStatuses": "SSS",
            })
        );
    }

    /// An oversized batch with a failed item records `F` and the status
    /// `FAILED`.
    #[tokio::test]
    async fn oversized_map_with_failure_writes_failed_status() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let big = "x".repeat(300 * 1024);
        ctx.map(vec![0_u32, 1], move |_child, item, _idx| {
            let big = big.clone();
            async move {
                if item == 0 {
                    Ok(big)
                } else {
                    Err("boom".into())
                }
            }
        })
        .completion(
            crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1),
        )
        .await_batch()
        .await
        .expect("a tolerated failure is a batch result");
        let summary = parent_payload_json(&client);
        assert_eq!(summary["status"], "FAILED", "summary: {summary}");
        assert_eq!(summary["itemStatuses"], "SF");
        assert_eq!(summary["failureCount"], 1);
    }

    #[tokio::test]
    async fn oversized_parallel_writes_started_count() {
        use crate::future::Branch;
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let big = "x".repeat(300 * 1024);
        let branches = vec![
            Branch::new("big", move |_c| async move { Ok(big) }),
            Branch::new("slow", |c| async move {
                Ok(
                    c.step(|_| std::future::pending::<Result<String, BoxError>>())
                        .name("forever")
                        .await?,
                )
            }),
        ];
        let batch = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            ctx.parallel(branches)
                .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(1))
                .await_batch(),
        )
        .await
        .expect("the batch does not wait for the running branch")
        .expect("the batch succeeds");
        assert_eq!(batch.items[1].status, BatchItemStatus::Started);
        assert_eq!(
            parent_payload_json(&client),
            serde_json::json!({
                "type": "ParallelResult",
                "totalCount": 2,
                "successCount": 1,
                "failureCount": 0,
                "startedCount": 1,
                "completionReason": "MIN_SUCCESSFUL_REACHED",
                "status": "SUCCEEDED",
                "itemStatuses": "S-",
            })
        );
    }

    /// The oversized summary's `status` comes from `BatchResult::status()`:
    /// a predicate that completed the batch as succeeded writes
    /// `SUCCEEDED` although an item failed, and one that completed it as
    /// failed writes `FAILED` although no item failed.
    #[test]
    fn oversized_summary_status_follows_custom_completion() {
        let failed_item = || {
            BatchItem::<u32>::new(
                0,
                "a".to_owned(),
                BatchItemStatus::Failed,
                None,
                Some("boom".to_owned()),
                None,
            )
        };
        let succeeded_item = || {
            BatchItem::<u32>::new(
                0,
                "a".to_owned(),
                BatchItemStatus::Succeeded,
                Some(1),
                None,
                None,
            )
        };
        let summary = |items, reason| -> serde_json::Value {
            let result = BatchResult::new(items, reason);
            serde_json::from_str(&oversized_batch_summary(&result, MAP_SUB_TYPE))
                .expect("the summary is JSON")
        };
        let custom_ok = summary(
            vec![failed_item()],
            CompletionReason::CustomCompletionSucceeded,
        );
        assert_eq!(custom_ok["status"], "SUCCEEDED", "summary: {custom_ok}");
        assert_eq!(custom_ok["itemStatuses"], "F");
        assert_eq!(custom_ok["completionReason"], "CUSTOM_COMPLETION_SUCCEEDED");
        let custom_failed = summary(
            vec![succeeded_item()],
            CompletionReason::CustomCompletionFailed,
        );
        assert_eq!(
            custom_failed["status"], "FAILED",
            "summary: {custom_failed}"
        );
        assert_eq!(custom_failed["itemStatuses"], "S");
        assert_eq!(
            custom_failed["completionReason"],
            "CUSTOM_COMPLETION_FAILED"
        );
    }

    /// A failed item writes its non-empty frames and its data at both
    /// error levels, under `ChildContextError` with one cause.
    #[test]
    fn failed_item_writes_stack_trace_at_both_levels() {
        let mut item = BatchItem::<u32>::new(
            0,
            "a".to_owned(),
            BatchItemStatus::Failed,
            None,
            Some("it broke".to_owned()),
            Some("CustomFailure".to_owned()),
        );
        item.error_data = Some("data".to_owned());
        item.stack_trace = vec!["frame-a".to_owned(), "frame-b".to_owned()];
        let result = BatchResult::new(vec![item], CompletionReason::AllCompleted);
        let payload = record_json(&from_batch_result(&result, &[None], &ItemEncoding::AllJson));
        assert_eq!(
            payload.pointer("/all/0/error"),
            Some(&serde_json::json!({
                "ErrorType": "ChildContextError",
                "ErrorMessage": "it broke",
                "ErrorData": "data",
                "StackTrace": ["frame-a", "frame-b"],
                "Cause": {
                    "ErrorType": "CustomFailure",
                    "ErrorMessage": "it broke",
                    "ErrorData": "data",
                    "StackTrace": ["frame-a", "frame-b"],
                },
            }))
        );
    }

    /// A failed item reads type, message and data from `Cause` when
    /// `Cause` has them, and from the top level when `Cause` lacks them.
    /// Its frames come from `Cause` whenever `Cause` is present.
    #[tokio::test]
    async fn js_error_prefers_cause_and_falls_back_per_field() {
        let payload = serde_json::json!({
            "all": [
                {
                    "index": 0,
                    "status": "FAILED",
                    "name": "map-item-0",
                    "error": {
                        "ErrorType": "TopError",
                        "ErrorMessage": "top",
                        "ErrorData": "top-data",
                        "StackTrace": ["top-frame"],
                        "Cause": {
                            "ErrorType": "CauseError",
                            "ErrorMessage": "cause",
                            "ErrorData": "cause-data",
                            "StackTrace": ["cause-frame"],
                        },
                    },
                },
                {
                    "index": 1,
                    "status": "FAILED",
                    "name": "map-item-1",
                    "error": {
                        "ErrorType": "TopError",
                        "ErrorMessage": "top",
                        "ErrorData": "top-data",
                        "StackTrace": ["top-frame"],
                        "Cause": {},
                    },
                },
            ],
            "completionReason": "ALL_COMPLETED",
        });
        let replayed: BatchResult<u32> = test_ctx(terminal_batch_log(&payload.to_string()))
            .map(vec![1_u32, 2], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .completion(
                crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(2),
            )
            .await_batch()
            .await
            .expect("replay succeeds");
        let fields = |item: &BatchItem<u32>| {
            (
                item.error_type.clone(),
                item.error_message.clone(),
                item.error_data.clone(),
                item.stack_trace.clone(),
            )
        };
        assert_eq!(
            fields(&replayed.items[0]),
            (
                Some("CauseError".to_owned()),
                Some("cause".to_owned()),
                Some("cause-data".to_owned()),
                vec!["cause-frame".to_owned()],
            )
        );
        assert_eq!(
            fields(&replayed.items[1]),
            (
                Some("TopError".to_owned()),
                Some("top".to_owned()),
                Some("top-data".to_owned()),
                Vec::new(),
            )
        );
    }

    /// Replays a JS-form batch record `payload` through a map over
    /// `inputs` `u32` items with a recording item serdes. Returns the
    /// error text and the serdes calls.
    async fn js_replay_error_over(
        payload: &serde_json::Value,
        inputs: u32,
    ) -> (String, Vec<(&'static str, String)>) {
        async fn never_runs(
            _child: DurableContext,
            _item: u32,
            _idx: usize,
        ) -> Result<u32, BoxError> {
            unreachable!("a terminal batch replays")
        }
        let calls = CallLog::default();
        let err = test_ctx(terminal_batch_log(&payload.to_string()))
            .map((0..inputs).collect::<Vec<_>>(), never_runs)
            .serdes(calls.clone())
            .await_batch()
            .await
            .expect_err("the record fails replay");
        (crate::error::chain_string(&err), calls.calls())
    }

    /// A JS-form record with one succeeded item per index in `indexes`, in
    /// that order.
    fn js_record_with_indexes(indexes: &[usize]) -> serde_json::Value {
        let all: Vec<_> = indexes
            .iter()
            .map(|index| {
                serde_json::json!({
                    "index": index,
                    "status": "SUCCEEDED",
                    "name": format!("map-item-{index}"),
                    "result": "1",
                })
            })
            .collect();
        serde_json::json!({"all": all, "completionReason": "ALL_COMPLETED"})
    }

    /// A record whose `all` is not the started prefix in index order fails
    /// the replay before any item value is decoded.
    #[tokio::test]
    async fn js_record_with_malformed_indexes_fails_replay() {
        let cases: [(&[usize], &str); 4] = [
            (&[0, 2], "batch record item at position 1 has index 2"),
            (&[0, 0], "batch record item at position 1 has index 0"),
            (&[1, 0], "batch record item at position 0 has index 1"),
            (
                &[usize::MAX],
                "batch record item at position 0 has index 18446744073709551615",
            ),
        ];
        for (indexes, expected) in cases {
            let (err, calls) = js_replay_error_over(&js_record_with_indexes(indexes), 3).await;
            assert!(err.contains(expected), "indexes {indexes:?}: {err}");
            assert_eq!(calls, vec![], "indexes {indexes:?}");
        }
    }

    /// A record with more items than the current input fails the replay
    /// before any item value is decoded. A record as long as the input
    /// replays.
    #[tokio::test]
    async fn js_record_longer_than_the_input_fails_replay() {
        let (err, calls) = js_replay_error_over(&js_record_with_indexes(&[0, 1, 2, 3]), 3).await;
        assert!(
            err.contains("batch record has 4 items but the input has 3"),
            "{err}"
        );
        assert_eq!(calls, vec![]);

        let replayed: BatchResult<u32> = test_ctx(terminal_batch_log(
            &js_record_with_indexes(&[0, 1, 2]).to_string(),
        ))
        .map(vec![0_u32, 1, 2], |_child, _item, _idx| async move {
            unreachable!("a terminal batch replays")
        })
        .serdes(CallLog::default())
        .await_batch()
        .await
        .expect("a record as long as the input replays");
        assert_eq!(replayed.items.len(), 3);
    }

    /// The records of a batch whose items 0 and 2 completed while item 1
    /// ran, with `payload` as the oversized parent's payload.
    fn started_middle_log(payload: &str) -> CheckpointLog {
        CheckpointLog::from_records(vec![
            decision_record_parent("1", payload),
            succeeded_record("1-1", "\"a\""),
            succeeded_record("1-3", "\"c\""),
        ])
    }

    /// Replays a map of 4 items from `log`, without running a body.
    async fn replay_started_middle(log: CheckpointLog) -> BatchResult<String> {
        test_ctx(log)
            .map(vec![0_usize, 1, 2, 3], |_child, _item, _idx| async move {
                unreachable!("replay must not re-run any item body")
            })
            .completion(crate::builders::map_parallel::CompletionConfig::with_min_successful(2))
            .await_batch()
            .await
            .expect("replay succeeds")
    }

    fn started_middle_rows(
        batch: &BatchResult<String>,
    ) -> Vec<(usize, BatchItemStatus, Option<String>)> {
        batch
            .items
            .iter()
            .map(|i| (i.index, i.status, i.result.clone()))
            .collect()
    }

    const STARTED_MIDDLE: [(usize, BatchItemStatus, Option<&str>); 3] = [
        (0, BatchItemStatus::Succeeded, Some("a")),
        (1, BatchItemStatus::Started, None),
        (2, BatchItemStatus::Succeeded, Some("c")),
    ];

    fn started_middle_expected() -> Vec<(usize, BatchItemStatus, Option<String>)> {
        STARTED_MIDDLE
            .iter()
            .map(|(i, s, r)| (*i, *s, r.map(str::to_owned)))
            .collect()
    }

    #[tokio::test]
    async fn python_decision_record_replays() {
        let batch = replay_started_middle(started_middle_log(
            r#"{"totalCount":3,"completionReason":"MIN_SUCCESSFUL_REACHED","startedIndexes":[1]}"#,
        ))
        .await;
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        assert_eq!(started_middle_rows(&batch), started_middle_expected());
    }

    /// The JS summary replays to the same result as the Python decision
    /// record.
    #[tokio::test]
    async fn js_summary_replays_like_the_decision_record() {
        let batch = replay_started_middle(started_middle_log(
            r#"{"type":"MapResult","totalCount":3,"successCount":2,"failureCount":0,"completionReason":"MIN_SUCCESSFUL_REACHED","status":"SUCCEEDED","itemStatuses":"S-S"}"#,
        ))
        .await;
        assert_eq!(batch.reason, CompletionReason::MinSuccessfulReached);
        assert_eq!(started_middle_rows(&batch), started_middle_expected());
    }

    /// The order of rule 10: JS summary, Python decision record, corrupt.
    #[test]
    fn replay_children_payload_is_read_in_order() {
        let record = parse_replay_children_payload(Some(
            r#"{"itemStatuses":"S-F","completionReason":"ALL_COMPLETED","totalCount":99}"#,
        ))
        .expect("a JS summary parses");
        assert_eq!(record.total_count, 3, "the length of itemStatuses wins");
        assert_eq!(record.completed_indexes, Some(vec![0, 2]));
        assert_eq!(record.started_indexes, None);
        assert_eq!(record.completion_reason, "ALL_COMPLETED");

        for statuses in [r#""""#, r#""SX""#, "1", "null"] {
            let payload =
                format!(r#"{{"itemStatuses":{statuses},"completionReason":"ALL_COMPLETED"}}"#);
            let err = parse_replay_children_payload(Some(&payload))
                .expect_err("an invalid itemStatuses is corrupt");
            assert!(
                crate::error::chain_string(&err).contains("itemStatuses must be a non-empty"),
                "{statuses}"
            );
        }

        // Without a completion reason, the plan rejects the record.
        let record =
            parse_replay_children_payload(Some(r#"{"itemStatuses":"S"}"#)).expect("parses");
        let err = ReplayPlan::from_record(&record, 1).expect_err("no reason");
        assert!(crate::error::chain_string(&err).contains("unknown completionReason"));

        let record = parse_replay_children_payload(Some(
            r#"{"totalCount":2,"completionReason":"ALL_COMPLETED","completedIndexes":[0]}"#,
        ))
        .expect("a decision record parses");
        assert_eq!(record.completed_indexes, Some(vec![0]));
        let record = parse_replay_children_payload(Some(
            r#"{"totalCount":2,"completionReason":"ALL_COMPLETED","startedIndexes":[1]}"#,
        ))
        .expect("a decision record parses");
        assert_eq!(record.started_indexes, Some(vec![1]));

        for corrupt in [
            "not json",
            r#"{"unrelated":true}"#,
            r#"{"totalCount":2,"completionReason":"ALL_COMPLETED"}"#,
            r#"{"startedIndexes":[]}"#,
            "[1]",
        ] {
            let err = parse_replay_children_payload(Some(corrupt))
                .expect_err("an unrecognized payload is corrupt");
            assert!(
                crate::error::chain_string(&err).contains("malformed batch decision record"),
                "{corrupt}"
            );
        }
    }

    #[tokio::test]
    async fn item_serdes_called_once_per_item() {
        let calls = CallLog::default();
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let out: Vec<u32> = ctx
            .map(
                vec![10_u32, 20],
                |_child, item, _idx| async move { Ok(item) },
            )
            .serdes(calls.clone())
            .await
            .expect("map succeeds");
        assert_eq!(out, vec![10, 20]);
        let (c0, c1) = (wire_of("1-1"), wire_of("1-2"));
        let mut live = vec![
            ("deserialize", c0.clone()),
            ("deserialize", c1.clone()),
            ("serialize", c0.clone()),
            ("serialize", c1.clone()),
        ];
        live.sort();
        assert_eq!(calls.calls(), live);

        let replay_calls = CallLog::default();
        let replayed: Vec<u32> = test_ctx(terminal_batch_log(&parent_payload(&client)))
            .map(vec![10_u32, 20], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .serdes(replay_calls.clone())
            .await
            .expect("replay succeeds");
        assert_eq!(replayed, vec![10, 20]);
        let mut replay = vec![("deserialize", c0), ("deserialize", c1)];
        replay.sort();
        assert_eq!(replay_calls.calls(), replay);
        assert!(
            calls
                .calls()
                .iter()
                .chain(replay_calls.calls().iter())
                .all(|(_, id)| !id.contains("/item-"))
        );
    }

    #[tokio::test]
    async fn batch_item_name_survives_summary_replay() {
        use crate::future::Branch;
        let branches = || {
            vec![
                Branch::new("a", |_c| async { Err::<u32, BoxError>("a failed".into()) }),
                Branch::unnamed(|_c| async { Err::<u32, BoxError>("b failed".into()) }),
            ]
        };
        let tolerate = || {
            crate::builders::map_parallel::CompletionConfig::with_tolerated_failure_percentage(
                100.0,
            )
        };
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        ctx.parallel(branches())
            .completion(tolerate())
            .await_batch()
            .await
            .expect("tolerated failures are a batch result");

        let replayed = test_ctx(terminal_batch_log(&parent_payload(&client)))
            .parallel(vec![
                Branch::<u32>::new("a", |_c| async { unreachable!("a terminal batch replays") }),
                Branch::<u32>::unnamed(|_c| async { unreachable!("a terminal batch replays") }),
            ])
            .completion(tolerate())
            .await_batch()
            .await
            .expect("replay succeeds");
        let names: Vec<&str> = replayed.items.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["a", "parallel-branch-1"]);
        let error_names: Vec<&str> = replayed.errors().iter().map(|e| e.name).collect();
        assert_eq!(error_names, names);
    }

    /// A Flat map writes each item value into the record, and replays it.
    #[tokio::test]
    async fn flat_map_values_replay_from_the_record() {
        let (ctx, client) = test_ctx_with_client(CheckpointLog::empty());
        let live: Vec<u32> = ctx
            .map(
                vec![1_u32, 2],
                |_child, item, _idx| async move { Ok(item * 10) },
            )
            .nesting(NestingMode::Flat)
            .await
            .expect("map succeeds");
        assert_eq!(live, vec![10, 20]);
        let payload = parent_payload(&client);
        let replayed: Vec<u32> = test_ctx(terminal_batch_log(&payload))
            .map(vec![1_u32, 2], |_child, _item, _idx| async move {
                unreachable!("a terminal batch replays")
            })
            .nesting(NestingMode::Flat)
            .await
            .expect("replay succeeds");
        assert_eq!(replayed, live);
    }

    /// An item replayed from its terminal child record writes the record's
    /// wire string into the batch record.
    #[tokio::test]
    async fn resumed_terminal_item_writes_its_recorded_value() {
        let log = CheckpointLog::from_records(vec![
            (
                wire_of("1"),
                CheckpointRecord {
                    status: CheckpointStatus::Started,
                    result: None,
                    ..succeeded_record("1", "").1
                },
            ),
            succeeded_record("1-1", "\"hello\""),
        ]);
        let (ctx, client) = test_ctx_with_client(log);
        let out: Vec<String> = ctx
            .map(vec![0_u32], |_child, _item, _idx| async move {
                unreachable!("a terminal child replays")
            })
            .await
            .expect("map succeeds");
        assert_eq!(out, vec!["hello"]);
        assert_eq!(parent_payload_json(&client)["all"][0]["result"], "hello");
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

/// The batch terminals (`.await`, `await_batch()` and `spawn_batch()`)
/// driven through `LocalRunner`.
#[cfg(all(test, feature = "test-util"))]
mod batch_terminal_tests {
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::{BatchItemStatus, BatchResult, CompletionReason, MAX_CONCURRENCY_ZERO_MESSAGE};
    use crate::builders::map_parallel::CompletionConfig;
    use crate::test_util::{ExecutionStatus, LocalRunner};
    use crate::{BoxError, Branch, DurableContext};

    /// `(index, name, status, result)` of each item, and the reason.
    type Record = (
        Vec<(usize, String, BatchItemStatus, Option<i32>)>,
        CompletionReason,
    );

    fn record_of(batch: &BatchResult<i32>) -> Record {
        let items = batch
            .items
            .iter()
            .map(|item| (item.index, item.name.clone(), item.status, item.result))
            .collect();
        (items, batch.reason)
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    type ItemFuture = std::pin::Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>;

    /// `map(vec![1, 2])` named `m`, where each item's step appends the
    /// current invocation number to `log` and returns `i * 10`.
    fn logged_map(
        ctx: &DurableContext,
        invocation: usize,
        log: &Arc<Mutex<Vec<usize>>>,
    ) -> crate::builders::MapBuilder<
        i32,
        i32,
        impl Fn(DurableContext, i32, usize) -> ItemFuture + Send + Sync + 'static,
        ItemFuture,
    > {
        let log = Arc::clone(log);
        ctx.map(vec![1, 2], move |child, i: i32, _idx| -> ItemFuture {
            let log = Arc::clone(&log);
            Box::pin(async move {
                child
                    .step(move |_| async move {
                        lock(&log).push(invocation);
                        Ok(i * 10)
                    })
                    .name("scale")
                    .await
                    .map_err(Into::into)
            })
        })
        .name("m")
    }

    #[tokio::test]
    async fn spawn_batch_runs_while_the_handler_waits() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let log = Arc::new(Mutex::new(Vec::new()));
        let (inv, step_log) = (Arc::clone(&invocations), Arc::clone(&log));
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let invocation = inv.fetch_add(1, Ordering::SeqCst) + 1;
                    let log = Arc::clone(&step_log);
                    async move {
                        let batch = logged_map(&ctx, invocation, &log).spawn_batch();
                        ctx.wait(Duration::from_secs(1)).await?;
                        let batch = batch.await?;
                        Ok::<_, BoxError>(batch.results().into_iter().copied().collect::<Vec<_>>())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        assert_eq!(result.output(), Some(&vec![10, 20]));
        assert_eq!(result.invocation_count(), 2);
        assert_eq!(*lock(&log), [1, 1]);
    }

    #[tokio::test]
    async fn spawn_batch_equals_await_batch() {
        let spawned: Arc<Mutex<Vec<Record>>> = Arc::default();
        let awaited: Arc<Mutex<Vec<Record>>> = Arc::default();

        let records = Arc::clone(&spawned);
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let records = Arc::clone(&records);
                    async move {
                        let log = Arc::default();
                        let batch = logged_map(&ctx, 0, &log).spawn_batch();
                        ctx.wait(Duration::from_secs(1)).await?;
                        let batch = batch.await?;
                        lock(&records).push(record_of(&batch));
                        Ok::<_, BoxError>(())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);

        let records = Arc::clone(&awaited);
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let records = Arc::clone(&records);
                    async move {
                        let log = Arc::default();
                        let batch = logged_map(&ctx, 0, &log).await_batch().await?;
                        lock(&records).push(record_of(&batch));
                        ctx.wait(Duration::from_secs(1)).await?;
                        Ok::<_, BoxError>(())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);

        let spawned = lock(&spawned).clone();
        let awaited = lock(&awaited).clone();
        let [spawned] = spawned.as_slice() else {
            unreachable!("A awaits the batch once, after the wait: {spawned:?}");
        };
        assert_eq!(awaited.len(), 2, "B awaits the batch in both invocations");
        for record in &awaited {
            assert_eq!(record, spawned);
        }
    }

    #[tokio::test]
    async fn parallel_tolerated_failure_percentage_fails_the_batch() {
        let seen: Arc<Mutex<Option<Record>>> = Arc::default();
        let slot = Arc::clone(&seen);
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let slot = Arc::clone(&slot);
                    async move {
                        let branches = vec![
                            Branch::new("ok", |_| async { Ok(1) }),
                            Branch::new("bad-1", |_| async { Err::<i32, BoxError>("boom".into()) }),
                            Branch::new("bad-2", |_| async { Err::<i32, BoxError>("boom".into()) }),
                        ];
                        let batch = ctx
                            .parallel(branches)
                            .max_concurrency(1)
                            .completion(CompletionConfig::with_tolerated_failure_percentage(50.0))
                            .await_batch()
                            .await?;
                        *lock(&slot) = Some(record_of(&batch));
                        Ok::<_, BoxError>(())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        let (items, reason) = lock(&seen).clone().expect("the handler recorded the batch");
        let statuses: Vec<_> = items
            .iter()
            .map(|(index, _, status, value)| (*index, *status, *value))
            .collect();
        assert_eq!(
            statuses,
            [
                (0, BatchItemStatus::Succeeded, Some(1)),
                (1, BatchItemStatus::Failed, None),
                (2, BatchItemStatus::Failed, None),
            ]
        );
        assert_eq!(reason, CompletionReason::FailureToleranceExceeded);
    }

    /// Records a step at the first position in invocation 1, then runs a
    /// map named `m` there on replay, through `await_batch()` or `.await`.
    async fn replay_identity_case(
        use_await_batch: bool,
    ) -> (Option<String>, Option<String>, usize) {
        let invocations = Arc::new(AtomicUsize::new(0));
        let bodies = Arc::new(AtomicUsize::new(0));
        let ran = Arc::clone(&bodies);
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let first = invocations.fetch_add(1, Ordering::SeqCst) == 0;
                    let ran = Arc::clone(&ran);
                    async move {
                        if first {
                            ctx.step(|_| async { Ok(1_i32) }).name("s").await?;
                            ctx.wait(Duration::from_secs(1)).await?;
                            return Ok::<_, BoxError>(());
                        }
                        let map = ctx
                            .map(vec![1], move |_child, i: i32, _idx| {
                                let ran = Arc::clone(&ran);
                                async move {
                                    ran.fetch_add(1, Ordering::SeqCst);
                                    Ok(i)
                                }
                            })
                            .name("m");
                        if use_await_batch {
                            map.await_batch().await?;
                        } else {
                            map.await?;
                        }
                        Ok(())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Failed);
        (
            result.error_type().map(str::to_owned),
            result.error_message().map(str::to_owned),
            bodies.load(Ordering::SeqCst),
        )
    }

    #[tokio::test]
    async fn await_batch_checks_replay_identity_like_await() {
        let batch = replay_identity_case(true).await;
        let plain = replay_identity_case(false).await;
        assert_eq!(batch.0.as_deref(), Some("NonDeterministicExecutionError"));
        assert_eq!(plain.0.as_deref(), Some("NonDeterministicExecutionError"));
        assert!(batch.1.is_some());
        assert_eq!(batch.1, plain.1);
        assert_eq!((batch.2, plain.2), (0, 0), "no item body ran");
    }

    #[tokio::test]
    async fn parallel_spawn_batch_returns_branch_statuses() {
        let seen: Arc<Mutex<Option<Record>>> = Arc::default();
        let slot = Arc::clone(&seen);
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let slot = Arc::clone(&slot);
                    async move {
                        let branches = vec![
                            Branch::new("ok", |_| async { Ok(1) }),
                            Branch::new("bad", |_| async { Err::<i32, BoxError>("boom".into()) }),
                        ];
                        let batch = ctx
                            .parallel(branches)
                            .completion(CompletionConfig::with_tolerated_failure_count(1))
                            .spawn_batch();
                        ctx.wait(Duration::from_secs(1)).await?;
                        let batch = batch.await?;
                        *lock(&slot) = Some(record_of(&batch));
                        Ok::<_, BoxError>(())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Succeeded);
        let (items, reason) = lock(&seen).clone().expect("the handler recorded the batch");
        let statuses: Vec<_> = items
            .iter()
            .map(|(index, _, status, value)| (*index, *status, *value))
            .collect();
        assert_eq!(
            statuses,
            [
                (0, BatchItemStatus::Succeeded, Some(1)),
                (1, BatchItemStatus::Failed, None),
            ]
        );
        assert_eq!(reason, CompletionReason::AllCompleted);
    }

    #[derive(Clone, Copy)]
    enum Terminal {
        Await,
        AwaitBatch,
        SpawnBatch,
    }

    async fn max_concurrency_zero_case(terminal: Terminal) {
        let after = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&after);
        let result = LocalRunner::new()
            .run(
                move |(): (), ctx: DurableContext| {
                    let counter = Arc::clone(&counter);
                    async move {
                        let map = ctx
                            .map(vec![1, 2], |_child, i: i32, _idx| async move { Ok(i) })
                            .name("m")
                            .max_concurrency(0);
                        match terminal {
                            Terminal::Await => drop(map.await),
                            Terminal::AwaitBatch => drop(map.await_batch().await),
                            Terminal::SpawnBatch => drop(map.spawn_batch().await),
                        }
                        counter.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, BoxError>(())
                    }
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Failed);
        assert_eq!(result.error_type(), Some("Error"));
        assert_eq!(result.error_message(), Some(MAX_CONCURRENCY_ZERO_MESSAGE));
        assert_eq!(
            MAX_CONCURRENCY_ZERO_MESSAGE,
            "Invalid max_concurrency: 0. Must be a positive number or unset for unlimited \
             concurrency."
        );
        assert!(
            result.operation("m").is_none(),
            "no batch record is written"
        );
        assert_eq!(after.load(Ordering::SeqCst), 0, "the await never resolves");
    }

    #[tokio::test]
    async fn max_concurrency_zero_fails_the_execution() {
        max_concurrency_zero_case(Terminal::Await).await;
        max_concurrency_zero_case(Terminal::AwaitBatch).await;
        max_concurrency_zero_case(Terminal::SpawnBatch).await;
    }

    /// The same check on `parallel`, which has its own builder.
    #[tokio::test]
    async fn parallel_max_concurrency_zero_fails_the_execution() {
        let result = LocalRunner::new()
            .run(
                |(): (), ctx: DurableContext| async move {
                    let branches = vec![Branch::new("ok", |_| async { Ok(1) })];
                    drop(ctx.parallel(branches).name("p").max_concurrency(0).await);
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("the run finishes");
        assert_eq!(result.status(), ExecutionStatus::Failed);
        assert_eq!(result.error_message(), Some(MAX_CONCURRENCY_ZERO_MESSAGE));
        assert!(result.operation("p").is_none());
    }
}
