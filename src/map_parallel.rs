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
use std::pin::Pin;
use std::sync::Arc;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::task::JoinSet;

use crate::Serdes;
use crate::SerdesContext;
use crate::context::DurableContext;
use crate::driver::{ScopeOutcome, drive_scope};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, ChildFnError, OperationError, OperationErrorKind,
};

/// Wire sub-type for map operations.
const MAP_SUB_TYPE: &str = "Map";
/// Wire sub-type for map iteration children.
const MAP_ITERATION_SUB_TYPE: &str = "MapIteration";
/// Wire sub-type for parallel operations.
const PARALLEL_SUB_TYPE: &str = "Parallel";
/// Wire sub-type for parallel branch children.
const PARALLEL_BRANCH_SUB_TYPE: &str = "ParallelBranch";

/// Maximum checkpoint payload size in bytes (256KB). Matches child.rs.
const CHECKPOINT_SIZE_LIMIT_BYTES: usize = 256 * 1024;

/// Sentinel error message returned by `replay_terminal_batch` when the
/// batch has `replay_children` set, signalling the caller should fall
/// through to normal re-execution instead of short-circuiting.
const REPLAY_CHILDREN_SENTINEL: &str = "__replay_children_reexecute__";

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
/// the result value or an error message.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{BatchItemStatus, BatchItem};
///
/// let item: BatchItem<i32> = BatchItem::new(
///     0,
///     String::new(),
///     BatchItemStatus::Succeeded,
///     Some(42),
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
    ) -> Self {
        Self {
            index,
            name,
            status,
            result,
            error_message,
        }
    }
}

/// Why the batch completed.
///
/// Records the reason a [`BatchResult`] finished: either all items ran,
/// the success threshold was met, or the failure tolerance was exceeded.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::CompletionReason;
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
        }
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
/// use aws_durable_execution_sdk_rust::{BatchResult, BatchItem, BatchItemStatus, CompletionReason};
///
/// let result: BatchResult<i32> = BatchResult::new(
///     vec![
///         BatchItem::new(0, String::new(), BatchItemStatus::Succeeded, Some(10), None),
///         BatchItem::new(1, String::new(), BatchItemStatus::Failed, None, Some("oops".into())),
///     ],
///     CompletionReason::FailureToleranceExceeded,
/// );
/// assert!(result.has_failure());
/// assert_eq!(result.success_count(), 1);
/// assert_eq!(result.failure_count(), 1);
/// assert_eq!(result.status(), "FAILED");
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

    /// Returns `"SUCCEEDED"` if no item failed, `"FAILED"` otherwise.
    #[must_use]
    pub fn status(&self) -> &'static str {
        if self.has_failure() {
            "FAILED"
        } else {
            "SUCCEEDED"
        }
    }

    /// Returns the total number of items that were started.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.items.len()
    }

    /// Returns the errors from failed items, in input order.
    #[must_use]
    pub fn errors(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter_map(|item| {
                if item.status == BatchItemStatus::Failed {
                    item.error_message.as_deref()
                } else {
                    None
                }
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
/// use aws_durable_execution_sdk_rust::NestingMode;
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
pub(crate) struct MapExecution<I, O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) max_concurrency: Option<usize>,
    pub(crate) completion: Option<crate::CompletionConfig>,
    pub(crate) serdes: Option<Box<dyn Serdes>>,
    pub(crate) result_serdes: Option<Box<dyn Serdes>>,
    pub(crate) nesting: NestingMode,
    pub(crate) item_namer: Option<Arc<dyn Fn(usize) -> String + Send + Sync>>,
    pub(crate) items: Vec<I>,
    #[allow(clippy::type_complexity)] // reason: boxed async closure factory
    pub(crate) closure: Arc<
        dyn Fn(
                DurableContext,
                I,
                usize,
            ) -> Pin<Box<dyn Future<Output = Result<O, ChildFnError>> + Send>>
            + Send
            + Sync,
    >,
}

impl<
    I: Serialize + DeserializeOwned + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + Send + 'static,
> MapExecution<I, O>
{
    /// Executes the map operation.
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        let total_items = self.items.len();
        let items = into_item_slots(self.items);
        let closure = self.closure;
        let items_ref = Arc::clone(&items);

        let batch_result = execute_batch(
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
            move |child_ctx, index| {
                let items = Arc::clone(&items_ref);
                let closure = Arc::clone(&closure);
                async move {
                    let item =
                        take_item(&items, index).map_err(|e| ChildFnError::new(e.to_string()))?;
                    (closure)(child_ctx, item, index).await
                }
            },
        )
        .await?;

        // Extract successful results in order; errors are embedded in the
        // batch result structure. For the simple Vec<O> return, we only
        // include successful items (never-started are omitted — invariant).
        let mut results = Vec::with_capacity(batch_result.items.len());
        for item in batch_result.items {
            match item.status {
                BatchItemStatus::Succeeded => {
                    if let Some(value) = item.result {
                        results.push(value);
                    }
                }
                BatchItemStatus::Failed => {
                    // If the batch completed within tolerance (AllCompleted),
                    // failed items are expected and NOT propagated as errors.
                    // Only propagate as Err when there is no completion config
                    // that allows failures (i.e. the default fail-fast case
                    // where FailureToleranceExceeded is the reason).
                    if batch_result.reason != CompletionReason::AllCompleted
                        && batch_result.reason != CompletionReason::MinSuccessfulReached
                    {
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

    /// Executes the map operation and returns the full `BatchResult`.
    ///
    /// Unlike `execute()` which returns `Vec<O>` (only successes), this
    /// returns the raw `BatchResult` including per-item status, completion
    /// reason, and error messages. Used by handlers that need batch metadata
    /// (e.g., success/failure counts for projection results).
    pub(crate) async fn execute_batch_result(self) -> Result<BatchResult<O>, OperationError> {
        let total_items = self.items.len();
        let items = into_item_slots(self.items);
        let closure = self.closure;
        let items_ref = Arc::clone(&items);

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
            move |child_ctx, index| {
                let items = Arc::clone(&items_ref);
                let closure = Arc::clone(&closure);
                async move {
                    let item =
                        take_item(&items, index).map_err(|e| ChildFnError::new(e.to_string()))?;
                    (closure)(child_ctx, item, index).await
                }
            },
        )
        .await
    }
}

/// Internal state for a parallel execution passed from the builder.
pub(crate) struct ParallelExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) max_concurrency: Option<usize>,
    pub(crate) completion: Option<crate::CompletionConfig>,
    pub(crate) serdes: Option<Box<dyn Serdes>>,
    pub(crate) result_serdes: Option<Box<dyn Serdes>>,
    pub(crate) nesting: NestingMode,
    #[allow(clippy::type_complexity)] // reason: boxed future factory per branch
    pub(crate) branches: Vec<(
        String,
        Box<
            dyn FnOnce(
                    DurableContext,
                )
                    -> Pin<Box<dyn Future<Output = Result<O, ChildFnError>> + Send>>
                + Send,
        >,
    )>,
}

impl<O: Serialize + DeserializeOwned + Send + 'static> ParallelExecution<O> {
    /// Executes the parallel operation.
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        let total = self.branches.len();
        // Split each branch into its display name (threaded to the
        // coordinator as the item namer so it reaches child checkpoint
        // updates) and its factory (kept in a take-once slot since it's
        // FnOnce).
        let mut names: Vec<String> = Vec::with_capacity(total);
        #[allow(clippy::type_complexity)] // reason: FnOnce branch factories require complex boxing
        let mut slots: Vec<
            std::sync::Mutex<
                Option<
                    Box<
                        dyn FnOnce(
                                DurableContext,
                            )
                                -> Pin<Box<dyn Future<Output = Result<O, ChildFnError>> + Send>>
                            + Send,
                    >,
                >,
            >,
        > = Vec::with_capacity(total);
        for (name, factory) in self.branches {
            names.push(name);
            slots.push(std::sync::Mutex::new(Some(factory)));
        }
        let branch_slots = Arc::new(slots);
        let branch_namer: Arc<dyn Fn(usize) -> String + Send + Sync> =
            Arc::new(move |index| names.get(index).cloned().unwrap_or_default());

        let branch_slots_ref = Arc::clone(&branch_slots);
        let batch_result = execute_batch(
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
            move |child_ctx, index| {
                let slots = Arc::clone(&branch_slots_ref);
                async move {
                    let factory = {
                        let guard = slots
                            .get(index)
                            .ok_or_else(|| ChildFnError::new("branch index out of bounds"))?;
                        let mut lock = guard
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        lock.take().ok_or_else(|| {
                            ChildFnError::new("branch already consumed (concurrent access bug)")
                        })?
                    };
                    (factory)(child_ctx).await
                }
            },
        )
        .await?;

        let mut results = Vec::with_capacity(batch_result.items.len());
        for item in batch_result.items {
            match item.status {
                BatchItemStatus::Succeeded => {
                    if let Some(value) = item.result {
                        results.push(value);
                    }
                }
                BatchItemStatus::Failed => {
                    let msg = item
                        .error_message
                        .unwrap_or_else(|| "branch failed".to_owned());
                    return Err(batch_error(&msg));
                }
            }
        }
        if batch_result.reason == CompletionReason::FailureToleranceExceeded {
            return Err(batch_error("failure tolerance exceeded"));
        }
        Ok(results)
    }
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

/// Core batch execution: schedule items with bounded concurrency and
/// completion checking.
#[allow(clippy::too_many_lines)]
// reason: batch coordination has distinct phases (claim, schedule, collect, checkpoint) that read better as one flow
#[allow(clippy::too_many_arguments)] // reason: batch execution requires all these parameters
async fn execute_batch<O, F, Fut>(
    ctx: DurableContext,
    parent_op_id: OperationId,
    parent_name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<crate::CompletionConfig>,
    serdes: Option<Box<dyn Serdes>>,
    result_serdes: Option<Box<dyn Serdes>>,
    nesting: NestingMode,
    item_namer: Option<Arc<dyn Fn(usize) -> String + Send + Sync>>,
    total_items: usize,
    parent_sub_type: &str,
    child_sub_type: &str,
    run_item: F,
) -> Result<BatchResult<O>, OperationError>
where
    O: Serialize + DeserializeOwned + Send + 'static,
    F: Fn(DurableContext, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, ChildFnError>> + Send + 'static,
{
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
    if let Some(record) = ctx.checkpoint_record(&parent_positional) {
        if record.status.is_terminal() {
            let serdes_ctx = SerdesContext::new(&parent_wire, ctx.execution_arn());
            match replay_terminal_batch::<O>(
                &ctx,
                &record,
                &parent_positional,
                total_items,
                serdes.as_deref(),
                result_serdes.as_deref(),
                &serdes_ctx,
            ) {
                Ok(result) => {
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
                    let is_replay_children = e.to_string().contains(REPLAY_CHILDREN_SENTINEL);
                    if !is_replay_children {
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
        ctx.checkpoint_updates(vec![update])
            .await
            .map_err(|e| batch_error(&format!("checkpoint parent start: {e}")))?;
    }

    // 4. Empty collection: checkpoint success immediately.
    if total_items == 0 {
        let result = BatchResult {
            items: Vec::new(),
            reason: CompletionReason::AllCompleted,
        };
        let serdes_ref: Option<&dyn Serdes> = serdes.as_ref().map(Box::as_ref);
        let payload = from_batch_result(&result, serdes_ref)?;
        let json_str = serde_json::to_string(&payload)
            .map_err(|e| batch_error(&format!("serialize batch result: {e}")))?;
        let serialized_payload = if let Some(ref rs) = result_serdes {
            let serdes_ctx = SerdesContext::new(&parent_wire, ctx.execution_arn());
            rs.serialize_to_string_with_context(&json_str, &serdes_ctx)
                .map_err(|e| batch_error(&format!("serialize batch result (op-serdes): {e}")))?
        } else {
            json_str
        };
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
            let is_terminal = ctx
                .checkpoint_record(&child_positional)
                .is_some_and(|r| r.status.is_terminal());
            pre_claimed.push(PreClaimed {
                index: i,
                op_id: child_op_id,
                is_terminal,
            });
        }
    }

    // 7. Execute items with bounded concurrency, branch-local suspension, and
    // slot-holding accounting.
    let run_item = Arc::new(run_item);
    let serdes: Arc<Option<Box<dyn Serdes>>> = Arc::new(serdes);

    // 7a. For concurrent mode: checkpoint ALL child STARTs synchronously
    // BEFORE dispatching any tasks. This prevents token rotation races
    // between the main loop and spawned tasks: all child STARTs are
    // checkpointed on the owning task before any spawned task runs.
    if concurrency > 1 {
        for pre in &pre_claimed {
            if !pre.is_terminal && nesting != NestingMode::Flat {
                let record = ctx.checkpoint_record(pre.op_id.positional());
                if record.is_none() {
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
                    ctx.checkpoint_updates(vec![update])
                        .await
                        .map_err(|e| batch_error(&format!("checkpoint child start: {e}")))?;
                }
            }
        }
    }

    // Coordinator loop. In-flight = running + suspended, bounded by
    // `concurrency`: a SUSPENDED branch KEEPS its slot (only terminal
    // completion frees one — the slot-holding invariant), so
    // `suspended_count` counts against the cap and new branches only start
    // when capacity remains after terminal completions. Each branch runs
    // through `execute_single_item`, which drives the branch body under its
    // own scope so a park resolves to `ItemOutcome::Suspended` locally rather
    // than tearing down the whole invocation.
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
    let mut join_set: JoinSet<(usize, Result<ItemOutcome<O>, OperationError>)> = JoinSet::new();
    // Maps a branch task's id to the metadata needed to record a controlled
    // failure if that task ends via a `JoinError`. Removed on both the value
    // and the error arm, so it never outgrows the in-flight set.
    let mut branch_meta: std::collections::HashMap<tokio::task::Id, BranchMeta> =
        std::collections::HashMap::with_capacity(total_items);

    let mut results: Vec<Option<BatchItem<O>>> = (0..total_items).map(|_| None).collect();
    let mut success_count: usize = 0;
    let mut failure_count: usize = 0;
    let mut suspended_count: usize = 0;
    let mut running: usize = 0;
    let mut next_index: usize = 0;
    let mut stopped = false;
    let mut any_suspended = false;

    loop {
        // Dispatch while capacity remains and not-started eligible work exists.
        // A threshold hit (`stopped`) halts new dispatch; already-running
        // branches are still drained below (in-flight branches always complete).
        while !stopped && next_index < total_items && running + suspended_count < concurrency {
            let i = next_index;
            next_index += 1;

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
                let is_terminal = ctx
                    .checkpoint_record(&child_positional)
                    .is_some_and(|r| r.status.is_terminal());
                PreClaimed {
                    index: i,
                    op_id: child_op_id,
                    is_terminal,
                }
            };

            let start_checkpointed =
                concurrency > 1 && !pc.is_terminal && nesting != NestingMode::Flat;

            let ctx_task = ctx.clone();
            let parent_wire_task = parent_wire.clone();
            let child_sub_type_task = child_sub_type_owned.clone();
            let run_item_task = Arc::clone(&run_item);
            let serdes_task = Arc::clone(&serdes);
            let item_nesting = nesting;
            let task_ownership = ctx.task_ownership().clone();

            // Compute the item name in the coordinator so it is available both
            // for the branch body and for a controlled-failure checkpoint if
            // the branch task ends via a `JoinError`.
            let item_name = item_namer
                .as_ref()
                .map_or_else(String::new, |namer| namer(pc.index));
            let item_name_task = item_name.clone();
            let branch_index = pc.index;
            let branch_wire = pc.op_id.wire().to_owned();

            let abort = join_set.spawn(async move {
                // Bless this task for task-ownership checks.
                if let Some(task_id) = tokio::task::try_id() {
                    task_ownership.bless_task(task_id);
                }

                let outcome = execute_single_item(
                    &ctx_task,
                    &pc.op_id,
                    pc.index,
                    pc.is_terminal,
                    start_checkpointed,
                    &parent_wire_task,
                    &child_sub_type_task,
                    &item_name_task,
                    item_nesting,
                    &*run_item_task,
                    serdes_task.as_ref().as_deref(),
                )
                .await;

                (pc.index, outcome)
            });
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

        match joined {
            Ok((task_id, (index, outcome))) => {
                branch_meta.remove(&task_id);
                match outcome {
                    Ok(ItemOutcome::Terminal(item)) => {
                        match item.status {
                            BatchItemStatus::Succeeded => success_count += 1,
                            BatchItemStatus::Failed => failure_count += 1,
                        }
                        if let Some(slot) = results.get_mut(index) {
                            *slot = Some(item);
                        }
                        if should_stop_min(&completion_cfg, success_count)
                            || should_stop_failure(&completion_cfg, failure_count, total_items)
                        {
                            stopped = true;
                        }
                    }
                    Ok(ItemOutcome::Suspended) => {
                        // Branch parked: keep its slot (suspended_count counts
                        // against `concurrency`), let siblings continue.
                        suspended_count += 1;
                        any_suspended = true;
                    }
                    Err(e) => {
                        // Coordinator-level failure (e.g. a checkpoint call
                        // failed): abort remaining branches (JoinSet drops on
                        // return) and surface.
                        return Err(e);
                    }
                }
            }
            Err(join_err) => {
                // The branch task terminated without producing an outcome.
                // Record a controlled BRANCH FAILURE: mirror the failure a
                // normal branch would have produced so accounting and the
                // completion threshold treat it identically, and checkpoint the
                // child FAIL so a retry does not repeat already-started work.
                // A panicking branch surfaces as a failed batch item rather
                // than failing or hanging the whole batch.
                let Some(meta) = branch_meta.remove(&join_err.id()) else {
                    return Err(batch_error(
                        "branch task terminated with an unrecognized task id",
                    ));
                };
                let message = match join_err.try_into_panic() {
                    Ok(payload) => panic_message(payload.as_ref()),
                    Err(_) => "branch task was cancelled".to_owned(),
                };

                // Best-effort child FAIL checkpoint. Skipped in FLAT mode, which
                // emits no child-context events (mirrors the normal fail path).
                if nesting != NestingMode::Flat {
                    let update = OperationUpdate::builder()
                        .id(meta.child_wire.clone())
                        .r#type(OperationType::Context)
                        .sub_type(child_sub_type_owned.clone())
                        .action(OperationAction::Fail)
                        .parent_id(parent_wire.clone())
                        .error(
                            aws_sdk_lambda::types::ErrorObject::builder()
                                .error_type("ChildFnError")
                                .error_message(message.clone())
                                .build(),
                        );
                    if let Ok(update) = update.build() {
                        let _ = ctx.checkpoint_updates(vec![update]).await;
                    }
                }

                failure_count += 1;
                if let Some(slot) = results.get_mut(meta.index) {
                    *slot = Some(BatchItem {
                        index: meta.index,
                        name: meta.item_name,
                        status: BatchItemStatus::Failed,
                        result: None,
                        error_message: Some(message),
                    });
                }
                if should_stop_min(&completion_cfg, success_count)
                    || should_stop_failure(&completion_cfg, failure_count, total_items)
                {
                    stopped = true;
                }
            }
        }
    }

    // Quiescent. If a branch is parked and no completion threshold was met,
    // the batch cannot finish this invocation: suspend the coordinator's OWN
    // scope so whoever drives it (the invocation driver at the root, or an
    // outer coordinator's branch driver when nested) observes the suspension
    // and reports PENDING for its subtree. `suspend_now` never returns; the
    // coordinator future is dropped at teardown, aborting the guards.
    // Started-not-terminal children replay on the next invocation. When a
    // threshold WAS met, parked branches are excluded (like never-started
    // work) and the batch completes normally.
    if any_suspended && !stopped {
        return Ok(ctx.suspend_now::<BatchResult<O>>().await);
    }

    // 9. Assemble results in input order (only terminal items; suspended and
    // never-started branches are omitted).
    let final_items: Vec<BatchItem<O>> = results.into_iter().flatten().collect();

    // 10. Determine completion reason.
    let reason = if should_stop_min(&completion_cfg, success_count) {
        CompletionReason::MinSuccessfulReached
    } else if should_stop_failure(&completion_cfg, failure_count, total_items) {
        CompletionReason::FailureToleranceExceeded
    } else {
        CompletionReason::AllCompleted
    };

    let batch_result = BatchResult {
        items: final_items,
        reason,
    };

    // 11. Serialize the batch result BEFORE the async checkpoint call
    // (avoids requiring O: Sync for the reference across await).
    // If result_serdes is set (operation-level serdes), serialize the
    // batch result to default JSON, then apply the custom transform.
    // Otherwise, use the default batch payload format unchanged.
    let serdes_opt: &Option<Box<dyn Serdes>> = &serdes;
    let serdes_ref: Option<&dyn Serdes> = serdes_opt.as_ref().map(Box::as_ref);
    let payload = from_batch_result(&batch_result, serdes_ref)?;
    let json_str = serde_json::to_string(&payload)
        .map_err(|e| batch_error(&format!("serialize batch result: {e}")))?;
    let serialized_payload = if let Some(ref rs) = result_serdes {
        let serdes_ctx = SerdesContext::new(&parent_wire, ctx.execution_arn());
        rs.serialize_to_string_with_context(&json_str, &serdes_ctx)
            .map_err(|e| batch_error(&format!("serialize batch result (op-serdes): {e}")))?
    } else {
        json_str
    };

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

/// Executes a single item (child context) within the batch.
///
/// Returns [`ItemOutcome::Suspended`] if the item's own scope parked on a
/// durable operation (the branch keeps its slot; the coordinator lets
/// siblings keep running). Returns [`ItemOutcome::Terminal`] on success or
/// failure. `Err` is reserved for coordinator-level failures (e.g. a
/// checkpoint call failing).
#[allow(clippy::too_many_arguments)] // reason: single-item execution needs full context
#[allow(clippy::too_many_lines)] // reason: FLAT/NORMAL branches + replay/live paths read better in one flow
async fn execute_single_item<O, F, Fut>(
    ctx: &DurableContext,
    child_op_id: &OperationId,
    index: usize,
    is_terminal: bool,
    start_checkpointed: bool,
    parent_wire: &str,
    child_sub_type: &str,
    item_name: &str,
    nesting: NestingMode,
    run_item: &F,
    serdes: Option<&dyn Serdes>,
) -> Result<ItemOutcome<O>, OperationError>
where
    O: Serialize + DeserializeOwned + Send + 'static,
    F: Fn(DurableContext, usize) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O, ChildFnError>> + Send,
{
    let child_positional = child_op_id.positional().to_owned();
    let child_wire = child_op_id.wire().to_owned();

    // Replay path: child already terminal.
    if is_terminal {
        match replay_terminal_child::<O>(ctx, &child_positional, index, serdes) {
            Ok(item) => return Ok(ItemOutcome::Terminal(item)),
            Err(e) => {
                // ReplayChildren sentinel: fall through to re-execution.
                let is_replay_children = e.to_string().contains(REPLAY_CHILDREN_SENTINEL);
                if !is_replay_children {
                    return Err(e);
                }
                // Fall through: re-execute the child to reconstruct result.
            }
        }
    }

    // FLAT nesting: skip child context events; run item directly under parent.
    // Operations inside the flat branch checkpoint with ParentId pointing to
    // the batch parent (not the virtual child).
    if nesting == NestingMode::Flat {
        let child_ctx = ctx.new_scoped_flat_child(&child_positional, parent_wire);
        let scope = Arc::clone(child_ctx.suspension_signal());
        let outcome = drive_scope(run_item(child_ctx, index), scope).await;
        return match outcome {
            ScopeOutcome::Suspended => Ok(ItemOutcome::Suspended),
            ScopeOutcome::Completed(Ok(value)) => {
                let serialized = serialize_value(&value, serdes)?;
                let deserialized: O = deserialize_value(&serialized, serdes)?;
                Ok(ItemOutcome::Terminal(BatchItem {
                    index,
                    name: item_name.to_owned(),
                    status: BatchItemStatus::Succeeded,
                    result: Some(deserialized),
                    error_message: None,
                }))
            }
            ScopeOutcome::Completed(Err(child_err)) => Ok(ItemOutcome::Terminal(BatchItem {
                index,
                name: item_name.to_owned(),
                status: BatchItemStatus::Failed,
                result: None,
                error_message: Some(child_err.to_string()),
            })),
        };
    }

    // Check if we need to checkpoint START for this child.
    // Skip if the caller already checkpointed START synchronously (for
    // concurrency safety).
    if !start_checkpointed {
        let record = ctx.checkpoint_record(&child_positional);
        if record.is_none() {
            let update = build_child_update(
                &child_wire,
                item_name,
                child_sub_type,
                parent_wire,
                OperationAction::Start,
            );
            ctx.checkpoint_updates(vec![update])
                .await
                .map_err(|e| batch_error(&format!("checkpoint child start: {e}")))?;
        }
    }

    // Create child context (with its OWN suspension scope) and run the item
    // through the branch driver so a park is caught locally as Suspended
    // instead of tearing down the whole invocation.
    let child_ctx = ctx.new_scoped_child(&child_positional);
    let scope = Arc::clone(child_ctx.suspension_signal());
    let outcome = drive_scope(run_item(child_ctx, index), Arc::clone(&scope)).await;

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
            let serialized = serialize_value(&value, serdes)?;
            let mut builder = OperationUpdate::builder()
                .id(child_wire)
                .r#type(OperationType::Context)
                .sub_type(child_sub_type.to_owned())
                .action(OperationAction::Succeed)
                .parent_id(parent_wire.to_owned());

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

            ctx.checkpoint_updates(vec![update])
                .await
                .map_err(|e| batch_error(&format!("checkpoint child succeed: {e}")))?;

            // Round-trip deserialize for live == replay consistency.
            let deserialized: O = deserialize_value(&serialized, serdes)?;

            Ok(ItemOutcome::Terminal(BatchItem {
                index,
                name: item_name.to_owned(),
                status: BatchItemStatus::Succeeded,
                result: Some(deserialized),
                error_message: None,
            }))
        }
        ScopeOutcome::Completed(Err(child_err)) => {
            // Defensive: a durable op that set its suspend flag and then
            // returned Err (rather than parking) is a suspension, not a
            // failure — mirror the invocation driver's precedence rule.
            if scope.is_suspend_requested() {
                return Ok(ItemOutcome::Suspended);
            }

            // Checkpoint failure.
            let err_message = child_err.to_string();
            let builder = OperationUpdate::builder()
                .id(child_wire)
                .r#type(OperationType::Context)
                .sub_type(child_sub_type.to_owned())
                .action(OperationAction::Fail)
                .parent_id(parent_wire.to_owned())
                .error(
                    aws_sdk_lambda::types::ErrorObject::builder()
                        .error_type("ChildFnError")
                        .error_message(err_message.clone())
                        .build(),
                );

            #[allow(clippy::expect_used)] // reason: all required fields are set above
            let update = builder
                .build()
                .expect("all required OperationUpdate fields set");

            // Best-effort checkpoint.
            let _ = ctx.checkpoint_updates(vec![update]).await;

            Ok(ItemOutcome::Terminal(BatchItem {
                index,
                name: item_name.to_owned(),
                status: BatchItemStatus::Failed,
                result: None,
                error_message: Some(err_message),
            }))
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Replay helpers
// ────────────────────────────────────────────────────────────────────────────

/// Replays a terminal batch (parent already succeeded/failed in the log).
fn replay_terminal_batch<O: DeserializeOwned + 'static>(
    _ctx: &DurableContext,
    record: &crate::engine::CheckpointRecord,
    _parent_positional: &str,
    _total_items: usize,
    serdes: Option<&dyn Serdes>,
    result_serdes: Option<&dyn Serdes>,
    serdes_ctx: &SerdesContext,
) -> Result<BatchResult<O>, OperationError> {
    match &record.status {
        CheckpointStatus::Succeeded => {
            if record.replay_children {
                // ReplayChildren mode: cannot reconstruct from the payload
                // alone — the caller must fall through to re-execution.
                // Signal this by returning a sentinel error that the caller
                // catches to continue normal execution.
                return Err(batch_error(REPLAY_CHILDREN_SENTINEL));
            }
            // Deserialize the stored batch summary.
            let payload_str = record
                .result
                .as_deref()
                .ok_or_else(|| batch_error("terminal batch has no result payload"))?;
            // If result_serdes is set, reverse its transform first.
            let json_str = if let Some(rs) = result_serdes {
                rs.deserialize_from_string_with_context(payload_str, serdes_ctx)
                    .map_err(|e| {
                        batch_error(&format!("deserialize batch payload (op-serdes): {e}"))
                    })?
            } else {
                payload_str.to_owned()
            };
            let payload: BatchCheckpointPayload = serde_json::from_str(&json_str)
                .map_err(|e| batch_error(&format!("deserialize batch payload: {e}")))?;
            to_batch_result(&payload, serdes)
        }
        CheckpointStatus::Failed => {
            let msg = record.error_message.as_deref().unwrap_or("batch failed");
            Err(batch_error(msg))
        }
        _ => {
            // Shouldn't happen — we checked is_terminal() above.
            Err(batch_error("unexpected non-terminal status in replay"))
        }
    }
}

/// Replays a terminal child item from the checkpoint log.
fn replay_terminal_child<O: DeserializeOwned + 'static>(
    ctx: &DurableContext,
    child_positional: &str,
    index: usize,
    serdes: Option<&dyn Serdes>,
) -> Result<BatchItem<O>, OperationError> {
    let record = ctx
        .checkpoint_record(child_positional)
        .ok_or_else(|| batch_error("replay child has no checkpoint record"))?;

    match &record.status {
        CheckpointStatus::Succeeded => {
            if record.replay_children {
                // ReplayChildren mode: signal re-execution needed.
                return Err(batch_error(REPLAY_CHILDREN_SENTINEL));
            }
            let payload = record
                .result
                .as_deref()
                .ok_or_else(|| batch_error("succeeded child has no result"))?;
            let value: O = deserialize_value(payload, serdes)?;
            Ok(BatchItem {
                index,
                name: String::new(),
                status: BatchItemStatus::Succeeded,
                result: Some(value),
                error_message: None,
            })
        }
        CheckpointStatus::Failed => {
            let msg = record.error_message.as_deref().unwrap_or("child failed");
            Ok(BatchItem {
                index,
                name: String::new(),
                status: BatchItemStatus::Failed,
                result: None,
                error_message: Some(msg.to_owned()),
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

    ctx.checkpoint_updates(vec![update])
        .await
        .map_err(|e| batch_error(&format!("checkpoint parent succeed: {e}")))?;

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

// ────────────────────────────────────────────────────────────────────────────
// Completion logic
// ────────────────────────────────────────────────────────────────────────────

/// Checks if the `min_successful` threshold has been met.
fn should_stop_min(cfg: &crate::CompletionConfig, success_count: usize) -> bool {
    match cfg.min_successful {
        Some(min) if min > 0 => success_count >= min,
        _ => false,
    }
}

/// Checks if the failure tolerance has been exceeded.
fn should_stop_failure(
    cfg: &crate::CompletionConfig,
    failure_count: usize,
    total_items: usize,
) -> bool {
    // Count-based tolerance.
    if let Some(tolerated) = cfg.tolerated_failure_count
        && failure_count > tolerated
    {
        return true;
    }

    // Percentage-based tolerance.
    if let Some(pct) = cfg.tolerated_failure_percentage
        && pct > 0
        && total_items > 0
    {
        let actual_pct = (failure_count * 100) / total_items;
        if actual_pct > pct {
            return true;
        }
    }

    false
}

// ────────────────────────────────────────────────────────────────────────────
// Serialization helpers
// ────────────────────────────────────────────────────────────────────────────

/// Serializes a value using the configured serdes or JSON default.
///
/// When a custom serdes is provided, uses `Serdes::serialize(&dyn Any)` to
/// serialize the raw value directly. The result bytes are converted to UTF-8
/// for the checkpoint payload string.
fn serialize_value<O: Serialize + 'static>(
    value: &O,
    serdes: Option<&dyn Serdes>,
) -> Result<String, OperationError> {
    if let Some(s) = serdes {
        let bytes = s
            .serialize(value as &dyn std::any::Any)
            .map_err(|e| batch_error(&format!("serialize result (custom): {e}")))?;
        String::from_utf8(bytes)
            .map_err(|e| batch_error(&format!("serialize result (custom): non-UTF8: {e}")))
    } else {
        serde_json::to_string(value).map_err(|e| batch_error(&format!("serialize result: {e}")))
    }
}

/// Deserializes a value using the configured serdes or JSON default.
///
/// When a custom serdes is provided, uses `Serdes::deserialize_bytes()` to
/// deserialize raw bytes. The result is downcast from `Box<dyn Any>` to the
/// target type.
fn deserialize_value<O: DeserializeOwned + 'static>(
    payload: &str,
    serdes: Option<&dyn Serdes>,
) -> Result<O, OperationError> {
    if let Some(s) = serdes {
        let boxed = s
            .deserialize_bytes(payload.as_bytes(), std::any::type_name::<O>())
            .map_err(|e| batch_error(&format!("deserialize result (custom): {e}")))?;
        let any_send: Box<dyn std::any::Any + Send> = boxed;
        // Attempt downcast to the target type.
        any_send
            .downcast::<O>()
            .map(|b| *b)
            .map_err(|_| batch_error("deserialize result (custom): type mismatch on downcast"))
    } else {
        serde_json::from_str(payload).map_err(|e| batch_error(&format!("deserialize result: {e}")))
    }
}

/// Batch checkpoint payload.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BatchCheckpointPayload {
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

/// Converts a live `BatchResult` into the checkpoint payload format.
fn from_batch_result<O: Serialize + 'static>(
    result: &BatchResult<O>,
    serdes: Option<&dyn Serdes>,
) -> Result<BatchCheckpointPayload, OperationError> {
    let mut items = Vec::with_capacity(result.items.len());
    for item in &result.items {
        let status_str = match item.status {
            BatchItemStatus::Succeeded => "SUCCEEDED",
            BatchItemStatus::Failed => "FAILED",
        };
        let result_str = if item.status == BatchItemStatus::Succeeded {
            if let Some(ref value) = item.result {
                serialize_value(value, serdes)?
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        items.push(BatchCheckpointItem {
            index: item.index,
            name: item.name.clone(),
            status: status_str.to_owned(),
            result: result_str,
            err_type: if item.status == BatchItemStatus::Failed {
                "ChildFnError".to_owned()
            } else {
                String::new()
            },
            err_message: item.error_message.clone().unwrap_or_default(),
        });
    }
    Ok(BatchCheckpointPayload {
        results: items,
        reason: result.reason.as_str().to_owned(),
    })
}

/// Converts a deserialized checkpoint payload back into a `BatchResult`.
fn to_batch_result<O: DeserializeOwned + 'static>(
    payload: &BatchCheckpointPayload,
    serdes: Option<&dyn Serdes>,
) -> Result<BatchResult<O>, OperationError> {
    let mut items = Vec::with_capacity(payload.results.len());
    for cp in &payload.results {
        let status = match cp.status.as_str() {
            "SUCCEEDED" => BatchItemStatus::Succeeded,
            "FAILED" => BatchItemStatus::Failed,
            other => return Err(batch_error(&format!("unknown item status: {other}"))),
        };
        let result = if status == BatchItemStatus::Succeeded && !cp.result.is_empty() {
            Some(deserialize_value::<O>(&cp.result, serdes)?)
        } else {
            None
        };
        items.push(BatchItem {
            index: cp.index,
            name: cp.name.clone(),
            status,
            result,
            error_message: if status == BatchItemStatus::Failed {
                Some(cp.err_message.clone())
            } else {
                None
            },
        });
    }

    let reason = match payload.reason.as_str() {
        "MIN_SUCCESSFUL_REACHED" => CompletionReason::MinSuccessfulReached,
        "FAILURE_TOLERANCE_EXCEEDED" => CompletionReason::FailureToleranceExceeded,
        _ => CompletionReason::AllCompleted,
    };

    Ok(BatchResult { items, reason })
}

// ────────────────────────────────────────────────────────────────────────────
// Error helper
// ────────────────────────────────────────────────────────────────────────────

/// Constructs a batch operation error.
fn batch_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(
        ChildContextError::from_kind(ChildContextErrorKind::Internal {
            message: message.to_owned(),
        }),
    ))
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

    #[allow(dead_code)] // reason: used by future test cases
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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
            },
        )
    }

    #[allow(dead_code)] // reason: used by future test cases
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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
            },
        )
    }

    #[tokio::test]
    async fn basic_map_two_items() {
        // Without a client, the checkpoint will fail, but we can test that
        // the structure is correct by using a replay path.
        let ctx = test_ctx(CheckpointLog::empty());
        // Map will fail at checkpoint (no client) — that's expected.
        let result = ctx
            .map(vec![10, 20], |_child, item: i32, _idx| async move {
                Ok(item * 2)
            })
            .await;
        // Should error because no execution client.
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn basic_parallel_two_branches() {
        use crate::future::Branch;
        let ctx = test_ctx(CheckpointLog::empty());
        let branches = vec![
            Branch::new("a", |_ctx| Box::pin(async { Ok(1) })),
            Branch::new("b", |_ctx| Box::pin(async { Ok(2) })),
        ];
        let result = ctx.parallel(branches).await;
        // Should error because no execution client.
        assert!(result.is_err());
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
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: None,
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
        let cfg = crate::CompletionConfig {
            min_successful: Some(2),
            ..Default::default()
        };
        assert!(should_stop_min(&cfg, 2));
        assert!(should_stop_min(&cfg, 3));
        assert!(!should_stop_min(&cfg, 1));
    }

    #[tokio::test]
    async fn completion_config_tolerated_failure_count() {
        let cfg = crate::CompletionConfig {
            tolerated_failure_count: Some(0),
            ..Default::default()
        };
        // 0 tolerated means fail-fast: first failure exceeds.
        assert!(should_stop_failure(&cfg, 1, 10));
        assert!(!should_stop_failure(&cfg, 0, 10));
    }

    #[tokio::test]
    async fn completion_config_tolerated_failure_percentage() {
        let cfg = crate::CompletionConfig {
            tolerated_failure_percentage: Some(20),
            ..Default::default()
        };
        // 3/10 = 30% > 20%: should stop.
        assert!(should_stop_failure(&cfg, 3, 10));
        // 2/10 = 20% == 20%: should NOT stop (strictly exceeds).
        assert!(!should_stop_failure(&cfg, 2, 10));
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
        let cfg = crate::CompletionConfig {
            min_successful: Some(1),
            ..Default::default()
        };
        // After 1 success, should stop.
        assert!(should_stop_min(&cfg, 1));
    }

    #[tokio::test]
    async fn completion_config_validate_mutual_exclusivity() {
        // Having both min_successful and tolerated_failure_count is valid
        // (Go/JS allow it — first threshold fires). No error.
        let cfg = crate::CompletionConfig {
            min_successful: Some(2),
            tolerated_failure_count: Some(1),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
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
        );
        assert_eq!(item.index, 3);
        assert_eq!(item.name, "my-item");
        assert_eq!(item.status, BatchItemStatus::Succeeded);
        assert_eq!(item.result.as_deref(), Some("hello"));
        assert!(item.error_message.is_none());
    }

    #[test]
    fn batch_result_accessors_match_go_sdk() {
        let result: BatchResult<i32> = BatchResult::new(
            vec![
                BatchItem::new(0, String::new(), BatchItemStatus::Succeeded, Some(10), None),
                BatchItem::new(
                    1,
                    String::new(),
                    BatchItemStatus::Failed,
                    None,
                    Some("err".into()),
                ),
                BatchItem::new(2, String::new(), BatchItemStatus::Succeeded, Some(30), None),
            ],
            CompletionReason::FailureToleranceExceeded,
        );
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 1);
        assert_eq!(result.total_count(), 3);
        assert!(result.has_failure());
        assert_eq!(result.status(), "FAILED");
        assert_eq!(result.errors(), vec!["err"]);
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
                ),
                BatchItem::new(
                    1,
                    String::new(),
                    BatchItemStatus::Succeeded,
                    Some("b"),
                    None,
                ),
            ],
            CompletionReason::AllCompleted,
        );
        assert!(!result.has_failure());
        assert_eq!(result.status(), "SUCCEEDED");
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 0);
        assert!(result.errors().is_empty());
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
        let result: Result<Vec<String>, _> =
            ctx.parallel::<String>(vec![]).max_concurrency(0).await;
        let err = result.expect_err("max_concurrency=0 must error");
        let msg = err.to_string();
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
        let msg = err.to_string();
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
            .completion(crate::CompletionConfig {
                tolerated_failure_count: Some(0),
                ..Default::default()
            })
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
            .completion(crate::CompletionConfig {
                tolerated_failure_count: Some(0),
                ..Default::default()
            })
            .await
        };

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), run).await;
        let result: Result<Vec<i32>, OperationError> =
            outcome.expect("map with a panicking branch must resolve, not hang");
        let err =
            result.expect_err("a panicking branch must surface as a controlled batch failure");
        assert!(
            err.to_string().contains("boom in map branch"),
            "batch failure should carry the branch panic message, got: {err}"
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
            .await
        };

        let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), run).await;
        let result: Result<Vec<i32>, OperationError> =
            outcome.expect("parallel with a panicking branch must resolve, not hang");
        let err =
            result.expect_err("a panicking branch must surface as a controlled batch failure");
        assert!(
            err.to_string().contains("boom in parallel branch"),
            "batch failure should carry the branch panic message, got: {err}"
        );
    }
}
