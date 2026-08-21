//! The batch operations: [`MapBuilder`] and [`ParallelBuilder`], returned
//! by [`DurableContext::map`](crate::DurableContext::map) and
//! [`DurableContext::parallel`](crate::DurableContext::parallel), plus
//! their shared completion configuration ([`CompletionConfig`]) and the
//! batch result types.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::BoxError;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

pub use crate::map_parallel::{
    BatchError, BatchItem, BatchItemStatus, BatchResult, BatchStats, BatchStatus, BatchSummary,
    CompletionReason, NestingMode, SettledOutcome,
};

// ============================================================
// MapBuilder
// ============================================================

/// Builder for a durable map operation.
///
/// Created by [`DurableContext::map`]. Applies a function to each item
/// with configurable concurrency and completion behavior.
///
/// The builder is generic over the item closure `F` and its future `Fut`
/// so the closure is stored **without type erasure**; both parameters are
/// inferred at the call site and never written by users. At execution the
/// closure is shared as `Arc<F>` and every item produces a concrete
/// future, so the internal `JoinSet` holds unboxed futures. The single
/// erasure point is `.future()` / `.await`, which produces the one
/// [`DurableFuture`] box.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct Item { id: u64 }
///
/// #[derive(Serialize, Deserialize)]
/// struct Output { processed: bool }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     let items = vec![Item { id: 1 }, Item { id: 2 }];
///     let _results: Vec<Output> = ctx.map(items, |child, item, _idx| async move {
///         Ok(Output { processed: true })
///     }).name("process-all")
///       .max_concurrency(4)
///       .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct MapBuilder<I, O, F, Fut, IS = JsonSerdes, RS = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    serdes: IS,
    result_serdes: RS,
    nesting: NestingMode,
    item_namer: Option<Arc<dyn Fn(usize) -> String + Send + Sync>>,
    items: Vec<I>,
    closure: F,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<I, O, F, Fut, IS, RS> std::fmt::Debug for MapBuilder<I, O, F, Fut, IS, RS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapBuilder")
            .field("name", &self.name)
            .field("max_concurrency", &self.max_concurrency)
            .finish_non_exhaustive()
    }
}

impl<I, O, F, Fut> MapBuilder<I, O, F, Fut>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the items and closure here
    /// keeps the closure field non-optional: a builder without a body is
    /// unrepresentable.
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, items: Vec<I>, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            serdes: JsonSerdes,
            result_serdes: JsonSerdes,
            nesting: NestingMode::Normal,
            item_namer: None,
            items,
            closure,
            _marker: PhantomData,
        }
    }
}

impl<I, O, F, Fut, IS, RS> MapBuilder<I, O, F, Fut, IS, RS>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this map operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the maximum number of concurrent items to process.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Sets the completion configuration.
    ///
    /// **Default: fail fast.** Without this call — or with an empty
    /// [`CompletionConfig`] (no threshold, no predicate) — the first item
    /// failure fails the batch, matching the Python and JS SDKs.
    /// Configuring any criterion replaces that implicit fail-fast with the
    /// configured criteria:
    ///
    /// - explicit fail-fast:
    ///   [`CompletionConfig::with_tolerated_failure_count(0)`](CompletionConfig::with_tolerated_failure_count)
    /// - tolerate all failures:
    ///   [`CompletionConfig::with_tolerated_failure_percentage(100)`](CompletionConfig::with_tolerated_failure_percentage)
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for item results.
    ///
    /// The serdes must implement [`Serdes<O>`](crate::Serdes) for this
    /// map's item output type — attaching a serdes for a different type
    /// fails at compile time. It goes through the same transform boundary
    /// as every other operation, so a [`Serdes`] attached here behaves
    /// exactly as it does on a step, invoke, callback, or `result_serdes`.
    pub fn serdes<IS2>(self, serdes: IS2) -> MapBuilder<I, O, F, Fut, IS2, RS>
    where
        IS2: Serdes<O>,
    {
        MapBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Sets a custom serializer/deserializer for the whole batch result.
    ///
    /// This is the operation-level serdes: it serializes and deserializes
    /// the entire batch summary ([`BatchSummary`]) rather than individual
    /// items, so it must implement `Serdes<BatchSummary>` — typically
    /// through a type-agnostic blanket `impl<T> Serdes<T>`.
    pub fn result_serdes<RS2>(self, serdes: RS2) -> MapBuilder<I, O, F, Fut, IS, RS2>
    where
        RS2: Serdes<BatchSummary>,
    {
        MapBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            result_serdes: serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Sets the nesting mode for the map operation.
    ///
    /// [`NestingMode::Flat`] causes items to run in virtual contexts
    /// without per-item context events.
    pub fn nesting(mut self, mode: NestingMode) -> Self {
        self.nesting = mode;
        self
    }

    /// Sets a custom item namer for per-iteration display names.
    ///
    /// The namer function receives the zero-based item index and returns
    /// a display name for that iteration.
    pub fn item_namer(mut self, namer: impl Fn(usize) -> String + Send + Sync + 'static) -> Self {
        self.item_namer = Some(Arc::new(namer));
        self
    }

    /// Executes the map and returns the full [`BatchResult`] including
    /// completion metadata (reason, success/failure counts, per-item status).
    ///
    /// Use this when you need to inspect batch completion details (e.g., when
    /// using a completion config that tolerates failures). The standard
    /// `.await` returns `Vec<O>` with only the successful items and cannot
    /// report which items failed; it returns an error only when the batch
    /// ends because the configured failure tolerance was exceeded.
    ///
    /// Always returns the full `BatchResult<O>` including per-item outcomes.
    /// [`BatchResult::status`] reports the overall outcome as a
    /// [`BatchStatus`], and [`BatchResult::errors`] associates each failure
    /// with the index and name of the item that produced it.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch execution encounters an infrastructure
    /// failure (checkpoint client error, task-ownership violation, invalid
    /// configuration). Item-level failures are NOT errors — they appear as
    /// `BatchItemStatus::Failed` entries in the result, and the batch's
    /// [`CompletionReason`] records why the batch ended.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// async fn handler(
    ///     event: Vec<String>,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let batch = ctx
    ///         .map(event, |child, item, _idx| async move {
    ///             let len = child
    ///                 .step(move |_| async move { Ok(item.len()) })
    ///                 .name("measure")
    ///                 .await?;
    ///             Ok(len)
    ///         })
    ///         .name("measure-all")
    ///         .completion(durable::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(2))
    ///         .await_batch()
    ///         .await?;
    ///
    ///     if batch.status() == durable::builders::map_parallel::BatchStatus::Failed {
    ///         for error in batch.errors() {
    ///             println!(
    ///                 "item {} failed with {}: {}",
    ///                 error.index,
    ///                 error.error_type.unwrap_or("unknown error type"),
    ///                 error.message,
    ///             );
    ///         }
    ///     }
    ///     println!("batch ended because {:?}", batch.reason);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// [`BatchResult`]: BatchResult
    /// [`BatchResult::status`]: BatchResult::status
    /// [`BatchResult::errors`]: BatchResult::errors
    /// [`BatchStatus`]: BatchStatus
    /// [`CompletionReason`]: CompletionReason
    pub async fn await_batch(self) -> Result<BatchResult<O>, OperationError>
    where
        IS: Serdes<O>,
        RS: Serdes<BatchSummary>,
    {
        use crate::map_parallel::MapExecution;

        let execution = MapExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: Arc::new(self.closure),
            _marker: PhantomData,
        };

        execution.execute_batch_result().await
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<BatchSummary>,
    {
        self.into_future()
    }

    /// Eagerly spawns the map operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<BatchSummary>,
    {
        spawn_terminal!(self)
    }
}

impl<I, O, F, Fut, IS, RS> IntoFuture for MapBuilder<I, O, F, Fut, IS, RS>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<BatchSummary>,
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::map_parallel::MapExecution;

        preflight_identity!(self, "Context", crate::map_parallel::MAP_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = MapExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: Arc::new(self.closure),
            _marker: PhantomData,
        };

        DurableFuture::lazy_scoped(
            async move { execution.execute().await },
            owner_scope,
            op_scope,
        )
    }
}

// ============================================================
// ParallelBuilder
// ============================================================

/// Builder for a parallel operation with named branches.
///
/// Created by [`DurableContext::parallel`]. Each branch gets its own child
/// context.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     let branches = vec![
///         durable::Branch::new("a", |_| async { Ok(1) }),
///         durable::Branch::new("b", |_| async { Ok(2) }),
///     ];
///     let _results: Vec<i32> = ctx.parallel(branches)
///         .name("fan-out")
///         .max_concurrency(2)
///         .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct ParallelBuilder<O, IS = JsonSerdes, RS = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    serdes: IS,
    result_serdes: RS,
    nesting: NestingMode,
    branches: Vec<(String, crate::future::BranchBody<O>)>,
    _marker: PhantomData<O>,
}

impl<O, IS, RS> std::fmt::Debug for ParallelBuilder<O, IS, RS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> ParallelBuilder<O> {
    /// Creates a new builder (internal). Taking the branches here keeps the
    /// builder complete from construction: `context.parallel()` always has
    /// them in hand.
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        branches: Vec<(String, crate::future::BranchBody<O>)>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            serdes: JsonSerdes,
            result_serdes: JsonSerdes,
            nesting: NestingMode::Normal,
            branches,
            _marker: PhantomData,
        }
    }
}

impl<O: Send + 'static, IS, RS> ParallelBuilder<O, IS, RS> {
    /// Sets a human-readable name for this parallel operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the maximum number of concurrent branches.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Sets the completion configuration.
    ///
    /// **Default: fail fast.** Without this call — or with an empty
    /// [`CompletionConfig`] (no threshold, no predicate) — the first branch
    /// failure fails the batch, matching the Python and JS SDKs.
    /// Configuring any criterion replaces that implicit fail-fast with the
    /// configured criteria:
    ///
    /// - explicit fail-fast:
    ///   [`CompletionConfig::with_tolerated_failure_count(0)`](CompletionConfig::with_tolerated_failure_count)
    /// - tolerate all failures:
    ///   [`CompletionConfig::with_tolerated_failure_percentage(100)`](CompletionConfig::with_tolerated_failure_percentage)
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for branch results.
    ///
    /// The serdes must implement [`Serdes<O>`](crate::Serdes) for this
    /// operation's branch output type — attaching a serdes for a different
    /// type fails at compile time. It goes through the same transform
    /// boundary as every other operation, so a [`Serdes`] attached here
    /// behaves exactly as it does on a step, invoke, callback, or
    /// `result_serdes`.
    pub fn serdes<IS2>(self, serdes: IS2) -> ParallelBuilder<O, IS2, RS>
    where
        IS2: Serdes<O>,
    {
        ParallelBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
            _marker: PhantomData,
        }
    }

    /// Sets a custom serializer/deserializer for the whole batch result.
    ///
    /// This is the operation-level serdes: it serializes and deserializes
    /// the entire batch summary ([`BatchSummary`]) rather than individual
    /// items, so it must implement `Serdes<BatchSummary>` — typically
    /// through a type-agnostic blanket `impl<T> Serdes<T>`.
    pub fn result_serdes<RS2>(self, serdes: RS2) -> ParallelBuilder<O, IS, RS2>
    where
        RS2: Serdes<BatchSummary>,
    {
        ParallelBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            result_serdes: serdes,
            nesting: self.nesting,
            branches: self.branches,
            _marker: PhantomData,
        }
    }

    /// Sets the nesting mode for the parallel operation.
    ///
    /// [`NestingMode::Flat`] causes branches to run in virtual contexts
    /// without per-branch context events.
    pub fn nesting(mut self, mode: NestingMode) -> Self {
        self.nesting = mode;
        self
    }

    /// Executes the parallel operation and returns the full [`BatchResult`]
    /// including completion metadata (reason, success/failure counts,
    /// per-branch status).
    ///
    /// Use this when you need to inspect batch completion details (e.g., when
    /// using a completion config that tolerates failures). The standard
    /// `.await` returns `Vec<O>` with only the successful branches and
    /// cannot report which branches failed; it returns an error only when
    /// the batch ends because the configured failure tolerance was
    /// exceeded.
    ///
    /// Always returns the full `BatchResult<O>` including per-branch
    /// outcomes. [`BatchResult::status`] reports the overall outcome as a
    /// [`BatchStatus`], and [`BatchResult::errors`] associates each failure
    /// with the index and name of the branch that produced it.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch execution encounters an infrastructure
    /// failure (checkpoint client error, task-ownership violation, invalid
    /// configuration). Branch-level failures are NOT errors — they appear as
    /// `BatchItemStatus::Failed` entries in the result, and the batch's
    /// [`CompletionReason`] records why the batch ended.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let branches = vec![
    ///         durable::Branch::new("a", |_| async { Ok(1) }),
    ///         durable::Branch::new("b", |_| async { Ok(2) }),
    ///     ];
    ///     let batch = ctx
    ///         .parallel(branches)
    ///         .name("fan-out")
    ///         .completion(durable::builders::map_parallel::CompletionConfig::with_tolerated_failure_count(1))
    ///         .await_batch()
    ///         .await?;
    ///     if batch.status() == durable::builders::map_parallel::BatchStatus::Failed {
    ///         for error in batch.errors() {
    ///             println!(
    ///                 "branch {} failed with {}: {}",
    ///                 error.name,
    ///                 error.error_type.unwrap_or("unknown error type"),
    ///                 error.message,
    ///             );
    ///         }
    ///     }
    ///     println!("batch ended because {:?}", batch.reason);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// [`BatchResult`]: BatchResult
    /// [`BatchResult::status`]: BatchResult::status
    /// [`BatchResult::errors`]: BatchResult::errors
    /// [`BatchStatus`]: BatchStatus
    /// [`CompletionReason`]: CompletionReason
    pub async fn await_batch(self) -> Result<BatchResult<O>, OperationError>
    where
        IS: Serdes<O>,
        RS: Serdes<BatchSummary>,
    {
        use crate::map_parallel::ParallelExecution;

        let execution = ParallelExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
        };

        execution.execute_batch_result().await
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<BatchSummary>,
    {
        self.into_future()
    }

    /// Eagerly spawns the parallel operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<BatchSummary>,
    {
        spawn_terminal!(self)
    }
}

impl<O, IS, RS> IntoFuture for ParallelBuilder<O, IS, RS>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<BatchSummary>,
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::map_parallel::ParallelExecution;

        preflight_identity!(self, "Context", crate::map_parallel::PARALLEL_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = ParallelExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
        };

        DurableFuture::lazy_scoped(
            async move { execution.execute().await },
            owner_scope,
            op_scope,
        )
    }
}

/// A shared completion predicate consulted by map/parallel batch execution.
///
/// Receives the running [`BatchStats`] after each item settles and returns
/// `true` to end the batch early.
///
/// Crate-internal: the `Arc` wrapping is an implementation detail (it is what
/// keeps [`CompletionConfig`] cheaply cloneable with a closure inside, the
/// same way per-operation serdes are stored). Public setters
/// ([`CompletionConfig::with_completion_predicate`],
/// [`CompletionConfigBuilder::completion_predicate`]) take a generic closure
/// and wrap it here.
pub(crate) type CompletionPredicate = Arc<dyn Fn(&BatchStats<'_>) -> bool + Send + Sync>;

/// Configuration for completion behavior in map and parallel operations.
///
/// Controls early-completion thresholds: how many items must succeed and
/// how many failures are tolerated before stopping. Thresholds may be
/// combined — when multiple are set (including a
/// [completion predicate](Self::with_completion_predicate)), the first
/// trigger to fire wins.
///
/// # Default: fail fast
///
/// An **empty** config — [`CompletionConfig::default()`], with no threshold
/// and no predicate — fails the batch on the **first item failure**, exactly
/// as running with no config at all does. This matches the Python and JS
/// SDKs. Configuring any criterion opts out of that implicit fail-fast and
/// hands the decision to the configured criteria:
///
/// - explicit fail-fast: [`with_tolerated_failure_count(0)`](Self::with_tolerated_failure_count)
/// - tolerate all failures: [`with_tolerated_failure_percentage(100)`](Self::with_tolerated_failure_percentage)
///
/// Construct with [`CompletionConfig::builder`] (which combines thresholds
/// and validates them at [`build`](CompletionConfigBuilder::build) time) or
/// with one of the single-threshold constructors. Fields are private per
/// C-STRUCT-PRIVATE; read values back through the accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfig;
///
/// // Fail-fast: stop on the first failure.
/// let fail_fast = CompletionConfig::with_tolerated_failure_count(0);
/// assert_eq!(fail_fast.tolerated_failure_count(), Some(0));
///
/// // Early completion: stop after 2 successes.
/// let min_success = CompletionConfig::with_min_successful(2);
/// assert_eq!(min_success.min_successful(), Some(2));
///
/// // Custom predicate: stop once two items have settled either way.
/// let custom = CompletionConfig::with_completion_predicate(|stats| stats.settled() >= 2);
/// assert!(custom.has_completion_predicate());
///
/// // Combined thresholds — first to fire wins.
/// let combined = CompletionConfig::builder()
///     .min_successful(2)
///     .tolerated_failure_count(1)
///     .build()?;
/// assert_eq!(combined.min_successful(), Some(2));
/// assert_eq!(combined.tolerated_failure_count(), Some(1));
/// # Ok::<(), aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfigValidationError>(())
/// ```
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct CompletionConfig {
    /// Completes the batch early once this many items succeed.
    /// `None` means no minimum-success threshold.
    min_successful: Option<usize>,

    /// Fails the batch once more than this many items fail.
    /// `Some(0)` means fail-fast (stop on first failure).
    /// `None` means no count-based failure tolerance.
    tolerated_failure_count: Option<usize>,

    /// Fails the batch once the failure percentage strictly exceeds this
    /// threshold (integer 0-100).  Uses cross-multiplication to avoid
    /// integer-division truncation.
    /// `Some(0)` means fail-fast (stop on first failure).
    /// `None` means no percentage-based failure tolerance.
    tolerated_failure_percentage: Option<usize>,

    /// User-supplied completion predicate over the running batch
    /// statistics. Consulted after the fixed thresholds (first trigger
    /// wins); returning `true` completes the batch with
    /// [`CompletionReason::PredicateMatched`].
    /// `None` means no custom predicate.
    completion_predicate: Option<CompletionPredicate>,
}

impl std::fmt::Debug for CompletionConfig {
    // Hand-written because the stored predicate closure has no `Debug`;
    // its presence is reported instead (same approach the builders take
    // for their stored closures).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionConfig")
            .field("min_successful", &self.min_successful)
            .field("tolerated_failure_count", &self.tolerated_failure_count)
            .field(
                "tolerated_failure_percentage",
                &self.tolerated_failure_percentage,
            )
            .field(
                "completion_predicate",
                &self.completion_predicate.as_ref().map(|_| "<closure>"),
            )
            .finish()
    }
}

impl CompletionConfig {
    /// Creates a new [`CompletionConfigBuilder`].
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfig;
    ///
    /// let config = CompletionConfig::builder().min_successful(3).build()?;
    /// assert_eq!(config.min_successful(), Some(3));
    /// # Ok::<(), aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfigValidationError>(())
    /// ```
    pub fn builder() -> CompletionConfigBuilder {
        CompletionConfigBuilder::default()
    }

    /// Creates a config with just a `min_successful` threshold.
    ///
    /// **Default when unset:** the batch runs until every item settles.
    /// Note that a config carrying only this threshold has no failure
    /// tolerance configured, so item failures are tolerated while the
    /// batch waits to reach `min` successes — the implicit fail-fast of an
    /// empty config no longer applies once any criterion is set.
    #[must_use]
    pub fn with_min_successful(min: usize) -> Self {
        Self {
            min_successful: Some(min),
            ..Self::default()
        }
    }

    /// Creates a config with just a `tolerated_failure_count` threshold.
    ///
    /// Use `0` for explicit fail-fast behavior (stop on first failure).
    ///
    /// **Default when unset:** an empty config fails fast on the first
    /// failure; a config with some other criterion set applies only that
    /// criterion.
    #[must_use]
    pub fn with_tolerated_failure_count(count: usize) -> Self {
        Self {
            tolerated_failure_count: Some(count),
            ..Self::default()
        }
    }

    /// Creates a config with just a `tolerated_failure_percentage` threshold.
    ///
    /// The batch stops once the true failure rate **strictly exceeds** the
    /// given percentage.  Internally this uses cross-multiplication
    /// (`failure_count * 100 > pct * total_items`) to avoid integer-division
    /// truncation — so a failure rate of 33.3% (1 of 3) correctly exceeds a
    /// 33% threshold.
    ///
    /// Use `0` for explicit fail-fast behavior (stop on first failure).
    /// Use `100` to tolerate all failures (the failure rate can never
    /// strictly exceed 100%).
    ///
    /// **Default when unset:** an empty config fails fast on the first
    /// failure; a config with some other criterion set applies only that
    /// criterion.
    #[must_use]
    pub fn with_tolerated_failure_percentage(pct: usize) -> Self {
        Self {
            tolerated_failure_percentage: Some(pct),
            ..Self::default()
        }
    }

    /// Creates a config with just a custom completion predicate.
    ///
    /// The predicate receives the running [`BatchStats`] and returns `true`
    /// to end the batch early. A batch completed
    /// this way records [`CompletionReason::PredicateMatched`], and — like a
    /// [`min_successful`](Self::with_min_successful) completion — item
    /// failures inside it are tolerated rather than propagated as errors.
    /// Setting a predicate counts as a completion criterion, so it also
    /// replaces the implicit fail-fast of an empty config — the predicate
    /// owns the completion decision.
    ///
    /// When combined with fixed thresholds (via
    /// [`CompletionConfig::builder`]), the first trigger to fire wins,
    /// matching the existing threshold semantics: the SDK checks
    /// `min_successful`, then the failure tolerances, then this predicate.
    ///
    /// # Determinism — read this before using
    ///
    /// **The predicate MUST be a deterministic, pure function of the
    /// [`BatchStats`] it receives.** If the predicate consults anything
    /// else — the clock, random numbers, environment state, an external
    /// service, or mutable captured state — replays can diverge from the
    /// original run, which corrupts the execution history. Put
    /// nondeterminism inside a step body, never inside a completion
    /// predicate.
    ///
    /// The SDK evaluates the predicate only on state derivable from
    /// recorded checkpoint results: item outcomes feed the statistics
    /// strictly in input order, whatever order the items actually finished
    /// in at run time, so a pure predicate sees the identical sequence of
    /// [`BatchStats`] snapshots on the original run and on every replay.
    /// See [`CompletionConfigBuilder::completion_predicate`] for the full
    /// ordering contract.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfig;
    ///
    /// // End the batch once half the items have settled.
    /// let config =
    ///     CompletionConfig::with_completion_predicate(|stats| {
    ///         stats.settled() * 2 >= stats.total_items()
    ///     });
    /// assert!(config.has_completion_predicate());
    /// ```
    #[must_use]
    pub fn with_completion_predicate<F>(predicate: F) -> Self
    where
        F: Fn(&BatchStats<'_>) -> bool + Send + Sync + 'static,
    {
        Self {
            completion_predicate: Some(Arc::new(predicate)),
            ..Self::default()
        }
    }

    /// Returns the minimum-success threshold, if set.
    #[must_use]
    pub fn min_successful(&self) -> Option<usize> {
        self.min_successful
    }

    /// Returns the count-based failure tolerance, if set.
    #[must_use]
    pub fn tolerated_failure_count(&self) -> Option<usize> {
        self.tolerated_failure_count
    }

    /// Returns the percentage-based failure tolerance, if set.
    #[must_use]
    pub fn tolerated_failure_percentage(&self) -> Option<usize> {
        self.tolerated_failure_percentage
    }

    /// Reports whether a custom completion predicate is set.
    ///
    /// The predicate itself is not exposed: its boxing is an implementation
    /// detail, matching how the crate stores other user closures (for
    /// example retry strategies).
    #[must_use]
    pub fn has_completion_predicate(&self) -> bool {
        self.completion_predicate.is_some()
    }

    /// Reports whether ANY completion criterion is configured — a threshold
    /// or a custom predicate (crate-internal).
    ///
    /// An empty config (nothing set) fails the batch on the first item
    /// failure, exactly as running with no config does; configuring any
    /// criterion opts out of that implicit fail-fast (issue #52, matching
    /// the Python and JS SDKs).
    pub(crate) fn has_criteria(&self) -> bool {
        self.min_successful.is_some()
            || self.tolerated_failure_count.is_some()
            || self.tolerated_failure_percentage.is_some()
            || self.completion_predicate.is_some()
    }

    /// Evaluates the custom completion predicate against the running batch
    /// statistics. Returns `false` when no predicate is configured
    /// (crate-internal; called by the batch coordinator after each settled
    /// item).
    pub(crate) fn predicate_matches(&self, stats: &BatchStats<'_>) -> bool {
        self.completion_predicate
            .as_ref()
            .is_some_and(|predicate| predicate(stats))
    }

    /// Validates the completion config, returning an error message when the
    /// config is invalid (crate-internal; callers convert the message into a
    /// typed batch error).
    ///
    /// [`CompletionConfigBuilder::build`] performs the same range check at
    /// construction time; this execute-time check remains as the guard for
    /// configs made through the single-threshold constructors (for example
    /// [`CompletionConfig::with_tolerated_failure_percentage`]), which do
    /// not validate.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if let Some(pct) = self.tolerated_failure_percentage
            && pct > 100
        {
            return Err(format!(
                "tolerated_failure_percentage must be 0-100, got {pct}"
            ));
        }
        Ok(())
    }
}

/// Error returned by [`CompletionConfigBuilder::build`] when the configured
/// thresholds are invalid.
///
/// Mirrors [`OptionsValidationError`](crate::OptionsValidationError): misconfiguration fails at
/// construction time rather than mid-execution.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfig;
///
/// let err = CompletionConfig::builder()
///     .tolerated_failure_percentage(101)
///     .build()
///     .unwrap_err();
/// assert!(err.to_string().contains("tolerated_failure_percentage"));
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CompletionConfigValidationError {
    message: String,
}

impl std::fmt::Display for CompletionConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid CompletionConfig: {}", self.message)
    }
}

impl std::error::Error for CompletionConfigValidationError {}

/// Builder for [`CompletionConfig`].
///
/// Follows the Rust API Guidelines C-BUILDER pattern. All methods consume
/// and return `self` for chaining, so multiple thresholds combine in one
/// expression instead of requiring post-construction mutation.
/// [`build`](Self::build) validates the combination and rejects a
/// misconfiguration at construction time.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfig;
///
/// let config = CompletionConfig::builder()
///     .min_successful(2)
///     .tolerated_failure_percentage(25)
///     .build()?;
/// assert_eq!(config.tolerated_failure_percentage(), Some(25));
/// # Ok::<(), aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfigValidationError>(())
/// ```
#[derive(Clone, Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct CompletionConfigBuilder {
    min_successful: Option<usize>,
    tolerated_failure_count: Option<usize>,
    tolerated_failure_percentage: Option<usize>,
    completion_predicate: Option<CompletionPredicate>,
}

impl std::fmt::Debug for CompletionConfigBuilder {
    // Hand-written because the stored predicate closure has no `Debug`;
    // its presence is reported instead.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionConfigBuilder")
            .field("min_successful", &self.min_successful)
            .field("tolerated_failure_count", &self.tolerated_failure_count)
            .field(
                "tolerated_failure_percentage",
                &self.tolerated_failure_percentage,
            )
            .field(
                "completion_predicate",
                &self.completion_predicate.as_ref().map(|_| "<closure>"),
            )
            .finish()
    }
}

impl CompletionConfigBuilder {
    /// Completes the batch early once this many items succeed.
    ///
    /// **Default when unset:** the batch runs until every item settles.
    /// Setting this (or any other criterion) replaces the implicit
    /// fail-fast of an empty config; a config carrying only this threshold
    /// tolerates item failures while waiting to reach `min` successes.
    pub fn min_successful(mut self, min: usize) -> Self {
        self.min_successful = Some(min);
        self
    }

    /// Fails the batch once more than this many items fail.
    ///
    /// Use `0` for explicit fail-fast behavior (stop on first failure).
    ///
    /// **Default when unset:** an empty config fails fast on the first
    /// failure; a config with some other criterion set applies only that
    /// criterion.
    pub fn tolerated_failure_count(mut self, count: usize) -> Self {
        self.tolerated_failure_count = Some(count);
        self
    }

    /// Fails the batch once the failure percentage strictly exceeds this
    /// threshold (integer 0-100).
    ///
    /// Use `0` for explicit fail-fast behavior (stop on first failure).
    /// Use `100` to tolerate all failures (the failure rate can never
    /// strictly exceed 100%).
    ///
    /// **Default when unset:** an empty config fails fast on the first
    /// failure; a config with some other criterion set applies only that
    /// criterion.
    ///
    /// A value above 100 is rejected by [`build`](Self::build).
    pub fn tolerated_failure_percentage(mut self, pct: usize) -> Self {
        self.tolerated_failure_percentage = Some(pct);
        self
    }

    /// Sets a custom completion predicate over the running batch statistics.
    ///
    /// The predicate receives the running [`BatchStats`] and returns `true`
    /// to end the batch early with
    /// [`CompletionReason::PredicateMatched`]. It composes with the fixed
    /// thresholds: the SDK checks `min_successful`, then the failure
    /// tolerances, then this predicate — the first trigger to fire wins,
    /// matching the existing threshold semantics. Setting a predicate
    /// counts as a completion criterion, so it also replaces the implicit
    /// fail-fast of an empty config — the predicate owns the completion
    /// decision.
    ///
    /// # Determinism — read this before using
    ///
    /// **The predicate MUST be a deterministic, pure function of the
    /// [`BatchStats`] it receives.** If the predicate consults anything
    /// else — the clock, random numbers, environment state, an external
    /// service, or mutable captured state — replays can diverge from the
    /// original run, which corrupts the execution history. Put
    /// nondeterminism inside a step body, never inside a completion
    /// predicate.
    ///
    /// The SDK holds up its half of that contract by evaluating the
    /// predicate only on state derivable from recorded checkpoint results:
    /// item outcomes feed the statistics strictly in input order (item `i`
    /// enters only after items `0..i` have all settled), whatever order
    /// the items actually finished in at run time. Live settlement order
    /// is scheduler-timed and unrecorded, so it never influences the
    /// statistics — a pure predicate therefore sees the identical sequence
    /// of [`BatchStats`] snapshots on the original run and on every
    /// replay. The corollary: a slow or suspended item holds later items'
    /// outcomes out of the statistics until it settles.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfig;
    ///
    /// // Stop early once 2 items succeed OR any 3 items settle,
    /// // whichever fires first.
    /// let config = CompletionConfig::builder()
    ///     .min_successful(2)
    ///     .completion_predicate(|stats| stats.settled() >= 3)
    ///     .build()?;
    /// assert!(config.has_completion_predicate());
    /// # Ok::<(), aws_durable_execution_sdk_rust::builders::map_parallel::CompletionConfigValidationError>(())
    /// ```
    pub fn completion_predicate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&BatchStats<'_>) -> bool + Send + Sync + 'static,
    {
        self.completion_predicate = Some(Arc::new(predicate));
        self
    }

    /// Builds the [`CompletionConfig`] from the configured thresholds,
    /// validating them at construction time.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionConfigValidationError`] when
    /// `tolerated_failure_percentage` is outside the 0-100 range.
    /// Item-count-dependent checks (for example `min_successful` against
    /// the actual number of items) happen at execute time, where the item
    /// count is first known.
    pub fn build(self) -> Result<CompletionConfig, CompletionConfigValidationError> {
        if let Some(pct) = self.tolerated_failure_percentage
            && pct > 100
        {
            return Err(CompletionConfigValidationError {
                message: format!("tolerated_failure_percentage must be 0-100, got {pct}"),
            });
        }
        Ok(CompletionConfig {
            min_successful: self.min_successful,
            tolerated_failure_count: self.tolerated_failure_count,
            tolerated_failure_percentage: self.tolerated_failure_percentage,
            completion_predicate: self.completion_predicate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `CompletionConfig` builder combines thresholds in one expression —
    /// no `Default`-then-mutate — and the built config drives the same
    /// completion decisions the single-threshold constructors do.
    #[test]
    fn completion_config_builder_combines_thresholds() {
        let combined = CompletionConfig::builder()
            .min_successful(2)
            .tolerated_failure_count(1)
            .tolerated_failure_percentage(25)
            .build()
            .expect("valid config");

        assert_eq!(combined.min_successful(), Some(2));
        assert_eq!(combined.tolerated_failure_count(), Some(1));
        assert_eq!(combined.tolerated_failure_percentage(), Some(25));
        assert!(combined.validate().is_ok());

        // An unset threshold stays `None`.
        let only_min = CompletionConfig::builder()
            .min_successful(3)
            .build()
            .expect("valid config");
        assert_eq!(only_min.min_successful(), Some(3));
        assert_eq!(only_min.tolerated_failure_count(), None);
        assert_eq!(only_min.tolerated_failure_percentage(), None);
        assert!(!only_min.has_completion_predicate());

        // Builder output matches the equivalent single-threshold constructor.
        assert_eq!(
            CompletionConfig::builder()
                .tolerated_failure_count(0)
                .build()
                .expect("valid config")
                .tolerated_failure_count(),
            CompletionConfig::with_tolerated_failure_count(0).tolerated_failure_count(),
        );

        // Default is still "no thresholds".
        let default = CompletionConfig::default();
        assert_eq!(default.min_successful(), None);
        assert_eq!(default.tolerated_failure_count(), None);
        assert_eq!(default.tolerated_failure_percentage(), None);
        assert!(!default.has_completion_predicate());
    }

    /// An out-of-range percentage is rejected at construction time by
    /// `CompletionConfigBuilder::build`, and the error names the offending
    /// field. The execute-time `validate()` applies the same check for
    /// configs made through the single-threshold constructor, which does
    /// not validate.
    #[test]
    fn completion_config_builder_percentage_is_validated() {
        let err = CompletionConfig::builder()
            .tolerated_failure_percentage(101)
            .build()
            .expect_err("a percentage above 100 must be rejected at build time");
        assert!(
            err.to_string().contains("tolerated_failure_percentage"),
            "error should name the offending field, got: {err}"
        );

        // The boundary value 100 is accepted.
        assert!(
            CompletionConfig::builder()
                .tolerated_failure_percentage(100)
                .build()
                .is_ok()
        );

        // The constructor path is still guarded by the execute-time check.
        let constructed = CompletionConfig::with_tolerated_failure_percentage(101);
        let msg = constructed
            .validate()
            .expect_err("execute-time validation must reject the constructor path");
        assert!(
            msg.contains("tolerated_failure_percentage"),
            "error should name the offending field, got: {msg}"
        );
    }

    /// A stored completion predicate keeps `CompletionConfig`'s `Clone` and
    /// `Debug` story intact: the clone shares the predicate (`Arc`), and
    /// `Debug` reports the predicate's presence instead of requiring the
    /// closure to implement `Debug`.
    #[test]
    fn completion_config_predicate_clone_and_debug() {
        let config = CompletionConfig::builder()
            .min_successful(2)
            .completion_predicate(|stats| stats.settled() >= 3)
            .build()
            .expect("valid config");
        assert!(config.has_completion_predicate());

        let cloned = config.clone();
        assert!(cloned.has_completion_predicate());
        assert_eq!(cloned.min_successful(), Some(2));

        let debugged = format!("{config:?}");
        assert!(
            debugged.contains("completion_predicate") && debugged.contains("<closure>"),
            "Debug should report predicate presence, got: {debugged}"
        );

        // Constructor parity: `with_completion_predicate` sets only the
        // predicate.
        let only_predicate =
            CompletionConfig::with_completion_predicate(|stats| stats.failed() > stats.succeeded());
        assert!(only_predicate.has_completion_predicate());
        assert_eq!(only_predicate.min_successful(), None);

        // The predicate is evaluated against the stats it receives.
        let outcomes = [
            SettledOutcome::new(0, BatchItemStatus::Failed),
            SettledOutcome::new(1, BatchItemStatus::Failed),
            SettledOutcome::new(2, BatchItemStatus::Succeeded),
        ];
        let stats = BatchStats::new(1, 2, 5, &outcomes);
        assert!(only_predicate.predicate_matches(&stats));
        let stats_even = BatchStats::new(2, 2, 5, &outcomes);
        assert!(!only_predicate.predicate_matches(&stats_even));
        // No predicate configured → never matches.
        assert!(!CompletionConfig::default().predicate_matches(&stats));
    }
}
