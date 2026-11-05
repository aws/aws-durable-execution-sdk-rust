//! The batch operations: [`MapBuilder`] and [`ParallelBuilder`], returned
//! by [`DurableContext::map`](crate::DurableContext::map) and
//! [`DurableContext::parallel`](crate::DurableContext::parallel), plus
//! their shared completion configuration ([`CompletionConfig`]) and the
//! batch result types.
//!
//! The [map operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/map/) and the
//! [parallel operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/parallel/)
//! describe these operations independently of any language SDK.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::BoxError;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::map_parallel::ItemEncoding;
use crate::serdes::{JsonSerdes, PassthroughSerdes};

pub use crate::map_parallel::{
    BatchError, BatchItem, BatchItemStatus, BatchResult, BatchStats, BatchStatus,
    CompletionDecision, CompletionItem, CompletionOutcome, CompletionReason, NestingMode,
};

/// Builder for a durable map operation.
///
/// Created by [`DurableContext::map`]. Applies a function to each item
/// with configurable concurrency and completion behavior.
///
/// The builder is generic over the item closure `F` and its future `Fut`;
/// both parameters are inferred at the call site and never written by
/// users.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
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
pub struct MapBuilder<I, O, F, Fut, IS = JsonSerdes, RS = PassthroughSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    serdes: IS,
    encoding: ItemEncoding,
    result_serdes: RS,
    nesting: NestingMode,
    item_namer: Option<crate::map_parallel::ItemNamer<I>>,
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
    pub(crate) fn new_internal(
        ctx: DurableContext,
        op_id: OperationId,
        items: Vec<I>,
        closure: F,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            serdes: JsonSerdes,
            encoding: ItemEncoding::AllJson,
            result_serdes: PassthroughSerdes,
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
    ///
    /// `0` fails the execution with the error `Invalid max_concurrency: 0.
    /// Must be a positive number or unset for unlimited concurrency.`; it
    /// cannot be caught, and the map records nothing. Unset means
    /// unlimited.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Sets the completion configuration.
    ///
    /// **Default: fail fast.** Without this call, or with an empty
    /// [`CompletionConfig`] (no threshold, no predicate), the first item
    /// failure fails the batch, matching the Python and JS SDKs.
    /// Configuring any criterion replaces that implicit fail-fast with the
    /// configured criteria:
    ///
    /// - explicit fail-fast:
    ///   [`CompletionConfig::with_tolerated_failure_count(0)`](CompletionConfig::with_tolerated_failure_count)
    /// - tolerate all failures:
    ///   [`CompletionConfig::with_tolerated_failure_percentage(100.0)`](CompletionConfig::with_tolerated_failure_percentage)
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for item results.
    ///
    /// The serdes must implement [`Serdes<O>`](crate::Serdes) for this
    /// map's item output type: attaching a serdes for a different type
    /// fails at compile time. It goes through the same transform boundary
    /// as every other operation, so a [`Serdes`] attached here behaves
    /// exactly as it does on a step, invoke or callback. It is a different
    /// serdes from `result_serdes`, which transforms the whole batch record.
    ///
    /// `serdes` corresponds to JS `itemSerdes` and Python `item_serdes`.
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
            encoding: ItemEncoding::AllSerdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Sets a custom serializer/deserializer for the whole batch record.
    ///
    /// The serdes is a [`Serdes<String>`](crate::Serdes). Its serializer
    /// receives the batch record as JSON text, in the shape of the JS SDK's
    /// batch result (`{"all": [...], "completionReason": "..."}`), and
    /// returns the payload to store. Its deserializer receives that stored
    /// payload on replay, and must reconstruct the exact input record text:
    /// `deserialize(serialize(record_text))` must return `record_text`
    /// unchanged. The stored payload may use any format, for example a
    /// compressed or binary encoding of the text. The SDK does not read the
    /// record back on the live path, so a serdes that changes the record
    /// text is not supported.
    ///
    /// The default, [`PassthroughSerdes`], stores
    /// the record text unchanged. [`JsonSerdes`] stores it as a quoted JSON
    /// string.
    ///
    /// `result_serdes` corresponds to JS and Python `serdes`. Those receive
    /// the batch result object.
    pub fn result_serdes<RS2>(self, serdes: RS2) -> MapBuilder<I, O, F, Fut, IS, RS2>
    where
        RS2: Serdes<String>,
    {
        MapBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            encoding: self.encoding,
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
    /// The namer receives a reference to the item and its zero-based index,
    /// and returns the name history records for that item's child context.
    /// It runs once per input item when the map starts, before any item
    /// runs, including items that never start. It also runs on every
    /// replay, so it must be deterministic: a name that changes between
    /// invocations fails replay with a non-determinism error.
    ///
    /// A missing namer, or a namer that returns an empty string, records
    /// `map-item-{index}`. A completion predicate sees the namer's result,
    /// including `Some("")`, or `None` when no namer is set.
    pub fn item_namer(
        mut self,
        namer: impl Fn(&I, usize) -> String + Send + Sync + 'static,
    ) -> Self {
        self.item_namer = Some(Box::new(namer));
        self
    }

    /// Executes the map and returns the full [`BatchResult`] including
    /// completion metadata (reason, success/failure counts, per-item status).
    ///
    /// Use this when you need to inspect batch completion details (e.g., when
    /// using a completion config that tolerates failures). The standard
    /// `.await` returns `Vec<O>` with only the successful items and cannot
    /// report which items failed. It returns an error when the batch ends
    /// because the configured failure tolerance was exceeded, or because
    /// the completion predicate completed it as failed
    /// ([`CompletionReason::CustomCompletionFailed`]). This method returns
    /// the `BatchResult` in both cases.
    ///
    /// Always returns the full `BatchResult<O>` including per-item outcomes.
    /// [`BatchResult::status`] reports the overall outcome as a
    /// [`BatchStatus`], and [`BatchResult::errors`] associates each failure
    /// with the index and name of the item that produced it. When the
    /// completion rule ends the batch early, an item that is still running
    /// or parked is cancelled at its next `.await` and appears live as
    /// [`BatchItemStatus::Started`].
    ///
    /// The batch starts when the returned future is first polled. To start
    /// it at once and run it in the background, use
    /// [`spawn_batch`](Self::spawn_batch).
    ///
    /// # Errors
    ///
    /// Returns an error if the batch execution encounters an infrastructure
    /// failure (checkpoint client error, task-ownership violation) or an
    /// invalid completion configuration, such as a failure percentage outside
    /// `0.0..=100.0`. `max_concurrency(0)` returns no error: it fails the
    /// execution, and the handler cannot catch it. Item-level failures are
    /// NOT errors: they appear as
    /// `BatchItemStatus::Failed` entries in the result, and the batch's
    /// [`CompletionReason`] records why the batch ended.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
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
        RS: Serdes<String>,
    {
        self.batch_future().await
    }

    /// Starts the operation on its own task and returns its full
    /// [`BatchResult`].
    ///
    /// This is [`await_batch`](Self::await_batch) started at once, as
    /// [`spawn`](Self::spawn) is for the `Vec<O>` form. A handler can run
    /// other operations before it awaits the returned future.
    ///
    /// Dropping the returned future loses only the result: the operation
    /// runs until it settles. The SDK stops it earlier only when the scope
    /// it was started in closes, such as a child context whose body
    /// returned, or when the invocation ends. The SDK also stops the
    /// `.future()` inputs a combinator started. A stopped task stops at its
    /// next `.await`, and its record can stay `STARTED`. The SDK cannot stop
    /// code between two `.await`s, `spawn_blocking` work, tasks or threads
    /// the handler starts itself, or a request already sent to another
    /// service.
    ///
    /// `.spawn_batch()` runs the operation's start step on the caller's task
    /// before it returns, as JS starts the operation inside the call. When
    /// the operation's current state requires a `START` (it has no record,
    /// or its record begins a new attempt), that `START` is admitted to the
    /// write order before `.spawn_batch()` returns, so it reaches the service
    /// before every update the caller sends afterwards, on any runtime
    /// flavor. The task-ownership check runs on the caller too: a
    /// `.spawn_batch()` from a task the SDK did not start returns a handle that
    /// resolves to the task-ownership error, queues no `START` and runs no
    /// body. A start hook that JS fires when it queues the `START`, and the
    /// replay start hook of a record that resumes work, fire before
    /// `.spawn_batch()` returns. The rest of the operation, including
    /// `on_attempt_start` and the body, runs on the spawned task. The
    /// exceptions are `invoke`, `wait_for_condition` and the combinators,
    /// whose `.spawn()` starts the whole operation on the spawned task.
    ///
    /// # Errors
    ///
    /// When no Tokio runtime is running, the returned future resolves to an
    /// error and the operation records nothing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::time::Duration;
    ///
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<Vec<u32>, durable::BoxError> {
    ///     let batch = ctx
    ///         .map(vec![1_u32, 2], |child, i, _idx| async move {
    ///             Ok(child.step(move |_| async move { Ok(i * 10) }).name("scale").await?)
    ///         })
    ///         .name("scale-all")
    ///         .spawn_batch();
    ///     ctx.wait(Duration::from_secs(1)).name("meanwhile").await?;
    ///     let batch = batch.await?.into_result()?;
    ///     Ok(batch.results().into_iter().copied().collect())
    /// }
    /// ```
    pub fn spawn_batch(self) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        spawn_terminal!(self, started_batch)
    }

    /// The lazy terminal of the full [`BatchResult`]: the body of
    /// `into_future`, with the full `BatchResult` in place of its `Vec<O>` view.
    /// So `.await`, `await_batch()` and `spawn_batch()` run the same
    /// replay identity check and the same admission check.
    fn batch_future(self) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        self.batch_future_with(false)
    }

    /// [`Self::batch_future`] with the start step run now, on the caller's
    /// task (see [`crate::start`]). `spawn_batch()` and `.spawn()` use it.
    fn started_batch_future(self) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        self.batch_future_with(true)
    }

    /// The body of [`Self::batch_future`] and [`Self::started_batch_future`].
    fn batch_future_with(self, started: bool) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        use crate::map_parallel::MapExecution;

        preflight_identity!(self, "Context", crate::map_parallel::MAP_SUB_TYPE);

        let execution = MapExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            encoding: self.encoding,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: Arc::new(self.closure),
            _marker: PhantomData,
        };

        if started {
            DurableFuture::for_started_operation(execution.execute_batch_result_started())
        } else {
            DurableFuture::for_start_step(move || execution.execute_batch_result_started())
        }
    }

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    ///
    /// A combinator begins its `.future()` inputs' start phases, in input
    /// order, before its own start step. So this operation's `START` is
    /// queued before the combinator's, and a start hook it fires when it
    /// queues `START` runs before the combinator's; a start hook that waits
    /// for the `START` response keeps that timing. A replay of the
    /// combinator's final record starts no `.future()` input.
    ///
    /// A handle that [`.spawn()`](Self::spawn) returned already runs on its
    /// own task. A combinator that receives it never starts it again, and
    /// it keeps running independently: when the combinator resolves first,
    /// it does not cancel the handle, and the handle's result is dropped.
    pub fn future(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        self.into_future()
    }

    /// Eagerly spawns the map operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// Dropping the returned future loses only the result: the operation
    /// runs until it settles. The SDK stops it earlier only when the scope
    /// it was started in closes, such as a child context whose body
    /// returned, or when the invocation ends. The SDK also stops the
    /// `.future()` inputs a combinator started. A stopped task stops at its
    /// next `.await`, and its record can stay `STARTED`. The SDK cannot stop
    /// code between two `.await`s, `spawn_blocking` work, tasks or threads
    /// the handler starts itself, or a request already sent to another
    /// service.
    ///
    /// `.spawn()` runs the operation's start step on the caller's task
    /// before it returns, as JS starts the operation inside the call. When
    /// the operation's current state requires a `START` (it has no record,
    /// or its record begins a new attempt), that `START` is admitted to the
    /// write order before `.spawn()` returns, so it reaches the service
    /// before every update the caller sends afterwards, on any runtime
    /// flavor. The task-ownership check runs on the caller too: a
    /// `.spawn()` from a task the SDK did not start returns a handle that
    /// resolves to the task-ownership error, queues no `START` and runs no
    /// body. A start hook that JS fires when it queues the `START`, and the
    /// replay start hook of a record that resumes work, fire before
    /// `.spawn()` returns. The rest of the operation, including
    /// `on_attempt_start` and the body, runs on the spawned task. The
    /// exceptions are `invoke`, `wait_for_condition` and the combinators,
    /// whose `.spawn()` starts the whole operation on the spawned task.
    ///
    /// # Errors
    ///
    /// When no Tokio runtime is running, the returned future resolves to an
    /// error and the operation records nothing.
    pub fn spawn(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        spawn_terminal!(self, started)
    }

    /// The `Vec<O>` future of `.spawn()`: the full batch, started now on
    /// the caller's task, projected as `.await` projects it.
    fn started_future(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        // `map_output` keeps the batch's task context, so a combinator that
        // receives the handle runs it in the frames of its creation.
        self.started_batch_future()
            .map_output(|outcome| crate::map_parallel::collect_successful(outcome?))
    }
}

impl<I, O, F, Fut, IS, RS> IntoFuture for MapBuilder<I, O, F, Fut, IS, RS>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<String>,
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::map_parallel::MapExecution;

        preflight_identity!(self, "Context", crate::map_parallel::MAP_SUB_TYPE);

        let execution = MapExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            encoding: self.encoding,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            item_namer: self.item_namer,
            items: self.items,
            closure: Arc::new(self.closure),
            _marker: PhantomData,
        };

        DurableFuture::for_start_step(move || {
            let batch = execution.execute_batch_result_started();
            async move { crate::map_parallel::collect_successful(batch.await?) }
        })
    }
}

/// Builder for a parallel operation with named branches.
///
/// Created by [`DurableContext::parallel`]. Each branch gets its own child
/// context.
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
pub struct ParallelBuilder<O, IS = JsonSerdes, RS = PassthroughSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    serdes: IS,
    encoding: ItemEncoding,
    result_serdes: RS,
    nesting: NestingMode,
    branches: Vec<(Option<String>, crate::future::BranchBody<O>)>,
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
    pub(crate) fn new_internal(
        ctx: DurableContext,
        op_id: OperationId,
        branches: Vec<(Option<String>, crate::future::BranchBody<O>)>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            serdes: JsonSerdes,
            encoding: ItemEncoding::AllJson,
            result_serdes: PassthroughSerdes,
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
    ///
    /// `0` fails the execution with the error `Invalid max_concurrency: 0.
    /// Must be a positive number or unset for unlimited concurrency.`; it
    /// cannot be caught, and the parallel records nothing. Unset means
    /// unlimited.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Sets the completion configuration.
    ///
    /// **Default: fail fast.** Without this call, or with an empty
    /// [`CompletionConfig`] (no threshold, no predicate), the first branch
    /// failure fails the batch, matching the Python and JS SDKs.
    /// Configuring any criterion replaces that implicit fail-fast with the
    /// configured criteria:
    ///
    /// - explicit fail-fast:
    ///   [`CompletionConfig::with_tolerated_failure_count(0)`](CompletionConfig::with_tolerated_failure_count)
    /// - tolerate all failures:
    ///   [`CompletionConfig::with_tolerated_failure_percentage(100.0)`](CompletionConfig::with_tolerated_failure_percentage)
    ///
    /// A failure percentage applies to `parallel` as it does to `map`, as
    /// in JS and Python `ParallelConfig`.
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for branch results.
    ///
    /// The serdes must implement [`Serdes<O>`](crate::Serdes) for this
    /// operation's branch output type: attaching a serdes for a different
    /// type fails at compile time. It goes through the same transform
    /// boundary as every other operation, so a [`Serdes`] attached here
    /// behaves exactly as it does on a step, invoke or callback. It is a
    /// different serdes from `result_serdes`, which transforms the whole
    /// batch record.
    ///
    /// `serdes` corresponds to JS `itemSerdes` and Python `item_serdes`.
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
            encoding: ItemEncoding::AllSerdes,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
            _marker: PhantomData,
        }
    }

    /// Sets a custom serializer/deserializer for the whole batch record.
    ///
    /// The serdes is a [`Serdes<String>`](crate::Serdes). Its serializer
    /// receives the batch record as JSON text, in the shape of the JS SDK's
    /// batch result (`{"all": [...], "completionReason": "..."}`), and
    /// returns the payload to store. Its deserializer receives that stored
    /// payload on replay, and must reconstruct the exact input record text:
    /// `deserialize(serialize(record_text))` must return `record_text`
    /// unchanged. The stored payload may use any format, for example a
    /// compressed or binary encoding of the text. The SDK does not read the
    /// record back on the live path, so a serdes that changes the record
    /// text is not supported.
    ///
    /// The default, [`PassthroughSerdes`], stores
    /// the record text unchanged. [`JsonSerdes`] stores it as a quoted JSON
    /// string.
    ///
    /// `result_serdes` corresponds to JS and Python `serdes`. Those receive
    /// the batch result object.
    pub fn result_serdes<RS2>(self, serdes: RS2) -> ParallelBuilder<O, IS, RS2>
    where
        RS2: Serdes<String>,
    {
        ParallelBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            encoding: self.encoding,
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
    /// cannot report which branches failed. It returns an error when the
    /// batch ends because the configured failure tolerance was exceeded, or
    /// because the completion predicate completed it as failed
    /// ([`CompletionReason::CustomCompletionFailed`]). This method returns
    /// the `BatchResult` in both cases.
    ///
    /// Always returns the full `BatchResult<O>` including per-branch
    /// outcomes. [`BatchResult::status`] reports the overall outcome as a
    /// [`BatchStatus`], and [`BatchResult::errors`] associates each failure
    /// with the index and name of the branch that produced it. When the
    /// completion rule ends the batch early, a branch that is still running
    /// or parked is cancelled at its next `.await` and appears live as
    /// [`BatchItemStatus::Started`].
    ///
    /// The batch starts when the returned future is first polled. To start
    /// it at once and run it in the background, use
    /// [`spawn_batch`](Self::spawn_batch).
    ///
    /// # Errors
    ///
    /// Returns an error if the batch execution encounters an infrastructure
    /// failure (checkpoint client error, task-ownership violation) or an
    /// invalid completion configuration, such as a failure percentage outside
    /// `0.0..=100.0`. `max_concurrency(0)` returns no error: it fails the
    /// execution, and the handler cannot catch it. Branch-level failures are
    /// NOT errors: they appear as
    /// `BatchItemStatus::Failed` entries in the result, and the batch's
    /// [`CompletionReason`] records why the batch ended.
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
        RS: Serdes<String>,
    {
        self.batch_future().await
    }

    /// Starts the operation on its own task and returns its full
    /// [`BatchResult`].
    ///
    /// This is [`await_batch`](Self::await_batch) started at once, as
    /// [`spawn`](Self::spawn) is for the `Vec<O>` form. A handler can run
    /// other operations before it awaits the returned future.
    ///
    /// Dropping the returned future loses only the result: the operation
    /// runs until it settles. The SDK stops it earlier only when the scope
    /// it was started in closes, such as a child context whose body
    /// returned, or when the invocation ends. The SDK also stops the
    /// `.future()` inputs a combinator started. A stopped task stops at its
    /// next `.await`, and its record can stay `STARTED`. The SDK cannot stop
    /// code between two `.await`s, `spawn_blocking` work, tasks or threads
    /// the handler starts itself, or a request already sent to another
    /// service.
    ///
    /// `.spawn_batch()` runs the operation's start step on the caller's task
    /// before it returns, as JS starts the operation inside the call. When
    /// the operation's current state requires a `START` (it has no record,
    /// or its record begins a new attempt), that `START` is admitted to the
    /// write order before `.spawn_batch()` returns, so it reaches the service
    /// before every update the caller sends afterwards, on any runtime
    /// flavor. The task-ownership check runs on the caller too: a
    /// `.spawn_batch()` from a task the SDK did not start returns a handle that
    /// resolves to the task-ownership error, queues no `START` and runs no
    /// body. A start hook that JS fires when it queues the `START`, and the
    /// replay start hook of a record that resumes work, fire before
    /// `.spawn_batch()` returns. The rest of the operation, including
    /// `on_attempt_start` and the body, runs on the spawned task. The
    /// exceptions are `invoke`, `wait_for_condition` and the combinators,
    /// whose `.spawn()` starts the whole operation on the spawned task.
    ///
    /// # Errors
    ///
    /// When no Tokio runtime is running, the returned future resolves to an
    /// error and the operation records nothing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::time::Duration;
    ///
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<usize, durable::BoxError> {
    ///     let branches = vec![
    ///         durable::Branch::new("a", |_| async { Ok(1) }),
    ///         durable::Branch::new("b", |_| async { Ok(2) }),
    ///     ];
    ///     let batch = ctx.parallel(branches).name("fan-out").spawn_batch();
    ///     ctx.wait(Duration::from_secs(1)).name("meanwhile").await?;
    ///     let batch = batch.await?;
    ///     Ok(batch.succeeded().count())
    /// }
    /// ```
    pub fn spawn_batch(self) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        spawn_terminal!(self, started_batch)
    }

    /// The lazy terminal of the full [`BatchResult`]: the body of
    /// `into_future`, with the full `BatchResult` in place of its `Vec<O>` view.
    /// So `.await`, `await_batch()` and `spawn_batch()` run the same
    /// replay identity check and the same admission check.
    fn batch_future(self) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        self.batch_future_with(false)
    }

    /// [`Self::batch_future`] with the start step run now, on the caller's
    /// task (see [`crate::start`]). `spawn_batch()` and `.spawn()` use it.
    fn started_batch_future(self) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        self.batch_future_with(true)
    }

    /// The body of [`Self::batch_future`] and [`Self::started_batch_future`].
    fn batch_future_with(self, started: bool) -> DurableFuture<BatchResult<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        use crate::map_parallel::{ItemIds, ParallelExecution};

        preflight_identity!(self, "Context", crate::map_parallel::PARALLEL_SUB_TYPE);

        let item_ids = ItemIds::new(&self.op_id, self.branches.len());
        let execution = ParallelExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            encoding: self.encoding,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
            item_ids,
        };

        if started {
            DurableFuture::for_started_operation(execution.execute_batch_result_started())
        } else {
            DurableFuture::for_start_step(move || execution.execute_batch_result_started())
        }
    }

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    ///
    /// A combinator begins its `.future()` inputs' start phases, in input
    /// order, before its own start step. So this operation's `START` is
    /// queued before the combinator's, and a start hook it fires when it
    /// queues `START` runs before the combinator's; a start hook that waits
    /// for the `START` response keeps that timing. A replay of the
    /// combinator's final record starts no `.future()` input.
    ///
    /// A handle that [`.spawn()`](Self::spawn) returned already runs on its
    /// own task. A combinator that receives it never starts it again, and
    /// it keeps running independently: when the combinator resolves first,
    /// it does not cancel the handle, and the handle's result is dropped.
    pub fn future(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        self.into_future()
    }

    /// Eagerly spawns the parallel operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// Dropping the returned future loses only the result: the operation
    /// runs until it settles. The SDK stops it earlier only when the scope
    /// it was started in closes, such as a child context whose body
    /// returned, or when the invocation ends. The SDK also stops the
    /// `.future()` inputs a combinator started. A stopped task stops at its
    /// next `.await`, and its record can stay `STARTED`. The SDK cannot stop
    /// code between two `.await`s, `spawn_blocking` work, tasks or threads
    /// the handler starts itself, or a request already sent to another
    /// service.
    ///
    /// `.spawn()` runs the operation's start step on the caller's task
    /// before it returns, as JS starts the operation inside the call. When
    /// the operation's current state requires a `START` (it has no record,
    /// or its record begins a new attempt), that `START` is admitted to the
    /// write order before `.spawn()` returns, so it reaches the service
    /// before every update the caller sends afterwards, on any runtime
    /// flavor. The task-ownership check runs on the caller too: a
    /// `.spawn()` from a task the SDK did not start returns a handle that
    /// resolves to the task-ownership error, queues no `START` and runs no
    /// body. A start hook that JS fires when it queues the `START`, and the
    /// replay start hook of a record that resumes work, fire before
    /// `.spawn()` returns. The rest of the operation, including
    /// `on_attempt_start` and the body, runs on the spawned task. The
    /// exceptions are `invoke`, `wait_for_condition` and the combinators,
    /// whose `.spawn()` starts the whole operation on the spawned task.
    ///
    /// # Errors
    ///
    /// When no Tokio runtime is running, the returned future resolves to an
    /// error and the operation records nothing.
    pub fn spawn(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        spawn_terminal!(self, started)
    }

    /// The `Vec<O>` future of `.spawn()`: the full batch, started now on
    /// the caller's task, projected as `.await` projects it.
    fn started_future(self) -> DurableFuture<Vec<O>>
    where
        IS: Serdes<O>,
        RS: Serdes<String>,
    {
        // `map_output` keeps the batch's task context, so a combinator that
        // receives the handle runs it in the frames of its creation.
        self.started_batch_future()
            .map_output(|outcome| crate::map_parallel::collect_successful(outcome?))
    }
}

impl<O, IS, RS> IntoFuture for ParallelBuilder<O, IS, RS>
where
    O: Send + 'static,
    IS: Serdes<O>,
    RS: Serdes<String>,
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::map_parallel::{ItemIds, ParallelExecution};

        preflight_identity!(self, "Context", crate::map_parallel::PARALLEL_SUB_TYPE);

        let item_ids = ItemIds::new(&self.op_id, self.branches.len());
        let execution = ParallelExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes: self.serdes,
            encoding: self.encoding,
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
            item_ids,
        };

        DurableFuture::for_start_step(move || {
            let batch = execution.execute_batch_result_started();
            async move { crate::map_parallel::collect_successful(batch.await?) }
        })
    }
}

/// Builder for a parallel operation over branches whose results have
/// different types.
///
/// Created by [`DurableContext::parallel_mixed`]. Each branch's result is
/// read through the [`BranchHandle`](crate::BranchHandle) that
/// [`BranchSet`](crate::BranchSet) returned when the branch was added, so
/// the builder has no item serdes and no `await_batch`. `.await` returns
/// the batch's [`CompletionReason`].
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
///     let mut branches = durable::BranchSet::new();
///     let count = branches.add("count", |_| async { Ok(7_u32) });
///     let label = branches.add("label", |_| async { Ok("ready".to_owned()) });
///     ctx.parallel_mixed(branches)
///         .name("fan-out")
///         .max_concurrency(2)
///         .await?;
///     Ok(format!("{}: {}", count.into_result()?, label.into_result()?))
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct ParallelMixedBuilder<RS = PassthroughSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    result_serdes: RS,
    nesting: NestingMode,
    branches: Vec<(
        Option<String>,
        crate::future::BranchBody<crate::future::MixedValue>,
    )>,
    decoders: Arc<Vec<Arc<dyn crate::future::BranchDecoder>>>,
    json: Arc<[bool]>,
}

impl<RS> std::fmt::Debug for ParallelMixedBuilder<RS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelMixedBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ParallelMixedBuilder {
    /// Creates a new builder (internal).
    pub(crate) fn new_internal(
        ctx: DurableContext,
        op_id: OperationId,
        branches: crate::future::BranchSet,
    ) -> Self {
        let (branches, decoders, json) = branches.into_parts();
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            result_serdes: PassthroughSerdes,
            nesting: NestingMode::Normal,
            branches,
            decoders,
            json,
        }
    }
}

impl<RS> ParallelMixedBuilder<RS> {
    /// Sets a human-readable name for this parallel operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the maximum number of concurrent branches.
    ///
    /// `0` fails the execution with the error `Invalid max_concurrency: 0.
    /// Must be a positive number or unset for unlimited concurrency.`; it
    /// cannot be caught, and the parallel records nothing. Unset means
    /// unlimited.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Sets the completion configuration.
    ///
    /// **Default: fail fast.** Without this call, or with an empty
    /// [`CompletionConfig`] (no threshold, no predicate), the first branch
    /// failure fails the batch, matching the Python and JS SDKs.
    /// Configuring any criterion replaces that implicit fail-fast with the
    /// configured criteria:
    ///
    /// - explicit fail-fast:
    ///   [`CompletionConfig::with_tolerated_failure_count(0)`](CompletionConfig::with_tolerated_failure_count)
    /// - tolerate all failures:
    ///   [`CompletionConfig::with_tolerated_failure_percentage(100.0)`](CompletionConfig::with_tolerated_failure_percentage)
    ///
    /// A failure percentage applies to `parallel_mixed` as it does to
    /// `map`, as in JS and Python `ParallelConfig`.
    ///
    /// A [`BranchHandle`](crate::BranchHandle) has a result exactly when its
    /// branch settled, whatever the parallel's own outcome. A succeeded
    /// branch's handle returns its value, and a failed branch's handle
    /// returns its error. A branch that was waiting or running when the
    /// batch ended, or that never started, returns
    /// [`ChildContextErrorKind::NotCompleted`](crate::ChildContextErrorKind::NotCompleted).
    /// This holds when the completion rule fails the batch too: after a
    /// predicate completes the batch as failed with no failed branch,
    /// `.await` returns
    /// [`ChildContextErrorKind::BatchCompletionFailed`](crate::ChildContextErrorKind::BatchCompletionFailed),
    /// and every branch that succeeded still returns its value.
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for the whole batch record.
    ///
    /// The serdes is a [`Serdes<String>`](crate::Serdes). Its serializer
    /// receives the batch record as JSON text, in the shape of the JS SDK's
    /// batch result (`{"all": [...], "completionReason": "..."}`), and
    /// returns the payload to store. Its deserializer receives that stored
    /// payload on replay, and must reconstruct the exact input record text:
    /// `deserialize(serialize(record_text))` must return `record_text`
    /// unchanged. The stored payload may use any format, for example a
    /// compressed or binary encoding of the text. The SDK does not read the
    /// record back on the live path, so a serdes that changes the record
    /// text is not supported.
    ///
    /// The default, [`PassthroughSerdes`], stores
    /// the record text unchanged. [`JsonSerdes`] stores it as a quoted JSON
    /// string.
    ///
    /// `result_serdes` corresponds to JS and Python `serdes`. Those receive
    /// the batch result object.
    pub fn result_serdes<RS2>(self, serdes: RS2) -> ParallelMixedBuilder<RS2>
    where
        RS2: Serdes<String>,
    {
        ParallelMixedBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            result_serdes: serdes,
            nesting: self.nesting,
            branches: self.branches,
            decoders: self.decoders,
            json: self.json,
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

    /// Eagerly spawns the parallel operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations. The handles receive their results when the returned
    /// future is awaited, not when the spawned task finishes.
    ///
    /// Dropping the returned future loses only the result: the operation
    /// runs until it settles. The SDK stops it earlier only when the scope
    /// it was started in closes, such as a child context whose body
    /// returned, or when the invocation ends. The SDK also stops the
    /// `.future()` inputs a combinator started. A stopped task stops at its
    /// next `.await`, and its record can stay `STARTED`. The SDK cannot stop
    /// code between two `.await`s, `spawn_blocking` work, tasks or threads
    /// the handler starts itself, or a request already sent to another
    /// service.
    ///
    /// `.spawn()` runs the operation's start step on the caller's task
    /// before it returns, as JS starts the operation inside the call. When
    /// the operation's current state requires a `START` (it has no record,
    /// or its record begins a new attempt), that `START` is admitted to the
    /// write order before `.spawn()` returns, so it reaches the service
    /// before every update the caller sends afterwards, on any runtime
    /// flavor. The task-ownership check runs on the caller too: a
    /// `.spawn()` from a task the SDK did not start returns a handle that
    /// resolves to the task-ownership error, queues no `START` and runs no
    /// body. A start hook that JS fires when it queues the `START`, and the
    /// replay start hook of a record that resumes work, fire before
    /// `.spawn()` returns. The rest of the operation, including
    /// `on_attempt_start` and the body, runs on the spawned task. The
    /// exceptions are `invoke`, `wait_for_condition` and the combinators,
    /// whose `.spawn()` starts the whole operation on the spawned task.
    ///
    /// # Errors
    ///
    /// When no Tokio runtime is running, the returned future resolves to an
    /// error and the operation records nothing.
    pub fn spawn(self) -> DurableFuture<CompletionReason>
    where
        RS: Serdes<String>,
    {
        use crate::map_parallel::publish_outcome;

        let decoders = Arc::clone(&self.decoders);
        let spawned = spawn_terminal!(self, started_batch);
        // `map_output` keeps the handle's task context, so a combinator that
        // takes it runs it inside the creator's frames.
        spawned.map_output(move |outcome| publish_outcome(outcome, &decoders))
    }

    /// The lazy terminal of the full [`BatchResult`], shared by `.await`
    /// and `.spawn()`. So both run the same replay identity check and the
    /// same admission check.
    fn batch_future(self) -> DurableFuture<BatchResult<crate::future::MixedValue>>
    where
        RS: Serdes<String>,
    {
        self.batch_future_with(false)
    }

    /// [`Self::batch_future`] with the start step run now, on the caller's
    /// task (see [`crate::start`]). `spawn_batch()` and `.spawn()` use it.
    fn started_batch_future(self) -> DurableFuture<BatchResult<crate::future::MixedValue>>
    where
        RS: Serdes<String>,
    {
        self.batch_future_with(true)
    }

    /// The body of [`Self::batch_future`] and [`Self::started_batch_future`].
    fn batch_future_with(
        self,
        started: bool,
    ) -> DurableFuture<BatchResult<crate::future::MixedValue>>
    where
        RS: Serdes<String>,
    {
        use crate::map_parallel::{ItemIds, MixedItemSerdes, PARALLEL_SUB_TYPE, ParallelExecution};

        preflight_identity!(self, "Context", PARALLEL_SUB_TYPE);

        // One table of branch IDs, read by the batch engine and by the item
        // serdes, which finds each branch's serdes from it.
        let item_ids = ItemIds::new(&self.op_id, self.branches.len());
        let serdes = MixedItemSerdes::new(&item_ids, self.decoders);
        let execution = ParallelExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            max_concurrency: self.max_concurrency,
            completion: self.completion,
            serdes,
            encoding: ItemEncoding::PerItem(self.json),
            result_serdes: self.result_serdes,
            nesting: self.nesting,
            branches: self.branches,
            item_ids,
        };

        if started {
            DurableFuture::for_started_operation(execution.execute_batch_result_started())
        } else {
            DurableFuture::for_start_step(move || execution.execute_batch_result_started())
        }
    }
}

impl<RS> IntoFuture for ParallelMixedBuilder<RS>
where
    RS: Serdes<String>,
{
    type Output = Result<CompletionReason, OperationError>;
    type IntoFuture = DurableFuture<CompletionReason>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::map_parallel::publish_outcome;

        let decoders = Arc::clone(&self.decoders);
        let batch = self.batch_future();
        // `map_output` keeps the batch's task context, so a
        // combinator that starts it runs the branches inside the frames of
        // the batch's creation.
        batch.map_output(move |outcome| publish_outcome(outcome, &decoders))
    }
}

/// A shared completion predicate consulted by map/parallel batch execution.
///
/// Receives a [`BatchStats`] snapshot before each item starts and after
/// each item settles, and returns a [`CompletionDecision`].
///
/// Crate-internal: the `Arc` wrapping is an implementation detail (it is what
/// keeps [`CompletionConfig`] cheaply cloneable with a closure inside, the
/// same way per-operation serdes are stored). Public setters
/// ([`CompletionConfig::with_completion_predicate`],
/// [`CompletionConfigBuilder::completion_predicate`]) take a generic closure
/// and wrap it here.
pub(crate) type CompletionPredicate =
    Arc<dyn Fn(&BatchStats<'_>) -> CompletionDecision + Send + Sync>;

/// Configuration for completion behavior in map and parallel operations.
///
/// Controls early-completion thresholds: how many items must succeed and
/// how many failures are tolerated before stopping. Thresholds can be
/// combined, and the first to fire wins. A
/// [completion predicate](Self::with_completion_predicate) cannot be
/// combined with a threshold: [`CompletionConfigBuilder::build`] rejects
/// the mix.
///
/// # Default: fail fast
///
/// An **empty** config: [`CompletionConfig::default()`], with no threshold
/// and no predicate: fails the batch on the **first item failure**, exactly
/// as running with no config at all does. This matches the Python and JS
/// SDKs. Configuring any criterion opts out of that implicit fail-fast and
/// hands the decision to the configured criteria:
///
/// - explicit fail-fast: [`with_tolerated_failure_count(0)`](Self::with_tolerated_failure_count)
/// - tolerate all failures: [`with_tolerated_failure_percentage(100.0)`](Self::with_tolerated_failure_percentage)
///
/// Construct with [`CompletionConfig::builder`] (which combines thresholds
/// and validates them at [`build`](CompletionConfigBuilder::build) time) or
/// with one of the single-threshold constructors. Read values back
/// through the accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::map_parallel::{
///     CompletionConfig, CompletionDecision, CompletionOutcome,
/// };
///
/// // Fail-fast: stop on the first failure.
/// let fail_fast = CompletionConfig::with_tolerated_failure_count(0);
/// assert_eq!(fail_fast.tolerated_failure_count(), Some(0));
///
/// // Early completion: stop after 2 successes.
/// let min_success = CompletionConfig::with_min_successful(2);
/// assert_eq!(min_success.min_successful(), Some(2));
///
/// // Custom predicate: complete as a success once two items have settled.
/// let custom = CompletionConfig::with_completion_predicate(|stats| {
///     if stats.settled() >= 2 {
///         CompletionDecision::Complete(CompletionOutcome::Succeeded)
///     } else {
///         CompletionDecision::Continue
///     }
/// });
/// assert!(custom.has_completion_predicate());
///
/// // Combined thresholds: first to fire wins.
/// let combined = CompletionConfig::builder()
///     .min_successful(2)
///     .tolerated_failure_count(1)
///     .build()?;
/// assert_eq!(combined.min_successful(), Some(2));
/// assert_eq!(combined.tolerated_failure_count(), Some(1));
/// # Ok::<(), aws_durable_execution_sdk::builders::map_parallel::CompletionConfigValidationError>(())
/// ```
#[derive(Clone, Default)]
#[non_exhaustive]
// Fields are private so the struct can grow without breaking callers;
// see the Rust API Guidelines on structs with private fields:
// https://rust-lang.github.io/api-guidelines/future-proofing.html#c-struct-private
pub struct CompletionConfig {
    /// Completes the batch early once this many items succeed.
    /// `None` means no minimum-success threshold.
    min_successful: Option<usize>,

    /// Fails the batch once more than this many items fail.
    /// `Some(0)` means fail-fast (stop on first failure).
    /// `None` means no count-based failure tolerance.
    tolerated_failure_count: Option<usize>,

    /// Fails the batch once the failure percentage strictly exceeds this
    /// threshold (0.0 to 100.0, fractions allowed). Uses
    /// cross-multiplication rather than a division.
    /// `Some(0.0)` means fail-fast (stop on first failure).
    /// `None` means no percentage-based failure tolerance.
    tolerated_failure_percentage: Option<f64>,

    /// User-supplied completion predicate over the batch view. It cannot be
    /// combined with a threshold. A `Complete` decision records
    /// [`CompletionReason::CustomCompletionSucceeded`] or
    /// [`CompletionReason::CustomCompletionFailed`].
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
    /// use aws_durable_execution_sdk::builders::map_parallel::CompletionConfig;
    ///
    /// let config = CompletionConfig::builder().min_successful(3).build()?;
    /// assert_eq!(config.min_successful(), Some(3));
    /// # Ok::<(), aws_durable_execution_sdk::builders::map_parallel::CompletionConfigValidationError>(())
    /// ```
    pub fn builder() -> CompletionConfigBuilder {
        CompletionConfigBuilder::default()
    }

    /// Creates a config with just a `min_successful` threshold.
    ///
    /// **Default when unset:** the batch runs until every item settles.
    /// Note that a config carrying only this threshold has no failure
    /// tolerance configured, so item failures are tolerated while the
    /// batch waits to reach `min` successes: the implicit fail-fast of an
    /// empty config no longer applies once any criterion is set.
    ///
    /// `with_min_successful(0)` completes a non-empty batch at its first
    /// settle, as in JS. A batch of one item then records
    /// [`CompletionReason::AllCompleted`], because the batch checks "every
    /// item finished" before the minimum.
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
    /// given percentage. It accepts a fraction such as `12.5`; the batch
    /// fails when the failure rate strictly exceeds it. Internally this uses
    /// cross-multiplication (`failure_count * 100.0 > pct * total_items`)
    /// rather than a division, so 7 failures of 50 do not exceed `14.0`.
    /// The check is exact for integer percentages and totals below 2^53.
    ///
    /// Use `0.0` for explicit fail-fast behavior (stop on first failure).
    /// Use `100.0` to tolerate all failures (the failure rate can never
    /// strictly exceed 100%). A value that is not finite or lies outside
    /// `0.0..=100.0` fails the batch at execute time.
    ///
    /// The percentage applies to `parallel` as it does to `map`, as in JS
    /// and Python `ParallelConfig`.
    ///
    /// **Default when unset:** an empty config fails fast on the first
    /// failure; a config with some other criterion set applies only that
    /// criterion.
    #[must_use]
    pub fn with_tolerated_failure_percentage(pct: f64) -> Self {
        Self {
            tolerated_failure_percentage: Some(pct),
            ..Self::default()
        }
    }

    /// Creates a config with just a custom completion predicate.
    ///
    /// The predicate receives a [`BatchStats`] snapshot and returns a
    /// [`CompletionDecision`]:
    ///
    /// - [`CompletionDecision::Continue`] keeps starting and awaiting items.
    ///   A batch whose predicate never completes it records
    ///   [`CompletionReason::AllCompleted`].
    /// - `Complete(CompletionOutcome::Succeeded)` ends the batch now and
    ///   records [`CompletionReason::CustomCompletionSucceeded`]. Item
    ///   failures inside it are tolerated rather than propagated as errors.
    /// - `Complete(CompletionOutcome::Failed)` ends the batch now and records
    ///   [`CompletionReason::CustomCompletionFailed`]. Awaiting the batch
    ///   returns the first failed item's error, or a
    ///   [`BatchCompletionFailed`](crate::ChildContextErrorKind::BatchCompletionFailed)
    ///   error when no item failed.
    ///
    /// A predicate counts as a completion criterion, so the implicit
    /// fail-fast of an empty config does not apply. A predicate cannot be
    /// combined with a threshold. A zero-item batch asks the predicate once,
    /// on an empty view.
    ///
    /// # When the predicate runs
    ///
    /// The SDK asks the predicate before each item starts and after each
    /// item settles. The counts follow the settle order, and
    /// [`BatchStats::items`] holds every input item by index. The SDK stops
    /// asking once the first trigger fires, so within one run a predicate is
    /// never asked to reverse a decision.
    ///
    /// # Determinism: read this before using
    ///
    /// The live settle order is not recorded. On resume, the recorded items
    /// settle first, in index order. So the predicate MUST be a pure function
    /// of the [`BatchStats`] it receives, and it MUST be monotonic. A
    /// predicate is monotonic when, after it returns `Complete(outcome)` for
    /// a snapshot, it returns `Complete(outcome)` with the same outcome for
    /// every later snapshot of the same batch. A later snapshot is one where
    /// each item's status is the same or further along, in the order `None`,
    /// `Started`, then `Succeeded` or `Failed`. Under that rule the recorded
    /// reason and outcome do not depend on the settle order, so two runs
    /// agree. See [`CompletionConfigBuilder::completion_predicate`] for a
    /// counterexample.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::builders::map_parallel::{
    ///     CompletionConfig, CompletionDecision, CompletionOutcome,
    /// };
    ///
    /// // Complete as a success once half the items have settled.
    /// let config = CompletionConfig::with_completion_predicate(|stats| {
    ///     if stats.settled() * 2 >= stats.total_items() {
    ///         CompletionDecision::Complete(CompletionOutcome::Succeeded)
    ///     } else {
    ///         CompletionDecision::Continue
    ///     }
    /// });
    /// assert!(config.has_completion_predicate());
    /// ```
    #[must_use]
    pub fn with_completion_predicate<F>(predicate: F) -> Self
    where
        F: Fn(&BatchStats<'_>) -> CompletionDecision + Send + Sync + 'static,
    {
        Self {
            completion_predicate: Some(Arc::new(predicate)),
            ..Self::default()
        }
    }

    /// Returns the minimum-success threshold, if set.
    pub fn min_successful(&self) -> Option<usize> {
        self.min_successful
    }

    /// Returns the count-based failure tolerance, if set.
    pub fn tolerated_failure_count(&self) -> Option<usize> {
        self.tolerated_failure_count
    }

    /// Returns the percentage-based failure tolerance, if set.
    pub fn tolerated_failure_percentage(&self) -> Option<f64> {
        self.tolerated_failure_percentage
    }

    /// Reports whether a custom completion predicate is set.
    ///
    /// The predicate itself is not exposed: its boxing is an implementation
    /// detail, matching how the crate stores other user closures (for
    /// example retry strategies).
    pub fn has_completion_predicate(&self) -> bool {
        self.completion_predicate.is_some()
    }

    /// Reports whether ANY completion criterion is configured: a threshold
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

    /// Calls the predicate. Returns `CompletionDecision::Continue` when none is set.
    pub(crate) fn predicate_decision(&self, stats: &BatchStats<'_>) -> CompletionDecision {
        self.completion_predicate
            .as_ref()
            .map_or(CompletionDecision::Continue, |predicate| predicate(stats))
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
        check_percentage(self.tolerated_failure_percentage)
    }
}

/// Checks the failure percentage: it must be finite and lie in
/// `0.0..=100.0`. Returns the message both validation paths report.
fn check_percentage(pct: Option<f64>) -> Result<(), String> {
    match pct {
        Some(pct) if !(pct.is_finite() && (0.0..=100.0).contains(&pct)) => Err(format!(
            "tolerated_failure_percentage must be 0-100, got {pct}"
        )),
        _ => Ok(()),
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
/// use aws_durable_execution_sdk::builders::map_parallel::CompletionConfig;
///
/// let err = CompletionConfig::builder()
///     .tolerated_failure_percentage(101.0)
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
/// use aws_durable_execution_sdk::builders::map_parallel::CompletionConfig;
///
/// let config = CompletionConfig::builder()
///     .min_successful(2)
///     .tolerated_failure_percentage(25.0)
///     .build()?;
/// assert_eq!(config.tolerated_failure_percentage(), Some(25.0));
/// # Ok::<(), aws_durable_execution_sdk::builders::map_parallel::CompletionConfigValidationError>(())
/// ```
#[derive(Clone, Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct CompletionConfigBuilder {
    min_successful: Option<usize>,
    tolerated_failure_count: Option<usize>,
    tolerated_failure_percentage: Option<f64>,
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
    ///
    /// `min_successful(0)` completes a non-empty batch at its first settle,
    /// as in JS. A batch of one item then records
    /// [`CompletionReason::AllCompleted`].
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
    /// threshold (0.0 to 100.0).
    ///
    /// It accepts a fraction such as `12.5`; the batch fails when the
    /// failure rate strictly exceeds it. Use `0.0` for explicit fail-fast
    /// behavior (stop on first failure). Use `100.0` to tolerate all
    /// failures (the failure rate can never strictly exceed 100%).
    ///
    /// **Default when unset:** an empty config fails fast on the first
    /// failure; a config with some other criterion set applies only that
    /// criterion.
    ///
    /// A value that is not finite or lies outside `0.0..=100.0` is rejected
    /// by [`build`](Self::build).
    ///
    /// The percentage applies to `parallel` as it does to `map`, as in JS
    /// and Python `ParallelConfig`.
    pub fn tolerated_failure_percentage(mut self, pct: f64) -> Self {
        self.tolerated_failure_percentage = Some(pct);
        self
    }

    /// Sets a custom completion predicate over the batch view.
    ///
    /// The predicate receives a [`BatchStats`] snapshot and returns a
    /// [`CompletionDecision`]. `Continue` keeps the batch running, and a
    /// batch whose predicate never completes it records
    /// [`CompletionReason::AllCompleted`].
    /// `Complete(CompletionOutcome::Succeeded)` ends the batch and records
    /// [`CompletionReason::CustomCompletionSucceeded`].
    /// `Complete(CompletionOutcome::Failed)` ends the batch and records
    /// [`CompletionReason::CustomCompletionFailed`]; awaiting the batch then
    /// returns an error. A predicate counts as a completion criterion, so
    /// the implicit fail-fast of an empty config does not apply. A
    /// zero-item batch asks the predicate once, on an empty view.
    ///
    /// A predicate cannot be combined with `min_successful`,
    /// `tolerated_failure_count` or `tolerated_failure_percentage`:
    /// [`build`](Self::build) rejects the mix, as JS does.
    ///
    /// # When the predicate runs
    ///
    /// The SDK asks the predicate before each item starts and after each
    /// item settles. The counts follow the settle order, and
    /// [`BatchStats::items`] holds every input item by index, with its name
    /// and status. The SDK calls the predicate only until the first trigger
    /// fires, so a predicate is never asked to reverse a decision within one
    /// run.
    ///
    /// # Determinism: read this before using
    ///
    /// The live settle order is not recorded. On resume, the recorded items
    /// settle first, in index order. So the sequence of snapshots can differ
    /// between the original run and a resume.
    ///
    /// So the predicate MUST be a pure function of the [`BatchStats`] it
    /// receives, and it MUST be monotonic. A predicate is monotonic when,
    /// after it returns `Complete(outcome)` for a snapshot, it returns
    /// `Complete(outcome)` with the same outcome for every later snapshot of
    /// the same batch. A later snapshot is one where each item's status is
    /// the same or further along, in the order `None`, `Started`, then
    /// `Succeeded` or `Failed`. Under that rule the recorded reason and
    /// outcome do not depend on the settle order. Monotonicity is what keeps
    /// two runs in agreement.
    ///
    /// A counterexample, as JS gives one: `succeeded() == 1 && failed() == 0`
    /// is not monotonic. One run can see one success first and complete. A
    /// resume can see the failure first, then `(1, 1)`, and never complete.
    /// See JS's own advice in
    /// [batch.ts](https://github.com/aws/aws-durable-execution-sdk-js/blob/67549feacd5568ea305b7c2d533d3549974f8c9c/packages/aws-durable-execution-sdk-js/src/types/batch.ts#L276-L282).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::builders::map_parallel::{
    ///     BatchItemStatus, CompletionConfig, CompletionDecision, CompletionOutcome,
    /// };
    ///
    /// // Complete when item 0 succeeds, or when items 1 and 2 both succeed.
    /// // The rule stays met once it is met, so it is monotonic.
    /// let config = CompletionConfig::builder()
    ///     .completion_predicate(|stats| {
    ///         let ok = |i: usize| {
    ///             stats
    ///                 .items()
    ///                 .get(i)
    ///                 .is_some_and(|item| item.status == Some(BatchItemStatus::Succeeded))
    ///         };
    ///         if ok(0) || (ok(1) && ok(2)) {
    ///             CompletionDecision::Complete(CompletionOutcome::Succeeded)
    ///         } else {
    ///             CompletionDecision::Continue
    ///         }
    ///     })
    ///     .build()?;
    /// assert!(config.has_completion_predicate());
    /// # Ok::<(), aws_durable_execution_sdk::builders::map_parallel::CompletionConfigValidationError>(())
    /// ```
    pub fn completion_predicate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&BatchStats<'_>) -> CompletionDecision + Send + Sync + 'static,
    {
        self.completion_predicate = Some(Arc::new(predicate));
        self
    }

    /// Builds the [`CompletionConfig`] from the configured thresholds,
    /// validating them at construction time.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionConfigValidationError`] in two cases, checked in
    /// this order:
    ///
    /// 1. `tolerated_failure_percentage` is not finite or lies outside
    ///    `0.0..=100.0`.
    /// 2. A completion predicate is set together with `min_successful`,
    ///    `tolerated_failure_count` or `tolerated_failure_percentage`. The
    ///    message is `completion_predicate is mutually exclusive with
    ///    min_successful, tolerated_failure_count and
    ///    tolerated_failure_percentage`.
    ///
    /// Item-count-dependent checks (for example `min_successful` against
    /// the actual number of items) happen at execute time, where the item
    /// count is first known.
    pub fn build(self) -> Result<CompletionConfig, CompletionConfigValidationError> {
        check_percentage(self.tolerated_failure_percentage)
            .map_err(|message| CompletionConfigValidationError { message })?;
        if self.completion_predicate.is_some()
            && (self.min_successful.is_some()
                || self.tolerated_failure_count.is_some()
                || self.tolerated_failure_percentage.is_some())
        {
            return Err(CompletionConfigValidationError {
                message: "completion_predicate is mutually exclusive with min_successful, \
                          tolerated_failure_count and tolerated_failure_percentage"
                    .to_owned(),
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

    /// The `CompletionConfig` builder combines thresholds in one expression,
    /// no `Default`-then-mutate, and the built config drives the same
    /// completion decisions the single-threshold constructors do.
    #[test]
    fn completion_config_builder_combines_thresholds() {
        let combined = CompletionConfig::builder()
            .min_successful(2)
            .tolerated_failure_count(1)
            .tolerated_failure_percentage(25.0)
            .build()
            .expect("valid config");

        assert_eq!(combined.min_successful(), Some(2));
        assert_eq!(combined.tolerated_failure_count(), Some(1));
        assert_eq!(combined.tolerated_failure_percentage(), Some(25.0));
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
            .tolerated_failure_percentage(101.0)
            .build()
            .expect_err("a percentage above 100 must be rejected at build time");
        assert!(
            err.to_string().contains("tolerated_failure_percentage"),
            "error should name the offending field, got: {err}"
        );

        // The boundary value 100 is accepted.
        assert!(
            CompletionConfig::builder()
                .tolerated_failure_percentage(100.0)
                .build()
                .is_ok()
        );

        // The constructor path is still guarded by the execute-time check.
        let constructed = CompletionConfig::with_tolerated_failure_percentage(101.0);
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
            .completion_predicate(|stats| {
                if stats.settled() >= 3 {
                    CompletionDecision::Complete(CompletionOutcome::Succeeded)
                } else {
                    CompletionDecision::Continue
                }
            })
            .build()
            .expect("valid config");
        assert!(config.has_completion_predicate());

        let cloned = config.clone();
        assert!(cloned.has_completion_predicate());
        assert_eq!(cloned.min_successful(), None);

        let debugged = format!("{config:?}");
        assert!(
            debugged.contains("completion_predicate") && debugged.contains("<closure>"),
            "Debug should report predicate presence, got: {debugged}"
        );

        // Constructor parity: `with_completion_predicate` sets only the
        // predicate.
        let only_predicate = CompletionConfig::with_completion_predicate(|stats| {
            if stats.failed() > stats.succeeded() {
                CompletionDecision::Complete(CompletionOutcome::Succeeded)
            } else {
                CompletionDecision::Continue
            }
        });
        assert!(only_predicate.has_completion_predicate());
        assert_eq!(only_predicate.min_successful(), None);

        // The predicate is evaluated against the stats it receives.
        let more_failed = [
            CompletionItem::new(0, None, Some(BatchItemStatus::Failed)),
            CompletionItem::new(1, None, Some(BatchItemStatus::Failed)),
            CompletionItem::new(2, None, Some(BatchItemStatus::Succeeded)),
        ];
        let stats = BatchStats::new(&more_failed);
        assert_eq!(
            only_predicate.predicate_decision(&stats),
            CompletionDecision::Complete(CompletionOutcome::Succeeded)
        );
        let even = [
            CompletionItem::new(0, None, Some(BatchItemStatus::Failed)),
            CompletionItem::new(1, None, Some(BatchItemStatus::Succeeded)),
        ];
        assert_eq!(
            only_predicate.predicate_decision(&BatchStats::new(&even)),
            CompletionDecision::Continue
        );
        // No predicate configured: the decision is `Continue`.
        assert_eq!(
            CompletionConfig::default().predicate_decision(&stats),
            CompletionDecision::Continue
        );
    }

    fn complete_on_first_settle(stats: &BatchStats<'_>) -> CompletionDecision {
        if stats.settled() >= 1 {
            CompletionDecision::Complete(CompletionOutcome::Succeeded)
        } else {
            CompletionDecision::Continue
        }
    }

    /// `build()` rejects a predicate together with `min_successful`, as JS
    /// rejects `shouldComplete` with a threshold.
    #[test]
    fn build_rejects_predicate_with_min_successful() {
        let err = CompletionConfig::builder()
            .min_successful(1)
            .completion_predicate(complete_on_first_settle)
            .build()
            .expect_err("a predicate with min_successful must be rejected");
        assert!(
            err.to_string().contains("mutually exclusive"),
            "unexpected message: {err}"
        );
        assert_eq!(
            err.to_string(),
            "invalid CompletionConfig: completion_predicate is mutually exclusive with \
             min_successful, tolerated_failure_count and tolerated_failure_percentage"
        );
    }

    /// `build()` rejects a predicate together with `tolerated_failure_count`.
    #[test]
    fn build_rejects_predicate_with_failure_count() {
        let result = CompletionConfig::builder()
            .tolerated_failure_count(0)
            .completion_predicate(complete_on_first_settle)
            .build();
        assert!(result.is_err());
    }

    /// `build()` rejects a predicate together with
    /// `tolerated_failure_percentage`.
    #[test]
    fn build_rejects_predicate_with_failure_percentage() {
        let result = CompletionConfig::builder()
            .tolerated_failure_percentage(10.0)
            .completion_predicate(complete_on_first_settle)
            .build();
        assert!(result.is_err());
    }

    /// A predicate alone is a valid config.
    #[test]
    fn build_accepts_predicate_alone() {
        let config = CompletionConfig::builder()
            .completion_predicate(complete_on_first_settle)
            .build()
            .expect("a predicate alone is valid");
        assert!(config.has_completion_predicate());
    }

    /// `build()` checks the percentage before the predicate exclusion, so a
    /// config that breaks both rules reports the percentage.
    #[test]
    fn build_checks_percentage_before_predicate_exclusion() {
        let err = CompletionConfig::builder()
            .tolerated_failure_percentage(150.0)
            .completion_predicate(complete_on_first_settle)
            .build()
            .expect_err("both rules are broken");
        assert!(
            err.to_string()
                .contains("tolerated_failure_percentage must be 0-100"),
            "unexpected message: {err}"
        );
    }

    /// A context whose log holds a succeeded STEP record at position 1,
    /// where the batch under test claims a `Context` operation.
    fn ctx_with_step_record_at_1() -> DurableContext {
        use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};

        let wire = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire.clone(),
            status: CheckpointStatus::Succeeded,
            result: Some("0".to_owned()),
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
        DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(vec![(wire, record)])),
        )
    }

    /// `batch_future` checks the replay identity when it is called, before
    /// the returned future is polled, as `into_future` does. The log holds a
    /// STEP where the map claims a `Context`. So the fatal mismatch is
    /// recorded at once. Without the preflight, the mismatch is recorded only
    /// when the batch first runs.
    #[tokio::test]
    async fn map_batch_future_checks_replay_identity_before_first_poll() {
        let ctx = ctx_with_step_record_at_1();
        let future = ctx
            .map(
                vec![1_i32],
                |_child, item: i32, _idx| async move { Ok(item) },
            )
            .batch_future();
        let fatal = ctx
            .coordinator()
            .fatal_error()
            .expect("the mismatch is recorded before the first poll");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        drop(future);
    }

    /// The parallel counterpart of
    /// `map_batch_future_checks_replay_identity_before_first_poll`.
    #[tokio::test]
    async fn parallel_batch_future_checks_replay_identity_before_first_poll() {
        let ctx = ctx_with_step_record_at_1();
        let branches = vec![crate::future::Branch::new("b", |_child| async {
            Ok(1_i32)
        })];
        let future = ctx.parallel(branches).batch_future();
        let fatal = ctx
            .coordinator()
            .fatal_error()
            .expect("the mismatch is recorded before the first poll");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        drop(future);
    }

    /// A `BranchSet` with one `i32` branch, for the `parallel_mixed` tests.
    fn one_branch_set() -> crate::BranchSet {
        let mut set = crate::BranchSet::new();
        let _handle = set.add("b", |_child| async { Ok(1_i32) });
        set
    }

    /// `parallel_mixed`'s `into_future` checks the replay identity when it is
    /// called, before the returned future is polled. The log holds a STEP
    /// where the parallel claims a `Context`. So the fatal mismatch is
    /// recorded at once. If `into_future` called `batch_future` inside the
    /// returned async block, the mismatch would be recorded only at the first
    /// poll, and this test would fail.
    #[tokio::test]
    async fn parallel_mixed_into_future_checks_replay_identity_before_first_poll() {
        let ctx = ctx_with_step_record_at_1();
        let future = ctx.parallel_mixed(one_branch_set()).into_future();
        let fatal = ctx
            .coordinator()
            .fatal_error()
            .expect("the mismatch is recorded before the first poll");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        drop(future);
    }

    /// The `.spawn()` counterpart of
    /// `parallel_mixed_into_future_checks_replay_identity_before_first_poll`.
    #[tokio::test]
    async fn parallel_mixed_spawn_checks_replay_identity_before_first_poll() {
        let ctx = ctx_with_step_record_at_1();
        let future = ctx.parallel_mixed(one_branch_set()).spawn();
        let fatal = ctx
            .coordinator()
            .fatal_error()
            .expect("the mismatch is recorded before the first poll");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        drop(future);
    }
}
