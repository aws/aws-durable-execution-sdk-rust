//! The retrying child-context operation: [`WithRetryBuilder`], returned by
//! [`DurableContext::with_retry`](crate::DurableContext::with_retry).
//!
//! `with_retry` composes the
//! [child context operation](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/child-context/)
//! with step-style retries; it has no separate page in the
//! language-agnostic operation guide.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::BoxError;
use crate::OperationSubType;
use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

/// Builder for a block-level retry operation.
///
/// Created by [`DurableContext::with_retry`]. Runs a closure against a
/// child context and applies a retry strategy to the closure's **overall**
/// outcome, so a multi-operation block retries as a unit. Each attempt
/// starts clean: nothing recorded by a failed attempt is replayed into the
/// next one, so every operation in the block re-runs on retry. Delays
/// between attempts suspend the execution, and retry progress survives
/// suspension.
///
/// Without an explicit strategy the block uses the same default a step
/// uses: 6 total attempts with exponential backoff (5s initial delay, 60s
/// cap, factor 2).
///
/// When retries exhaust, the operation fails with a
/// [`ChildContextError`](crate::ChildContextError) whose message carries
/// the attempt count and the last attempt's error.
///
/// The builder is generic over the block closure `F` and its future
/// `Fut`; both parameters are inferred at the call site and never written
/// by users.
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
///     let result = ctx.with_retry(|child| async move {
///         let a = child.step(|_| async { Ok(1_u32) }).name("fetch").await?;
///         let b = child.step(move |_| async move { Ok(a + 1) }).name("apply").await?;
///         Ok(format!("{a}+{b}"))
///     })
///     .name("fetch-and-apply")
///     .retry_strategy(|_err, attempt| {
///         if attempt >= 3 {
///             RetryDecision::Stop
///         } else {
///             RetryDecision::Retry { delay: Duration::from_secs(2) }
///         }
///     })
///     .await?;
///     Ok(result)
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct WithRetryBuilder<O, F, Fut, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    sub_type: Option<OperationSubType>,
    retry_strategy: Option<RetryStrategy>,
    serdes: S,
    closure: F,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<O, F, Fut, S> std::fmt::Debug for WithRetryBuilder<O, F, Fut, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithRetryBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O, F, Fut> WithRetryBuilder<O, F, Fut>
where
    O: Send + 'static,
    F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the closure here keeps the
    /// field non-optional: a builder without a body is unrepresentable.
    pub(crate) fn new_internal(ctx: DurableContext, op_id: OperationId, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            sub_type: None,
            retry_strategy: None,
            serdes: JsonSerdes,
            closure,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> WithRetryBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this block.
    ///
    /// A non-empty name also names each wait between attempts
    /// `{name}-backoff-{attempt}`. Without a name, or with an empty name,
    /// those waits are unnamed. Names appear in logs, traces, and execution
    /// history for debugging.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Records `sub_type` instead of the default subtype.
    ///
    /// The subtype labels this block's child context in its checkpoint, in
    /// place of `RunInChildContext`. So the blocks of an operation composed
    /// from several durable operations can be told apart. Execution history
    /// and plugins read it. Without this call the block records
    /// [`OperationSubType::RUN_IN_CHILD_CONTEXT`]. A second call replaces
    /// the first.
    ///
    /// The subtype labels the block, not its attempts. Each attempt's
    /// child context `attempt-N` keeps `RunInChildContext`, each delay
    /// keeps `Wait`, and every operation inside an attempt keeps its own
    /// subtype. A delay is named `{name}-backoff-N` for a block with a
    /// non-empty name, and is unnamed for an unnamed block.
    ///
    /// Replay compares the recorded subtype with this one, as it compares
    /// the name. So changing the subtype of an existing block breaks the
    /// executions in flight: their replay fails with
    /// `NonDeterministicExecutionError`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use aws_durable_execution_sdk::OperationSubType;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u32, durable::BoxError> {
    ///     let total = ctx
    ///         .with_retry(|child| async move {
    ///             let v = child.step(|_| async { Ok(42_u32) }).name("work").await?;
    ///             Ok(v)
    ///         })
    ///         .name("block")
    ///         .sub_type(OperationSubType::new("Checkout")?)
    ///         .await?;
    ///     Ok(total)
    /// }
    /// ```
    pub fn sub_type(mut self, sub_type: OperationSubType) -> Self {
        self.sub_type = Some(sub_type);
        self
    }

    /// Sets the retry strategy for this block.
    ///
    /// The strategy closure is called after each failed attempt with the
    /// block's error and the 1-based attempt number, and decides whether to
    /// retry the whole block. It takes the same shape a step's strategy
    /// does, and the same determinism rule applies: the decision must be a
    /// pure function of the error and the attempt number, because it is
    /// re-evaluated against the recorded error during replay.
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
    /// ) -> Result<u32, durable::BoxError> {
    ///     let result = ctx.with_retry(|child| async move {
    ///         let v = child.step(|_| async { Ok(42_u32) }).await?;
    ///         Ok(v)
    ///     })
    ///     .retry_strategy(|_err, attempt| {
    ///         if attempt >= 3 {
    ///             RetryDecision::Stop
    ///         } else {
    ///             RetryDecision::Retry { delay: Duration::from_secs(1) }
    ///         }
    ///     })
    ///     .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn retry_strategy<R>(mut self, strategy: R) -> Self
    where
        R: Fn(&crate::StepError, u32) -> crate::RetryDecision + Send + Sync + 'static,
    {
        self.retry_strategy = Some(Box::new(strategy));
        self
    }

    /// Sets the retry strategy for this block from a
    /// [`RetryStrategyConfig`](crate::builders::RetryStrategyConfig).
    ///
    /// Use this for the common case, shaping retry delays (attempt count,
    /// initial delay, cap, backoff rate, jitter), without hand-writing a
    /// closure. For decisions a delay schedule cannot express, such as
    /// inspecting the error, use [`retry_strategy`](Self::retry_strategy)
    /// instead.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u32, durable::BoxError> {
    ///     let result = ctx.with_retry(|child| async move {
    ///         let v = child.step(|_| async { Ok(42_u32) }).await?;
    ///         Ok(v)
    ///     })
    ///     .retry_strategy_config(
    ///         RetryStrategyConfig::builder()
    ///             .max_attempts(3)
    ///             .initial_delay(Duration::from_secs(1))
    ///             .build(),
    ///     )
    ///     .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn retry_strategy_config(mut self, config: crate::builders::RetryStrategyConfig) -> Self {
        self.retry_strategy = Some(config.into_retry_strategy());
        self
    }

    /// Overrides the serialization strategy for the block's result.
    ///
    /// Replaces the builder's serdes type parameter with `S2`, which must
    /// implement [`Serdes<O>`](crate::Serdes) for this block's output type:
    /// attaching a serdes for a different type fails at compile time. To
    /// share one instance across operations, wrap it in an
    /// [`Arc`] and clone the `Arc` handle into each builder.
    pub fn serdes<S2>(self, serdes: S2) -> WithRetryBuilder<O, F, Fut, S2>
    where
        S2: Serdes<O>,
    {
        WithRetryBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            sub_type: self.sub_type,
            retry_strategy: self.retry_strategy,
            serdes,
            closure: self.closure,
            _marker: PhantomData,
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
    pub fn future(self) -> DurableFuture<O>
    where
        S: Serdes<O>,
    {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the retry block on a tokio task.
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
    /// # Errors
    ///
    /// When no Tokio runtime is running, the returned future resolves to an
    /// error and the operation records nothing.
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
    ///     let handle = ctx.with_retry(|child| async move {
    ///         let v = child.step(|_| async { Ok(1_u32) }).await?;
    ///         Ok(v)
    ///     }).name("bg").spawn();
    ///     let _result = handle.await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn spawn(self) -> DurableFuture<O>
    where
        S: Serdes<O>,
    {
        spawn_terminal!(self)
    }
}

impl<O, F, Fut, S> IntoFuture for WithRetryBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::child::ChildExecution;

        // The block is a child context on the wire; validate against the
        // recorded identity exactly as `run_in_child_context` does.
        preflight_identity!(
            self,
            "Context",
            self.sub_type
                .as_ref()
                .map_or(crate::child::CHILD_SUB_TYPE, OperationSubType::as_str)
        );

        let closure = Arc::new(self.closure);
        let strategy: Arc<RetryStrategy> = Arc::new(
            self.retry_strategy
                .unwrap_or_else(crate::step::default_retry_strategy),
        );
        // One instance serves both the outer block result and every
        // attempt's round trip, shared through the forwarding
        // `impl Serdes<T> for Arc<S>`.
        let serdes = Arc::new(self.serdes);
        let loop_serdes = Arc::clone(&serdes);

        let gate = self.ctx.clone();

        let loop_name = self.name.clone();

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            sub_type: self
                .sub_type
                .unwrap_or(OperationSubType::RUN_IN_CHILD_CONTEXT),
            serdes,
            // A concrete closure returning the concrete retry-loop future:
            // no erasure here: the one box is the DurableFuture below.
            closure: move |outer_ctx| {
                crate::with_retry::retry_loop(outer_ctx, loop_name, closure, strategy, loop_serdes)
            },
            _marker: PhantomData,
        };

        DurableFuture::for_operation(&gate, execution.execute())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The end of a `with_retry` block closes the block's own scope: a step
    /// started with `.spawn()` directly on the block's context stops,
    /// although its handle outlives the block.
    #[tokio::test(start_paused = true)]
    async fn with_retry_block_end_stops_direct_block_task() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        use crate::child::ChildExecution;
        use crate::driver::InvocationOutcome;
        use crate::driver::test_support::{RefreshClient, drive_handler, refresh_ctx, wire_err};

        let client = Arc::new(RefreshClient::new(|_| Ok(Vec::new())));
        let ctx = refresh_ctx(&client);
        let effects = Arc::new(AtomicUsize::new(0));
        let block_effects = Arc::clone(&effects);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let holder: Arc<Mutex<Vec<DurableFuture<()>>>> = Arc::new(Mutex::new(Vec::new()));
            let block_holder = Arc::clone(&holder);
            let attempt = Arc::new(|_attempt_ctx: DurableContext| async { Ok::<(), BoxError>(()) });
            // The outer child execution `WithRetryBuilder` builds, with a
            // closure that starts `escaped` on the block's own context
            // before it runs the retry loop.
            let execution = ChildExecution {
                ctx: ctx.clone(),
                op_id: ctx.claim_operation_id("with_retry"),
                name: Some("block".to_owned()),
                sub_type: OperationSubType::RUN_IN_CHILD_CONTEXT,
                serdes: Arc::new(JsonSerdes),
                closure: move |outer_ctx: DurableContext| async move {
                    let handle = outer_ctx
                        .step(move |_| async move {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            block_effects.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })
                        .name("escaped")
                        .spawn();
                    block_holder.lock().expect("lock").push(handle);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    crate::with_retry::retry_loop(
                        outer_ctx,
                        Some("block".to_owned()),
                        attempt,
                        Arc::new(crate::step::default_retry_strategy()),
                        Arc::new(JsonSerdes),
                    )
                    .await
                },
                _marker: PhantomData,
            };
            execution.execute().await.map_err(wire_err)?;
            ctx.step(|_| async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok(())
            })
            .name("keep")
            .await
            .map_err(wire_err)?;
            drop(holder);
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(
            effects.load(Ordering::SeqCst),
            0,
            "the escaped body stopped"
        );
    }

    /// A second `sub_type` call replaces the first, and `serdes` keeps
    /// the subtype.
    #[test]
    fn sub_type_setter_replaces_and_survives_serdes() {
        let ctx = DurableContext::test_context();
        let builder = ctx
            .with_retry(|_child| async { Ok(1_u32) })
            .sub_type(OperationSubType::new("First").unwrap())
            .sub_type(OperationSubType::new("Second").unwrap())
            .serdes(Arc::new(JsonSerdes));
        assert_eq!(
            builder.sub_type.as_ref().map(OperationSubType::as_str),
            Some("Second")
        );
    }
}
