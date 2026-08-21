//! The retrying child-context operation: [`WithRetryBuilder`], returned by
//! [`DurableContext::with_retry`](crate::DurableContext::with_retry).

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::BoxError;
use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

// ============================================================
// WithRetryBuilder
// ============================================================

/// Builder for a block-level retry operation.
///
/// Created by [`DurableContext::with_retry`]. Runs a closure against a
/// child context and applies a retry strategy to the closure's **overall**
/// outcome, so a multi-operation block retries as a unit. Each attempt
/// receives a fresh child operation namespace: a failed attempt's recorded
/// operations are never replayed into the next attempt, so every operation
/// in the block re-runs on retry. Delays between attempts suspend the
/// execution (the backend owns the timer), and the retry progress itself is
/// derived from checkpointed results, so it survives suspension.
///
/// Without an explicit strategy the block uses the same default a step
/// uses: 6 total attempts with exponential backoff (5s initial delay, 60s
/// cap, factor 2).
///
/// On the wire the block appears as a child context (sub-type
/// `RunInChildContext`) containing one nested child context per attempt
/// (named `attempt-N`) and one wait per retry delay (named
/// `retry-delay-N`). When retries exhaust, the operation fails with a
/// [`ChildContextError`](crate::ChildContextError) whose message carries
/// the attempt count and the last attempt's error.
///
/// The builder is generic over the block closure `F` and its future `Fut`
/// so the body is stored **without type erasure**; both parameters are
/// inferred at the call site and never written by users. The single
/// erasure point is `.future()` / `.await`, which produces the one
/// [`DurableFuture`] box.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use aws_durable_execution_sdk_rust::RetryDecision;
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
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
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
    /// Sets a human-readable name for this operation.
    ///
    /// Names appear in logs, traces, and the execution history for
    /// debugging purposes.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
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
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::RetryDecision;
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
    /// Use this for the common case — shaping retry delays (attempt count,
    /// initial delay, cap, backoff rate, jitter) — without hand-writing a
    /// closure. For decisions a delay schedule cannot express, such as
    /// inspecting the error, use [`retry_strategy`](Self::retry_strategy)
    /// instead.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::builders::RetryStrategyConfig;
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
    /// implement [`Serdes<O>`](crate::Serdes) for this block's output type —
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
            retry_strategy: self.retry_strategy,
            serdes,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for fan-out
    /// patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<O>
    where
        S: Serdes<O>,
    {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the retry block on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running. The operation ID
    /// was already claimed at builder creation, so spawn order does not
    /// affect replay.
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

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::child::ChildExecution;

        // The block is a child context on the wire; validate against the
        // recorded identity exactly as `run_in_child_context` does.
        preflight_identity!(self, "Context", crate::child::CHILD_SUB_TYPE);

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

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            serdes,
            // A concrete closure returning the concrete retry-loop future:
            // no erasure here — the one box is the DurableFuture below.
            closure: move |outer_ctx| {
                crate::with_retry::retry_loop(outer_ctx, closure, strategy, loop_serdes)
            },
            _marker: PhantomData,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}
