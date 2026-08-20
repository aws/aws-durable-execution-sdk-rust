//! The retrying child-context operation: [`WithRetryBuilder`], returned by
//! [`DurableContext::with_retry`](crate::DurableContext::with_retry).

use std::future::IntoFuture;
use std::sync::Arc;

use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;

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
pub struct WithRetryBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: Option<Arc<dyn Serdes>>,
    closure: crate::with_retry::WithRetryClosure<O>,
}

impl<O> std::fmt::Debug for WithRetryBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithRetryBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> WithRetryBuilder<O> {
    /// Creates a new builder (internal). Taking the closure here keeps the
    /// field non-optional: a builder without a body is unrepresentable.
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        closure: crate::with_retry::WithRetryClosure<O>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: None,
            closure,
        }
    }

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
    pub fn retry_strategy<F>(mut self, strategy: F) -> Self
    where
        F: Fn(&crate::StepError, u32) -> crate::RetryDecision + Send + Sync + 'static,
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
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Arc::new(serdes));
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for fan-out
    /// patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<O>
    where
        O: serde::Serialize + serde::de::DeserializeOwned,
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
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for WithRetryBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::child::ChildExecution;

        // The block is a child context on the wire; validate against the
        // recorded identity exactly as `run_in_child_context` does.
        preflight_identity!(self, "Context", crate::child::CHILD_SUB_TYPE);

        let closure = self.closure;
        let strategy: Arc<RetryStrategy> = Arc::new(
            self.retry_strategy
                .unwrap_or_else(crate::step::default_retry_strategy),
        );

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            serdes: self.serdes,
            closure: Box::new(move |outer_ctx| {
                Box::pin(crate::with_retry::retry_loop(outer_ctx, closure, strategy))
            }),
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}
