//! The step operation: [`StepBuilder`], returned by
//! [`DurableContext::step`](crate::DurableContext::step).
//!
//! The [step operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/step/)
//! describes this operation independently of any language SDK.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;

use crate::OperationSubType;
use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;
use crate::{BoxError, context::StepContext};

/// Builder for a durable step operation.
///
/// Created by [`DurableContext::step`]. Chain optional configuration
/// methods, then `.await` or `.spawn()`.
///
/// The builder is generic over the step closure `F` and its future `Fut`;
/// both are inferred at the call site and never written by users. It also
/// carries the serdes implementation `S` (defaulting to [`JsonSerdes`]);
/// [`serdes`](Self::serdes) swaps that parameter. Whatever `S` is, the
/// finalized future is [`DurableFuture<O>`], so futures configured with
/// different serdes implementations coexist in one combinator input
/// collection.
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
/// ) -> Result<i32, durable::BoxError> {
///     let result = ctx.step(|_| async { Ok(42) })
///         .name("compute")
///         .retry_strategy(|_err, attempt| {
///             if attempt >= 3 {
///                 RetryDecision::Stop
///             } else {
///                 RetryDecision::Retry { delay: Duration::from_secs(1) }
///             }
///         })
///         .await?;
///     Ok(result)
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct StepBuilder<O, F, Fut, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: S,
    semantics: crate::step::StepSemantics,
    sub_type: Option<OperationSubType>,
    closure: F,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<O, F, Fut, S> std::fmt::Debug for StepBuilder<O, F, Fut, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O, F, Fut> StepBuilder<O, F, Fut>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Creates a new step builder (internal). Taking the closure here keeps
    /// the field non-optional: a builder without a body is unrepresentable.
    pub(crate) fn new_internal(ctx: DurableContext, op_id: OperationId, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: JsonSerdes,
            semantics: crate::step::StepSemantics::default(),
            sub_type: None,
            closure,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> StepBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this step.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the retry strategy for this step.
    ///
    /// On each failure the strategy receives the
    /// [`StepError`](crate::StepError) and the attempt number that just
    /// failed (attempt numbers start at 1), and returns a
    /// [`RetryDecision`](crate::RetryDecision).
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
    /// ) -> Result<i32, durable::BoxError> {
    ///     let result = ctx.step(|_| async { Ok(42) })
    ///         .retry_strategy(|_err, attempt| {
    ///             if attempt >= 3 {
    ///                 RetryDecision::Stop
    ///             } else {
    ///                 RetryDecision::Retry {
    ///                     delay: Duration::from_millis(100 * u64::from(2_u32.pow(attempt - 1))),
    ///                 }
    ///             }
    ///         })
    ///         .await?;
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

    /// Sets the retry strategy for this step from a
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
    /// ) -> Result<i32, durable::BoxError> {
    ///     let result = ctx.step(|_| async { Ok(42) })
    ///         .retry_strategy_config(
    ///             RetryStrategyConfig::builder()
    ///                 .max_attempts(3)
    ///                 .initial_delay(Duration::from_secs(1))
    ///                 .build(),
    ///         )
    ///         .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn retry_strategy_config(mut self, config: crate::builders::RetryStrategyConfig) -> Self {
        self.retry_strategy = Some(config.into_retry_strategy());
        self
    }

    /// Sets a custom serializer/deserializer for this step's result.
    ///
    /// Replaces the builder's serdes type parameter with `S2`, which must
    /// implement [`Serdes<O>`](crate::Serdes) for this step's output type:
    /// attaching a serdes for a different type fails at compile time. To
    /// share one instance across operations, wrap it in an
    /// [`Arc`](std::sync::Arc) and clone the `Arc` handle into each builder.
    pub fn serdes<S2>(self, serdes: S2) -> StepBuilder<O, F, Fut, S2>
    where
        S2: Serdes<O>,
    {
        StepBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            retry_strategy: self.retry_strategy,
            serdes,
            semantics: self.semantics,
            sub_type: self.sub_type,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Sets the execution semantics for this step.
    ///
    /// Controls how the SDK handles a replay where the previous attempt was
    /// interrupted (checkpoint status `Started` with no recorded outcome).
    ///
    /// - [`StepSemantics::AtLeastOncePerRetry`](crate::StepSemantics::AtLeastOncePerRetry) (default): re-execute the
    ///   step body on replay. The body starts without waiting for the
    ///   `START` checkpoint. The outcome is written only after `START` is
    ///   acknowledged; if `START` fails, the outcome is not written and the
    ///   invocation fails.
    /// - [`StepSemantics::AtMostOncePerRetry`](crate::StepSemantics::AtMostOncePerRetry): treat the interruption as a
    ///   failure and consult the retry strategy without re-executing. The
    ///   body starts only after `START` is acknowledged.
    ///
    /// This is a client-side configuration only: it does not affect the
    /// wire protocol.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use aws_durable_execution_sdk::StepSemantics;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.step(|_| async { Ok("done".to_owned()) })
    ///         .name("charge-card")
    ///         .semantics(StepSemantics::AtMostOncePerRetry)
    ///         .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn semantics(mut self, semantics: crate::step::StepSemantics) -> Self {
        self.semantics = semantics;
        self
    }

    /// Records `sub_type` instead of the default subtype.
    ///
    /// The subtype labels this step in its checkpoint, in place of `Step`.
    /// So the steps of an operation composed from several durable
    /// operations can be told apart. Execution history and plugins read
    /// it. Without this call the step records [`OperationSubType::STEP`].
    /// A second call replaces the first.
    ///
    /// Replay compares the recorded subtype with this one, as it compares
    /// the name. So changing the subtype of an existing step breaks the
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
    /// ) -> Result<String, durable::BoxError> {
    ///     let receipt = ctx
    ///         .step(|_| async { Ok("receipt-1".to_owned()) })
    ///         .name("charge")
    ///         .sub_type(OperationSubType::new("Payment")?)
    ///         .await?;
    ///     Ok(receipt)
    /// }
    /// ```
    pub fn sub_type(mut self, sub_type: OperationSubType) -> Self {
        self.sub_type = Some(sub_type);
        self
    }
}

impl<O, F, Fut, S> StepBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<O> {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the step on a tokio task.
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
    ///     let handle = ctx.step(|_| async { Ok(1) }).name("bg").spawn();
    ///     // handle is already running
    ///     let result = handle.await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O, F, Fut, S> IntoFuture for StepBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::step::StepExecution;

        preflight_identity!(
            self,
            "Step",
            self.sub_type
                .as_ref()
                .map_or(crate::step::STEP_SUB_TYPE, OperationSubType::as_str)
        );

        let gate = self.ctx.clone();

        let execution = StepExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            retry_strategy: self.retry_strategy,
            serdes: self.serdes,
            semantics: self.semantics,
            sub_type: self.sub_type.unwrap_or(OperationSubType::STEP),
            plugin_name: None,
            closure: self.closure,
            _marker: PhantomData,
        };

        DurableFuture::for_operation(&gate, execution.execute())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A second `sub_type` call replaces the first, and `serdes` keeps
    /// the subtype.
    #[test]
    fn sub_type_setter_replaces_and_survives_serdes() {
        let ctx = DurableContext::test_context();
        let builder = ctx
            .step(|_| async { Ok("a".to_owned()) })
            .sub_type(OperationSubType::new("First").unwrap())
            .sub_type(OperationSubType::new("Second").unwrap())
            .serdes(crate::PassthroughSerdes);
        assert_eq!(
            builder.sub_type.as_ref().map(OperationSubType::as_str),
            Some("Second")
        );
    }

    /// The closure setters accept a bare (unboxed) closure and box it
    /// internally: a plain `|err, attempt| ...` compiles and the installed
    /// strategy returns that closure's decisions.
    #[test]
    fn retry_strategy_setter_accepts_bare_closure() {
        let ctx = DurableContext::test_context();

        let builder = ctx
            .step(|_| async { Ok(1_i32) })
            .retry_strategy(|_err, attempt| {
                if attempt >= 2 {
                    crate::RetryDecision::Stop
                } else {
                    crate::RetryDecision::Retry {
                        delay: Duration::from_secs(7),
                    }
                }
            });

        let strategy = builder
            .retry_strategy
            .as_ref()
            .expect("retry_strategy must install a strategy");
        let err = crate::StepError::new(crate::StepErrorKind::ExecutionFailed, Some("boom".into()));

        assert_eq!(
            strategy(&err, 1),
            crate::RetryDecision::Retry {
                delay: Duration::from_secs(7)
            }
        );
        assert_eq!(strategy(&err, 2), crate::RetryDecision::Stop);
    }
}
