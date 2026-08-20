//! The step operation: [`StepBuilder`], returned by
//! [`DurableContext::step`](crate::DurableContext::step).

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::{BoxError, context::StepContext};

// ============================================================
// StepBuilder
// ============================================================

/// Builder for a durable step operation.
///
/// Created by [`DurableContext::step`]. Chain optional configuration
/// methods, then `.await` or `.spawn()`.
///
/// The builder is generic over the step closure `F` and its future `Fut`
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
pub struct StepBuilder<O, F, Fut> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: Option<Arc<dyn Serdes>>,
    semantics: crate::step::StepSemantics,
    closure: F,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<O, F, Fut> std::fmt::Debug for StepBuilder<O, F, Fut> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O, F, Fut> StepBuilder<O, F, Fut>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Creates a new step builder (internal). Taking the closure here keeps
    /// the field non-optional: a builder without a body is unrepresentable.
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: None,
            semantics: crate::step::StepSemantics::default(),
            closure,
            _marker: PhantomData,
        }
    }

    /// Sets a human-readable name for this step.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the retry strategy for this step.
    ///
    /// The strategy closure is called on each failure with the error and
    /// attempt number to decide whether to retry. The SDK boxes the closure
    /// internally — no `Box::new` at the call site.
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
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Arc::new(serdes));
        self
    }

    /// Sets the execution semantics for this step.
    ///
    /// Controls how the SDK handles a replay where the previous attempt was
    /// interrupted (checkpoint status `Started` with no recorded outcome).
    ///
    /// - [`StepSemantics::AtLeastOncePerRetry`](crate::StepSemantics::AtLeastOncePerRetry) (default): re-execute the
    ///   step body on replay.
    /// - [`StepSemantics::AtMostOncePerRetry`](crate::StepSemantics::AtMostOncePerRetry): treat the interruption as a
    ///   failure and consult the retry strategy without re-executing.
    ///
    /// This is a client-side configuration only — it does not affect the
    /// wire protocol.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::StepSemantics;
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
}

impl<O, F, Fut> StepBuilder<O, F, Fut>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: FnOnce(StepContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<O> {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the step on a tokio task.
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
    ///     let handle = ctx.step(|_| async { Ok(1) }).name("bg").spawn();
    ///     // handle is already running
    ///     let result = handle.await?;
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

impl<O, F, Fut> IntoFuture for StepBuilder<O, F, Fut>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: FnOnce(StepContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::step::StepExecution;

        preflight_identity!(self, "Step", crate::step::STEP_SUB_TYPE);

        let execution = StepExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            retry_strategy: self.retry_strategy,
            serdes: self.serdes,
            semantics: self.semantics,
            closure: self.closure,
            _marker: PhantomData,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
mod tests {
    use std::time::Duration;

    use super::*;

    /// The closure setters accept a bare (unboxed) closure and box it
    /// internally: a plain `|err, attempt| ...` compiles and the installed
    /// strategy returns that closure's decisions.
    #[test]
    fn retry_strategy_setter_accepts_bare_closure() {
        let ctx = DurableContext::__test_context();

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
        let err = crate::StepError::from_kind(crate::StepErrorKind::ExecutionFailed {
            message: "boom".to_owned(),
        });

        assert_eq!(
            strategy(&err, 1),
            crate::RetryDecision::Retry {
                delay: Duration::from_secs(7)
            }
        );
        assert_eq!(strategy(&err, 2), crate::RetryDecision::Stop);
    }
}
