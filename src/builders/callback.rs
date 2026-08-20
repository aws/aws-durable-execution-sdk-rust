//! The callback operations: [`CreateCallbackBuilder`] and
//! [`WaitForCallbackBuilder`], returned by
//! [`DurableContext::create_callback`](crate::DurableContext::create_callback)
//! and
//! [`DurableContext::wait_for_callback`](crate::DurableContext::wait_for_callback),
//! plus the [`Callback`] handle the create-callback operation resolves to.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::time::Duration;

use crate::BoxError;
use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::context::StepContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

pub use crate::future::Callback;

// ============================================================
// CreateCallbackBuilder
// ============================================================

/// Builder for creating a callback token.
///
/// Created by [`DurableContext::create_callback`]. The resulting
/// [`Callback`] provides the token ID and a future for
/// the result.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use serde::{Deserialize, Serialize};
/// use std::time::Duration;
///
/// #[derive(Serialize, Deserialize)]
/// struct Approval { ok: bool }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     let cb = ctx.create_callback::<Approval>()
///         .name("approval-cb")
///         .timeout(Duration::from_secs(3600))
///         .await?;
///     let _id = cb.id();
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct CreateCallbackBuilder<O, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
    serdes: S,
    _marker: PhantomData<O>,
}

impl<O, S> std::fmt::Debug for CreateCallbackBuilder<O, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateCallbackBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> CreateCallbackBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: JsonSerdes,
            _marker: PhantomData,
        }
    }
}

impl<O: Send + 'static, S> CreateCallbackBuilder<O, S> {
    /// Sets a human-readable name for this callback.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the maximum time to wait for the callback to be completed.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the heartbeat interval for keep-alive signals.
    pub fn heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat = Some(interval);
        self
    }

    /// Sets a custom deserializer for the callback payload.
    ///
    /// The callback payload is produced by an external caller through the
    /// callback-completion API, so the SDK never serializes a value on the
    /// way out — only the deserialize half of this serdes acts on the
    /// delivered payload when the result is read. The serdes must implement
    /// [`Serdes<O>`](crate::Serdes) for this callback's payload type —
    /// attaching a serdes for a different type fails at compile time. The
    /// default is [`JsonSerdes`].
    pub fn serdes<S2>(self, serdes: S2) -> CreateCallbackBuilder<O, S2>
    where
        S2: Serdes<O>,
    {
        CreateCallbackBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat,
            serdes,
            _marker: PhantomData,
        }
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<Callback<O>>
    where
        S: Serdes<O>,
    {
        self.into_future()
    }

    /// Eagerly spawns the callback creation on a tokio task.
    pub fn spawn(self) -> DurableFuture<Callback<O>>
    where
        S: Serdes<O>,
    {
        spawn_terminal!(self)
    }
}

impl<O, S> IntoFuture for CreateCallbackBuilder<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    type Output = Result<Callback<O>, OperationError>;
    type IntoFuture = DurableFuture<Callback<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::callback::CreateCallbackExecution;

        preflight_identity!(self, "Callback", crate::callback::CALLBACK_SUB_TYPE);

        let execution = CreateCallbackExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat,
            serdes: self.serdes,
            _marker: PhantomData,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// WaitForCallbackBuilder
// ============================================================

/// Builder for a combined callback creation + wait operation.
///
/// Created by [`DurableContext::wait_for_callback`]. Registers the callback
/// and waits for completion in one step.
///
/// The builder is generic over the submitter closure `F` and its future
/// `Fut` so the submitter is stored **without type erasure**; both
/// parameters are inferred at the call site and never written by users.
/// The single erasure point is `.future()` / `.await`, which produces the
/// one [`DurableFuture`] box.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Deserialize, Serialize)]
/// struct Outcome { value: i32 }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     let _outcome: Outcome = ctx.wait_for_callback::<Outcome, _, _>(
///         |_step_ctx, cb_id| async move {
///             // send cb_id to external system
///             Ok(())
///         }
///     ).name("external-approval")
///      .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct WaitForCallbackBuilder<O, F, Fut, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
    submitter: F,
    submitter_retry: Option<RetryStrategy>,
    serdes: S,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<O, F, Fut, S> std::fmt::Debug for WaitForCallbackBuilder<O, F, Fut, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitForCallbackBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O, F, Fut> WaitForCallbackBuilder<O, F, Fut>
where
    O: Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the submitter here keeps
    /// the field non-optional: a builder without a submitter is
    /// unrepresentable.
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, submitter: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            submitter,
            submitter_retry: None,
            serdes: JsonSerdes,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> WaitForCallbackBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the retry strategy for the submitter step.
    ///
    /// Controls how many times the submitter is retried on failure before
    /// the operation is abandoned. If not set, the SDK default retry
    /// strategy is used (exponential backoff, 6 attempts). The SDK boxes the
    /// closure internally — no `Box::new` at the call site.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.wait_for_callback::<String, _, _>(
    ///         |_step_ctx, _cb_id| async { Ok(()) }
    ///     ).name("with-retry")
    ///      .submitter_retry(|_err, attempt| {
    ///          if attempt >= 2 {
    ///              durable::RetryDecision::Stop
    ///          } else {
    ///              durable::RetryDecision::Retry { delay: Duration::from_secs(1) }
    ///          }
    ///      })
    ///      .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn submitter_retry<R>(mut self, strategy: R) -> Self
    where
        R: Fn(&crate::StepError, u32) -> crate::RetryDecision + Send + Sync + 'static,
    {
        self.submitter_retry = Some(Box::new(strategy));
        self
    }

    /// Sets the submitter retry strategy from a
    /// [`RetryStrategyConfig`](crate::builders::RetryStrategyConfig).
    ///
    /// Use this for the common case — shaping retry delays (attempt count,
    /// initial delay, cap, backoff rate, jitter) — without hand-writing a
    /// closure. For decisions a delay schedule cannot express, such as
    /// inspecting the error, use [`submitter_retry`](Self::submitter_retry)
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
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.wait_for_callback::<String, _, _>(
    ///         |_step_ctx, _cb_id| async { Ok(()) }
    ///     ).name("with-retry")
    ///      .submitter_retry_config(
    ///          RetryStrategyConfig::builder()
    ///              .max_attempts(2)
    ///              .initial_delay(Duration::from_secs(1))
    ///              .build(),
    ///      )
    ///      .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn submitter_retry_config(mut self, config: crate::builders::RetryStrategyConfig) -> Self {
        self.submitter_retry = Some(config.into_retry_strategy());
        self
    }

    /// Sets the maximum time to wait for the callback.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the heartbeat interval for keep-alive signals.
    ///
    /// If the external system does not send heartbeats within this
    /// interval, the callback times out.
    pub fn heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat = Some(interval);
        self
    }

    /// Sets a custom deserializer for the callback payload.
    ///
    /// The callback payload is produced by an external caller through the
    /// callback-completion API, so the SDK never serializes a value on the
    /// way out — only the deserialize half of this serdes acts on the
    /// delivered payload. The serdes must implement
    /// [`Serdes<O>`](crate::Serdes) for this callback's payload type —
    /// attaching a serdes for a different type fails at compile time. The
    /// default is [`JsonSerdes`].
    pub fn serdes<S2>(self, serdes: S2) -> WaitForCallbackBuilder<O, F, Fut, S2>
    where
        S2: Serdes<O>,
    {
        WaitForCallbackBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat,
            submitter: self.submitter,
            submitter_retry: self.submitter_retry,
            serdes,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> WaitForCallbackBuilder<O, F, Fut, S>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O, F, Fut, S> IntoFuture for WaitForCallbackBuilder<O, F, Fut, S>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::callback::WaitForCallbackExecution;

        preflight_identity!(self, "Context", crate::callback::WFCB_SUB_TYPE);

        let execution = WaitForCallbackExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat,
            submitter: self.submitter,
            submitter_retry: self.submitter_retry,
            serdes: self.serdes,
            _marker: PhantomData,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
mod tests {
    use super::*;

    /// The callback submitter retry setter likewise takes a bare closure.
    #[test]
    fn submitter_retry_setter_accepts_bare_closure() {
        let ctx = DurableContext::__test_context();

        let builder = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .submitter_retry(|_err, _attempt| crate::RetryDecision::Stop);

        let strategy = builder
            .submitter_retry
            .as_ref()
            .expect("submitter_retry must install a strategy");
        let err = crate::StepError::from_kind(crate::StepErrorKind::ExecutionFailed {
            message: "boom".to_owned(),
        });

        assert_eq!(strategy(&err, 1), crate::RetryDecision::Stop);
    }
}
