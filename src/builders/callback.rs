//! The callback operations: [`CreateCallbackBuilder`] and
//! [`WaitForCallbackBuilder`], returned by
//! [`DurableContext::create_callback`](crate::DurableContext::create_callback)
//! and
//! [`DurableContext::wait_for_callback`](crate::DurableContext::wait_for_callback),
//! plus the [`Callback`] handle the create-callback operation resolves to.
//!
//! The [callback operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/callback/)
//! describes these operations independently of any language SDK.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::time::Duration;

use crate::BoxError;
use crate::OperationSubType;
use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::context::StepContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

pub use crate::future::Callback;

/// Builder for creating a callback token.
///
/// Created by [`DurableContext::create_callback`]. The resulting
/// [`Callback`] provides the token ID and a future for
/// the result.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
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
#[non_exhaustive]
pub struct CreateCallbackBuilder<O, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    timeout: Option<Duration>,
    heartbeat_timeout: Option<Duration>,
    sub_type: Option<OperationSubType>,
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
    pub(crate) fn new_internal(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            timeout: None,
            heartbeat_timeout: None,
            sub_type: None,
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

    /// Sets the heartbeat timeout. If the external system sends no
    /// heartbeat before the timeout elapses, the callback times out with
    /// [`CallbackErrorKind::HeartbeatTimedOut`](crate::CallbackErrorKind::HeartbeatTimedOut).
    pub fn heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.heartbeat_timeout = Some(timeout);
        self
    }

    /// Records `sub_type` instead of the default subtype.
    ///
    /// The subtype labels this callback in its checkpoint, in place of
    /// `Callback`. So the callbacks of an operation composed from several
    /// durable operations can be told apart. Execution history and
    /// plugins read it. Without this call the callback records
    /// [`OperationSubType::CALLBACK`]. A second call replaces the first.
    ///
    /// Replay compares the recorded subtype with this one, as it compares
    /// the name. So changing the subtype of an existing callback breaks the
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
    /// ) -> Result<bool, durable::BoxError> {
    ///     let cb = ctx
    ///         .create_callback::<bool>()
    ///         .name("approval")
    ///         .sub_type(OperationSubType::new("Approval")?)
    ///         .await?;
    ///     Ok(cb.result().await?)
    /// }
    /// ```
    pub fn sub_type(mut self, sub_type: OperationSubType) -> Self {
        self.sub_type = Some(sub_type);
        self
    }

    /// Sets a custom deserializer for the callback payload.
    ///
    /// The callback payload is produced by an external caller through the
    /// callback-completion API, so the SDK never serializes a value on the
    /// way out: only the deserialize half of this serdes acts on the
    /// delivered payload when the result is read. The serdes must implement
    /// [`Serdes<O>`](crate::Serdes) for this callback's payload type:
    /// attaching a serdes for a different type fails at compile time. The
    /// default is [`JsonSerdes`].
    ///
    /// `.serdes(PassthroughSerdes)` on a `create_callback::<String>()`
    /// returns the raw payload the external caller sent, as JS does by
    /// default. The Rust default is [`JsonSerdes`], so a typed `O` decodes
    /// without configuration.
    ///
    /// [`PassthroughSerdes`]: crate::PassthroughSerdes
    pub fn serdes<S2>(self, serdes: S2) -> CreateCallbackBuilder<O, S2>
    where
        S2: Serdes<O>,
    {
        CreateCallbackBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat_timeout: self.heartbeat_timeout,
            sub_type: self.sub_type,
            serdes,
            _marker: PhantomData,
        }
    }

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// Callback creation itself cannot be passed to a combinator because
    /// this method returns `DurableFuture<Callback<O>>`. Await creation
    /// first; [`Callback::result()`](crate::future::Callback::result) can then be
    /// used as a combinator input when `O: Serialize + DeserializeOwned`.
    pub fn future(self) -> DurableFuture<Callback<O>>
    where
        S: Serdes<O>,
    {
        self.into_future()
    }

    /// Eagerly spawns the callback creation on a tokio task.
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
    /// `.spawn()` from a task or thread that does not own the context
    /// fails the execution with a context usage error, returns a handle
    /// that resolves to the task-ownership error, queues no `START` and
    /// runs no body. A start hook that JS fires when it queues the `START`, and the
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
    pub fn spawn(self) -> DurableFuture<Callback<O>>
    where
        S: Serdes<O>,
    {
        spawn_terminal!(self, started)
    }

    /// The callback's future, with its start step run now on the caller's
    /// task (see [`crate::start`]).
    fn started_future(self) -> DurableFuture<Callback<O>>
    where
        S: Serdes<O>,
    {
        match self.execution() {
            Ok(execution) => DurableFuture::for_started_operation(
                &execution.ctx.clone(),
                execution.execute_started(),
            ),
            Err(future) => *future,
        }
    }

    /// The callback's execution, or the error future of a failed identity
    /// preflight.
    fn execution(
        self,
    ) -> Result<crate::callback::CreateCallbackExecution<O, S>, Box<DurableFuture<Callback<O>>>>
    {
        if let Err(err) = self.ctx.preflight_replay_identity(
            &self.op_id,
            "Callback",
            Some(
                self.sub_type
                    .as_ref()
                    .map_or(crate::callback::CALLBACK_SUB_TYPE, OperationSubType::as_str),
            ),
            self.name.as_deref(),
        ) {
            return Err(Box::new(DurableFuture::from_async(async move { Err(err) })));
        }
        Ok(crate::callback::CreateCallbackExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat_timeout,
            sub_type: self.sub_type.unwrap_or(OperationSubType::CALLBACK),
            plugin_name: None,
            serdes: self.serdes,
            _marker: PhantomData,
        })
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
        match self.execution() {
            Ok(execution) => DurableFuture::for_start_step(&execution.ctx.clone(), move || {
                execution.execute_started()
            }),
            Err(future) => *future,
        }
    }
}

/// Builder for a combined callback creation + wait operation.
///
/// Created by [`DurableContext::wait_for_callback`]. Registers the callback
/// and waits for completion in one step.
///
/// The outer operation records the raw callback payload, byte for byte, as
/// its result. The serdes is applied after it: each invocation that reads
/// the outer result, live or on replay, decodes the payload once with the
/// serdes, and never asks it to serialize.
///
/// The builder is generic over the submitter closure `F` and its future
/// `Fut`; both parameters are inferred at the call site and never written
/// by users.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
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
#[non_exhaustive]
pub struct WaitForCallbackBuilder<O, F, Fut, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    timeout: Option<Duration>,
    heartbeat_timeout: Option<Duration>,
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
    F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the submitter here keeps
    /// the field non-optional: a builder without a submitter is
    /// unrepresentable.
    pub(crate) fn new_internal(ctx: DurableContext, op_id: OperationId, submitter: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            timeout: None,
            heartbeat_timeout: None,
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
    F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the retry strategy for the submitter step.
    ///
    /// On each failure the strategy receives the
    /// [`StepError`](crate::StepError) and the attempt number that just
    /// failed (attempt numbers start at 1), and returns a
    /// [`RetryDecision`](crate::RetryDecision). If not set, the SDK default
    /// retry strategy is used (exponential backoff, 6 attempts).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
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
    /// Use this for the common case, shaping retry delays (attempt count,
    /// initial delay, cap, backoff rate, jitter) and limiting the retries
    /// to some errors with
    /// [`retry_if`](crate::builders::RetryStrategyConfigBuilder::retry_if),
    /// without hand-writing a closure. The closure setter
    /// [`submitter_retry`](Self::submitter_retry) remains for retry rules the config
    /// cannot express, such as a delay that depends on the error.
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

    /// Sets the heartbeat timeout. If the external system sends no
    /// heartbeat before the timeout elapses, the callback times out with
    /// [`CallbackErrorKind::HeartbeatTimedOut`](crate::CallbackErrorKind::HeartbeatTimedOut).
    pub fn heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.heartbeat_timeout = Some(timeout);
        self
    }

    /// Sets a custom deserializer for the callback payload.
    ///
    /// The callback payload is produced by an external caller through the
    /// callback-completion API, so the SDK never serializes a value on the
    /// way out: only the deserialize half of this serdes acts on the
    /// delivered payload. The serdes must implement
    /// [`Serdes<O>`](crate::Serdes) for this callback's payload type:
    /// attaching a serdes for a different type fails at compile time, and
    /// `O` needs no `serde` traits. The default is [`JsonSerdes`].
    ///
    /// `.serdes(PassthroughSerdes)` on a `wait_for_callback::<String, _, _>`
    /// returns the raw payload the external caller sent, as JS does by
    /// default. The Rust default is [`JsonSerdes`], so a typed `O` decodes
    /// without configuration.
    ///
    /// # Examples
    ///
    /// A payload type without `serde` derives, decoded by its own serdes:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::{Serdes, SerdesContext};
    ///
    /// struct Token(String);
    ///
    /// struct TokenSerdes;
    ///
    /// impl Serdes<Token> for TokenSerdes {
    ///     async fn serialize(
    ///         &self,
    ///         value: Token,
    ///         _context: SerdesContext,
    ///     ) -> Result<String, durable::BoxError> {
    ///         Ok(value.0)
    ///     }
    ///
    ///     async fn deserialize(
    ///         &self,
    ///         wire: String,
    ///         _context: SerdesContext,
    ///     ) -> Result<Token, durable::BoxError> {
    ///         Ok(Token(wire))
    ///     }
    /// }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let token = ctx
    ///         .wait_for_callback::<Token, _, _>(|_step_ctx, _callback_id| async { Ok(()) })
    ///         .name("token")
    ///         .serdes(TokenSerdes)
    ///         .await?;
    ///     Ok(token.0)
    /// }
    /// ```
    ///
    /// [`PassthroughSerdes`]: crate::PassthroughSerdes
    pub fn serdes<S2>(self, serdes: S2) -> WaitForCallbackBuilder<O, F, Fut, S2>
    where
        S2: Serdes<O>,
    {
        WaitForCallbackBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat_timeout: self.heartbeat_timeout,
            submitter: self.submitter,
            submitter_retry: self.submitter_retry,
            serdes,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> WaitForCallbackBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
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
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
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
    /// `.spawn()` from a task or thread that does not own the context
    /// fails the execution with a context usage error, returns a handle
    /// that resolves to the task-ownership error, queues no `START` and
    /// runs no body. A start hook that JS fires when it queues the `START`, and the
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
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self, started)
    }

    /// The operation's future, with its start step run now on the caller's
    /// task (see [`crate::start`]).
    fn started_future(self) -> DurableFuture<O> {
        match self.execution() {
            Ok(execution) => DurableFuture::for_started_operation(
                &execution.ctx.clone(),
                execution.execute_started(),
            ),
            Err(future) => *future,
        }
    }

    /// The operation's execution, or the error future of a failed identity
    /// preflight.
    fn execution(
        self,
    ) -> Result<crate::callback::WaitForCallbackExecution<O, F, S>, Box<DurableFuture<O>>> {
        if let Err(err) = self.ctx.preflight_replay_identity(
            &self.op_id,
            "Context",
            Some(crate::callback::WFCB_SUB_TYPE),
            self.name.as_deref(),
        ) {
            return Err(Box::new(DurableFuture::from_async(async move { Err(err) })));
        }
        Ok(crate::callback::WaitForCallbackExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat_timeout,
            submitter: self.submitter,
            submitter_retry: self.submitter_retry,
            serdes: self.serdes,
            _marker: PhantomData,
        })
    }
}

impl<O, F, Fut, S> IntoFuture for WaitForCallbackBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        match self.execution() {
            Ok(execution) => DurableFuture::for_start_step(&execution.ctx.clone(), move || {
                execution.execute_started()
            }),
            Err(future) => *future,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second `sub_type` call replaces the first, and `serdes` keeps
    /// the subtype.
    #[test]
    fn sub_type_setter_replaces_and_survives_serdes() {
        let ctx = DurableContext::test_context();
        let builder = ctx
            .create_callback::<String>()
            .sub_type(OperationSubType::new("First").unwrap())
            .sub_type(OperationSubType::new("Second").unwrap())
            .serdes(crate::PassthroughSerdes);
        assert_eq!(
            builder.sub_type.as_ref().map(OperationSubType::as_str),
            Some("Second")
        );
    }

    /// For both callback builders, a second `heartbeat_timeout` call
    /// replaces the first, and `serdes` keeps the value.
    #[test]
    fn heartbeat_timeout_setter_replaces_and_survives_serdes() {
        let ctx = DurableContext::test_context();
        let create = ctx
            .create_callback::<String>()
            .heartbeat_timeout(Duration::from_secs(30))
            .heartbeat_timeout(Duration::from_secs(90))
            .serdes(crate::PassthroughSerdes);
        assert_eq!(create.heartbeat_timeout, Some(Duration::from_secs(90)));

        let wait = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .heartbeat_timeout(Duration::from_secs(30))
            .heartbeat_timeout(Duration::from_secs(90))
            .serdes(crate::PassthroughSerdes);
        assert_eq!(wait.heartbeat_timeout, Some(Duration::from_secs(90)));
    }

    /// The callback submitter retry setter likewise takes a bare closure.
    #[test]
    fn submitter_retry_setter_accepts_bare_closure() {
        let ctx = DurableContext::test_context();

        let builder = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .submitter_retry(|_err, _attempt| crate::RetryDecision::Stop);

        let strategy = builder
            .submitter_retry
            .as_ref()
            .expect("submitter_retry must install a strategy");
        let err = crate::StepError::new(crate::StepErrorKind::ExecutionFailed, Some("boom".into()));

        assert_eq!(strategy(&err, 1), crate::RetryDecision::Stop);
    }
}
