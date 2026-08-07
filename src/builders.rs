//! Fluent operation builders.
//!
//! Every builder owns a [`DurableContext`] clone (cheap `Arc` — no
//! lifetimes), carries the pre-claimed [`OperationId`] (minted at the call
//! site), implements [`IntoFuture`] for `.await` support, and provides
//! a `.spawn()` eager terminal. Chain methods consume and return `self`.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::time::Duration;

use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::{CompletionConfig, RetryStrategy, Serdes, Settled, WaitStrategy};

/// Eagerly validates the builder's claimed replay identity when the builder
/// is finalized into a [`DurableFuture`].
///
/// Every `into_future` runs this FIRST — and `.future()`, `.spawn()`, and
/// `.await` all funnel through `into_future` — so a replay identity mismatch
/// is recorded on the execution-fatal slot synchronously at finalization,
/// before the operation future is ever polled. This is what makes fatal
/// propagation scheduler-independent: a short-circuiting combinator
/// (`select_ok`, `race`, `try_join_all`) aborts losers the moment a winner
/// settles, so a mismatching constituent might never be polled — but by then
/// its identity was already validated here.
///
/// On mismatch the returned future resolves immediately with the dedicated
/// error and never runs the operation (no START is checkpointed for an
/// operation the recorded history contradicts).
macro_rules! preflight_identity {
    ($builder:expr, $claimed_type:expr, $sub_type:expr) => {
        if let Err(err) = $builder.ctx.preflight_replay_identity(
            &$builder.op_id,
            $claimed_type,
            Some($sub_type),
            $builder.name.as_deref(),
        ) {
            return DurableFuture::from_async(async move { Err(err) });
        }
    };
}

/// The body shared by every builder's `.spawn()` terminal.
///
/// Rebinds the builder's context onto a FRESH child suspension scope, then
/// hands the operation future to
/// [`DurableFuture::spawn_blessed`](crate::future::DurableFuture) together with
/// the owner's scope (for quiescence accounting) and the new scope (which the
/// spawned task drives).
///
/// This is one helper rather than thirteen copies so that no `.spawn()`
/// terminal can drift out of the accounting: an eagerly spawned operation that
/// kept the owner's scope would park the owner — ending the invocation — the
/// moment it hit a durable suspension point, aborting runnable siblings.
///
/// The builder must have a `ctx: DurableContext` field and implement
/// [`IntoFuture`] with `IntoFuture = DurableFuture<_>`.
macro_rules! spawn_terminal {
    ($builder:expr) => {{
        let mut builder = $builder;
        let owner_scope = ::std::sync::Arc::clone(builder.ctx.suspension_signal());
        let task_ownership = ::std::sync::Arc::clone(builder.ctx.task_ownership());
        let (spawn_ctx, spawn_scope) = builder.ctx.spawn_scope();
        builder.ctx = spawn_ctx;
        let future = ::std::future::IntoFuture::into_future(builder);
        $crate::future::DurableFuture::spawn_blessed(
            future,
            task_ownership,
            owner_scope,
            spawn_scope,
        )
    }};
}

/// Like [`spawn_terminal!`] but also redirects park signals from the
/// builder's constituent `futures` onto the combinator's spawn scope.
///
/// Without this, a constituent that was itself `.spawn()`ed would park the
/// OUTER owner scope — the same scope that counts the combinator as
/// outstanding — creating a deadlock cycle: the owner waits for the
/// combinator to settle, while the combinator waits for the constituent to
/// complete, which it never will (it parked).
///
/// By redirecting, the constituent's park hits the combinator's spawn scope
/// instead. [`drive_scope`](crate::driver::drive_scope) detects the
/// suspension and the combinator settles as parked on the owner scope,
/// breaking the cycle.
macro_rules! spawn_combinator_terminal {
    ($builder:expr) => {{
        let mut builder = $builder;
        let owner_scope = ::std::sync::Arc::clone(builder.ctx.suspension_signal());
        let task_ownership = ::std::sync::Arc::clone(builder.ctx.task_ownership());
        let (spawn_ctx, spawn_scope) = builder.ctx.spawn_scope();
        // Redirect each constituent future's park to the combinator's scope.
        for future in &builder.futures {
            future.set_park_scope(::std::sync::Arc::clone(&spawn_scope));
        }
        builder.ctx = spawn_ctx;
        let future = ::std::future::IntoFuture::into_future(builder);
        $crate::future::DurableFuture::spawn_blessed(
            future,
            task_ownership,
            owner_scope,
            spawn_scope,
        )
    }};
}

// ============================================================
// StepBuilder
// ============================================================

/// Builder for a durable step operation.
///
/// Created by [`DurableContext::step`]. Chain optional configuration
/// methods, then `.await` or `.spawn()`.
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
pub struct StepBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: Option<Box<dyn Serdes>>,
    semantics: crate::step::StepSemantics,
    #[allow(clippy::type_complexity)] // reason: boxed future factory is inherently complex
    closure: Option<
        Box<
            dyn FnOnce(
                    crate::context::StepContext,
                )
                    -> std::pin::Pin<Box<dyn Future<Output = Result<O, crate::BoxError>> + Send>>
                + Send,
        >,
    >,
    _marker: PhantomData<O>,
}

impl<O> std::fmt::Debug for StepBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> StepBuilder<O> {
    /// Creates a new step builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: None,
            semantics: crate::step::StepSemantics::default(),
            closure: None,
            _marker: PhantomData,
        }
    }

    /// Sets the closure for this step (internal, called by `context.step()`).
    #[allow(clippy::type_complexity)] // reason: boxed future factory is inherently complex
    pub(crate) fn with_closure(
        mut self,
        closure: Box<
            dyn FnOnce(
                    crate::context::StepContext,
                )
                    -> std::pin::Pin<Box<dyn Future<Output = Result<O, crate::BoxError>> + Send>>
                + Send,
        >,
    ) -> Self {
        self.closure = Some(closure);
        self
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
    pub fn retry_strategy<F>(mut self, strategy: F) -> Self
    where
        F: Fn(&crate::StepError, u32) -> crate::RetryDecision + Send + Sync + 'static,
    {
        self.retry_strategy = Some(Box::new(strategy));
        self
    }

    /// Sets a custom serializer/deserializer for this step's result.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
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

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> StepBuilder<O> {
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

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for StepBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::step::StepExecution;

        preflight_identity!(self, "Step", crate::step::STEP_SUB_TYPE);

        let closure = self.closure.unwrap_or_else(|| {
            // If no closure was provided (shouldn't happen in normal use),
            // produce an immediate error.
            Box::new(|_| Box::pin(async { Err("step has no closure".into()) }))
        });

        let execution = StepExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            retry_strategy: self.retry_strategy,
            serdes: self.serdes,
            semantics: self.semantics,
            closure,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// WaitBuilder
// ============================================================

/// Builder for a durable wait (timer) operation.
///
/// Created by [`DurableContext::wait`]. The wait duration is set at
/// creation; chain `.name()` for identification.
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
/// ) -> Result<(), durable::BoxError> {
///     ctx.wait(Duration::from_secs(30))
///         .name("pause")
///         .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct WaitBuilder {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    duration_secs: i32,
}

impl std::fmt::Debug for WaitBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl WaitBuilder {
    /// Creates a new wait builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, duration_secs: i32) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            duration_secs,
        }
    }

    /// Sets a human-readable name for this wait.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<()> {
        self.into_future()
    }

    /// Eagerly spawns the wait on a tokio task.
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
    /// ) -> Result<(), durable::BoxError> {
    ///     let handle = ctx.wait(Duration::from_secs(10)).spawn();
    ///     // do other work while timer runs
    ///     handle.await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn spawn(self) -> DurableFuture<()> {
        spawn_terminal!(self)
    }
}

impl IntoFuture for WaitBuilder {
    type Output = Result<(), OperationError>;
    type IntoFuture = DurableFuture<()>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::wait::WaitExecution;

        preflight_identity!(self, "Wait", crate::wait::WAIT_SUB_TYPE);

        let execution = WaitExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            duration_secs: self.duration_secs,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// InvokeBuilder
// ============================================================

/// Builder for a durable invoke operation.
///
/// Created by [`DurableContext::invoke`]. Configure with `.name()` and
/// custom serialization before awaiting.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize)]
/// struct Input { id: u64 }
///
/// #[derive(Deserialize)]
/// struct Output { status: String }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     let out = ctx.invoke::<Output, _>("target-fn", Input { id: 1 })
///         .name("call-target")
///         .await?;
///     Ok(out.status)
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct InvokeBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    function_id: String,
    erased_input: Result<serde_json::Value, String>,
    payload_serdes: Option<Box<dyn Serdes>>,
    result_serdes: Option<Box<dyn Serdes>>,
    tenant_id: Option<String>,
    _marker: PhantomData<O>,
}

impl<O> std::fmt::Debug for InvokeBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvokeBuilder")
            .field("name", &self.name)
            .field("function_id", &self.function_id)
            .finish_non_exhaustive()
    }
}

impl<O: serde::de::DeserializeOwned + Send + 'static> InvokeBuilder<O> {
    /// Creates a new invoke builder (internal).
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        function_id: String,
        erased_input: Result<serde_json::Value, String>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            function_id,
            erased_input,
            payload_serdes: None,
            result_serdes: None,
            tenant_id: None,
            _marker: PhantomData,
        }
    }

    /// Sets a human-readable name for this invoke.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets a custom serializer/deserializer for this invoke's result
    /// deserialization.
    ///
    /// The serdes is applied when deserializing the invoke result payload
    /// returned by the target function. Use this to apply custom
    /// transformations on the result (e.g., uppercasing).
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.result_serdes = Some(Box::new(serdes));
        self
    }

    /// Sets a custom serializer/deserializer for this invoke's input
    /// payload serialization.
    ///
    /// The serdes is applied when serializing the input payload before
    /// sending it to the target function. This is independent of the
    /// result serdes set via [`.serdes()`](Self::serdes).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// # #[derive(Debug)]
    /// # struct UpperSerdes;
    /// # impl durable::Serdes for UpperSerdes {
    /// #     fn serialize(&self, v: &serde_json::Value, _c: &durable::SerdesContext) -> Result<String, durable::BoxError> { Ok(v.to_string().to_uppercase()) }
    /// # }
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.invoke::<String, _>("target-fn", "hello")
    ///         .payload_serdes(UpperSerdes)
    ///         .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn payload_serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.payload_serdes = Some(Box::new(serdes));
        self
    }

    /// Sets the tenant ID for tenant-isolated invocations.
    ///
    /// When set, the target function is invoked in the context of the
    /// specified tenant.
    pub fn tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<O> {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the invoke on a tokio task.
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
    ///     let handle = ctx.invoke::<String, _>("fn", "input")
    ///         .name("bg-call")
    ///         .spawn();
    ///     let _result = handle.await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O: serde::de::DeserializeOwned + Send + 'static> IntoFuture for InvokeBuilder<O> {
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::invoke::InvokeExecution;

        preflight_identity!(
            self,
            "ChainedInvoke",
            crate::invoke::CHAINED_INVOKE_SUB_TYPE
        );

        let execution = InvokeExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            function_id: self.function_id,
            erased_input: self.erased_input,
            payload_serdes: self.payload_serdes,
            result_serdes: self.result_serdes,
            tenant_id: self.tenant_id,
            _marker: PhantomData,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// ChildBuilder
// ============================================================

/// Builder for a child context (sub-orchestration) operation.
///
/// Created by [`DurableContext::run_in_child_context`]. Use `.spawn()` for
/// eager fan-out execution.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<i32, durable::BoxError> {
///     let result = ctx.run_in_child_context(|child| async move {
///         Ok(42)
///     }).name("sub-orchestration")
///       .await?;
///     Ok(result)
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct ChildBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    serdes: Option<Box<dyn Serdes>>,
    #[allow(clippy::type_complexity)] // reason: boxed future factory is inherently complex
    closure: Option<
        Box<
            dyn FnOnce(
                    DurableContext,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                            + Send,
                    >,
                > + Send,
        >,
    >,
}

impl<O> std::fmt::Debug for ChildBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> ChildBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            serdes: None,
            closure: None,
        }
    }

    /// Sets the closure for this builder (internal).
    #[allow(clippy::type_complexity)] // reason: boxed future factory is inherently complex
    pub(crate) fn with_closure(
        mut self,
        closure: Box<
            dyn FnOnce(
                    DurableContext,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                            + Send,
                    >,
                > + Send,
        >,
    ) -> Self {
        self.closure = Some(closure);
        self
    }

    /// Sets a human-readable name for this operation.
    ///
    /// Names appear in logs, traces, and the execution history for
    /// debugging purposes.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Overrides the serialization strategy for the child result.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for
    /// fan-out patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<O>
    where
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the child context on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running — this is
    /// the replay-safe alternative to bare `tokio::spawn` for
    /// durable operations. The operation ID was already claimed at
    /// builder creation.
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
    ///     let handle = ctx.run_in_child_context(|child| async move {
    ///         Ok(1)
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
    for ChildBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::child::ChildExecution;

        preflight_identity!(self, "Context", crate::child::CHILD_SUB_TYPE);

        let closure = self.closure.unwrap_or_else(|| {
            Box::new(|_| {
                Box::pin(async { Err(crate::error::ChildFnError::new("child has no closure")) })
            })
        });

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            serdes: self.serdes,
            closure,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// WaitForConditionBuilder
// ============================================================

/// Builder for a wait-for-condition operation.
///
/// Created by [`DurableContext::wait_for_condition`]. Configure the polling
/// strategy with `.wait_strategy_fn()`.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use aws_durable_execution_sdk_rust::WaitDecision;
/// use serde::{Serialize, Deserialize};
/// use std::time::Duration;
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct State { ready: bool }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     ctx.wait_for_condition(
///         |_, state: State| async move { Ok(State { ready: true }) },
///         State { ready: false },
///     ).name("wait-ready")
///      .wait_strategy_fn(|state: State, _attempt| {
///          if state.ready {
///              WaitDecision::complete()
///          } else {
///              WaitDecision::continue_with(Duration::from_secs(5))
///          }
///      })
///      .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct WaitForConditionBuilder<S> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    initial_state: S,
    wait_strategy: Option<crate::wait_for_condition::WaitStrategyFn<S>>,
    serdes: Option<Box<dyn Serdes>>,
    #[allow(clippy::type_complexity)] // reason: boxed Fn closure is inherently complex
    check: Option<
        Box<
            dyn Fn(
                    crate::context::StepContext,
                    S,
                )
                    -> std::pin::Pin<Box<dyn Future<Output = Result<S, crate::BoxError>> + Send>>
                + Send
                + Sync,
        >,
    >,
}

impl<S> std::fmt::Debug for WaitForConditionBuilder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitForConditionBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<S: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + Sync + 'static>
    WaitForConditionBuilder<S>
{
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, initial_state: S) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            initial_state,
            wait_strategy: None,
            serdes: None,
            check: None,
        }
    }

    /// Sets the check closure (internal, called by `context.wait_for_condition()`).
    #[allow(clippy::type_complexity)] // reason: boxed Fn closure is inherently complex
    pub(crate) fn with_check(
        mut self,
        check: Box<
            dyn Fn(
                    crate::context::StepContext,
                    S,
                )
                    -> std::pin::Pin<Box<dyn Future<Output = Result<S, crate::BoxError>> + Send>>
                + Send
                + Sync,
        >,
    ) -> Self {
        self.check = Some(check);
        self
    }

    /// Sets a human-readable name for this operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the wait strategy (polling interval and backoff config).
    ///
    /// This converts the [`WaitStrategy`] config struct into a functional
    /// strategy internally.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::WaitStrategy;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<i32, durable::BoxError> {
    ///     let done = ctx
    ///         .wait_for_condition(|_step_ctx, state: i32| async move { Ok(state + 1) }, 0_i32)
    ///         .wait_strategy(
    ///             WaitStrategy::builder()
    ///                 .initial_delay(Duration::from_secs(2))
    ///                 .max_delay(Duration::from_secs(30))
    ///                 .build(),
    ///         )
    ///         .await?;
    ///     Ok(done)
    /// }
    /// ```
    #[allow(clippy::needless_pass_by_value)] // reason: API consistency with other builder chain methods
    pub fn wait_strategy(mut self, strategy: WaitStrategy) -> Self {
        // Convert the config struct into a functional strategy with
        // exponential backoff.
        let initial = strategy.initial_delay();
        let max = strategy.max_delay();
        let factor = strategy.backoff_factor();
        self.wait_strategy = Some(Box::new(move |_state: S, attempt: u32| {
            // Default behavior: always continue with backoff.
            #[allow(clippy::cast_possible_truncation)] // reason: attempt is small
            let exponent = attempt.saturating_sub(1);
            let base_secs =
                initial.as_secs_f64() * factor.powi(i32::try_from(exponent).unwrap_or(0));
            let capped = base_secs.min(max.as_secs_f64());
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            let delay_secs = capped.ceil().max(1.0) as u64;
            crate::wait_for_condition::WaitDecision::Continue {
                delay: Duration::from_secs(delay_secs),
            }
        }));
        self
    }

    /// Sets a custom wait strategy closure.
    ///
    /// The strategy receives the current (deserialized) state and the
    /// 1-based attempt number, and returns a [`WaitDecision`](crate::WaitDecision).
    /// The SDK boxes the closure internally — no `Box::new` at the call site.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::WaitDecision;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<i32, durable::BoxError> {
    ///     let done = ctx
    ///         .wait_for_condition(|_step_ctx, state: i32| async move { Ok(state + 1) }, 0_i32)
    ///         .wait_strategy_fn(|state: i32, _attempt| {
    ///             if state >= 3 {
    ///                 WaitDecision::complete()
    ///             } else {
    ///                 WaitDecision::continue_with(Duration::from_secs(1))
    ///             }
    ///         })
    ///         .await?;
    ///     Ok(done)
    /// }
    /// ```
    pub fn wait_strategy_fn<F>(mut self, strategy: F) -> Self
    where
        F: Fn(S, u32) -> crate::WaitDecision + Send + Sync + 'static,
    {
        self.wait_strategy = Some(Box::new(strategy));
        self
    }

    /// Sets a custom serializer/deserializer for the state.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<S> {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<S> {
        spawn_terminal!(self)
    }
}

impl<S: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + Sync + 'static> IntoFuture
    for WaitForConditionBuilder<S>
{
    type Output = Result<S, OperationError>;
    type IntoFuture = DurableFuture<S>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::wait_for_condition::WaitForConditionExecution;

        preflight_identity!(self, "Step", crate::wait_for_condition::WFC_SUB_TYPE);

        let check = self
            .check
            .unwrap_or_else(|| Box::new(|_ctx, state| Box::pin(async move { Ok(state) })));

        let execution = WaitForConditionExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            initial_state: self.initial_state,
            wait_strategy: self.wait_strategy,
            serdes: self.serdes,
            check,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// CreateCallbackBuilder
// ============================================================

/// Builder for creating a callback token.
///
/// Created by [`DurableContext::create_callback`]. The resulting
/// [`Callback`](crate::Callback) provides the token ID and a future for
/// the result.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use serde::Deserialize;
/// use std::time::Duration;
///
/// #[derive(Deserialize)]
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
pub struct CreateCallbackBuilder<O> {
    #[allow(dead_code)] // reason: not yet read by the builder body
    ctx: DurableContext,
    #[allow(dead_code)] // reason: not yet read by the builder body
    op_id: OperationId,
    name: Option<String>,
    #[allow(dead_code)] // reason: not yet read by the builder body
    timeout: Option<Duration>,
    #[allow(dead_code)] // reason: not yet read by the builder body
    heartbeat: Option<Duration>,
    serdes: Option<Box<dyn Serdes>>,
    _marker: PhantomData<O>,
}

impl<O> std::fmt::Debug for CreateCallbackBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreateCallbackBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: serde::de::DeserializeOwned + Send + 'static> CreateCallbackBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: None,
            _marker: PhantomData,
        }
    }

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
    /// delivered payload when the result is read. When no serdes is set here,
    /// the callback decode falls back to the execution-wide serdes configured
    /// on [`Options`](crate::Options), whose own default is JSON.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<crate::Callback<O>> {
        self.into_future()
    }

    /// Eagerly spawns the callback creation on a tokio task.
    pub fn spawn(self) -> DurableFuture<crate::Callback<O>> {
        spawn_terminal!(self)
    }
}

impl<O: serde::de::DeserializeOwned + Send + 'static> IntoFuture for CreateCallbackBuilder<O> {
    type Output = Result<crate::Callback<O>, OperationError>;
    type IntoFuture = DurableFuture<crate::Callback<O>>;

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
pub struct WaitForCallbackBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
    submitter: Option<crate::callback::BoxedSubmitter>,
    submitter_retry: Option<RetryStrategy>,
    serdes: Option<Box<dyn Serdes>>,
    _marker: PhantomData<O>,
}

impl<O> std::fmt::Debug for WaitForCallbackBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitForCallbackBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> WaitForCallbackBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            submitter: None,
            submitter_retry: None,
            serdes: None,
            _marker: PhantomData,
        }
    }

    /// Sets the submitter closure (internal — called from `DurableContext`).
    pub(crate) fn with_submitter(mut self, submitter: crate::callback::BoxedSubmitter) -> Self {
        self.submitter = Some(submitter);
        self
    }

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
    pub fn submitter_retry<F>(mut self, strategy: F) -> Self
    where
        F: Fn(&crate::StepError, u32) -> crate::RetryDecision + Send + Sync + 'static,
    {
        self.submitter_retry = Some(Box::new(strategy));
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
    /// delivered payload. When no serdes is set here, the callback decode
    /// falls back to the execution-wide serdes configured on
    /// [`Options`](crate::Options), whose own default is JSON.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> WaitForCallbackBuilder<O> {
    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for WaitForCallbackBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::callback::WaitForCallbackExecution;

        preflight_identity!(self, "Context", crate::callback::WFCB_SUB_TYPE);

        // The submitter is required — if somehow missing (shouldn't happen
        // since context.rs always provides it), use a no-op.
        let submitter = self.submitter.unwrap_or_else(|| {
            Box::new(|_ctx, _id| {
                Box::pin(async { Ok(()) })
                    as std::pin::Pin<Box<dyn Future<Output = Result<(), crate::BoxError>> + Send>>
            })
        });

        let execution = WaitForCallbackExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            timeout: self.timeout,
            heartbeat: self.heartbeat,
            submitter,
            submitter_retry: self.submitter_retry,
            serdes: self.serdes,
            _marker: PhantomData,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// MapBuilder
// ============================================================

/// Builder for a durable map operation.
///
/// Created by [`DurableContext::map`]. Applies a function to each item
/// with configurable concurrency and completion behavior.
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
pub struct MapBuilder<I, O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    serdes: Option<Box<dyn Serdes>>,
    result_serdes: Option<Box<dyn Serdes>>,
    nesting: crate::map_parallel::NestingMode,
    item_namer: Option<std::sync::Arc<dyn Fn(usize) -> String + Send + Sync>>,
    items: Vec<I>,
    #[allow(clippy::type_complexity)] // reason: boxed async closure factory
    closure: Option<
        std::sync::Arc<
            dyn Fn(
                    DurableContext,
                    I,
                    usize,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                            + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,
    _marker: PhantomData<(I, O)>,
}

impl<I, O> std::fmt::Debug for MapBuilder<I, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapBuilder")
            .field("name", &self.name)
            .field("max_concurrency", &self.max_concurrency)
            .finish_non_exhaustive()
    }
}

impl<I: Send + 'static, O: Send + 'static> MapBuilder<I, O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            serdes: None,
            result_serdes: None,
            nesting: crate::map_parallel::NestingMode::Normal,
            item_namer: None,
            items: Vec::new(),
            closure: None,
            _marker: PhantomData,
        }
    }

    /// Sets the items and closure (internal, called by `context.map()`).
    #[allow(clippy::type_complexity)] // reason: boxed async closure factory
    pub(crate) fn with_items_and_closure(
        mut self,
        items: Vec<I>,
        closure: std::sync::Arc<
            dyn Fn(
                    DurableContext,
                    I,
                    usize,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                            + Send,
                    >,
                > + Send
                + Sync,
        >,
    ) -> Self {
        self.items = items;
        self.closure = Some(closure);
        self
    }

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
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for item results.
    ///
    /// Item results go through the same JSON-string transform model as every
    /// other operation, so a [`Serdes`] attached here behaves exactly as it
    /// does on a step, invoke, callback, or `result_serdes`.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }

    /// Sets a custom serializer/deserializer for the whole batch result.
    ///
    /// This is the operation-level serdes: it serializes and deserializes
    /// the entire [`crate::BatchResult`] rather than individual items.
    pub fn result_serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.result_serdes = Some(Box::new(serdes));
        self
    }

    /// Sets the nesting mode for the map operation.
    ///
    /// [`crate::NestingMode::Flat`] causes items to run in virtual contexts
    /// without per-item context events.
    pub fn nesting(mut self, mode: crate::map_parallel::NestingMode) -> Self {
        self.nesting = mode;
        self
    }

    /// Sets a custom item namer for per-iteration display names.
    ///
    /// The namer function receives the zero-based item index and returns
    /// a display name for that iteration.
    pub fn item_namer(mut self, namer: impl Fn(usize) -> String + Send + Sync + 'static) -> Self {
        self.item_namer = Some(std::sync::Arc::new(namer));
        self
    }

    /// Executes the map and returns the full [`BatchResult`] including
    /// completion metadata (reason, success/failure counts, per-item status).
    ///
    /// Use this when you need to inspect batch completion details (e.g., when
    /// using a completion config that tolerates failures). The standard
    /// `.await` returns only `Vec<O>` with successful items.
    ///
    /// Always returns the full `BatchResult<O>` including per-item outcomes.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch execution encounters an infrastructure
    /// failure (checkpoint client error, task-ownership violation, invalid
    /// configuration). Item-level failures within tolerance are NOT errors —
    /// they appear as `BatchItemStatus::Failed` entries in the result.
    ///
    /// [`BatchResult`]: crate::BatchResult
    pub async fn await_batch(self) -> Result<crate::BatchResult<O>, OperationError>
    where
        I: serde::Serialize + serde::de::DeserializeOwned + Sync,
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        use crate::map_parallel::MapExecution;

        let closure = self.closure.unwrap_or_else(|| {
            std::sync::Arc::new(|_ctx, _item, _idx| {
                Box::pin(async { Err(crate::error::ChildFnError::new("map has no closure")) })
                    as std::pin::Pin<
                        Box<
                            dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                                + Send,
                        >,
                    >
            })
        });

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
            closure,
        };

        execution.execute_batch_result().await
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<Vec<O>>
    where
        I: serde::Serialize + serde::de::DeserializeOwned + Sync,
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        self.into_future()
    }

    /// Eagerly spawns the map operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<Vec<O>>
    where
        I: serde::Serialize + serde::de::DeserializeOwned + Sync,
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        spawn_terminal!(self)
    }
}

impl<
    I: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + 'static,
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
> IntoFuture for MapBuilder<I, O>
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::map_parallel::MapExecution;

        preflight_identity!(self, "Context", crate::map_parallel::MAP_SUB_TYPE);

        let closure = self.closure.unwrap_or_else(|| {
            std::sync::Arc::new(|_ctx, _item, _idx| {
                Box::pin(async { Err(crate::error::ChildFnError::new("map has no closure")) })
                    as std::pin::Pin<
                        Box<
                            dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                                + Send,
                        >,
                    >
            })
        });

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
            closure,
        };

        DurableFuture::from_async(async move { execution.execute().await })
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
pub struct ParallelBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    max_concurrency: Option<usize>,
    completion: Option<CompletionConfig>,
    serdes: Option<Box<dyn Serdes>>,
    result_serdes: Option<Box<dyn Serdes>>,
    nesting: crate::map_parallel::NestingMode,
    #[allow(clippy::type_complexity)] // reason: boxed future factory per branch
    branches: Vec<(
        String,
        Box<
            dyn FnOnce(
                    DurableContext,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                            + Send,
                    >,
                > + Send,
        >,
    )>,
    _marker: PhantomData<O>,
}

impl<O> std::fmt::Debug for ParallelBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> ParallelBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            max_concurrency: None,
            completion: None,
            serdes: None,
            result_serdes: None,
            nesting: crate::map_parallel::NestingMode::Normal,
            branches: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Sets the branches (internal, called by `context.parallel()`).
    #[allow(clippy::type_complexity)] // reason: boxed future factory per branch
    pub(crate) fn with_branches(
        mut self,
        branches: Vec<(
            String,
            Box<
                dyn FnOnce(
                        DurableContext,
                    ) -> std::pin::Pin<
                        Box<
                            dyn std::future::Future<Output = Result<O, crate::error::ChildFnError>>
                                + Send,
                        >,
                    > + Send,
            >,
        )>,
    ) -> Self {
        self.branches = branches;
        self
    }

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
    pub fn completion(mut self, config: CompletionConfig) -> Self {
        self.completion = Some(config);
        self
    }

    /// Sets a custom serializer/deserializer for branch results.
    ///
    /// Branch results go through the same JSON-string transform model as every
    /// other operation, so a [`Serdes`] attached here behaves exactly as it
    /// does on a step, invoke, callback, or `result_serdes`.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }

    /// Sets a custom serializer/deserializer for the whole batch result.
    ///
    /// This is the operation-level serdes: it serializes and deserializes
    /// the entire [`crate::BatchResult`] rather than individual items.
    pub fn result_serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.result_serdes = Some(Box::new(serdes));
        self
    }

    /// Sets the nesting mode for the parallel operation.
    ///
    /// [`crate::NestingMode::Flat`] causes branches to run in virtual contexts
    /// without per-branch context events.
    pub fn nesting(mut self, mode: crate::map_parallel::NestingMode) -> Self {
        self.nesting = mode;
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<Vec<O>>
    where
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        self.into_future()
    }

    /// Eagerly spawns the parallel operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<Vec<O>>
    where
        O: serde::Serialize + serde::de::DeserializeOwned,
    {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for ParallelBuilder<O>
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::map_parallel::ParallelExecution;

        preflight_identity!(self, "Context", crate::map_parallel::PARALLEL_SUB_TYPE);

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

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// Combinator builders
// ============================================================

// ============================================================
// TryJoinAllBuilder (hand-written — output is Vec<O>, not O)
// ============================================================

/// Builder for [`DurableContext::try_join_all`] — fail-fast join.
///
/// Awaits all futures concurrently and returns `Vec<O>` on success,
/// or propagates the first [`OperationError`] encountered.
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
///     let a = ctx.step(|_| async { Ok(1) }).future();
///     let b = ctx.step(|_| async { Ok(2) }).future();
///     let results: Vec<i32> = ctx.try_join_all([a, b]).name("gather").await?;
///     assert_eq!(results.len(), 2);
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct TryJoinAllBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    futures: Vec<DurableFuture<O>>,
}

impl<O> std::fmt::Debug for TryJoinAllBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TryJoinAllBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> TryJoinAllBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        futures: Vec<DurableFuture<O>>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            futures,
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

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for
    /// fan-out patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<Vec<O>> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running — this is
    /// the replay-safe alternative to bare `tokio::spawn` for
    /// durable operations.
    pub fn spawn(self) -> DurableFuture<Vec<O>> {
        spawn_combinator_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for TryJoinAllBuilder<O>
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::TryJoinAllExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let execution = TryJoinAllExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// JoinAllBuilder (hand-written — output is Vec<Settled<O>>, not O)
// ============================================================

/// Builder for [`DurableContext::join_all`] — collect all outcomes.
///
/// Awaits all futures concurrently and returns `Vec<Settled<O>>`.
/// Never short-circuits — every future runs to completion regardless
/// of individual failures.
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
///     let a = ctx.step(|_| async { Ok(1) }).future();
///     let b = ctx.step(|_| async { Ok(2) }).future();
///     let settled: Vec<durable::Settled<i32>> = ctx.join_all([a, b]).await?;
///     assert_eq!(settled.len(), 2);
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct JoinAllBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    futures: Vec<DurableFuture<O>>,
}

impl<O> std::fmt::Debug for JoinAllBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinAllBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> JoinAllBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        futures: Vec<DurableFuture<O>>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            futures,
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

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for
    /// fan-out patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<Vec<Settled<O>>> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running — this is
    /// the replay-safe alternative to bare `tokio::spawn` for
    /// durable operations.
    pub fn spawn(self) -> DurableFuture<Vec<Settled<O>>> {
        spawn_combinator_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for JoinAllBuilder<O>
{
    type Output = Result<Vec<Settled<O>>, OperationError>;
    type IntoFuture = DurableFuture<Vec<Settled<O>>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::JoinAllExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let execution = JoinAllExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// SelectOkBuilder (hand-written — needs futures storage)
// ============================================================

/// Builder for [`DurableContext::select_ok`] — first success wins.
///
/// Returns the first successful `O`; losers are cancelled.
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
///     let a = ctx.step(|_| async { Ok("a".to_owned()) }).future();
///     let b = ctx.step(|_| async { Ok("b".to_owned()) }).future();
///     let _winner: String = ctx.select_ok([a, b]).await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct SelectOkBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    futures: Vec<DurableFuture<O>>,
}

impl<O> std::fmt::Debug for SelectOkBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectOkBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> SelectOkBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        futures: Vec<DurableFuture<O>>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            futures,
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

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for
    /// fan-out patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running — this is
    /// the replay-safe alternative to bare `tokio::spawn` for
    /// durable operations.
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_combinator_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for SelectOkBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::SelectOkExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let execution = SelectOkExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

// ============================================================
// RaceBuilder (hand-written — needs futures storage)
// ============================================================

/// Builder for [`DurableContext::race`] — first settled wins.
///
/// Returns the first result (success or failure); losers are cancelled.
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
///     let a = ctx.step(|_| async { Ok("fast".to_owned()) }).future();
///     let b = ctx.step(|_| async { Ok("slow".to_owned()) }).future();
///     let _winner: String = ctx.race([a, b]).await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
pub struct RaceBuilder<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    futures: Vec<DurableFuture<O>>,
}

impl<O> std::fmt::Debug for RaceBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaceBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> RaceBuilder<O> {
    /// Creates a new builder (internal).
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        futures: Vec<DurableFuture<O>>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            futures,
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

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for
    /// fan-out patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running — this is
    /// the replay-safe alternative to bare `tokio::spawn` for
    /// durable operations.
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_combinator_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for RaceBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::RaceExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let execution = RaceExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
#[allow(clippy::panic)] // reason: test assertions on unexpected variants
mod tests {
    use super::*;

    /// A [`WaitStrategy`] built through its builder drives the derived polling
    /// schedule: the first delay is `initial_delay`, each subsequent delay
    /// grows by `backoff_factor`, and the sequence is capped at `max_delay`.
    #[test]
    fn wait_strategy_builder_drives_polling_schedule() {
        let ctx = DurableContext::__test_context();

        let builder = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy(
                WaitStrategy::builder()
                    .initial_delay(Duration::from_secs(2))
                    .max_delay(Duration::from_secs(10))
                    .backoff_factor(3.0)
                    .build(),
            );

        let strategy = builder
            .wait_strategy
            .as_ref()
            .expect("wait_strategy must install a strategy");

        let delay_of = |attempt: u32| match strategy(0_i32, attempt) {
            crate::WaitDecision::Continue { delay } => delay,
            other => panic!("config-derived strategy must always continue, got {other:?}"),
        };

        // attempt 1 → initial (2s); attempt 2 → 2s * 3 = 6s;
        // attempt 3 → 18s, capped at max_delay (10s).
        assert_eq!(delay_of(1), Duration::from_secs(2));
        assert_eq!(delay_of(2), Duration::from_secs(6));
        assert_eq!(delay_of(3), Duration::from_secs(10));
        assert_eq!(delay_of(9), Duration::from_secs(10));
    }

    /// The default [`WaitStrategy`] keeps its pre-builder behavior: a 1 second
    /// first delay doubling up to the 1 minute cap.
    #[test]
    fn default_wait_strategy_schedule_unchanged() {
        let ctx = DurableContext::__test_context();

        let builder = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy(WaitStrategy::default());

        let strategy = builder
            .wait_strategy
            .as_ref()
            .expect("wait_strategy must install a strategy");

        let delay_of = |attempt: u32| match strategy(0_i32, attempt) {
            crate::WaitDecision::Continue { delay } => delay,
            other => panic!("config-derived strategy must always continue, got {other:?}"),
        };

        assert_eq!(delay_of(1), Duration::from_secs(1));
        assert_eq!(delay_of(2), Duration::from_secs(2));
        assert_eq!(delay_of(3), Duration::from_secs(4));
        assert_eq!(delay_of(20), Duration::from_mins(1));
    }

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
