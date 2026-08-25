//! The chained-invoke operation: [`InvokeBuilder`], returned by
//! [`DurableContext::invoke`](crate::DurableContext::invoke).
//!
//! The [invoke operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/invoke/)
//! describes this operation independently of any language SDK.

use std::future::IntoFuture;
use std::marker::PhantomData;

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

/// Builder for a durable invoke operation.
///
/// Created by [`DurableContext::invoke`]. Configure with `.name()` and
/// custom serialization before awaiting.
///
/// The builder carries the typed input `I` and two serdes implementations
/// as generic parameters, both defaulting to
/// [`JsonSerdes`]: `PS` serializes the input
/// payload ([`payload_serdes`](Self::payload_serdes)) and `RS`
/// deserializes the target function's result ([`serdes`](Self::serdes)).
/// Whatever the serdes types, the finalized future is [`DurableFuture<O>`].
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Input { id: u64 }
///
/// #[derive(Serialize, Deserialize)]
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
#[non_exhaustive]
pub struct InvokeBuilder<O, I, PS = JsonSerdes, RS = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    function_id: String,
    input: I,
    payload_serdes: PS,
    result_serdes: RS,
    tenant_id: Option<String>,
    _marker: PhantomData<O>,
}

impl<O, I, PS, RS> std::fmt::Debug for InvokeBuilder<O, I, PS, RS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvokeBuilder")
            .field("name", &self.name)
            .field("function_id", &self.function_id)
            .finish_non_exhaustive()
    }
}

impl<O, I> InvokeBuilder<O, I>
where
    O: Send + 'static,
    I: Send + 'static,
{
    /// Creates a new invoke builder (internal).
    pub(crate) fn new_internal(
        ctx: DurableContext,
        op_id: OperationId,
        function_id: String,
        input: I,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            function_id,
            input,
            payload_serdes: JsonSerdes,
            result_serdes: JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        }
    }
}

impl<O, I, PS, RS> InvokeBuilder<O, I, PS, RS>
where
    O: Send + 'static,
    I: Send + 'static,
{
    /// Sets a human-readable name for this invoke.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets a custom serializer/deserializer for this invoke's result
    /// deserialization.
    ///
    /// The serdes is applied when deserializing the invoke result payload
    /// returned by the target function. It must implement
    /// [`Serdes<O>`](crate::Serdes) for this invoke's output type:
    /// attaching a serdes for a different type fails at compile time. This
    /// is independent of the payload serdes set via
    /// [`payload_serdes`](Self::payload_serdes).
    pub fn serdes<RS2>(self, serdes: RS2) -> InvokeBuilder<O, I, PS, RS2>
    where
        RS2: Serdes<O>,
    {
        InvokeBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            function_id: self.function_id,
            input: self.input,
            payload_serdes: self.payload_serdes,
            result_serdes: serdes,
            tenant_id: self.tenant_id,
            _marker: PhantomData,
        }
    }

    /// Sets a custom serializer for this invoke's input payload
    /// serialization.
    ///
    /// The serdes is applied when serializing the input payload before
    /// sending it to the target function; the owned input transfers to the
    /// serdes directly (a write-only payload has no round-trip to
    /// preserve). It must implement [`Serdes<I>`](crate::Serdes) for the
    /// input's type. This is independent of the result serdes set via
    /// [`.serdes()`](Self::serdes).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::SerdesContext;
    ///
    /// # struct UpperSerdes;
    /// # impl durable::Serdes<String> for UpperSerdes {
    /// #     async fn serialize(&self, v: String, _c: SerdesContext) -> Result<String, durable::BoxError> { Ok(v.to_uppercase()) }
    /// #     async fn deserialize(&self, w: String, _c: SerdesContext) -> Result<String, durable::BoxError> { Ok(w.to_lowercase()) }
    /// # }
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.invoke::<String, _>("target-fn", "hello".to_owned())
    ///         .payload_serdes(UpperSerdes)
    ///         .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn payload_serdes<PS2>(self, serdes: PS2) -> InvokeBuilder<O, I, PS2, RS>
    where
        PS2: Serdes<I>,
    {
        InvokeBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            function_id: self.function_id,
            input: self.input,
            payload_serdes: serdes,
            result_serdes: self.result_serdes,
            tenant_id: self.tenant_id,
            _marker: PhantomData,
        }
    }

    /// Sets the tenant ID for tenant-isolated invocations.
    ///
    /// When set, the target function is invoked in the context of the
    /// specified tenant.
    pub fn tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
}

impl<O, I, PS, RS> InvokeBuilder<O, I, PS, RS>
where
    O: Send + 'static,
    I: Send + 'static,
    PS: Serdes<I>,
    RS: Serdes<O>,
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

    /// Eagerly spawns the invoke on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime.
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
    ///     let handle = ctx.invoke::<String, _>("fn", "input".to_owned())
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

impl<O, I, PS, RS> IntoFuture for InvokeBuilder<O, I, PS, RS>
where
    O: Send + 'static,
    I: Send + 'static,
    PS: Serdes<I>,
    RS: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::invoke::InvokeExecution;

        preflight_identity!(
            self,
            "ChainedInvoke",
            crate::invoke::CHAINED_INVOKE_SUB_TYPE
        );

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = InvokeExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            function_id: self.function_id,
            input: self.input,
            payload_serdes: self.payload_serdes,
            result_serdes: self.result_serdes,
            tenant_id: self.tenant_id,
            _marker: PhantomData,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}
