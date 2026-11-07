//! The chained-invoke operation: [`InvokeBuilder`], returned by
//! [`DurableContext::invoke`](crate::DurableContext::invoke).
//!
//! The [invoke operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/invoke/)
//! describes this operation independently of any language SDK.

use std::future::IntoFuture;
use std::marker::PhantomData;

use serde::Serialize;

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;
use crate::serdes::PayloadSerializer;

/// Builder for a durable invoke operation.
///
/// Created by [`DurableContext::invoke`]. Configure with `.name()` and
/// custom serialization before awaiting.
///
/// The builder carries the input `I` and two serializers as generic
/// parameters, both defaulting to [`JsonSerdes`]: the
/// [`PayloadSerializer`] `PS` serializes the input payload
/// ([`payload_serdes`](Self::payload_serdes)) and the [`Serdes<O>`] `RS`
/// deserializes the target function's result ([`serdes`](Self::serdes)).
/// [`DurableContext::invoke`] returns the input type as an opaque
/// `impl Serialize`, so callers write only the output type. Whatever the
/// serializer types, the finalized future is [`DurableFuture<O>`].
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize)]
/// struct Input { id: u64 }
///
/// #[derive(Serialize, Deserialize)]
/// struct Output { status: String }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     let out = ctx.invoke::<Output>("target-fn", Input { id: 1 })
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

    /// Sets the serializer for this invoke's input payload.
    ///
    /// The serializer runs when the invoke sends its input to the target
    /// function, and receives the owned input. It must implement
    /// [`PayloadSerializer`], which [`JsonSerdes`], the default, and
    /// `Arc<S>` of a payload serializer `S` implement. This is independent
    /// of the result serdes set via [`.serdes()`](Self::serdes).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::SerdesContext;
    ///
    /// struct UpperJson;
    ///
    /// impl durable::PayloadSerializer for UpperJson {
    ///     async fn serialize_payload<T>(
    ///         &self,
    ///         value: T,
    ///         _context: SerdesContext,
    ///     ) -> Result<String, durable::BoxError>
    ///     where
    ///         T: serde::Serialize + Send + 'static,
    ///     {
    ///         Ok(serde_json::to_string(&value)?.to_uppercase())
    ///     }
    /// }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.invoke::<String>("target-fn", "hello".to_owned())
    ///         .payload_serdes(UpperJson)
    ///         .await?;
    ///     Ok(result)
    /// }
    /// ```
    ///
    /// [`PassthroughSerdes`](crate::PassthroughSerdes) handles only strings,
    /// so it is not a payload serializer:
    ///
    /// ```compile_fail,E0277
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     Ok(ctx
    ///         .invoke::<String>("target-fn", "hello".to_owned())
    ///         .payload_serdes(durable::PassthroughSerdes)
    ///         .await?)
    /// }
    /// ```
    ///
    /// The target does not resolve the storage envelope of
    /// [`FileSystemSerdes`](crate::serdes::FileSystemSerdes), so it is not a
    /// payload serializer either:
    ///
    /// ```compile_fail,E0277
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::FileSystemSerdes;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     Ok(ctx
    ///         .invoke::<String>("target-fn", "hello".to_owned())
    ///         .payload_serdes(FileSystemSerdes::new("payloads"))
    ///         .await?)
    /// }
    /// ```
    ///
    /// Nor is a type that implements only [`Serdes<String>`]:
    ///
    /// ```compile_fail,E0277
    /// use aws_durable_execution_sdk as durable;
    /// use durable::serdes::{Serdes, SerdesContext};
    ///
    /// struct Upper;
    ///
    /// impl Serdes<String> for Upper {
    ///     async fn serialize(
    ///         &self,
    ///         value: String,
    ///         _context: SerdesContext,
    ///     ) -> Result<String, durable::BoxError> {
    ///         Ok(value.to_uppercase())
    ///     }
    ///
    ///     async fn deserialize(
    ///         &self,
    ///         wire: String,
    ///         _context: SerdesContext,
    ///     ) -> Result<String, durable::BoxError> {
    ///         Ok(wire)
    ///     }
    /// }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     Ok(ctx
    ///         .invoke::<String>("target-fn", "hello".to_owned())
    ///         .payload_serdes(Upper)
    ///         .await?)
    /// }
    /// ```
    pub fn payload_serdes<PS2: PayloadSerializer>(
        self,
        serdes: PS2,
    ) -> InvokeBuilder<O, I, PS2, RS> {
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
    I: Serialize + Send + 'static,
    PS: PayloadSerializer,
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
    ///
    /// A combinator begins its `.future()` inputs' start phases, in input
    /// order, before its own start step. This operation first serializes
    /// its input, so the combinator polls it once on its own task, and does
    /// not wait for its `START`: its `START` and its start hook can come
    /// after the combinator's. A replay of the combinator's final record
    /// starts no `.future()` input.
    ///
    /// A handle that [`.spawn()`](Self::spawn) returned already runs on its
    /// own task. A combinator that receives it never starts it again, and
    /// it keeps running independently: when the combinator resolves first,
    /// it does not cancel the handle, and the handle's result is dropped.
    pub fn future(self) -> DurableFuture<O> {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the invoke on a tokio task.
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
    /// Unlike `.spawn()` of a step, a wait or a child context, `.spawn()`
    /// of an invoke runs no start step on the caller's task: the spawned task
    /// runs the whole operation, including the `START`, so that `START` can
    /// follow updates the caller sends after `.spawn()` returns. The
    /// caller's task ownership is checked before the operation is built,
    /// not on the spawned task, so that error wins over a replay mismatch:
    /// a `.spawn()` from a task or thread that does not own the context
    /// fails the execution with a context usage error, returns a handle
    /// that resolves to the task-ownership error, queues no `START` and
    /// runs no body.
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
    ///     let handle = ctx.invoke::<String>("fn", "input".to_owned())
    ///         .name("bg-call")
    ///         .spawn();
    ///     let _result = handle.await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self, awaited)
    }
}

impl<O, I, PS, RS> IntoFuture for InvokeBuilder<O, I, PS, RS>
where
    O: Send + 'static,
    I: Serialize + Send + 'static,
    PS: PayloadSerializer,
    RS: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::invoke::InvokeExecution;

        preflight_identity!(
            self,
            "ChainedInvoke",
            crate::invoke::CHAINED_INVOKE_SUB_TYPE
        );

        let gate = self.ctx.clone();

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

        DurableFuture::for_first_poll_operation(&gate, execution.execute())
    }
}
