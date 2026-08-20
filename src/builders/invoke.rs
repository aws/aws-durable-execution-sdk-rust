//! The chained-invoke operation: [`InvokeBuilder`], returned by
//! [`DurableContext::invoke`](crate::DurableContext::invoke).

use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;

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
    payload_serdes: Option<Arc<dyn Serdes>>,
    result_serdes: Option<Arc<dyn Serdes>>,
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
        self.result_serdes = Some(Arc::new(serdes));
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
    /// #     fn serialize(&self, v: &serde_json::Value, _c: &durable::serdes::SerdesContext) -> Result<String, durable::BoxError> { Ok(v.to_string().to_uppercase()) }
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
        self.payload_serdes = Some(Arc::new(serdes));
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
