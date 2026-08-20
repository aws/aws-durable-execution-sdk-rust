//! The child-context operation: [`ChildBuilder`], returned by
//! [`DurableContext::run_in_child_context`](crate::DurableContext::run_in_child_context).

use std::future::IntoFuture;
use std::sync::Arc;

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;

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
    serdes: Option<Arc<dyn Serdes>>,
    closure: crate::child::BoxedChildBody<O>,
}

impl<O> std::fmt::Debug for ChildBuilder<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O: Send + 'static> ChildBuilder<O> {
    /// Creates a new builder (internal). Taking the closure here keeps the
    /// field non-optional: a builder without a body is unrepresentable.
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        closure: crate::child::BoxedChildBody<O>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
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

    /// Overrides the serialization strategy for the child result.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Arc::new(serdes));
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

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            serdes: self.serdes,
            closure: self.closure,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}
