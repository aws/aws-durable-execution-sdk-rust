//! The child-context operation: [`ChildBuilder`], returned by
//! [`DurableContext::run_in_child_context`](crate::DurableContext::run_in_child_context).

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;

use crate::BoxError;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

// ============================================================
// ChildBuilder
// ============================================================

/// Builder for a child context (sub-orchestration) operation.
///
/// Created by [`DurableContext::run_in_child_context`]. Use `.spawn()` for
/// eager fan-out execution.
///
/// The builder is generic over the body closure `F` and its future `Fut`
/// so the body is stored **without type erasure**; both parameters are
/// inferred at the call site and never written by users. The single
/// erasure point is `.future()` / `.await`, which produces the one
/// [`DurableFuture`] box.
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
#[non_exhaustive]
pub struct ChildBuilder<O, F, Fut, S = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    serdes: S,
    closure: F,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<O, F, Fut, S> std::fmt::Debug for ChildBuilder<O, F, Fut, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O, F, Fut> ChildBuilder<O, F, Fut>
where
    O: Send + 'static,
    F: FnOnce(DurableContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the closure here keeps the
    /// field non-optional: a builder without a body is unrepresentable.
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            serdes: JsonSerdes,
            closure,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> ChildBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(DurableContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this operation.
    ///
    /// Names appear in logs, traces, and the execution history for
    /// debugging purposes.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Overrides the serialization strategy for the child result.
    ///
    /// Replaces the builder's serdes type parameter with `S2`, which must
    /// implement [`Serdes<O>`](crate::Serdes) for this child's output type —
    /// attaching a serdes for a different type fails at compile time. To
    /// share one instance across operations, wrap it in an
    /// [`Arc`](std::sync::Arc) and clone the `Arc` handle into each builder.
    pub fn serdes<S2>(self, serdes: S2) -> ChildBuilder<O, F, Fut, S2>
    where
        S2: Serdes<O>,
    {
        ChildBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            serdes,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    ///
    /// Equivalent to `.into_future()` but more discoverable for
    /// fan-out patterns where you need to hold multiple futures.
    pub fn future(self) -> DurableFuture<O>
    where
        S: Serdes<O>,
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
        S: Serdes<O>,
    {
        spawn_terminal!(self)
    }
}

impl<O, F, Fut, S> IntoFuture for ChildBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(DurableContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::child::ChildExecution;

        preflight_identity!(self, "Context", crate::child::CHILD_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            serdes: self.serdes,
            closure: self.closure,
            _marker: PhantomData,
        };

        DurableFuture::lazy_scoped(
            async move { execution.execute().await },
            owner_scope,
            op_scope,
        )
    }
}
