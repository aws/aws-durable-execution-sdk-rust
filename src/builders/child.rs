//! The child-context operation: [`ChildBuilder`], returned by
//! [`DurableContext::run_in_child_context`](crate::DurableContext::run_in_child_context).
//!
//! The [child context operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/child-context/)
//! describes this operation independently of any language SDK.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;

use crate::BoxError;
use crate::OperationSubType;
use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

/// Builder for a child context (sub-orchestration) operation.
///
/// Created by [`DurableContext::run_in_child_context`]. Use `.spawn()` for
/// eager fan-out execution.
///
/// The builder is generic over the body closure `F` and its future
/// `Fut`; both parameters are inferred at the call site and never written
/// by users.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
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
    sub_type: Option<OperationSubType>,
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
    pub(crate) fn new_internal(ctx: DurableContext, op_id: OperationId, closure: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            sub_type: None,
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

    /// Records `sub_type` instead of the default subtype.
    ///
    /// The subtype labels this child context in its checkpoint, in place
    /// of `RunInChildContext`. So the child contexts of an operation
    /// composed from several durable operations can be told apart.
    /// Execution history and plugins read it. Without this call the child
    /// context records [`OperationSubType::RUN_IN_CHILD_CONTEXT`]. A second
    /// call replaces the first.
    ///
    /// Replay compares the recorded subtype with this one, as it compares
    /// the name. So changing the subtype of an existing child context
    /// breaks the executions in flight: their replay fails with
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
    ///     let order = ctx
    ///         .run_in_child_context(|child| async move {
    ///             let id = child
    ///                 .step(|_| async { Ok("order-1".to_owned()) })
    ///                 .name("reserve")
    ///                 .await?;
    ///             Ok(id)
    ///         })
    ///         .name("checkout")
    ///         .sub_type(OperationSubType::new("Checkout")?)
    ///         .await?;
    ///     Ok(order)
    /// }
    /// ```
    pub fn sub_type(mut self, sub_type: OperationSubType) -> Self {
        self.sub_type = Some(sub_type);
        self
    }

    /// Overrides the serialization strategy for the child result.
    ///
    /// Replaces the builder's serdes type parameter with `S2`, which must
    /// implement [`Serdes<O>`](crate::Serdes) for this child's output type:
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
            sub_type: self.sub_type,
            serdes,
            closure: self.closure,
            _marker: PhantomData,
        }
    }

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<O>
    where
        S: Serdes<O>,
    {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the child context on a tokio task.
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

    fn into_future(self) -> Self::IntoFuture {
        use crate::child::ChildExecution;

        preflight_identity!(
            self,
            "Context",
            self.sub_type
                .as_ref()
                .map_or(crate::child::CHILD_SUB_TYPE, OperationSubType::as_str)
        );

        let gate = self.ctx.clone();

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            sub_type: self
                .sub_type
                .unwrap_or(OperationSubType::RUN_IN_CHILD_CONTEXT),
            serdes: self.serdes,
            closure: self.closure,
            _marker: PhantomData,
        };

        DurableFuture::for_operation(&gate, execution.execute())
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
            .run_in_child_context(|_child| async { Ok(1_u32) })
            .sub_type(OperationSubType::new("First").unwrap())
            .sub_type(OperationSubType::new("Second").unwrap())
            .serdes(std::sync::Arc::new(JsonSerdes));
        assert_eq!(
            builder.sub_type.as_ref().map(OperationSubType::as_str),
            Some("Second")
        );
    }
}
