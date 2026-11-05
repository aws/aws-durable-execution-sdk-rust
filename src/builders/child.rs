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
    /// Whether the body is a user function. Only `with_retry` clears it,
    /// for its attempt children.
    user_function: crate::child::UserFunctionPolicy,
    summary: Option<crate::child::ChildSummary<O>>,
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
            user_function: crate::child::UserFunctionPolicy::Hooks,
            summary: None,
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
    /// Marks the body as no user function: no frame and no user-function
    /// hooks. `with_retry` sets it for its attempt children in both modes.
    /// For a block with its own record, the outer block is the user
    /// function, as JS wraps only that block. A block with
    /// `.without_child_context()` has no outer block, so no user function
    /// runs for it or its attempts.
    pub(crate) fn without_user_function(mut self) -> Self {
        self.user_function = crate::child::UserFunctionPolicy::None;
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
            user_function: self.user_function,
            summary: self.summary,
            _marker: PhantomData,
        }
    }

    /// Sets the payload recorded when the child's result is too large to
    /// checkpoint.
    ///
    /// Called only when the serialized result is over 256 KiB. The child
    /// then records its result with `ReplayChildren` and no value, so a
    /// replay runs the body again to rebuild the result. The summary
    /// receives the value the child returns, after the serdes round trip,
    /// and its output becomes the record's payload. Without a summary the
    /// payload is `""`. A result that fits is recorded as usual, and the
    /// summary is not called.
    ///
    /// The output is observability only: replay never reads it. A replay
    /// that rebuilds the result of a completed `ReplayChildren` record does
    /// not call the summary; a child that is still running when an
    /// invocation resumes it calls the summary when it completes with an
    /// oversized result. Keep it under 256 KiB, because the service rejects
    /// a larger payload. A second call replaces the first.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<usize, durable::BoxError> {
    ///     let report = ctx
    ///         .run_in_child_context(|child| async move {
    ///             let report = child
    ///                 .step(|_| async { Ok("x".repeat(300 * 1024)) })
    ///                 .name("render")
    ///                 .await?;
    ///             Ok(report)
    ///         })
    ///         .name("report")
    ///         .summary(|report: &String| format!("len={}", report.len()))
    ///         .await?;
    ///     Ok(report.len())
    /// }
    /// ```
    pub fn summary(mut self, summary: impl Fn(&O) -> String + Send + Sync + 'static) -> Self {
        self.summary = Some(Box::new(summary));
        self
    }

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
        spawn_terminal!(self, started)
    }
}

impl<O, F, Fut, S> ChildBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: FnOnce(DurableContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// The operation's future. With `started`, its start step runs now, on
    /// the caller's task (see [`crate::start`]); `.spawn()` uses that.
    fn build(self, started: bool) -> DurableFuture<O> {
        use crate::child::ChildExecution;

        preflight_identity!(
            self,
            "Context",
            self.sub_type
                .as_ref()
                .map_or(crate::child::CHILD_SUB_TYPE, OperationSubType::as_str)
        );

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            sub_type: self
                .sub_type
                .unwrap_or(OperationSubType::RUN_IN_CHILD_CONTEXT),
            serdes: self.serdes,
            closure: self.closure,
            user_function: self.user_function,
            summary: self.summary,
            _marker: PhantomData,
        };

        if started {
            DurableFuture::for_started_operation(
                &execution.ctx.clone(),
                execution.execute_started(),
            )
        } else {
            DurableFuture::for_start_step(&execution.ctx.clone(), move || {
                execution.execute_started()
            })
        }
    }

    /// The operation's future, with its start step run now. An unwrapped
    /// `with_retry` block also starts its first attempt this way.
    pub(crate) fn started_future(self) -> DurableFuture<O> {
        self.build(true)
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
        self.build(false)
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

    /// A second `summary` call replaces the first, and `serdes` keeps the
    /// summary.
    #[test]
    fn summary_setter_replaces_and_survives_serdes() {
        let ctx = DurableContext::test_context();
        let builder = ctx
            .run_in_child_context(|_child| async { Ok(1_u32) })
            .summary(|v: &u32| format!("first {v}"))
            .summary(|v: &u32| format!("second {v}"))
            .serdes(std::sync::Arc::new(JsonSerdes));
        let summary = builder.summary.as_ref().expect("the summary is kept");
        assert_eq!(summary(&7), "second 7");
    }
}
