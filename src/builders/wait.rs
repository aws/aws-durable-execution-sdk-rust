//! The wait (durable timer) operation: [`WaitBuilder`], returned by
//! [`DurableContext::wait`](crate::DurableContext::wait).
//!
//! The [wait operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/wait/)
//! describes this operation independently of any language SDK.

use std::future::IntoFuture;

use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;

/// Builder for a durable wait (timer) operation.
///
/// Created by [`DurableContext::wait`]. The wait duration is set at
/// creation; chain `.name()` for identification.
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
/// ) -> Result<(), durable::BoxError> {
///     ctx.wait(Duration::from_secs(30))
///         .name("pause")
///         .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
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
    pub(crate) fn new_internal(
        ctx: DurableContext,
        op_id: OperationId,
        duration_secs: i32,
    ) -> Self {
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

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<()> {
        self.into_future()
    }

    /// Eagerly spawns the wait on a tokio task.
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

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::wait::WaitExecution;

        preflight_identity!(self, "Wait", crate::wait::WAIT_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = WaitExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            duration_secs: self.duration_secs,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}
