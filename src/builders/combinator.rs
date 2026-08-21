//! The future combinators: [`TryJoinAllBuilder`], [`JoinAllBuilder`],
//! [`SelectOkBuilder`], and [`RaceBuilder`], returned by
//! [`DurableContext::try_join_all`](crate::DurableContext::try_join_all),
//! [`DurableContext::join_all`](crate::DurableContext::join_all),
//! [`DurableContext::select_ok`](crate::DurableContext::select_ok), and
//! [`DurableContext::race`](crate::DurableContext::race).
//!
//! The combinators are this SDK's composition surface over
//! [`DurableFuture`] values; the
//! [operations section of the SDK reference](https://docs.aws.amazon.com/durable-execution/sdk-reference/)
//! describes the underlying operations independently of any language SDK.

use std::future::IntoFuture;

use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::{DurableFuture, Settled};

/// Builder for [`DurableContext::try_join_all`]: fail-fast join.
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
#[non_exhaustive]
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

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<Vec<O>> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime.
    pub fn spawn(self) -> DurableFuture<Vec<O>> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for TryJoinAllBuilder<O>
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::combinator::TryJoinAllExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = TryJoinAllExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}

/// Builder for [`DurableContext::join_all`]: collect all outcomes.
///
/// Awaits all futures concurrently and returns `Vec<Settled<O>>`.
/// Never short-circuits: every future runs to completion regardless
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
#[non_exhaustive]
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

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<Vec<Settled<O>>> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime.
    pub fn spawn(self) -> DurableFuture<Vec<Settled<O>>> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for JoinAllBuilder<O>
{
    type Output = Result<Vec<Settled<O>>, OperationError>;
    type IntoFuture = DurableFuture<Vec<Settled<O>>>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::combinator::JoinAllExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = JoinAllExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}

/// Builder for [`DurableContext::select_ok`]: first success wins.
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
#[non_exhaustive]
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

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime.
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for SelectOkBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::combinator::SelectOkExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = SelectOkExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}

/// Builder for [`DurableContext::race`]: first settled wins.
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
#[non_exhaustive]
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

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime.
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for RaceBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::combinator::RaceExecution;

        preflight_identity!(self, "Context", crate::combinator::COMBINATOR_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = RaceExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            futures: self.futures,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}
