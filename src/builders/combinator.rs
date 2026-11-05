//! The future combinators: [`TryJoinAllBuilder`], [`JoinAllBuilder`],
//! [`SelectOkBuilder`], and [`RaceBuilder`], returned by
//! [`DurableContext::try_join_all`](crate::DurableContext::try_join_all),
//! [`DurableContext::join_all`](crate::DurableContext::join_all),
//! [`DurableContext::select_ok`](crate::DurableContext::select_ok), and
//! [`DurableContext::race`](crate::DurableContext::race).
//!
//! A combinator claims its operation ID when it is first awaited or
//! polled, or when it is spawned, not when the context method is called.
//! Every other operation claims its ID when its builder is created. A
//! combinator builder or future that is dropped before that point claims no
//! ID and records nothing.
//!
//! The combinators are this SDK's composition surface over
//! [`DurableFuture`] values; the
//! [operations section of the SDK reference](https://docs.aws.amazon.com/durable-execution/sdk-reference/)
//! describes the underlying operations independently of any language SDK.

use std::future::IntoFuture;

use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::{DurableFuture, PendingClaim};

/// The operation ID a combinator claimed on its first poll or at
/// `.spawn()`, with the context and name it was claimed with.
struct Claimed {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
}

/// Builder for [`DurableContext::try_join_all`]: fail-fast join.
///
/// Awaits all futures concurrently and returns `Vec<O>` on success,
/// or propagates the first [`OperationError`] encountered.
///
/// The returned value has passed through the combinator's JSON encoding,
/// so it equals the value replay returns.
///
/// When the combinator resolves, it stops every input that it started from
/// `.future()` and that is still running. A stopped input stops at its next
/// `.await`. An input started with `.spawn()` keeps running, and its result
/// is dropped.
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
    pub(crate) fn new_internal(ctx: DurableContext, futures: Vec<DurableFuture<O>>) -> Self {
        Self {
            ctx,
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
    ///
    /// The returned future claims the combinator's operation ID on its
    /// first poll, not here. Dropped before its first poll, it claims no ID
    /// and records nothing. Do not pass it to another combinator: an outer
    /// combinator that would start a still-unclaimed combinator fails the
    /// execution with a context usage error. Pass the handle that
    /// [`.spawn()`](Self::spawn) returns instead.
    pub fn future(self) -> DurableFuture<Vec<O>> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// The combinator claims its operation ID at this call, and checks it
    /// against the recorded history before the task starts. The returned
    /// handle has claimed, so it is a valid input to another combinator. A
    /// `.spawn()` from a task or thread that does not own the context
    /// claims no ID, fails the execution with a context usage error,
    /// returns a handle that resolves to the task-ownership error, queues
    /// no `START` and runs no body.
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
    pub fn spawn(self) -> DurableFuture<Vec<O>> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for TryJoinAllBuilder<O>
{
    type Output = Result<Vec<O>, OperationError>;
    type IntoFuture = DurableFuture<Vec<O>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::TryJoinAllExecution;

        let Self { ctx, name, futures } = self;
        let context = crate::context::display_prefix(ctx.prefix()).to_owned();
        DurableFuture::deferred(PendingClaim::new("try_join_all", context, move || {
            let claimed = Claimed {
                op_id: ctx.claim_operation_id("try_join_all"),
                ctx,
                name,
            };
            preflight_identity!(claimed, "Context", crate::combinator::COMBINATOR_SUB_TYPE);
            let gate = claimed.ctx.clone();
            let execution = TryJoinAllExecution {
                ctx: claimed.ctx,
                op_id: claimed.op_id,
                name: claimed.name,
                futures,
            };
            DurableFuture::for_operation(&gate, execution.execute())
        }))
    }
}

/// Builder for [`DurableContext::join_all`]: collect all outcomes.
///
/// Awaits all futures concurrently and returns one
/// `Result<O, OperationError>` per input, in input order. Never
/// short-circuits: every future runs to completion regardless of
/// individual failures.
///
/// Awaiting the builder returns
/// `Result<Vec<Result<O, OperationError>>, OperationError>`. The two
/// levels report different failures:
///
/// - The outer `Err` means the combinator itself failed, for example
///   replay nondeterminism or a checkpoint write failure.
/// - An input that returns `Err` appears only as that input's inner `Err`,
///   and the outer result is `Ok`.
/// - A panicking input ends the invocation, as a panic does everywhere else
///   in the SDK. The SDK resumes the original panic payload, and the
///   combinator writes no record and returns no result.
/// - Nothing cancels an input that `join_all` awaits: `join_all` drops no
///   input before it settles.
///
/// An inner `Err` is a `ChildContext` error of kind `ChildFailed` whose
/// `wire()` carries the input's recorded failure, on the first run and on
/// replay. Match on the `err.wire()` fields, not on the kind of the input
/// that failed.
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
///     let a = ctx.step(|_| async { Ok(1) }).future();
///     let b = ctx.step(|_| async { Ok(2) }).future();
///     let outcomes: Vec<Result<i32, durable::OperationError>> =
///         ctx.join_all([a, b]).await?;
///     let rendered: Vec<String> = outcomes
///         .iter()
///         .map(|outcome| match outcome {
///             Ok(value) => format!("ok:{value}"),
///             Err(err) => format!("err:{err}"),
///         })
///         .collect();
///     assert_eq!(rendered.len(), 2);
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct JoinAllBuilder<O> {
    ctx: DurableContext,
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
    pub(crate) fn new_internal(ctx: DurableContext, futures: Vec<DurableFuture<O>>) -> Self {
        Self {
            ctx,
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
    ///
    /// The returned future claims the combinator's operation ID on its
    /// first poll, not here. Dropped before its first poll, it claims no ID
    /// and records nothing. Do not pass it to another combinator: an outer
    /// combinator that would start a still-unclaimed combinator fails the
    /// execution with a context usage error. Pass the handle that
    /// [`.spawn()`](Self::spawn) returns instead.
    pub fn future(self) -> DurableFuture<Vec<Result<O, OperationError>>> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// The combinator claims its operation ID at this call, and checks it
    /// against the recorded history before the task starts. The returned
    /// handle has claimed, so it is a valid input to another combinator. A
    /// `.spawn()` from a task or thread that does not own the context
    /// claims no ID, fails the execution with a context usage error,
    /// returns a handle that resolves to the task-ownership error, queues
    /// no `START` and runs no body.
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
    pub fn spawn(self) -> DurableFuture<Vec<Result<O, OperationError>>> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for JoinAllBuilder<O>
{
    type Output = Result<Vec<Result<O, OperationError>>, OperationError>;
    type IntoFuture = DurableFuture<Vec<Result<O, OperationError>>>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::JoinAllExecution;

        let Self { ctx, name, futures } = self;
        let context = crate::context::display_prefix(ctx.prefix()).to_owned();
        DurableFuture::deferred(PendingClaim::new("join_all", context, move || {
            let claimed = Claimed {
                op_id: ctx.claim_operation_id("join_all"),
                ctx,
                name,
            };
            preflight_identity!(claimed, "Context", crate::combinator::COMBINATOR_SUB_TYPE);
            let gate = claimed.ctx.clone();
            let execution = JoinAllExecution {
                ctx: claimed.ctx,
                op_id: claimed.op_id,
                name: claimed.name,
                futures,
            };
            DurableFuture::for_operation(&gate, execution.execute())
        }))
    }
}

/// Builder for [`DurableContext::select_ok`]: first success wins.
///
/// Returns the first successful `O`.
///
/// The returned value has passed through the combinator's JSON encoding,
/// so it equals the value replay returns.
///
/// When the combinator resolves, it stops every input that it started from
/// `.future()` and that is still running. A stopped input stops at its next
/// `.await`. An input started with `.spawn()` keeps running, and its result
/// is dropped.
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
    pub(crate) fn new_internal(ctx: DurableContext, futures: Vec<DurableFuture<O>>) -> Self {
        Self {
            ctx,
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
    ///
    /// The returned future claims the combinator's operation ID on its
    /// first poll, not here. Dropped before its first poll, it claims no ID
    /// and records nothing. Do not pass it to another combinator: an outer
    /// combinator that would start a still-unclaimed combinator fails the
    /// execution with a context usage error. Pass the handle that
    /// [`.spawn()`](Self::spawn) returns instead.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// The combinator claims its operation ID at this call, and checks it
    /// against the recorded history before the task starts. The returned
    /// handle has claimed, so it is a valid input to another combinator. A
    /// `.spawn()` from a task or thread that does not own the context
    /// claims no ID, fails the execution with a context usage error,
    /// returns a handle that resolves to the task-ownership error, queues
    /// no `START` and runs no body.
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
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for SelectOkBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::SelectOkExecution;

        let Self { ctx, name, futures } = self;
        let context = crate::context::display_prefix(ctx.prefix()).to_owned();
        DurableFuture::deferred(PendingClaim::new("select_ok", context, move || {
            let claimed = Claimed {
                op_id: ctx.claim_operation_id("select_ok"),
                ctx,
                name,
            };
            preflight_identity!(claimed, "Context", crate::combinator::COMBINATOR_SUB_TYPE);
            let gate = claimed.ctx.clone();
            let execution = SelectOkExecution {
                ctx: claimed.ctx,
                op_id: claimed.op_id,
                name: claimed.name,
                futures,
            };
            DurableFuture::for_operation(&gate, execution.execute())
        }))
    }
}

/// Builder for [`DurableContext::race`]: first settled wins.
///
/// Returns the first result (success or failure).
///
/// The returned value has passed through the combinator's JSON encoding,
/// so it equals the value replay returns.
///
/// When the combinator resolves, it stops every input that it started from
/// `.future()` and that is still running. A stopped input stops at its next
/// `.await`. An input started with `.spawn()` keeps running, and its result
/// is dropped.
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
    pub(crate) fn new_internal(ctx: DurableContext, futures: Vec<DurableFuture<O>>) -> Self {
        Self {
            ctx,
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
    ///
    /// The returned future claims the combinator's operation ID on its
    /// first poll, not here. Dropped before its first poll, it claims no ID
    /// and records nothing. Do not pass it to another combinator: an outer
    /// combinator that would start a still-unclaimed combinator fails the
    /// execution with a context usage error. Pass the handle that
    /// [`.spawn()`](Self::spawn) returns instead.
    pub fn future(self) -> DurableFuture<O> {
        self.into_future()
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// The combinator claims its operation ID at this call, and checks it
    /// against the recorded history before the task starts. The returned
    /// handle has claimed, so it is a valid input to another combinator. A
    /// `.spawn()` from a task or thread that does not own the context
    /// claims no ID, fails the execution with a context usage error,
    /// returns a handle that resolves to the task-ownership error, queues
    /// no `START` and runs no body.
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
    pub fn spawn(self) -> DurableFuture<O> {
        spawn_terminal!(self)
    }
}

impl<O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> IntoFuture
    for RaceBuilder<O>
{
    type Output = Result<O, OperationError>;
    type IntoFuture = DurableFuture<O>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::combinator::RaceExecution;

        let Self { ctx, name, futures } = self;
        let context = crate::context::display_prefix(ctx.prefix()).to_owned();
        DurableFuture::deferred(PendingClaim::new("race", context, move || {
            let claimed = Claimed {
                op_id: ctx.claim_operation_id("race"),
                ctx,
                name,
            };
            preflight_identity!(claimed, "Context", crate::combinator::COMBINATOR_SUB_TYPE);
            let gate = claimed.ctx.clone();
            let execution = RaceExecution {
                ctx: claimed.ctx,
                op_id: claimed.op_id,
                name: claimed.name,
                futures,
            };
            DurableFuture::for_operation(&gate, execution.execute())
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::DurableFuture;
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};

    /// Test 18: a builder and a `.future()` dropped before their first poll
    /// claim no ID.
    #[test]
    fn dropped_combinator_builder_claims_nothing() {
        let ctx = DurableContext::new_root(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
        );
        drop(ctx.try_join_all(Vec::<DurableFuture<i32>>::new()));
        drop(ctx.join_all(Vec::<DurableFuture<i32>>::new()).future());
        drop(
            ctx.select_ok(Vec::<DurableFuture<i32>>::new())
                .name("s")
                .future(),
        );
        drop(ctx.race(Vec::<DurableFuture<i32>>::new()));
        assert_eq!(ctx.mint_id().positional(), "1", "the counter is unchanged");
    }

    /// A context whose log records a succeeded step at positional ID "1".
    fn ctx_with_step_at_1() -> DurableContext {
        let wire_id = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_id.clone(),
            status: CheckpointStatus::Succeeded,
            result: Some("1".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        DurableContext::new_root(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(vec![(wire_id, record)])),
        )
    }

    /// Asserts that `claim` recorded a replay identity mismatch on `ctx`.
    fn assert_mismatch_recorded(method: &str, ctx: &DurableContext) {
        assert!(
            ctx.coordinator().fatal_error().is_some(),
            "{method}: the claim records the mismatch before any poll"
        );
    }

    /// The claim of each combinator checks its ID against the recorded
    /// history at once, before the future is polled. A step recorded at
    /// the claimed ID is a replay identity mismatch.
    #[test]
    fn claim_checks_identity_before_poll() {
        let empty = Vec::<DurableFuture<i32>>::new;

        let ctx = ctx_with_step_at_1();
        ctx.try_join_all(empty()).future().claim_now();
        assert_mismatch_recorded("try_join_all", &ctx);

        let ctx = ctx_with_step_at_1();
        ctx.join_all(empty()).future().claim_now();
        assert_mismatch_recorded("join_all", &ctx);

        let ctx = ctx_with_step_at_1();
        ctx.select_ok(empty()).future().claim_now();
        assert_mismatch_recorded("select_ok", &ctx);

        let ctx = ctx_with_step_at_1();
        ctx.race(empty()).future().claim_now();
        assert_mismatch_recorded("race", &ctx);
    }
}
