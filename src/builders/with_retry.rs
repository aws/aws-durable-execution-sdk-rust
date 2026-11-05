//! The retrying child-context operation: [`WithRetryBuilder`], returned by
//! [`DurableContext::with_retry`](crate::DurableContext::with_retry).
//!
//! `with_retry` composes the
//! [child context operation](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/child-context/)
//! with step-style retries; it has no separate page in the
//! language-agnostic operation guide.

use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::BoxError;
use crate::OperationSubType;
use crate::RetryStrategy;
use crate::Serdes;
use crate::context::DurableContext;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;
use crate::with_retry::{LoopFailure, RetryWrap};

/// Builder for a block-level retry operation.
///
/// Created by [`DurableContext::with_retry`]. Runs a closure against a
/// child context and applies a retry strategy to the closure's **overall**
/// outcome, so a multi-operation block retries as a unit. Each attempt
/// starts clean: nothing recorded by a failed attempt is replayed into the
/// next one, so every operation in the block re-runs on retry. Delays
/// between attempts suspend the execution, and retry progress survives
/// suspension.
///
/// The closure receives the attempt's context and the 1-based attempt
/// number, which is `N` in the attempt's name `attempt-N`. The number is
/// metadata for names and logs: a resumed attempt receives the same
/// number, and operation IDs do not depend on it.
///
/// Without an explicit strategy the block uses the same default a step
/// uses: 6 total attempts with exponential backoff (5s initial delay, 60s
/// cap, factor 2).
///
/// When retries exhaust, a block with its own record fails with a
/// [`ChildContextError`](crate::ChildContextError) of kind `ChildFailed`
/// whose source is the last attempt's [`OperationError`]. A block with
/// [`without_child_context`](WithRetryBuilder::without_child_context)
/// returns the last attempt's [`OperationError`] unchanged instead: a
/// `ChildContextError` of kind `ChildFailed` whose source is the attempt
/// body's error.
///
/// The builder is generic over the block closure `F` and its future
/// `Fut`; both parameters are inferred at the call site and never written
/// by users.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use aws_durable_execution_sdk::RetryDecision;
/// use std::time::Duration;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     let result = ctx.with_retry(|child, _attempt| async move {
///         let a = child.step(|_| async { Ok(1_u32) }).name("fetch").await?;
///         let b = child.step(move |_| async move { Ok(a + 1) }).name("apply").await?;
///         Ok(format!("{a}+{b}"))
///     })
///     .name("fetch-and-apply")
///     .retry_strategy(|_err, attempt| {
///         if attempt >= 3 {
///             RetryDecision::Stop
///         } else {
///             RetryDecision::Retry { delay: Duration::from_secs(2) }
///         }
///     })
///     .await?;
///     Ok(result)
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct WithRetryBuilder<O, F, Fut, S = JsonSerdes> {
    ctx: DurableContext,
    name: Option<String>,
    /// Whether the block writes its own child context record.
    wrap: RetryWrap,
    sub_type: Option<OperationSubType>,
    retry_strategy: Option<RetryStrategy>,
    serdes: S,
    closure: F,
    summary: Option<crate::child::ChildSummary<O>>,
    _marker: PhantomData<fn() -> (O, Fut)>,
}

impl<O, F, Fut, S> std::fmt::Debug for WithRetryBuilder<O, F, Fut, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithRetryBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<O, F, Fut> WithRetryBuilder<O, F, Fut>
where
    O: Send + 'static,
    F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the closure here keeps the
    /// field non-optional: a builder without a body is unrepresentable.
    ///
    /// The builder claims no operation ID: a block with its own child
    /// context claims its ID when the builder becomes its future, because
    /// only then is its mode known.
    pub(crate) fn new_internal(ctx: DurableContext, closure: F) -> Self {
        Self {
            ctx,
            name: None,
            wrap: RetryWrap::Child,
            sub_type: None,
            retry_strategy: None,
            serdes: JsonSerdes,
            closure,
            summary: None,
            _marker: PhantomData,
        }
    }
}

impl<O, F, Fut, S> WithRetryBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this block.
    ///
    /// A non-empty name also names each wait between attempts
    /// `{name}-backoff-{attempt}`. Without a name, or with an empty name,
    /// those waits are unnamed. Names appear in logs, traces, and execution
    /// history for debugging.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Records `sub_type` instead of the default subtype.
    ///
    /// The subtype labels this block's child context in its checkpoint, in
    /// place of `RunInChildContext`. So the blocks of an operation composed
    /// from several durable operations can be told apart. Execution history
    /// and plugins read it. Without this call the block records
    /// [`OperationSubType::RUN_IN_CHILD_CONTEXT`]. A second call replaces
    /// the first.
    ///
    /// The subtype labels the block, not its attempts. Each attempt's
    /// child context `attempt-N` keeps `RunInChildContext`, each delay
    /// keeps `Wait`, and every operation inside an attempt keeps its own
    /// subtype. A delay is named `{name}-backoff-N` for a block with a
    /// non-empty name, and is unnamed for an unnamed block.
    ///
    /// Replay compares the recorded subtype with this one, as it compares
    /// the name. So changing the subtype of an existing block breaks the
    /// executions in flight: their replay fails with
    /// `NonDeterministicExecutionError`.
    ///
    /// With [`without_child_context`](Self::without_child_context) the
    /// block has no record, so the subtype has no recorded effect.
    /// With [`virtual_context`](Self::virtual_context) the block has no
    /// record either, so the subtype appears only in errors and plugin hook
    /// infos.
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
    /// ) -> Result<u32, durable::BoxError> {
    ///     let total = ctx
    ///         .with_retry(|child, _attempt| async move {
    ///             let v = child.step(|_| async { Ok(42_u32) }).name("work").await?;
    ///             Ok(v)
    ///         })
    ///         .name("block")
    ///         .sub_type(OperationSubType::new("Checkout")?)
    ///         .await?;
    ///     Ok(total)
    /// }
    /// ```
    pub fn sub_type(mut self, sub_type: OperationSubType) -> Self {
        self.sub_type = Some(sub_type);
        self
    }

    /// Runs the attempts in the caller's context, without a `with_retry`
    /// record, as JS `wrapWithRunInChildContext: false`. When the block is
    /// named `b`, its attempt children are named `b-attempt-N`; without
    /// this setter they are `attempt-N` inside the block's own context. In
    /// every mode the waits of a block named `b` are named `b-backoff-N`,
    /// and the waits of an unnamed block have no name.
    ///
    /// The block claims no operation ID, so `attempt-1` takes the caller's
    /// next positional ID, and each later attempt and wait takes the
    /// caller's next ID when it starts. Await the block before the caller
    /// creates another operation, so those IDs stay the same on every run.
    /// No operation hook, user-function hook or `wrap_user_function` runs
    /// for the block itself. On exhaustion the block returns the last
    /// `attempt-N` child's error unchanged: a
    /// [`ChildContextError`](crate::ChildContextError) of kind
    /// `ChildFailed` whose source is the body's error.
    ///
    /// The last call of [`virtual_context`](Self::virtual_context) and
    /// `without_child_context` wins.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u32, durable::BoxError> {
    ///     // Records `charge-attempt-1`, `charge-attempt-2`, ... directly in
    ///     // the handler's context.
    ///     let total = ctx
    ///         .with_retry(|child, _attempt| async move {
    ///             Ok(child.step(|_| async { Ok(42_u32) }).name("work").await?)
    ///         })
    ///         .name("charge")
    ///         .without_child_context()
    ///         .await?;
    ///     Ok(total)
    /// }
    /// ```
    pub fn without_child_context(mut self) -> Self {
        self.wrap = RetryWrap::None;
        self
    }

    /// Runs the body without recording this child context itself.
    ///
    /// The block still runs its retry loop in a context with its own
    /// operation namespace, so its attempts and waits keep IDs independent
    /// of the caller's. No record is written for the block: its attempts
    /// and waits name the caller's parent as their parent, and the block
    /// adds no level to
    /// [`plugin_child_operations_depth`](crate::OptionsBuilder::plugin_child_operations_depth).
    /// Because nothing records the block's result, the loop runs inline on
    /// the awaiting task in every invocation, and its attempts and waits
    /// replay their recorded outcomes.
    ///
    /// When retries exhaust, the block fails with a
    /// [`ChildContextError`](crate::ChildContextError) of kind
    /// `ChildFailed` whose source is the last attempt's error. A result
    /// that the serdes cannot serialize or deserialize returns the serdes
    /// error, or ends the invocation when the serdes marks the error with
    /// [`RetryableSerdesError`](crate::serdes::RetryableSerdesError).
    ///
    /// The block is stopped with its caller's scope, not when its own loop
    /// returns: an operation the block started with `.spawn()` and did not
    /// await keeps running and checkpointing after the block returns.
    ///
    /// Replay fails with `NonDeterministicExecutionError` when the log
    /// holds a record at the block's ID. So switching an existing block to
    /// virtual fails the replay of the executions in flight. A summary set
    /// with [`summary`](Self::summary) is never called, because no record
    /// exists to hold it, and a subtype set with
    /// [`sub_type`](Self::sub_type) appears only in errors and plugin hook
    /// infos.
    ///
    /// Plugins see the block: `on_operation_start` fires before the loop
    /// and `on_operation_end` after it, with `Succeeded` and no result, or
    /// `Failed` and the error, both with
    /// [`is_virtual`](crate::plugin::OperationInfo::is_virtual) true and
    /// [`is_replay`](crate::plugin::OperationInfo::is_replay) the
    /// invocation's replay state when each fires. The block is a user
    /// function: its user-function hooks and `wrap_user_function` run in
    /// every invocation, because the loop runs in every invocation.
    ///
    /// When the block is named `b`, its attempt children are named
    /// `b-attempt-N` and its waits `b-backoff-N`; the waits of an unnamed
    /// block have no name. The last call of `virtual_context` and
    /// [`without_child_context`](Self::without_child_context) wins.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u32, durable::BoxError> {
    ///     // Records `charge-attempt-1`, `charge-attempt-2`, ... and no
    ///     // record for the block itself.
    ///     let total = ctx
    ///         .with_retry(|child, _attempt| async move {
    ///             Ok(child.step(|_| async { Ok(42_u32) }).name("work").await?)
    ///         })
    ///         .name("charge")
    ///         .virtual_context()
    ///         .await?;
    ///     Ok(total)
    /// }
    /// ```
    pub fn virtual_context(mut self) -> Self {
        self.wrap = RetryWrap::Virtual;
        self
    }

    /// Sets the retry strategy for this block.
    ///
    /// The strategy closure is called after each failed attempt with the
    /// block's error and the 1-based attempt number, and decides whether to
    /// retry the whole block. It takes the same shape a step's strategy
    /// does, and the same determinism rule applies: the decision must be a
    /// pure function of the error and the attempt number, because it is
    /// re-evaluated against the recorded error during replay.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use aws_durable_execution_sdk::RetryDecision;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u32, durable::BoxError> {
    ///     let result = ctx.with_retry(|child, _attempt| async move {
    ///         let v = child.step(|_| async { Ok(42_u32) }).await?;
    ///         Ok(v)
    ///     })
    ///     .retry_strategy(|_err, attempt| {
    ///         if attempt >= 3 {
    ///             RetryDecision::Stop
    ///         } else {
    ///             RetryDecision::Retry { delay: Duration::from_secs(1) }
    ///         }
    ///     })
    ///     .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn retry_strategy<R>(mut self, strategy: R) -> Self
    where
        R: Fn(&crate::StepError, u32) -> crate::RetryDecision + Send + Sync + 'static,
    {
        self.retry_strategy = Some(Box::new(strategy));
        self
    }

    /// Sets the retry strategy for this block from a
    /// [`RetryStrategyConfig`](crate::builders::RetryStrategyConfig).
    ///
    /// Use this for the common case, shaping retry delays (attempt count,
    /// initial delay, cap, backoff rate, jitter) and limiting the retries
    /// to some errors with
    /// [`retry_if`](crate::builders::RetryStrategyConfigBuilder::retry_if),
    /// without hand-writing a closure. The closure setter
    /// [`retry_strategy`](Self::retry_strategy) remains for retry rules the config
    /// cannot express, such as a delay that depends on the error.
    ///
    /// The strategy receives the attempt's
    /// [`OperationError`] as the error's source, and
    /// an attempt that replays has no live source error. So a `retry_if`
    /// predicate here matches
    /// [`OperationError::wire`](crate::OperationError::wire), not a
    /// downcast.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk as durable;
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<u32, durable::BoxError> {
    ///     let result = ctx.with_retry(|child, _attempt| async move {
    ///         let v = child.step(|_| async { Ok(42_u32) }).await?;
    ///         Ok(v)
    ///     })
    ///     .retry_strategy_config(
    ///         RetryStrategyConfig::builder()
    ///             .max_attempts(3)
    ///             .initial_delay(Duration::from_secs(1))
    ///             .build(),
    ///     )
    ///     .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn retry_strategy_config(mut self, config: crate::builders::RetryStrategyConfig) -> Self {
        self.retry_strategy = Some(config.into_retry_strategy());
        self
    }

    /// Overrides the serialization strategy for the block's result.
    ///
    /// Replaces the builder's serdes type parameter with `S2`, which must
    /// implement [`Serdes<O>`](crate::Serdes) for this block's output type:
    /// attaching a serdes for a different type fails at compile time. To
    /// share one instance across operations, wrap it in an
    /// [`Arc`] and clone the `Arc` handle into each builder.
    pub fn serdes<S2>(self, serdes: S2) -> WithRetryBuilder<O, F, Fut, S2>
    where
        S2: Serdes<O>,
    {
        WithRetryBuilder {
            ctx: self.ctx,
            name: self.name,
            wrap: self.wrap,
            sub_type: self.sub_type,
            retry_strategy: self.retry_strategy,
            serdes,
            closure: self.closure,
            summary: self.summary,
            _marker: PhantomData,
        }
    }

    /// Sets the payload recorded when the block's result is too large to
    /// checkpoint.
    ///
    /// Called only when the serialized result is over 256 KiB. The summary
    /// applies to the block's own child context, as
    /// [`ChildBuilder::summary`](crate::builders::ChildBuilder::summary)
    /// does to a child context; the `attempt-N` contexts record their
    /// results without one. With
    /// [`without_child_context`](Self::without_child_context) the block
    /// writes no record of its own, so the summary is never called. With
    /// [`virtual_context`](Self::virtual_context) the block writes no
    /// record either, and the summary is never called.
    ///
    /// The output is observability only: replay never reads it. A replay
    /// that rebuilds the result of a completed `ReplayChildren` record does
    /// not call the summary; a block that is still running when an
    /// invocation resumes it calls the summary when it completes with an
    /// oversized result. Keep it under 256 KiB, because the service rejects
    /// a larger payload. A second call replaces the first.
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

    /// Eagerly spawns the retry block on a tokio task.
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
    ///     let handle = ctx.with_retry(|child, _attempt| async move {
    ///         let v = child.step(|_| async { Ok(1_u32) }).await?;
    ///         Ok(v)
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

impl<O, F, Fut, S> WithRetryBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// The operation's future. With `started`, its start step runs now, on
    /// the caller's task (see [`crate::start`]); `.spawn()` uses that.
    ///
    /// A block with its own child context claims its operation ID here,
    /// when the builder becomes its future. An unwrapped block claims
    /// nothing: its first attempt claims the caller's next ID when it
    /// starts.
    fn build(self, started: bool) -> DurableFuture<O> {
        match self.wrap {
            RetryWrap::Child => self.build_child(started),
            RetryWrap::None => self.build_unwrapped(started),
            RetryWrap::Virtual => self.build_virtual(),
        }
    }

    /// The future of a virtual block: it claims its operation ID and
    /// rejects a record at it now, then runs the retry loop in a virtual
    /// child context when it is polled.
    fn build_virtual(self) -> DurableFuture<O> {
        use crate::child::ChildExecution;

        let op_id = self.ctx.claim_operation_id("with_retry");
        let sub_type = self
            .sub_type
            .unwrap_or(OperationSubType::RUN_IN_CHILD_CONTEXT);
        if let Err(err) =
            self.ctx
                .reject_record_at_virtual_child(&op_id, sub_type.as_str(), self.name.as_deref())
        {
            return DurableFuture::from_async(async move { Err(err) });
        }

        let closure = Arc::new(self.closure);
        let strategy: Arc<RetryStrategy> = Arc::new(
            self.retry_strategy
                .unwrap_or_else(crate::step::default_retry_strategy),
        );
        let serdes = Arc::new(self.serdes);
        let loop_serdes = Arc::clone(&serdes);
        let loop_name = self.name.clone();

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id,
            name: self.name,
            sub_type,
            serdes,
            user_function: crate::child::UserFunctionPolicy::Hooks,
            summary: None,
            closure: move |outer_ctx| async move {
                crate::with_retry::retry_loop(
                    outer_ctx,
                    loop_name,
                    RetryWrap::Virtual,
                    None,
                    closure,
                    strategy,
                    loop_serdes,
                )
                .await
                .map_err(LoopFailure::into_child_error)
            },
            _marker: PhantomData,
        };
        DurableFuture::for_operation(&execution.ctx.clone(), execution.execute_virtual())
    }

    /// The future of a block with its own child context record.
    fn build_child(self, started: bool) -> DurableFuture<O> {
        use crate::child::ChildExecution;

        let op_id = self.ctx.claim_operation_id("with_retry");
        // The block is a child context on the wire; validate against the
        // recorded identity exactly as `run_in_child_context` does.
        if let Err(err) = self.ctx.preflight_replay_identity(
            &op_id,
            "Context",
            Some(
                self.sub_type
                    .as_ref()
                    .map_or(crate::child::CHILD_SUB_TYPE, OperationSubType::as_str),
            ),
            self.name.as_deref(),
        ) {
            return DurableFuture::from_async(async move { Err(err) });
        }

        let closure = Arc::new(self.closure);
        let strategy: Arc<RetryStrategy> = Arc::new(
            self.retry_strategy
                .unwrap_or_else(crate::step::default_retry_strategy),
        );
        // One instance serves both the outer block result and every
        // attempt's round trip, shared through the forwarding
        // `impl Serdes<T> for Arc<S>`.
        let serdes = Arc::new(self.serdes);
        let loop_serdes = Arc::clone(&serdes);

        let loop_name = self.name.clone();

        let execution = ChildExecution {
            ctx: self.ctx,
            op_id,
            name: self.name,
            sub_type: self
                .sub_type
                .unwrap_or(OperationSubType::RUN_IN_CHILD_CONTEXT),
            serdes,
            user_function: crate::child::UserFunctionPolicy::Hooks,
            summary: self.summary,
            // A concrete closure returning the concrete retry-loop future:
            // no erasure here: the one box is the DurableFuture below.
            closure: move |outer_ctx| async move {
                crate::with_retry::retry_loop(
                    outer_ctx,
                    loop_name,
                    RetryWrap::Child,
                    None,
                    closure,
                    strategy,
                    loop_serdes,
                )
                .await
                .map_err(LoopFailure::into_child_error)
            },
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

    /// The future of an unwrapped block: the retry loop runs in the
    /// caller's context, and the block writes no record and fires no hook
    /// of its own. Its start step is attempt 1's start step.
    fn build_unwrapped(self, started: bool) -> DurableFuture<O> {
        let ctx = self.ctx;
        let name = self.name;
        let closure = Arc::new(self.closure);
        let strategy: Arc<RetryStrategy> = Arc::new(
            self.retry_strategy
                .unwrap_or_else(crate::step::default_retry_strategy),
        );
        let serdes = Arc::new(self.serdes);
        let loop_ctx = ctx.clone();
        let begin = move || {
            let first = crate::with_retry::start_attempt(
                &loop_ctx,
                crate::with_retry::attempt_name(name.as_deref(), RetryWrap::None, 1),
                1,
                &closure,
                &serdes,
                true,
            );
            async move {
                crate::with_retry::retry_loop(
                    loop_ctx,
                    name,
                    RetryWrap::None,
                    Some(first),
                    closure,
                    strategy,
                    serdes,
                )
                .await
                .map_err(LoopFailure::into_operation_error)
            }
        };
        if started {
            DurableFuture::for_started_operation(&ctx, begin())
        } else {
            DurableFuture::for_start_step(&ctx, begin)
        }
    }

    /// The operation's future, with its start step run now.
    fn started_future(self) -> DurableFuture<O> {
        self.build(true)
    }
}

impl<O, F, Fut, S> IntoFuture for WithRetryBuilder<O, F, Fut, S>
where
    O: Send + 'static,
    F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
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

    /// The end of a `with_retry` block closes the block's own scope: a step
    /// started with `.spawn()` directly on the block's context stops,
    /// although its handle outlives the block.
    #[tokio::test(start_paused = true)]
    async fn with_retry_block_end_stops_direct_block_task() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        use crate::child::ChildExecution;
        use crate::driver::InvocationOutcome;
        use crate::driver::test_support::{RefreshClient, drive_handler, refresh_ctx, wire_err};

        let client = Arc::new(RefreshClient::new(|_| Ok(Vec::new())));
        let ctx = refresh_ctx(&client);
        let effects = Arc::new(AtomicUsize::new(0));
        let block_effects = Arc::clone(&effects);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let holder: Arc<Mutex<Vec<DurableFuture<()>>>> = Arc::new(Mutex::new(Vec::new()));
            let block_holder = Arc::clone(&holder);
            let attempt = Arc::new(|_attempt_ctx: DurableContext, _attempt: u32| async {
                Ok::<(), BoxError>(())
            });
            // The outer child execution `WithRetryBuilder` builds, with a
            // closure that starts `escaped` on the block's own context
            // before it runs the retry loop.
            let execution = ChildExecution {
                ctx: ctx.clone(),
                op_id: ctx.claim_operation_id("with_retry"),
                name: Some("block".to_owned()),
                sub_type: OperationSubType::RUN_IN_CHILD_CONTEXT,
                serdes: Arc::new(JsonSerdes),
                user_function: crate::child::UserFunctionPolicy::Hooks,
                summary: None,
                closure: move |outer_ctx: DurableContext| async move {
                    let handle = outer_ctx
                        .step(move |_| async move {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            block_effects.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })
                        .name("escaped")
                        .spawn();
                    block_holder.lock().expect("lock").push(handle);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    crate::with_retry::retry_loop(
                        outer_ctx,
                        Some("block".to_owned()),
                        RetryWrap::Child,
                        None,
                        attempt,
                        Arc::new(crate::step::default_retry_strategy()),
                        Arc::new(JsonSerdes),
                    )
                    .await
                    .map_err(LoopFailure::into_child_error)
                },
                _marker: PhantomData,
            };
            execution.execute().await.map_err(wire_err)?;
            ctx.step(|_| async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok(())
            })
            .name("keep")
            .await
            .map_err(wire_err)?;
            drop(holder);
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(
            effects.load(Ordering::SeqCst),
            0,
            "the escaped body stopped"
        );
    }

    /// `serdes` keeps the wrap mode in both directions.
    #[test]
    fn serdes_keeps_wrap_mode() {
        let ctx = DurableContext::test_context();
        let unwrapped = ctx
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .without_child_context()
            .serdes(Arc::new(JsonSerdes));
        assert_eq!(unwrapped.wrap, RetryWrap::None);
        let wrapped = ctx
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .serdes(Arc::new(JsonSerdes));
        assert_eq!(wrapped.wrap, RetryWrap::Child);
    }

    /// The later of `without_child_context` and `virtual_context` wins,
    /// and `serdes` keeps the virtual mode.
    #[test]
    fn last_wrap_setter_wins() {
        let ctx = DurableContext::test_context();
        let virtual_last = ctx
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .without_child_context()
            .virtual_context();
        assert_eq!(virtual_last.wrap, RetryWrap::Virtual);
        let unwrapped_last = ctx
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .virtual_context()
            .without_child_context();
        assert_eq!(unwrapped_last.wrap, RetryWrap::None);
        let kept = ctx
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .virtual_context()
            .serdes(Arc::new(JsonSerdes));
        assert_eq!(kept.wrap, RetryWrap::Virtual);
    }

    /// A second `sub_type` call replaces the first, and `serdes` keeps
    /// the subtype.
    #[test]
    fn sub_type_setter_replaces_and_survives_serdes() {
        let ctx = DurableContext::test_context();
        let builder = ctx
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .sub_type(OperationSubType::new("First").unwrap())
            .sub_type(OperationSubType::new("Second").unwrap())
            .serdes(Arc::new(JsonSerdes));
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
            .with_retry(|_child, _attempt| async { Ok(1_u32) })
            .summary(|v: &u32| format!("first {v}"))
            .summary(|v: &u32| format!("second {v}"))
            .serdes(Arc::new(JsonSerdes));
        let summary = builder.summary.as_ref().expect("the summary is kept");
        assert_eq!(summary(&7), "second 7");
    }
}
