//! Context types for durable execution.
//!
//! [`DurableContext`] provides access to all durable operations.
//! [`StepContext`] is passed to step bodies and deliberately omits durable
//! operations — the type system enforces the "no nesting" rule at compile
//! time.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use serde::{Serialize, de::DeserializeOwned};

use crate::BoxError;
use crate::Serdes;
use crate::builders::{
    ChildBuilder, CreateCallbackBuilder, InvokeBuilder, JoinAllBuilder, MapBuilder,
    ParallelBuilder, RaceBuilder, SelectOkBuilder, StepBuilder, TryJoinAllBuilder, WaitBuilder,
    WaitForCallbackBuilder, WaitForConditionBuilder,
};
use crate::client::{CheckpointOutput, ClientError, ExecutionClient};
use crate::driver::{SuspensionSignal, TaskOwnership};
use crate::engine::{CheckpointLog, CheckpointRecord, EngineState, OperationId};
use crate::error::{ChildFnError, OperationError, OperationErrorKind, StepError, StepErrorKind};
use crate::future::{Branch, DurableFuture};

use aws_sdk_lambda::types::OperationUpdate;
use tokio::sync::Mutex;

/// Shared inner state for a durable execution context.
#[derive(Debug)]
struct Inner {
    execution_arn: String,
    lambda_context: lambda_runtime::Context,
    engine: EngineState,
    /// Suspension signal shared with the driver.
    suspension_signal: Arc<SuspensionSignal>,
    /// Task-ownership detector — catches user `tokio::spawn` misuse.
    task_ownership: Arc<TaskOwnership>,
    /// Execution client for checkpointing (None in test contexts without a client).
    execution_client: Option<Arc<dyn ExecutionClient>>,
    /// Mutable checkpoint token rotated on each checkpoint call.
    /// Uses `tokio::sync::Mutex` (not `std::sync::Mutex`) because the lock
    /// must be held across `await` points in [`DurableContext::checkpoint_updates`],
    /// serializing all concurrent checkpoint callers through one critical section.
    checkpoint_token: Arc<Mutex<String>>,
    /// Cached parent wire ID — the SHA-256 hash of this context's prefix
    /// (positional ID of the parent operation). `None` for root contexts.
    parent_wire_id: Option<String>,
    /// Execution-wide default serdes, applied by an operation only when it
    /// sets no serdes of its own. Threaded in from [`Options`](crate::Options)
    /// by [`wrap`](crate::wrap); shared with every child context.
    default_serdes: Option<Arc<dyn Serdes>>,
}

/// The durable execution context — a cheap-to-clone handle providing access
/// to all durable operations.
///
/// `DurableContext` is `Clone + Send + Sync` (backed by an `Arc`). Clone it
/// freely to share across async boundaries. Every durable operation method
/// claims its operation ID synchronously at the call site — polling order
/// never affects identity.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     // Access execution metadata
///     let arn = ctx.execution_arn();
///     let replaying = ctx.is_replaying();
///
///     // Perform a durable step
///     let result = ctx.step(|_| async { Ok("done".to_owned()) })
///         .name("example")
///         .await?;
///     Ok(result)
/// }
/// ```
#[derive(Debug, Clone)]
pub struct DurableContext {
    inner: Arc<Inner>,
}

impl DurableContext {
    /// Returns a new context for testing purposes.
    ///
    /// This is NOT part of the public API — used only in doctests.
    #[doc(hidden)]
    #[must_use]
    pub fn __test_context() -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn: String::from("arn:aws:lambda:us-east-1:123456789012:function:test"),
                lambda_context: lambda_runtime::Context::default(),
                engine: EngineState::new_root(Arc::new(CheckpointLog::empty())),
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: None,
                checkpoint_token: Arc::new(Mutex::new(String::new())),
                parent_wire_id: None,
                default_serdes: None,
            }),
        }
    }

    /// Creates a root context with the given execution state (internal).
    #[allow(dead_code)] // reason: used by the handler wrapper
    pub(crate) fn new_root(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine: EngineState::new_root(checkpoint_log),
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: None,
                checkpoint_token: Arc::new(Mutex::new(String::new())),
                parent_wire_id: None,
                default_serdes: None,
            }),
        }
    }

    /// Creates a root context with a client and token (for live execution).
    #[allow(dead_code)] // reason: used by step tests and the handler wrapper
    pub(crate) fn new_root_with_client(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine: EngineState::new_root(checkpoint_log),
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                parent_wire_id: None,
                default_serdes: None,
            }),
        }
    }

    /// Creates a root context with a client, token, and execution-wide default
    /// serdes threaded in from [`Options`](crate::Options) (internal).
    pub(crate) fn new_root_with_client_and_defaults(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
        default_serdes: Option<Arc<dyn Serdes>>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine: EngineState::new_root(checkpoint_log),
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                parent_wire_id: None,
                default_serdes,
            }),
        }
    }

    /// Returns the execution-wide default serdes, if one was configured via
    /// [`Options`](crate::Options). Operations fall back to this when they set
    /// no serdes of their own.
    pub(crate) fn default_serdes(&self) -> Option<&dyn Serdes> {
        self.inner.default_serdes.as_deref()
    }
    #[allow(dead_code)] // reason: used by run_in_child_context
    pub(crate) fn new_child(&self, parent_positional_id: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine: EngineState::new_child(
                    parent_positional_id,
                    Arc::clone(&self.inner.engine.checkpoint_log),
                ),
                suspension_signal: Arc::clone(&self.inner.suspension_signal),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                parent_wire_id: Some(crate::engine::compute_wire_id_public(parent_positional_id)),
                default_serdes: self.inner.default_serdes.clone(),
            }),
        }
    }

    /// Mints the next operation ID (internal engine concern).
    pub(crate) fn mint_id(&self) -> OperationId {
        self.inner.engine.mint_id()
    }

    /// Creates a child context with its OWN fresh suspension scope.
    ///
    /// Unlike [`Self::new_child`] (which shares the parent's scope so a
    /// sequential child's suspension propagates directly upward), this gives
    /// the child an independent scope. Used for each map/parallel BRANCH so
    /// that a `wait` inside one branch suspends only that branch — sibling
    /// branches keep running, and the coordinator's branch driver observes
    /// the branch scope. Everything else (checkpoint log, token, ownership,
    /// ARN) is shared with the parent, exactly like `new_child`.
    pub(crate) fn new_scoped_child(&self, parent_positional_id: &str) -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine: EngineState::new_child(
                    parent_positional_id,
                    Arc::clone(&self.inner.engine.checkpoint_log),
                ),
                suspension_signal: Arc::new(self.inner.suspension_signal.new_child_scope()),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                parent_wire_id: Some(crate::engine::compute_wire_id_public(parent_positional_id)),
                default_serdes: self.inner.default_serdes.clone(),
            }),
        }
    }

    /// FLAT-nesting counterpart of [`Self::new_scoped_child`]: a fresh-scope
    /// child that reports `parent_wire_id_override` as its operations' parent
    /// (the batch parent, not the virtual child).
    pub(crate) fn new_scoped_flat_child(
        &self,
        child_positional_id: &str,
        parent_wire_id_override: &str,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine: EngineState::new_child(
                    child_positional_id,
                    Arc::clone(&self.inner.engine.checkpoint_log),
                ),
                suspension_signal: Arc::new(self.inner.suspension_signal.new_child_scope()),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                parent_wire_id: Some(parent_wire_id_override.to_owned()),
                default_serdes: self.inner.default_serdes.clone(),
            }),
        }
    }

    /// Advances the ID counter by `n` positions without minting.
    ///
    /// Used after replaying a terminal batch: the child IDs consumed during
    /// the original execution must be skipped so the next operation gets
    /// the correct positional ID.
    pub(crate) fn advance_counter(&self, n: usize) {
        self.inner.engine.id_counter.advance(n);
    }

    /// Returns a reference to the suspension signal for this context.
    ///
    /// Operations use this to request suspension when they cannot proceed.
    #[allow(dead_code)] // reason: used by operation execution
    pub(crate) fn suspension_signal(&self) -> &Arc<SuspensionSignal> {
        &self.inner.suspension_signal
    }

    /// Returns a reference to the task-ownership tracker.
    #[allow(dead_code)] // reason: used by .spawn()
    pub(crate) fn task_ownership(&self) -> &Arc<TaskOwnership> {
        &self.inner.task_ownership
    }

    /// Checks task ownership and returns an `OperationError` if the caller
    /// is not authorized. Used by every durable operation entry point.
    #[allow(dead_code)] // reason: wired into the builders
    pub(crate) fn enforce_task_ownership(&self) -> Result<(), OperationError> {
        self.inner
            .task_ownership
            .check_current_task()
            .map_err(|msg| {
                OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
                    StepErrorKind::ExecutionFailed { message: msg },
                )))
            })
    }

    /// Returns whether the given positional ID has a terminal checkpoint
    /// record (internal).
    #[allow(dead_code)] // reason: used by operation execution
    pub(crate) fn is_replaying_at(&self, positional_id: &str) -> bool {
        self.inner.engine.is_replaying_at(positional_id)
    }

    /// Returns the checkpoint record for the given positional ID, if any.
    ///
    /// NOTE: The checkpoint log is keyed by wire ID (the hash), which is
    /// what the backend returns in Operations[].Id. We look up by wire ID
    /// computed from the positional ID.
    pub(crate) fn checkpoint_record(&self, positional_id: &str) -> Option<CheckpointRecord> {
        // The log is keyed by wire ID (hash of the positional string),
        // because parse_inline_operations uses the Id field from the backend.
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.get(&wire_id)
    }

    /// Requests suspension from the driver (used by operations that cannot
    /// proceed — e.g., pending retry timer).
    pub(crate) fn request_suspend(&self) {
        self.inner.suspension_signal.request_suspend();
    }

    /// Suspends the invocation and never returns control to the caller.
    ///
    /// Sets the suspension signal, then awaits a future that never resolves
    /// and never registers a waker. The driver observes the signal on the
    /// next `Poll::Pending` from the handler and drops the handler future at
    /// this await point, completing the invocation as `PENDING`. Because the
    /// future is dropped rather than resumed, an operation's suspension can
    /// never surface to user code and can never be caught or ignored. The
    /// `-> T` return type is inhabited only vacuously: the awaited future
    /// never completes, so no value is ever produced.
    pub(crate) async fn suspend_now<T>(&self) -> T {
        self.request_suspend();
        std::future::pending::<T>().await
    }

    /// Checkpoints operation updates via the execution client.
    ///
    /// Serializes all concurrent callers through a single critical section:
    /// the lock is held across the full read-token → API-call →
    /// rotate-token sequence. This prevents concurrent branches from racing
    /// on the checkpoint token.
    pub(crate) async fn checkpoint_updates(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ClientError> {
        let client = self
            .inner
            .execution_client
            .as_ref()
            .ok_or_else(|| ClientError::new_non_retryable("no execution client configured"))?;

        // Hold the async mutex across the entire checkpoint call to
        // serialize concurrent branch checkpoints. This is the Rust
        // equivalent of Go's sync.Mutex held across the blocking API call
        // in checkpointer.checkpoint().
        let mut token_guard = self.inner.checkpoint_token.lock().await;

        let output = client
            .checkpoint(&self.inner.execution_arn, &token_guard, updates)
            .await?;

        // Rotate the token while still holding the lock.
        token_guard.clone_from(&output.checkpoint_token);
        drop(token_guard);

        // Merge updated operations into the checkpoint log so that
        // subsequent reads (e.g. reading callback_id after START) see
        // backend-assigned fields.
        if !output.updated_operations.is_empty() {
            crate::client::merge_operations_into_log(
                &self.inner.engine.checkpoint_log,
                &output.updated_operations,
            );
        }

        Ok(output)
    }

    /// Returns the parent wire ID if this is a child context.
    pub(crate) fn parent_wire_id(&self) -> Option<&str> {
        self.inner.parent_wire_id.as_deref()
    }

    /// Returns the parent wire ID computed from the prefix, or None for root.
    pub(crate) fn parent_wire_id_computed(&self) -> Option<String> {
        let prefix = self.inner.engine.id_counter.prefix();
        if prefix.is_empty() {
            None
        } else {
            Some(crate::engine::compute_wire_id_public(prefix))
        }
    }

    /// Returns the attempt number for the given operation from the checkpoint
    /// log (0 if no prior attempt recorded).
    pub(crate) fn get_attempt(&self, op_id: &OperationId) -> u32 {
        // On re-invocation after a RETRY, the backend returns the operation
        // with step_details.attempt set to the last completed attempt number.
        self.checkpoint_record(op_id.positional())
            .map_or(0, |r| r.attempt)
    }

    /// Returns the execution ARN identifying this durable execution.
    ///
    /// The ARN uniquely identifies the execution across invocations and is
    /// stable for the lifetime of the orchestration.
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
    ///     tracing::info!(arn = ctx.execution_arn(), "starting execution");
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn execution_arn(&self) -> &str {
        &self.inner.execution_arn
    }

    /// Returns a reference to the Lambda invocation context.
    ///
    /// Provides access to the request ID, deadline, function ARN, and other
    /// Lambda metadata for the current invocation.
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
    ///     let request_id = &ctx.lambda_context().request_id;
    ///     tracing::info!(%request_id, "invoked");
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn lambda_context(&self) -> &lambda_runtime::Context {
        &self.inner.lambda_context
    }

    /// Returns whether the current invocation is in replay mode.
    ///
    /// During replay, previously-checkpointed operation results are returned
    /// without re-execution. User code can use this flag to suppress
    /// duplicate side effects (e.g., logging) during replay.
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
    ///     if !ctx.is_replaying() {
    ///         tracing::info!("first execution — not replay");
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn is_replaying(&self) -> bool {
        self.inner.engine.is_replaying()
    }

    /// Creates a durable step operation.
    ///
    /// The step body receives a [`StepContext`] and returns a result. The
    /// result is checkpointed on success; on failure the retry strategy
    /// determines whether to retry.
    ///
    /// The operation ID is claimed synchronously at this call site.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.step(|_step_ctx| async {
    ///         Ok("computed value".to_owned())
    ///     }).name("compute")
    ///       .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn step<O, F, Fut>(&self, f: F) -> StepBuilder<O>
    where
        F: FnOnce(StepContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        #[allow(clippy::type_complexity)] // reason: boxed future factory is inherently complex
        let closure: Box<
            dyn FnOnce(
                    StepContext,
                )
                    -> std::pin::Pin<Box<dyn Future<Output = Result<O, BoxError>> + Send>>
                + Send,
        > = Box::new(move |ctx| Box::pin(f(ctx)));
        StepBuilder::new(self.clone(), op_id).with_closure(closure)
    }

    /// Creates a durable wait (timer) operation.
    ///
    /// The execution pauses for the specified duration. The wait is
    /// checkpointed, so replay does not re-wait.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     ctx.wait(Duration::from_secs(60))
    ///         .name("cooldown")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn wait(&self, duration: Duration) -> WaitBuilder {
        // Round up to whole seconds. Zero duration passes 0 (no min guard).
        #[allow(clippy::cast_possible_truncation)] // reason: duration ≤ i32::MAX for practical timers
        #[allow(clippy::cast_sign_loss)] // reason: ceil is non-negative
        let secs = (duration.as_secs_f64().ceil() as i64).min(i64::from(i32::MAX)) as i32;
        let op_id = self.mint_id();
        WaitBuilder::new(self.clone(), op_id, secs)
    }

    /// Creates a durable invoke operation to call another durable function.
    ///
    /// The output type parameter `O` comes first so callers can turbofish:
    /// `ctx.invoke::<Receipt, _>(...)`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize)]
    /// struct ChargeInput { amount: u64 }
    ///
    /// #[derive(Deserialize)]
    /// struct Receipt { id: String }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let receipt = ctx.invoke::<Receipt, _>("payment-fn", ChargeInput { amount: 100 })
    ///         .name("charge")
    ///         .await?;
    ///     Ok(receipt.id)
    /// }
    /// ```
    pub fn invoke<O, I>(&self, function_id: &str, input: I) -> InvokeBuilder<O>
    where
        I: Serialize,
        O: DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        // Serialize input at call site (synchronous, before the future body):
        // the input type is erased past this point. The outcome is carried as
        // a `Result` so a serialization failure surfaces as an error at await
        // time rather than being replaced by a `null` payload.
        let serialized_input = serde_json::to_string(&input).map_err(|e| e.to_string());
        InvokeBuilder::new(
            self.clone(),
            op_id,
            function_id.to_owned(),
            serialized_input,
        )
    }

    /// Creates a child context for fan-out / sub-orchestration.
    ///
    /// The child closure receives its own [`DurableContext`] with an
    /// independent operation-ID namespace. Use `.spawn()` for eager
    /// (background) execution.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let result = ctx.run_in_child_context(|child_ctx| async move {
    ///         let v = child_ctx.step(|_| async { Ok(42) }).await?;
    ///         Ok(v.to_string())
    ///     }).name("branch")
    ///       .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn run_in_child_context<O, F, Fut>(&self, f: F) -> ChildBuilder<O>
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        // The SDK pins the future and erases the BoxError into the internal
        // child-error carrier, so the caller writes a plain `async move` body.
        ChildBuilder::new(self.clone(), op_id).with_closure(Box::new(move |ctx| {
            Box::pin(async move { f(ctx).await.map_err(|e| ChildFnError::new(e.to_string())) })
        }))
    }

    /// Creates a wait-for-condition operation that polls until a predicate
    /// is satisfied.
    ///
    /// The check function is called repeatedly with the current state until
    /// the condition is met (implementation-defined termination signal via
    /// the state type).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Serialize, Deserialize)]
    /// struct PollState { count: u32 }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     ctx.wait_for_condition(
    ///         |_step_ctx, state: PollState| async move {
    ///             Ok(PollState { count: state.count + 1 })
    ///         },
    ///         PollState { count: 0 },
    ///     ).name("poll-ready")
    ///      .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn wait_for_condition<S, F, Fut>(
        &self,
        check: F,
        initial_state: S,
    ) -> WaitForConditionBuilder<S>
    where
        F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
        S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
    {
        let op_id = self.mint_id();
        #[allow(clippy::type_complexity)] // reason: boxed Fn closure for async check function
        let boxed_check: Box<
            dyn Fn(
                    StepContext,
                    S,
                )
                    -> std::pin::Pin<Box<dyn Future<Output = Result<S, BoxError>> + Send>>
                + Send
                + Sync,
        > = Box::new(move |ctx, state| Box::pin(check(ctx, state)));
        WaitForConditionBuilder::new(self.clone(), op_id, initial_state).with_check(boxed_check)
    }

    /// Creates a callback token for external completion.
    ///
    /// The returned [`Callback`](crate::Callback) provides an ID that
    /// external systems use to complete the operation, plus a
    /// [`DurableFuture`] that resolves when the callback arrives.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct Approval { approved: bool }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<bool, durable::BoxError> {
    ///     let cb = ctx.create_callback::<Approval>()
    ///         .name("approval")
    ///         .await?;
    ///     // Send cb.id() to an external system...
    ///     let approval = cb.result().await?;
    ///     Ok(approval.approved)
    /// }
    /// ```
    pub fn create_callback<O>(&self) -> CreateCallbackBuilder<O>
    where
        O: DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        CreateCallbackBuilder::new(self.clone(), op_id)
    }

    /// Creates a wait-for-callback operation that registers and waits for
    /// an external callback in one step.
    ///
    /// The submitter closure receives the callback ID and is responsible
    /// for delivering it to the external system that will complete it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Deserialize, Serialize)]
    /// struct Approval { ok: bool }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<bool, durable::BoxError> {
    ///     let approval = ctx.wait_for_callback::<Approval, _, _>(
    ///         |_step_ctx, callback_id| async move {
    ///             // e.g., send callback_id to an approval queue
    ///             Ok(())
    ///         }
    ///     ).name("await-approval")
    ///      .await?;
    ///     Ok(approval.ok)
    /// }
    /// ```
    pub fn wait_for_callback<O, F, Fut>(&self, submitter: F) -> WaitForCallbackBuilder<O>
    where
        F: FnOnce(StepContext, &str) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
        O: DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        // Box the submitter so the builder can store it without generics.
        let boxed_submitter: crate::callback::BoxedSubmitter =
            Box::new(move |ctx, id| Box::pin(submitter(ctx, id)));
        WaitForCallbackBuilder::new(self.clone(), op_id).with_submitter(boxed_submitter)
    }

    /// Creates a map operation that applies a function to each item in a
    /// collection, with configurable concurrency.
    ///
    /// Each item gets its own child context with an independent operation
    /// namespace.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Clone, Serialize, Deserialize)]
    /// struct Image { url: String }
    ///
    /// #[derive(Serialize, Deserialize)]
    /// struct Thumbnail { url: String }
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<(), durable::BoxError> {
    ///     let images = vec![Image { url: "a.png".into() }, Image { url: "b.png".into() }];
    ///     let _thumbnails = ctx.map(images, |child_ctx, _img, _idx| async move {
    ///         let thumb = child_ctx
    ///             .step(|_| async { Ok(Thumbnail { url: "thumb.png".into() }) })
    ///             .await?;
    ///         Ok(thumb)
    ///     }).name("resize")
    ///       .max_concurrency(4)
    ///       .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn map<Items, I, O, F, Fut>(&self, items: Items, f: F) -> MapBuilder<I, O>
    where
        Items: IntoIterator<Item = I>,
        F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        I: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        // Wrap the user closure once; each per-item invocation pins its future
        // and erases the BoxError into the internal child-error carrier.
        let f = Arc::new(f);
        let closure = Arc::new(move |ctx: DurableContext, item: I, idx: usize| {
            let f = Arc::clone(&f);
            Box::pin(async move {
                f(ctx, item, idx)
                    .await
                    .map_err(|e| ChildFnError::new(e.to_string()))
            })
                as std::pin::Pin<Box<dyn Future<Output = Result<O, ChildFnError>> + Send>>
        });
        let items: Vec<I> = items.into_iter().collect();
        MapBuilder::new(self.clone(), op_id).with_items_and_closure(items, closure)
    }

    /// Creates a parallel operation that executes named branches
    /// concurrently.
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
    ///     let branches = vec![
    ///         durable::Branch::new("a", |child_ctx: durable::DurableContext| async move {
    ///             let v = child_ctx.step(|_| async { Ok(1) }).await?;
    ///             Ok(v)
    ///         }),
    ///     ];
    ///     let _results: Vec<i32> = ctx.parallel(branches)
    ///         .name("fan-out")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn parallel<O>(&self, branches: impl IntoIterator<Item = Branch<O>>) -> ParallelBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        let branch_tuples: Vec<_> = branches
            .into_iter()
            .map(|b| {
                let name = b.name().to_owned();
                let factory = b.into_factory();
                (name, factory)
            })
            .collect();
        ParallelBuilder::new(self.clone(), op_id).with_branches(branch_tuples)
    }

    /// Joins all futures, failing fast on the first error.
    ///
    /// Returns `Vec<O>` on success, or the first `OperationError`
    /// encountered.
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
    ///     let _results: Vec<i32> = ctx.try_join_all([a, b])
    ///         .name("gather")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn try_join_all<O>(
        &self,
        futures: impl IntoIterator<Item = DurableFuture<O>>,
    ) -> TryJoinAllBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        TryJoinAllBuilder::new(self.clone(), op_id, futures.into_iter().collect())
    }

    /// Joins all futures, collecting every outcome as [`Settled`](crate::Settled).
    ///
    /// Never fails fast — all futures run to completion regardless of
    /// individual errors.
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
    ///     let _settled: Vec<durable::Settled<i32>> = ctx.join_all([a, b])
    ///         .name("collect")
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn join_all<O>(
        &self,
        futures: impl IntoIterator<Item = DurableFuture<O>>,
    ) -> JoinAllBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        JoinAllBuilder::new(self.clone(), op_id, futures.into_iter().collect())
    }

    /// Returns the first successful result.
    ///
    /// Losers are dropped (cancelled) when the first success resolves.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let a = ctx.step(|_| async { Ok("fast".to_owned()) }).future();
    ///     let b = ctx.step(|_| async { Ok("slow".to_owned()) }).future();
    ///     let winner = ctx.select_ok([a, b])
    ///         .name("race-ok")
    ///         .await?;
    ///     Ok(winner)
    /// }
    /// ```
    pub fn select_ok<O>(
        &self,
        futures: impl IntoIterator<Item = DurableFuture<O>>,
    ) -> SelectOkBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        SelectOkBuilder::new(self.clone(), op_id, futures.into_iter().collect())
    }

    /// Returns the first settled result, whether success or failure.
    ///
    /// Losers are dropped (cancelled) when the first future resolves.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     let a = ctx.step(|_| async { Ok("first".to_owned()) }).future();
    ///     let b = ctx.step(|_| async { Ok("second".to_owned()) }).future();
    ///     let winner = ctx.race([a, b])
    ///         .name("fastest")
    ///         .await?;
    ///     Ok(winner)
    /// }
    /// ```
    pub fn race<O>(&self, futures: impl IntoIterator<Item = DurableFuture<O>>) -> RaceBuilder<O>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        RaceBuilder::new(self.clone(), op_id, futures.into_iter().collect())
    }
}

/// Context passed to step bodies.
///
/// `StepContext` deliberately does **not** expose any durable operations.
/// This provides compile-time enforcement of the rule that durable
/// operations cannot be nested inside step bodies — attempting to call
/// a durable operation method inside a step body is a type error.
///
/// Use `tracing` macros for logging inside steps — the SDK-created
/// operation span automatically attaches execution context fields.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     let result = ctx.step(|step_ctx: durable::StepContext| async move {
///         // step_ctx has no durable operations — type-system enforced
///         tracing::info!("inside step");
///         Ok("done".to_owned())
///     }).await?;
///     Ok(result)
/// }
/// ```
#[derive(Debug, Clone)]
pub struct StepContext {
    attempt: u32,
    _private: (),
}

impl StepContext {
    /// Creates a new step context (internal).
    pub(crate) fn new(attempt: u32) -> Self {
        Self {
            attempt,
            _private: (),
        }
    }

    /// Returns the current attempt number (1-based).
    #[must_use]
    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}
