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
use crate::builders::{
    ChildBuilder, CreateCallbackBuilder, InvokeBuilder, JoinAllBuilder, MapBuilder,
    ParallelBuilder, RaceBuilder, SelectOkBuilder, StepBuilder, TryJoinAllBuilder, WaitBuilder,
    WaitForCallbackBuilder, WaitForConditionBuilder, WithRetryBuilder,
};
use crate::checkpoint_coalescer::{
    BatchLimits, CheckpointBatch, CheckpointCoalescer, TrackedUpdate, split_into_requests,
};
use crate::client::{CheckpointOutput, ClientError, ExecutionClient};
use crate::driver::{SuspensionSignal, TaskOwnership};
use crate::engine::{
    CheckpointLog, CheckpointRecord, CheckpointStatus, CheckpointStatusView, EngineState,
    OperationId, TerminalReplaySnapshot,
};
use crate::error::{
    NonDeterministicExecutionError, NonDeterministicExecutionErrorKind, OperationError,
    OperationErrorKind, StepError, StepErrorKind,
};
use crate::future::{Branch, DurableFuture};

use aws_sdk_lambda::types::OperationUpdate;
use tokio::sync::Mutex;

/// Shared inner state for a durable execution context.
struct Inner {
    execution_arn: String,
    lambda_context: lambda_runtime::Context,
    /// Engine state (ID counter + checkpoint log) for this context's
    /// namespace. Shared behind an `Arc` so a context can be rebound onto a
    /// different suspension scope (see
    /// [`DurableContext::spawn_scope`]) without duplicating the ID counter —
    /// two contexts minting from separate counters would diverge on replay.
    engine: Arc<EngineState>,
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
    /// Checkpoint-write coalescer, present only when
    /// [`Options`](crate::Options) configured a `checkpoint_delay` and/or
    /// `checkpoint_batching`. Shared with every child context so updates
    /// from all namespaces coalesce into the same batches. `None` means
    /// every checkpoint writes immediately (the default).
    coalescer: Option<Arc<CheckpointCoalescer>>,
    /// Cached parent wire ID — the SHA-256 hash of this context's prefix
    /// (positional ID of the parent operation). `None` for root contexts.
    parent_wire_id: Option<String>,
    /// This namespace's `durable_execution` span. For a root context it is
    /// the handler-level span wrapping the invocation; for a child context
    /// (sequential child, map/parallel branch, callback body) it is a
    /// detached span wrapping that child's body. [`DurableContext::mint_id`]
    /// re-records its `isReplay` field after every operation claim minted
    /// through this context, so replay-aware filters (see
    /// `tracing_layer::ReplayFilterLayer`) can suppress log events emitted
    /// while THIS namespace is replaying. Per-namespace spans are what keep
    /// concurrent branches — each with its own replay high-water mark — from
    /// clobbering each other's flag.
    replay_span: tracing::Span,
}

impl std::fmt::Debug for Inner {
    /// Hand-written to keep `checkpoint_token` — a credential-like value —
    /// out of log output, and to skip the engine/client/signal internals
    /// that make a derived impl unreadable. See issue #30.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("execution_arn", &self.execution_arn)
            .field("parent_wire_id", &self.parent_wire_id)
            .field("is_replaying", &self.engine.is_replaying())
            .field("checkpoint_token", &"<redacted>")
            .finish_non_exhaustive()
    }
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
#[derive(Clone)]
#[non_exhaustive]
pub struct DurableContext {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for DurableContext {
    /// Hand-written so `tracing::debug!(?ctx)` in a handler cannot leak the
    /// checkpoint token into `CloudWatch` Logs: the token prints as
    /// `"<redacted>"` and engine/client internals are skipped. See issue #30.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableContext")
            .field("execution_arn", &self.inner.execution_arn)
            .field("parent_wire_id", &self.inner.parent_wire_id)
            .field("is_replaying", &self.inner.engine.is_replaying())
            .field("checkpoint_token", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Every checkpoint write failure suffered by an invocation's buffered
/// (coalesced) checkpoints, reported by
/// [`DurableContext::flush_pending_checkpoints`] for the issue #43
/// classification at the end-of-invocation flush point.
#[derive(Debug)]
pub(crate) struct FlushFailure {
    /// The write failures, in occurrence order, each carrying the updates
    /// it did not persist.
    pub(crate) failures: Vec<crate::checkpoint_coalescer::FailedFlush>,
}

impl FlushFailure {
    /// Whether any of the failures is non-retryable. Non-retryable wins
    /// the classification: re-invoking on a deterministic rejection would
    /// loop until the execution timeout (issue #43 defect 1), so a single
    /// non-retryable failure routes the whole flush through the
    /// terminal-FAIL-then-fail-the-execution path.
    pub(crate) fn any_non_retryable(&self) -> bool {
        self.failures.iter().any(|f| !f.error.is_retryable())
    }

    /// The error to report: the first non-retryable failure when one
    /// exists (it decides the classification), otherwise the first
    /// failure.
    ///
    /// # Panics
    ///
    /// Never in practice: `flush_pending_checkpoints` only constructs a
    /// `FlushFailure` with at least one failure.
    pub(crate) fn primary_error(&self) -> &ClientError {
        self.failures
            .iter()
            .find(|f| !f.error.is_retryable())
            .or_else(|| self.failures.first())
            .map_or_else(
                || unreachable!("FlushFailure is never constructed empty"),
                |f| &f.error,
            )
    }
}

impl DurableContext {
    /// Returns a new context for testing purposes.
    ///
    /// This is NOT part of the public API — used only in doctests.
    #[doc(hidden)]
    #[must_use]
    pub fn __test_context() -> Self {
        let execution_arn = String::from("arn:aws:lambda:us-east-1:123456789012:function:test");
        let lambda_context = lambda_runtime::Context::default();
        let engine = Arc::new(EngineState::new_root(Arc::new(CheckpointLog::empty())));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: None,
                checkpoint_token: Arc::new(Mutex::new(String::new())),
                coalescer: None,
                parent_wire_id: None,
                replay_span,
            }),
        }
    }

    /// Creates a root context with the given execution state (test-only:
    /// production roots are built by the handler wrapper with a client and
    /// token via [`Self::new_root_with_client`]-equivalent wiring in
    /// `lib.rs`).
    #[cfg(test)]
    pub(crate) fn new_root(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: None,
                checkpoint_token: Arc::new(Mutex::new(String::new())),
                coalescer: None,
                parent_wire_id: None,
                replay_span,
            }),
        }
    }

    /// Creates a root context with a client and token (test-only harness
    /// for exercising live-path operation execution against a test double).
    #[cfg(test)]
    pub(crate) fn new_root_with_client(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                coalescer: None,
                parent_wire_id: None,
                replay_span,
            }),
        }
    }

    /// Creates a root context with a client, token, and checkpoint
    /// buffering window threaded in from
    /// [`Options`](crate::Options) (internal). `checkpoint_buffer_window`
    /// is `None` for immediate writes (the default), `Some(window)` for a
    /// `checkpoint_delay` coalescing window, and `Some(Duration::ZERO)` for
    /// pure `checkpoint_batching` (no added delay; writes batch behind the
    /// single-writer lock).
    pub(crate) fn new_root_with_client_and_defaults(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
        checkpoint_buffer_window: Option<Duration>,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                coalescer: checkpoint_buffer_window.map(|d| Arc::new(CheckpointCoalescer::new(d))),
                parent_wire_id: None,
                replay_span,
            }),
        }
    }

    /// Test-only root constructor that accepts an explicit
    /// [`CheckpointCoalescer`], letting tests force small [`BatchLimits`]
    /// so batch splitting is observable with a handful of updates.
    #[cfg(test)]
    pub(crate) fn new_root_with_client_and_coalescer(
        execution_arn: String,
        lambda_context: lambda_runtime::Context,
        checkpoint_log: Arc<CheckpointLog>,
        client: Arc<dyn ExecutionClient>,
        checkpoint_token: String,
        coalescer: CheckpointCoalescer,
    ) -> Self {
        let engine = Arc::new(EngineState::new_root(checkpoint_log));
        let replay_span = crate::tracing_layer::execution_span(
            &execution_arn,
            &lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn,
                lambda_context,
                engine,
                suspension_signal: Arc::new(SuspensionSignal::new()),
                task_ownership: Arc::new(TaskOwnership::new_current()),
                execution_client: Some(client),
                checkpoint_token: Arc::new(Mutex::new(checkpoint_token)),
                coalescer: Some(Arc::new(coalescer)),
                parent_wire_id: None,
                replay_span,
            }),
        }
    }
    pub(crate) fn new_child(&self, parent_positional_id: &str) -> Self {
        let engine = Arc::new(EngineState::new_child(
            parent_positional_id,
            Arc::clone(&self.inner.engine.checkpoint_log),
        ));
        // The child namespace has its own replay high-water mark: nested
        // operations can still be replaying while the parent is already
        // live. Give it its own detached span, initialized from the CHILD
        // engine's replay status; the child's mints keep it current, and the
        // parent span keeps the value the parent's own mints gave it.
        let replay_span = crate::tracing_layer::scoped_execution_span(
            &self.inner.execution_arn,
            &self.inner.lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine,
                suspension_signal: Arc::clone(&self.inner.suspension_signal),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                coalescer: self.inner.coalescer.clone(),
                parent_wire_id: Some(crate::engine::compute_wire_id_public(parent_positional_id)),
                replay_span,
            }),
        }
    }

    /// Mints the next operation ID (internal engine concern).
    ///
    /// Also re-records THIS namespace's span `isReplay` flag: after each
    /// claim, the namespace is replaying iff the NEXT operation to be
    /// claimed in it has a checkpoint record. This keeps replay-aware log
    /// filters (see `tracing_layer::ReplayFilterLayer`) in step with the
    /// live replay status as each namespace crosses its own replay
    /// high-water mark.
    pub(crate) fn mint_id(&self) -> OperationId {
        let id = self.inner.engine.mint_id();
        self.inner.replay_span.record(
            crate::tracing_layer::fields::IS_REPLAY,
            self.inner.engine.is_replaying(),
        );
        id
    }

    /// Returns this namespace's `durable_execution` span. The handler future
    /// (for a root context) or the child body future (for a child context)
    /// is instrumented with it so log events inherit the execution ARN,
    /// request ID, and the namespace's live `isReplay` flag.
    pub(crate) fn replay_span(&self) -> tracing::Span {
        self.inner.replay_span.clone()
    }

    /// Emits the
    /// [`operation_replayed`](crate::observability::event_names::OPERATION_REPLAYED)
    /// lifecycle event: the operation's recorded terminal outcome is being
    /// returned without re-running it (see [`crate::observability`]).
    ///
    /// `recorded_attempt` is the checkpoint record's retry count; the event
    /// reports the 1-based attempt that produced the recorded outcome.
    pub(crate) fn emit_operation_replayed(
        &self,
        wire_id: &str,
        operation_name: Option<&str>,
        operation_type: &str,
        operation_sub_type: Option<&str>,
        recorded_attempt: u32,
    ) {
        crate::tracing_layer::operation_replayed_event(&crate::tracing_layer::OperationIdentity {
            execution_arn: self.execution_arn(),
            request_id: &self.lambda_context().request_id,
            operation_id: wire_id,
            operation_name,
            operation_type,
            operation_sub_type,
            attempt: recorded_attempt.saturating_add(1),
        });
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
        let engine = Arc::new(EngineState::new_child(
            parent_positional_id,
            Arc::clone(&self.inner.engine.checkpoint_log),
        ));
        // A branch runs as its own task with its own namespace and its own
        // replay high-water mark; give it its own detached span (initialized
        // from the BRANCH engine's replay status) so its mints track branch
        // replay state without clobbering the root handler span's flag.
        let replay_span = crate::tracing_layer::scoped_execution_span(
            &self.inner.execution_arn,
            &self.inner.lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine,
                suspension_signal: Arc::new(self.inner.suspension_signal.new_child_scope()),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                coalescer: self.inner.coalescer.clone(),
                parent_wire_id: Some(crate::engine::compute_wire_id_public(parent_positional_id)),
                replay_span,
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
        let engine = Arc::new(EngineState::new_child(
            child_positional_id,
            Arc::clone(&self.inner.engine.checkpoint_log),
        ));
        // Same reasoning as `new_scoped_child`: a flat branch has its own
        // namespace and replay high-water mark, so it gets its own span.
        let replay_span = crate::tracing_layer::scoped_execution_span(
            &self.inner.execution_arn,
            &self.inner.lambda_context.request_id,
            engine.is_replaying(),
        );
        Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine,
                suspension_signal: Arc::new(self.inner.suspension_signal.new_child_scope()),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                coalescer: self.inner.coalescer.clone(),
                parent_wire_id: Some(parent_wire_id_override.to_owned()),
                replay_span,
            }),
        }
    }

    /// Creates a clone of this context bound to a FRESH child suspension
    /// scope, for an eagerly spawned (`.spawn()`ed) operation.
    ///
    /// Returns the rebound context and the new scope. Everything else —
    /// engine state (so the ID counter and checkpoint log stay shared, and
    /// replay identity is unaffected), client, token, ownership, ARN — is the
    /// same as this context: only the suspension scope differs. A parking
    /// operation inside the spawned task therefore suspends ITS OWN scope
    /// (observed by the spawned task's [`drive_scope`](crate::driver::drive_scope))
    /// instead of the owner's, exactly like a map/parallel branch.
    pub(crate) fn spawn_scope(&self) -> (Self, Arc<SuspensionSignal>) {
        let scope = Arc::new(self.inner.suspension_signal.new_child_scope());
        let rebound = Self {
            inner: Arc::new(Inner {
                execution_arn: self.inner.execution_arn.clone(),
                lambda_context: self.inner.lambda_context.clone(),
                engine: Arc::clone(&self.inner.engine),
                suspension_signal: Arc::clone(&scope),
                task_ownership: Arc::clone(&self.inner.task_ownership),
                execution_client: self.inner.execution_client.clone(),
                checkpoint_token: Arc::clone(&self.inner.checkpoint_token),
                coalescer: self.inner.coalescer.clone(),
                parent_wire_id: self.inner.parent_wire_id.clone(),
                // Same engine (same ID counter and namespace), so mints
                // through the rebound context keep the shared flag current.
                replay_span: self.inner.replay_span.clone(),
            }),
        };
        (rebound, scope)
    }

    /// Advances the ID counter by `n` positions without minting.
    ///
    /// Used after replaying a terminal batch: the child IDs consumed during
    /// the original execution must be skipped so the next operation gets
    /// the correct positional ID.
    ///
    /// Like [`Self::mint_id`], re-records this namespace's span `isReplay`
    /// flag afterwards. The skipped positions are a flat batch's synthetic
    /// child IDs, which intentionally have no checkpoint records, so
    /// minting the terminal batch parent set the flag to `false`; only
    /// after the skip does the next positional ID name the caller's next
    /// logical operation, whose record (or absence) decides whether the
    /// namespace is still replaying.
    pub(crate) fn advance_counter(&self, n: usize) {
        self.inner.engine.id_counter.advance(n);
        self.inner.replay_span.record(
            crate::tracing_layer::fields::IS_REPLAY,
            self.inner.engine.is_replaying(),
        );
    }

    /// Peeks at the operation identity `offset` positions ahead of the next
    /// mint in this namespace, without advancing the counter or touching
    /// the replay span.
    ///
    /// Terminal-batch replay uses this to derive the child IDs a prior live
    /// run minted: reading them non-destructively lets a failed replay
    /// attempt fall back to re-execution with the counter untouched.
    pub(crate) fn peek_id_at(&self, offset: usize) -> OperationId {
        self.inner.engine.id_counter.peek_at(offset as u64)
    }

    /// Returns a reference to the suspension signal for this context.
    ///
    /// Operations use this to request suspension when they cannot proceed.
    pub(crate) fn suspension_signal(&self) -> &Arc<SuspensionSignal> {
        &self.inner.suspension_signal
    }

    /// Returns a reference to the task-ownership tracker.
    pub(crate) fn task_ownership(&self) -> &Arc<TaskOwnership> {
        &self.inner.task_ownership
    }

    /// Checks task ownership and returns an `OperationError` if the caller
    /// is not authorized. Used by every durable operation entry point.
    pub(crate) fn enforce_task_ownership(&self) -> Result<(), OperationError> {
        self.inner
            .task_ownership
            .check_current_task()
            .map_err(|msg| {
                OperationError::from_kind(OperationErrorKind::Step(StepError::new(
                    StepErrorKind::ExecutionFailed,
                    Some(msg.into()),
                )))
            })
    }

    /// Reads the checkpoint record for the given positional ID in place,
    /// applying `f` under the log's read guard.
    ///
    /// Returns `None` when no record exists. This (and the targeted
    /// accessors below) replaces the removed full-record clone: `f` borrows
    /// the stored record, so only what `f` returns is cloned. `f` must not
    /// touch the checkpoint log (the read guard is held while it runs).
    ///
    /// NOTE: The checkpoint log is keyed by wire ID (the hash), which is
    /// what the backend returns in Operations[].Id. We look up by wire ID
    /// computed from the positional ID.
    pub(crate) fn with_checkpoint_record<R>(
        &self,
        positional_id: &str,
        f: impl FnOnce(&CheckpointRecord) -> R,
    ) -> Option<R> {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.with_record(&wire_id, f)
    }

    /// Returns the compact view (status, attempt,
    /// `replay_children`) of the checkpoint record for the given positional
    /// ID, without cloning any of the record's owned strings.
    pub(crate) fn checkpoint_status_view(
        &self,
        positional_id: &str,
    ) -> Option<CheckpointStatusView> {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.status_view(&wire_id)
    }

    /// Returns whether a checkpoint record exists for the given positional
    /// ID, without cloning anything.
    ///
    /// Production operation paths read record presence through
    /// [`Self::checkpoint_view_validated`] (which also validates identity);
    /// this direct form is retained for unit tests.
    #[cfg(test)]
    pub(crate) fn has_checkpoint_record(&self, positional_id: &str) -> bool {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        self.inner.engine.checkpoint_log.contains(&wire_id)
    }

    /// Returns the recorded result payload for the given positional ID,
    /// cloning only that one string.
    ///
    /// `None` means either no record exists or the record carries no
    /// result — callers that need to distinguish should use
    /// [`Self::with_checkpoint_record`].
    pub(crate) fn checkpoint_result_payload(&self, positional_id: &str) -> Option<String> {
        self.with_checkpoint_record(positional_id, |record| record.result.clone())
            .flatten()
    }

    /// Returns the backend-assigned callback ID for the given positional
    /// ID, cloning only that one string.
    pub(crate) fn checkpoint_callback_id(&self, positional_id: &str) -> Option<String> {
        self.with_checkpoint_record(positional_id, |record| record.callback_id.clone())
            .flatten()
    }

    /// Returns the recorded wire failure record for the given positional
    /// ID, cloning only the failure strings. `None` means no record
    /// exists.
    pub(crate) fn checkpoint_wire_error(
        &self,
        positional_id: &str,
    ) -> Option<crate::error::WireError> {
        self.with_checkpoint_record(positional_id, |record| {
            crate::error::WireError::new(record.error_type.clone(), record.error_message.clone())
                .with_error_data(record.error_data.clone())
                .with_stack_trace(record.stack_trace.clone().unwrap_or_default())
        })
    }

    /// Returns the terminal-replay projection (status, `replay_children`,
    /// and the result/error strings) of the checkpoint record for the
    /// given positional ID, cloning only those payload strings.
    ///
    /// This is what the map/parallel replay helpers consume when
    /// reconstructing a terminal batch or child item; the record's invoke,
    /// callback, attempt, ID, and identity fields are never cloned. `None`
    /// means no record exists.
    pub(crate) fn checkpoint_terminal_replay(
        &self,
        positional_id: &str,
    ) -> Option<TerminalReplaySnapshot> {
        self.with_checkpoint_record(positional_id, |record| TerminalReplaySnapshot {
            status: record.status.clone(),
            replay_children: record.replay_children,
            result: record.result.clone(),
            error_message: record.error_message.clone(),
            error_type: record.error_type.clone(),
        })
    }

    /// Validates the claimed operation identity against the checkpoint
    /// record and returns the compact status view, in a single read-guard
    /// pass — the common preamble of every operation's replay check.
    ///
    /// Returns `Ok(None)` when no record exists (live position, nothing to
    /// validate), `Ok(Some(view))` when the identity matches, and the
    /// `NonDeterministicExecution` error (also recorded on the fatal slot,
    /// exactly as [`Self::validate_replay_identity`] does) on mismatch —
    /// or when the record carries a [`CheckpointStatus::Unknown`] status
    /// this SDK version cannot interpret, in which case the error names
    /// the raw status. Nothing is cloned on the match path beyond the
    /// status itself.
    pub(crate) fn checkpoint_view_validated(
        &self,
        positional_id: &str,
        wire_id: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<Option<CheckpointStatusView>, OperationError> {
        let Some((mismatch, view)) = self.with_checkpoint_record(positional_id, |record| {
            (
                replay_identity_mismatch(record, claimed_type, claimed_sub_type, claimed_name),
                CheckpointStatusView {
                    status: record.status.clone(),
                    attempt: record.attempt,
                    replay_children: record.replay_children,
                },
            )
        }) else {
            return Ok(None);
        };
        if let Some(expected) = mismatch {
            // Built (and the fatal slot written) OUTSIDE the read guard.
            return Err(self.replay_mismatch_error(
                wire_id,
                &expected,
                claimed_type,
                claimed_sub_type,
                claimed_name,
            ));
        }
        if let CheckpointStatus::Unknown(raw) = &view.status {
            // A status this SDK cannot interpret must never be acted on:
            // guessing non-terminal re-runs completed work, and guessing
            // terminal fabricates an outcome that was never recorded. Fail
            // the execution naming the raw status (issue #45).
            return Err(self.unrecognized_status_error(wire_id, raw));
        }
        Ok(Some(view))
    }

    /// Eagerly validates a claimed operation's replay identity against the
    /// checkpoint log, BEFORE the operation future first runs.
    ///
    /// Builders call this when they are finalized into a
    /// [`crate::DurableFuture`] (`.future()`, `.spawn()`, or `.await` via
    /// `IntoFuture`). Validating at finalization instead of first poll makes
    /// fatal propagation scheduler-independent: a short-circuiting combinator
    /// (`select_ok`, `race`, `try_join_all`) aborts losers as soon as a
    /// winner settles, so a mismatching constituent might otherwise never be
    /// polled and the mismatch never recorded. The fatal slot is written
    /// synchronously here (inside [`Self::validate_replay_identity`]), so the
    /// invocation driver fails the execution with the dedicated error no
    /// matter which sibling settles first.
    ///
    /// A position with no checkpoint record is live (nothing to validate),
    /// and a matching identity is a no-op — an unchanged handler behaves
    /// identically.
    pub(crate) fn preflight_replay_identity(
        &self,
        op_id: &OperationId,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<(), OperationError> {
        self.checkpoint_view_validated(
            op_id.positional(),
            op_id.wire(),
            claimed_type,
            claimed_sub_type,
            claimed_name,
        )?;
        Ok(())
    }

    /// Validates that the claimed operation's identity matches the
    /// checkpoint record. On mismatch, returns a
    /// `NonDeterministicExecution` error.
    ///
    /// `claimed_type` is the operation type string the SDK sends to the
    /// backend (e.g. `Step`, `Wait`, `Context`, `ChainedInvoke`,
    /// `Callback`).
    ///
    /// `claimed_sub_type` is the sub-type (e.g. `Step`, `Wait`, `Map`,
    /// `RunInChildContext`) or `None` for operations without a sub-type.
    ///
    /// `claimed_name` is the user-supplied `.name("...")` or `None`.
    ///
    /// Production paths validate through [`Self::checkpoint_view_validated`]
    /// (one read-guard pass, no record clone); this record-taking form is
    /// retained as the direct unit-test harness for the identity-matching
    /// rules both share.
    #[cfg(test)]
    pub(crate) fn validate_replay_identity(
        &self,
        record: &CheckpointRecord,
        wire_id: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<(), OperationError> {
        match replay_identity_mismatch(record, claimed_type, claimed_sub_type, claimed_name) {
            None => Ok(()),
            Some(expected) => Err(self.replay_mismatch_error(
                wire_id,
                &expected,
                claimed_type,
                claimed_sub_type,
                claimed_name,
            )),
        }
    }

    /// Builds the `NonDeterministicExecution` mismatch error and records it
    /// on the shared fatal slot.
    ///
    /// A replay identity mismatch is execution-fatal: recording it on the
    /// shared slot makes the invocation driver fail the execution with the
    /// dedicated error even if the returned `Err` is swallowed on its way
    /// up — stored as a rejected outcome by `join_all`, out-raced by a
    /// sibling's success in `select_ok`, stringified through a
    /// child-context boundary, or tolerated by a map/parallel completion
    /// config.
    fn replay_mismatch_error(
        &self,
        wire_id: &str,
        expected: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> OperationError {
        let err = OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
            NonDeterministicExecutionError::from_kind(
                NonDeterministicExecutionErrorKind::OperationMismatch(
                    crate::error::OperationMismatch::new(
                        wire_id,
                        expected,
                        format_op_identity(claimed_type, claimed_sub_type, claimed_name),
                    ),
                ),
            ),
        ));
        self.inner.suspension_signal.record_fatal(
            "NonDeterministicExecutionError".to_owned(),
            crate::error::chain_string(&err),
        );
        err
    }

    /// Builds the execution-fatal error for a checkpoint record whose
    /// status this SDK version does not recognize, recording it on the
    /// shared fatal slot (issue #45).
    ///
    /// Exactly like [`Self::replay_mismatch_error`], the fatal-slot write
    /// makes the invocation driver fail the execution with the dedicated
    /// error even if the returned `Err` is swallowed on its way up. The
    /// error message carries the raw status verbatim, so the operator can
    /// see which service-side status the SDK build predates.
    pub(crate) fn unrecognized_status_error(&self, wire_id: &str, raw: &str) -> OperationError {
        let err = OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
            NonDeterministicExecutionError::from_kind(
                NonDeterministicExecutionErrorKind::UnrecognizedStatus(
                    crate::error::UnrecognizedStatus::new(wire_id, raw),
                ),
            ),
        ));
        self.inner.suspension_signal.record_fatal(
            "NonDeterministicExecutionError".to_owned(),
            crate::error::chain_string(&err),
        );
        err
    }

    /// Requests suspension of THIS context's scope (used by operations that
    /// cannot proceed — e.g. a pending retry timer).
    ///
    /// Routes through the scope's quiescence gate: if operations were
    /// `.spawn()`ed into this scope and have not settled yet, the request is
    /// deferred until the last of them settles, so a park never aborts a
    /// runnable sibling. Every parking path in the SDK funnels through here
    /// (directly or via [`Self::suspend_now`]), so none can bypass the
    /// accounting.
    pub(crate) fn request_suspend(&self) {
        self.inner.suspension_signal.park_owner();
    }

    /// Suspends the invocation and never returns control to the caller.
    ///
    /// Requests suspension of this context's scope (through the scope's
    /// quiescence gate — see [`Self::request_suspend`]), then awaits a future
    /// that never resolves and never registers a waker. The driver observes
    /// the signal on the next `Poll::Pending` from the handler and drops the
    /// handler future at this await point, completing the invocation as
    /// `PENDING`. Because the future is dropped rather than resumed, an
    /// operation's suspension can never surface to user code and can never be
    /// caught or ignored. The `-> T` return type is inhabited only vacuously:
    /// the awaited future never completes, so no value is ever produced.
    ///
    /// When runnable `.spawn()`ed siblings are still outstanding in this
    /// scope, the suspension request is deferred until they settle; this call
    /// still never returns.
    pub(crate) async fn suspend_now<T>(&self) -> T {
        self.request_suspend();
        std::future::pending::<T>().await
    }

    /// Routes a checkpoint write failure through the unrecoverable path
    /// and never returns control to the caller (issue #43).
    ///
    /// A failed outcome write must never become a catchable operation
    /// error: a handler that catches it branches on a decision no
    /// checkpoint records, which replay cannot reproduce. Instead the
    /// failure's classification (see
    /// [`ClientError::is_retryable`](crate::client::ClientError)) picks
    /// the recovery scope:
    ///
    /// - **Retryable** (exhausted transient failure, stale token): the
    ///   write channel is down, so a follow-up write would fail the same
    ///   way. Nothing more is written; the invocation fails with a Lambda
    ///   runtime error and the service re-invokes — the same recovery as
    ///   an interruption.
    /// - **Non-retryable** (permanent rejection, e.g. oversized payload):
    ///   re-invoking would re-run the body into the same rejection, an
    ///   infinite loop with side effects firing once per lap. When the
    ///   caller supplies `terminal_fail` — required at every site where
    ///   user code already ran, so its side effects get a recorded
    ///   outcome — that small terminal `FAIL` is written first (it goes
    ///   through on a channel that rejected only the payload; a failure
    ///   here is logged, not propagated, because the execution dies
    ///   either way). Then the execution fails with
    ///   [`CHECKPOINT_FAILED_ERROR_TYPE`](crate::error::CHECKPOINT_FAILED_ERROR_TYPE).
    ///
    /// Like [`Self::suspend_now`], the returned future never resolves: the
    /// fatal slot is recorded and the invocation driver — which checks it
    /// with priority over completion and suspension, from any scope in the
    /// tree — drops the handler at its current await point, so user code
    /// can neither catch nor ignore the failure.
    pub(crate) async fn checkpoint_failure_unrecoverable<T>(
        &self,
        op_wire_id: &str,
        err: ClientError,
        terminal_fail: Option<OperationUpdate>,
    ) -> T {
        let message = format!("checkpoint write failed for operation {op_wire_id}: {err}");
        if err.is_retryable() {
            tracing::error!(
                operation_id = %op_wire_id,
                error = %err,
                "retryable checkpoint failure exhausted retries; failing the invocation \
                 so the service re-invokes"
            );
            self.inner
                .suspension_signal
                .record_invocation_fault(message);
        } else {
            tracing::error!(
                operation_id = %op_wire_id,
                error = %err,
                "non-retryable checkpoint failure; recording terminal FAIL and failing \
                 the execution"
            );
            if let Some(update) = terminal_fail {
                // Direct write: the coalescing buffer may hold the very
                // updates the service just rejected, and this record must
                // not queue behind them.
                if let Err(fail_err) = self.checkpoint_updates_direct(vec![update]).await {
                    tracing::error!(
                        operation_id = %op_wire_id,
                        error = %fail_err,
                        "terminal FAIL write also failed after non-retryable checkpoint \
                         failure; failing the execution without a recorded outcome"
                    );
                }
            }
            self.inner.suspension_signal.record_fatal(
                crate::error::CHECKPOINT_FAILED_ERROR_TYPE.to_owned(),
                message,
            );
        }
        std::future::pending::<T>().await
    }

    /// Checkpoints operation updates via the execution client.
    ///
    /// When [`Options`](crate::Options) configured a `checkpoint_delay`
    /// and/or `checkpoint_batching`, the updates join the shared coalescing
    /// batch and this call resolves with the batch's result once it flushes
    /// (after at most the configured delay, or earlier if a flush point
    /// drains the buffer). Without either knob, the write happens
    /// immediately via [`Self::checkpoint_updates_direct`].
    pub(crate) async fn checkpoint_updates(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ClientError> {
        self.checkpoint_with_urgency(updates, false).await
    }

    /// Checkpoints operation updates, flushing the coalescing buffer
    /// immediately instead of waiting out the delay window.
    ///
    /// This is the "callback creation" flush point of the
    /// [`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay)
    /// contract: the caller needs the backend's response right away (the
    /// service assigns the callback ID in it), so the batch — including any
    /// previously buffered updates, which keeps write order intact — is
    /// written now. Identical to [`Self::checkpoint_updates`] when no
    /// delay is configured.
    pub(crate) async fn checkpoint_updates_urgent(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ClientError> {
        self.checkpoint_with_urgency(updates, true).await
    }

    /// Shared body of the buffered checkpoint paths: pairs each update
    /// with its captured lifecycle-event metadata, then hands the pairs to
    /// the write path, which emits each event as soon as the write that
    /// persists its update succeeds.
    async fn checkpoint_with_urgency(
        &self,
        updates: Vec<OperationUpdate>,
        urgent: bool,
    ) -> Result<CheckpointOutput, ClientError> {
        // Lifecycle events (see `crate::observability`): every operation
        // transition the SDK records passes through here, so this one site
        // covers `operation_started` / `operation_succeeded` /
        // `operation_failed` / `operation_retry_scheduled` for every
        // operation type. The metadata is captured BEFORE the write (the
        // checkpoint response mutates the log the attempt is derived from,
        // and the updates themselves are consumed by it; the capture also
        // snapshots this call's span so the event keeps its originating
        // context). Ownership then travels WITH the update into the write
        // path — the buffered flush task, not this possibly-cancelled
        // future, emits the event — so a contributor dropped after joining
        // a batch (a lost `race`/`select_ok` branch) cannot suppress
        // telemetry for a transition the flush still persists, and events
        // are emitted per persisted chunk rather than per batch. A rejected
        // write records nothing, so it emits nothing.
        let tracked: Vec<TrackedUpdate> = updates
            .into_iter()
            .map(|update| {
                // The current attempt is one past the recorded retry count,
                // exactly as the step live path derives it.
                let attempt = self
                    .inner
                    .engine
                    .checkpoint_log
                    .status_view(update.id())
                    .map_or(0, |view| view.attempt)
                    .saturating_add(1);
                TrackedUpdate::capture(update, attempt)
            })
            .collect();

        self.write_with_urgency(tracked, urgent).await
    }

    /// Dispatches a checkpoint write: joins the coalescing batch when one
    /// is configured, otherwise writes directly. A non-urgent contributor
    /// under a non-zero delay arms the delay timer and requests the flush
    /// itself when the window elapses; an urgent contributor — or any
    /// contributor under a zero-delay coalescer (pure `checkpoint_batching`
    /// mode) — requests the flush right away. Either way the flush task
    /// claims and writes the batch under the coalescer's writer lock, so
    /// batching still emerges while an earlier write is in flight.
    ///
    /// Joining the batch transfers ownership of each update's lifecycle
    /// event to the coalescer: the flush task that persists the update
    /// emits it, so this future can be dropped mid-await without losing
    /// telemetry for a transition the flush still records.
    async fn write_with_urgency(
        &self,
        updates: Vec<TrackedUpdate>,
        urgent: bool,
    ) -> Result<CheckpointOutput, ClientError> {
        let Some(coalescer) = self.inner.coalescer.clone() else {
            return self.write_tracked_direct(updates).await;
        };

        let batch = coalescer.join(updates);
        let flush_now = urgent || coalescer.delay().is_zero();
        if flush_now {
            self.spawn_batch_flush(&coalescer, &batch);
        }

        loop {
            // `enable()` registers the waiter BEFORE the result check, so a
            // publish landing between the check and the await still wakes us.
            let mut notified = std::pin::pin!(batch.notified());
            notified.as_mut().enable();
            if let Some(result) = batch.result_clone() {
                return result;
            }
            if flush_now {
                notified.await;
            } else {
                tokio::select! {
                    () = notified => {}
                    () = tokio::time::sleep(coalescer.delay()) => {
                        self.spawn_batch_flush(&coalescer, &batch);
                    }
                }
            }
        }
    }

    /// Writes one request's worth of tracked updates immediately and, if —
    /// and only if — the checkpoint call that persists them succeeds, emits
    /// each update's lifecycle event (inside the span captured with it).
    /// This is the single site that pairs "the transition was persisted"
    /// with "its telemetry was emitted": both the unbuffered path and every
    /// chunk of a batched write funnel through it. The emission happens
    /// inside [`Self::checkpoint_direct_with_events`], immediately after
    /// the service accepts the write — before the fallible pagination
    /// hydration that follows it — so a transition the service recorded
    /// always emits its event even when hydrating the paginated state
    /// fails afterwards.
    async fn write_tracked_direct(
        &self,
        tracked: Vec<TrackedUpdate>,
    ) -> Result<CheckpointOutput, ClientError> {
        let (updates, events): (Vec<_>, Vec<_>) =
            tracked.into_iter().map(|t| (t.update, t.event)).unzip();
        self.checkpoint_direct_with_events(updates, &events).await
    }

    /// Spawns a task that claims `batch` (if it is still the open batch)
    /// and writes its buffered updates, publishing the result to every
    /// contributor. The claim and the write both happen while holding the
    /// coalescer's writer lock, so buffered writes are totally ordered and
    /// a flush point that acquires the lock waits for this write to finish.
    /// If an earlier buffered write already failed (the coalescer's
    /// failure latch, checked under the same lock), nothing is written:
    /// the claimed updates are retained as unwritten and the prior error
    /// is published to this batch's contributors (issue #43, "no further
    /// writes" after a checkpoint channel failure).
    ///
    /// The write runs on its own task deliberately: a contributor dropped
    /// mid-await (a lost `race`, a dropped `DurableFuture`) must not cancel
    /// an in-flight batch write that other contributors are waiting on.
    fn spawn_batch_flush(
        &self,
        coalescer: &Arc<CheckpointCoalescer>,
        batch: &Arc<CheckpointBatch>,
    ) {
        let ctx = self.clone();
        let coalescer = Arc::clone(coalescer);
        let batch = Arc::clone(batch);
        drop(tokio::spawn(async move {
            let _writer = coalescer.writer_lock().lock().await;
            if let Some(updates) = coalescer.take_batch(&batch) {
                // Failure latch (issue #43): once any buffered write has
                // failed, the channel is down for the rest of the
                // invocation — "no further writes". A flusher that queued
                // on the writer lock behind the failing write must not
                // perform another checkpoint call: doing so could persist
                // replay-visible transitions after the invocation is
                // already doomed. Instead it retains its updates for the
                // flush-point classification and publishes the prior
                // error, so its contributors route through the same
                // unrecoverable path the failing batch's contributors
                // took. The latch is set by the failing flusher while it
                // holds this same lock, so the check cannot race it.
                if let Some(prior) = coalescer.latched_failure() {
                    coalescer.record_failed_flush(
                        prior.clone(),
                        updates.into_iter().map(|t| t.update).collect(),
                    );
                    batch.publish(Err(prior));
                    return;
                }
                let result = match ctx.write_batched_updates(updates, coalescer.limits()).await {
                    Ok(output) => Ok(output),
                    Err((error, unwritten)) => {
                        // Retain the failure for the end-of-invocation
                        // flush point (issue #43): every contributor of
                        // this batch may already be dropped (a lost
                        // `race`/`select_ok` branch, a dropped
                        // `DurableFuture`), and a failure published to
                        // nobody would otherwise be fully discarded,
                        // leaving the affected operations' records
                        // claiming less than what executed.
                        coalescer.record_failed_flush(error.clone(), unwritten);
                        Err(error)
                    }
                };
                batch.publish(result);
            }
        }));
    }

    /// Writes a sealed batch, splitting it into request-sized chunks that
    /// respect the coalescer's [`BatchLimits`] (operation count and
    /// estimated payload bytes) while preserving join order. Returns the
    /// last chunk's output on success — backend-assigned fields from every
    /// chunk are already merged into the checkpoint log by
    /// [`Self::checkpoint_updates_direct`] — or the first chunk error,
    /// which aborts the remaining chunks so contributors observe it. The
    /// error carries the updates the write did NOT persist — the rejected
    /// chunk plus every chunk after it, never a chunk that already
    /// succeeded — so the #43 flush point can write terminal `FAIL`
    /// records for exactly the operations whose outcomes were lost.
    ///
    /// Each chunk's lifecycle events are emitted immediately after that
    /// chunk's write succeeds (see [`Self::write_tracked_direct`]): a chunk
    /// persisted before a later chunk fails still emits its events, even
    /// though the batch as a whole publishes the error, so telemetry stays
    /// faithful to what was actually recorded.
    ///
    /// Callers must hold the coalescer's writer lock (see the invariants in
    /// [`crate::checkpoint_coalescer`]).
    async fn write_batched_updates(
        &self,
        updates: Vec<TrackedUpdate>,
        limits: BatchLimits,
    ) -> Result<CheckpointOutput, (ClientError, Vec<OperationUpdate>)> {
        if updates.is_empty() {
            // An empty seal (possible only defensively) still performs one
            // call so a waiting contributor receives a published result.
            return self
                .checkpoint_updates_direct(Vec::new())
                .await
                .map_err(|e| (e, Vec::new()));
        }
        let mut chunks = split_into_requests(updates, &limits).into_iter();
        let mut last = None;
        while let Some(chunk) = chunks.next() {
            // Snapshot the chunk before the write consumes it: on
            // rejection this chunk is the unwritten head, and the
            // remaining chunks the unwritten tail.
            let snapshot: Vec<OperationUpdate> = chunk.iter().map(|t| t.update.clone()).collect();
            match self.write_tracked_direct(chunk).await {
                Ok(output) => last = Some(output),
                Err(err) => {
                    let mut unwritten = snapshot;
                    unwritten.extend(chunks.flatten().map(|t| t.update));
                    return Err((err, unwritten));
                }
            }
        }
        last.ok_or_else(|| {
            (
                ClientError::new_non_retryable("internal: batched checkpoint produced no requests"),
                Vec::new(),
            )
        })
    }

    /// Unconditionally drains the checkpoint coalescing buffer, writing any
    /// pending batches now, and waits for any in-flight batch write to
    /// finish before returning.
    ///
    /// This is the "suspension" and "execution completion" flush point of
    /// the [`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay) /
    /// [`checkpoint_batching`](crate::OptionsBuilder::checkpoint_batching)
    /// contract: the invocation wrapper calls it after the driver settles —
    /// before reporting `PENDING`, `SUCCEEDED`, or `FAILED` to the service —
    /// so a checkpoint that must land before the invocation ends is never
    /// held back by the buffer. Because every claimed batch is written while
    /// holding the coalescer's writer lock, acquiring that lock here makes
    /// this a true barrier: a batch a delay timer already claimed cannot
    /// still be in flight when this returns. A no-op when no buffering is
    /// configured or the buffer is idle.
    ///
    /// A failure is returned as a [`FlushFailure`] carrying every write
    /// failure this invocation's buffered checkpoints suffered — the one
    /// hit here directly, the remaining drained-but-unwritten batches, and
    /// any failure a spawned batch flush retained because its contributors
    /// were all dropped — together with the updates that were not
    /// persisted, so the caller can apply the issue #43 classification
    /// (retryable fails the invocation with no further writes;
    /// non-retryable persists terminal `FAIL` records for the affected
    /// operations, then fails the execution).
    pub(crate) async fn flush_pending_checkpoints(&self) -> Result<(), FlushFailure> {
        let Some(coalescer) = self.inner.coalescer.clone() else {
            return Ok(());
        };
        // Acquiring the writer lock waits out any in-flight batch write
        // (batches are only claimed and written under it), then holding it
        // across the drain keeps this flush ordered after those writes.
        let _writer = coalescer.writer_lock().lock().await;
        // Failure latch (issue #43): a buffered write that already failed
        // — in a spawned flush this drain never contributed to — poisons
        // the channel for the rest of the invocation. Seed the drain's
        // "already failed" state from it, so pending batches are published
        // the prior error and retained as unwritten instead of written.
        let mut prior: Option<ClientError> = coalescer.latched_failure();
        let mut failures: Vec<crate::checkpoint_coalescer::FailedFlush> = Vec::new();
        while let Some((batch, updates)) = coalescer.take_any() {
            if let Some(error) = prior.clone() {
                // The channel already failed: do not attempt further
                // outcome writes ("no further writes" for the retryable
                // case; the non-retryable case writes only the small
                // terminal FAILs, and those go through the caller, not
                // here). Publish the error and collect the updates as
                // unwritten.
                batch.publish(Err(error.clone()));
                failures.push(crate::checkpoint_coalescer::FailedFlush {
                    error,
                    unwritten: updates.into_iter().map(|t| t.update).collect(),
                });
                continue;
            }
            match self
                .write_batched_updates(updates, coalescer.limits())
                .await
            {
                Ok(output) => batch.publish(Ok(output)),
                Err((error, unwritten)) => {
                    batch.publish(Err(error.clone()));
                    prior = Some(error.clone());
                    failures.push(crate::checkpoint_coalescer::FailedFlush { error, unwritten });
                }
            }
        }
        // Failures a spawned flush retained because its contributors were
        // all dropped happened BEFORE this drain: report them first.
        let mut all = coalescer.take_failed_flushes();
        all.extend(failures);
        if all.is_empty() {
            Ok(())
        } else {
            Err(FlushFailure { failures: all })
        }
    }

    /// Drains ONLY the write failures retained by detached (spawned) batch
    /// flushes, attempting no writes and leaving any still-pending batches
    /// in the buffer untouched.
    ///
    /// This is the `Fault`-outcome counterpart of
    /// [`Self::flush_pending_checkpoints`] (issue #43): when a retryable
    /// checkpoint failure is already failing the invocation, the "no
    /// further writes" contract forbids flushing the buffer — a follow-up
    /// write would fail the same way, and re-invocation re-runs the
    /// buffered operations under the interruption contract. But a failure
    /// a spawned flush retained on behalf of dropped contributors (a lost
    /// `race`/`select_ok` branch, a dropped [`crate::DurableFuture`]) has
    /// already happened and must still be classified: silently discarding
    /// a NON-retryable one would leave its operations' records claiming
    /// less than what executed on every future invocation — the very loop
    /// the #43 terminalization exists to end.
    ///
    /// Acquiring the coalescer's writer lock first waits out any batch
    /// write still in flight (batches are only claimed and written under
    /// it), so a failure that write is about to retain is observed rather
    /// than raced past. Waiting is not writing: nothing is sent here.
    pub(crate) async fn take_retained_flush_failures(&self) -> Option<FlushFailure> {
        let coalescer = self.inner.coalescer.clone()?;
        let _writer = coalescer.writer_lock().lock().await;
        let failures = coalescer.take_failed_flushes();
        if failures.is_empty() {
            None
        } else {
            Some(FlushFailure { failures })
        }
    }

    /// Writes a small terminal `FAIL` record for every operation whose
    /// buffered OUTCOME write was lost to a non-retryable flush failure
    /// (issue #43). Called by the invocation wrapper before it fails the
    /// execution, so each affected operation's record claims exactly what
    /// executed instead of a dangling `Started`.
    ///
    /// Per operation (first-appearance order across the failures):
    /// - An operation with an unwritten outcome update (`Succeed`, `Fail`,
    ///   or `Retry`) gets a terminal `FAIL` derived from the flush failure
    ///   — user code ran, so its side effects need a recorded outcome. An
    ///   unwritten `Start` for the same operation is written first so the
    ///   `FAIL` has a record to terminate.
    /// - An operation whose only unwritten update is its `Start` gets
    ///   nothing: no user-visible outcome was discarded, and the execution
    ///   is failing anyway.
    /// - An operation the checkpoint log already shows terminal is
    ///   skipped: a live contributor's unrecoverable routing may already
    ///   have persisted its terminal `FAIL`, and a `FAIL` must never be
    ///   written over a recorded outcome.
    ///
    /// Writes go one operation per request so one rejected record cannot
    /// take its siblings' terminal `FAIL`s down with it; individual
    /// failures are logged, not propagated — the execution dies either way.
    pub(crate) async fn terminalize_unwritten_outcomes(&self, flush: &FlushFailure) {
        use aws_sdk_lambda::types::OperationAction;

        struct PerOp {
            start: Option<OperationUpdate>,
            outcome: Option<OperationUpdate>,
            error: ClientError,
        }
        let mut order: Vec<String> = Vec::new();
        let mut per_op: std::collections::HashMap<String, PerOp> = std::collections::HashMap::new();
        for failure in &flush.failures {
            for update in &failure.unwritten {
                let entry = per_op.entry(update.id.clone()).or_insert_with(|| {
                    order.push(update.id.clone());
                    PerOp {
                        start: None,
                        outcome: None,
                        error: failure.error.clone(),
                    }
                });
                match update.action {
                    OperationAction::Start => {
                        entry.start.get_or_insert_with(|| update.clone());
                    }
                    _ => {
                        entry.outcome.get_or_insert_with(|| update.clone());
                    }
                }
            }
        }

        for wire_id in order {
            let Some(op) = per_op.remove(&wire_id) else {
                continue;
            };
            let Some(outcome) = op.outcome else {
                // Start-only: no outcome was discarded (see doc above).
                continue;
            };
            if self
                .inner
                .engine
                .checkpoint_log
                .status_view(&wire_id)
                .is_some_and(|view| view.status.is_terminal())
            {
                continue;
            }
            let wire = crate::error::checkpoint_failure_wire(&op.error);
            let mut updates = Vec::new();
            if let Some(start) = op.start {
                updates.push(start);
            }
            let mut builder = OperationUpdate::builder()
                .id(outcome.id.clone())
                .r#type(outcome.r#type.clone())
                .action(OperationAction::Fail)
                .error(wire.to_error_object());
            if let Some(sub_type) = &outcome.sub_type {
                builder = builder.sub_type(sub_type.clone());
            }
            if let Some(name) = &outcome.name {
                builder = builder.name(name.clone());
            }
            if let Some(parent_id) = &outcome.parent_id {
                builder = builder.parent_id(parent_id.clone());
            }
            match builder.build() {
                Ok(fail_update) => {
                    updates.push(fail_update);
                    if let Err(err) = self.checkpoint_updates_direct(updates).await {
                        tracing::error!(
                            operation_id = %wire_id,
                            error = %err,
                            "terminal FAIL write failed after non-retryable flush \
                             failure; the execution fails without a recorded outcome \
                             for this operation"
                        );
                    }
                }
                Err(err) => {
                    tracing::error!(
                        operation_id = %wire_id,
                        error = %err,
                        "could not build terminal FAIL update after non-retryable \
                         flush failure"
                    );
                }
            }
        }
    }

    /// Writes operation updates through the execution client immediately.
    ///
    /// Serializes all concurrent callers through a single critical section:
    /// the lock is held across the full read-token → API-call →
    /// rotate-token sequence, and — when the response carries a pagination
    /// marker — through the follow-up `get_state` fetch as well, so no
    /// concurrent branch can rotate the token out from under the paginated
    /// state read.
    pub(crate) async fn checkpoint_updates_direct(
        &self,
        updates: Vec<OperationUpdate>,
    ) -> Result<CheckpointOutput, ClientError> {
        self.checkpoint_direct_with_events(updates, &[]).await
    }

    /// Body of [`Self::checkpoint_updates_direct`] that additionally owns
    /// the lifecycle events paired with the updates. Each event is emitted
    /// the moment `client.checkpoint` returns success — the point at which
    /// the service has durably recorded the transitions — and **before**
    /// the pagination hydration below, which can fail after the write
    /// already persisted. Emitting first keeps telemetry faithful to what
    /// was recorded: a rejected write emits nothing, and a persisted write
    /// emits everything even when the follow-up `get_state` fetch fails.
    async fn checkpoint_direct_with_events(
        &self,
        updates: Vec<OperationUpdate>,
        events: &[crate::tracing_layer::PendingTransitionEvent],
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

        // The service has recorded the transitions: emit their lifecycle
        // events now, before the fallible pagination hydration below can
        // turn an already-persisted write into a caller-visible error.
        for event in events {
            event.emit(self.execution_arn(), &self.lambda_context().request_id);
        }

        // Rotate the token while still holding the lock.
        token_guard.clone_from(&output.checkpoint_token);

        // Merge updated operations into the checkpoint log so that
        // subsequent reads (e.g. reading callback_id after START) see
        // backend-assigned fields.
        if !output.updated_operations.is_empty() {
            crate::client::merge_operations_into_log(
                &self.inner.engine.checkpoint_log,
                &output.updated_operations,
            );
        }

        // If the checkpoint response is paginated, fetch remaining pages
        // via get_state and merge them into the checkpoint log. The token
        // lock stays held through this fetch: releasing it first would let
        // a concurrent branch checkpoint and rotate the token, leaving this
        // get_state call with a stale token.
        if output.next_marker.is_some() {
            let full_state = client
                .get_state(&self.inner.execution_arn, &token_guard)
                .await?;
            if !full_state.operations.is_empty() {
                crate::client::merge_operations_into_log(
                    &self.inner.engine.checkpoint_log,
                    &full_state.operations,
                );
            }
        }

        drop(token_guard);

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
        self.checkpoint_status_view(op_id.positional())
            .map_or(0, |view| view.attempt)
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
    /// without re-execution. Replay mode lasts while the next operation to
    /// be claimed was already claimed by a prior invocation — including a
    /// started-but-unfinished child context, map, or parallel parent that
    /// the resumed invocation re-enters to replay its nested operations.
    /// User code can use this flag to suppress duplicate side effects
    /// (e.g., logging) during replay.
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
    pub fn step<O, F, Fut>(&self, f: F) -> StepBuilder<O, F, Fut>
    where
        F: FnOnce(StepContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.mint_id();
        // The closure is stored unerased; the single erasure point is the
        // builder's `.future()` / `.await`, producing one DurableFuture box.
        StepBuilder::new(self.clone(), op_id, f)
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
    /// #[derive(Serialize, Deserialize)]
    /// struct ChargeInput { amount: u64 }
    ///
    /// #[derive(Serialize, Deserialize)]
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
    pub fn invoke<O, I>(&self, function_id: &str, input: I) -> InvokeBuilder<O, I>
    where
        I: Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.mint_id();
        // The input is carried TYPED into the builder: the payload serdes
        // receives the owned value directly at execution time (a write-only
        // transfer), so no intermediate representation is constructed here.
        InvokeBuilder::new(self.clone(), op_id, function_id.to_owned(), input)
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
    pub fn run_in_child_context<O, F, Fut>(&self, f: F) -> ChildBuilder<O, F, Fut>
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.mint_id();
        // The closure is stored unerased; the child execution converts the
        // BoxError into the internal child-error carrier at the boundary,
        // and the single erasure point is the builder's `.future()` /
        // `.await`, producing one DurableFuture box.
        ChildBuilder::new(self.clone(), op_id, f)
    }

    /// Runs a closure against a child context and retries the closure's
    /// **overall** outcome as a unit.
    ///
    /// Where a step's retry strategy re-runs one operation, `with_retry`
    /// re-runs a whole block: if any operation in the block fails (or the
    /// closure returns an error), the retry strategy decides whether to run
    /// the entire block again. Each attempt receives a **fresh child
    /// operation namespace**, so operations recorded by a failed attempt
    /// are never replayed into the next one — every operation in the block
    /// re-runs on retry. The delay between attempts suspends the execution
    /// (the backend owns the timer, exactly as step retries do), and the
    /// retry progress is derived from checkpointed results, so it survives
    /// suspension and replays deterministically.
    ///
    /// The closure is `Fn` rather than `FnOnce` because the SDK calls it
    /// once per attempt. Configure the policy with
    /// [`retry_strategy`](WithRetryBuilder::retry_strategy) or
    /// [`retry_strategy_config`](WithRetryBuilder::retry_strategy_config);
    /// without one, the step default applies (6 total attempts with
    /// exponential backoff). When retries exhaust, the operation fails with
    /// a [`ChildContextError`](crate::ChildContextError) carrying the
    /// attempt count and the last attempt's error.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::RetryDecision;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<String, durable::BoxError> {
    ///     // If either step fails, BOTH re-run on the next attempt.
    ///     let result = ctx.with_retry(|child| async move {
    ///         let quote = child
    ///             .step(|_| async { Ok("quote-17".to_owned()) })
    ///             .name("reserve-quote")
    ///             .await?;
    ///         let receipt = child
    ///             .step(move |_| async move { Ok(format!("booked:{quote}")) })
    ///             .name("book")
    ///             .await?;
    ///         Ok(receipt)
    ///     })
    ///     .name("reserve-and-book")
    ///     .retry_strategy(|_err, attempt| {
    ///         if attempt >= 3 {
    ///             RetryDecision::Stop
    ///         } else {
    ///             RetryDecision::Retry { delay: Duration::from_secs(5) }
    ///         }
    ///     })
    ///     .await?;
    ///     Ok(result)
    /// }
    /// ```
    pub fn with_retry<O, F, Fut>(&self, f: F) -> WithRetryBuilder<O, F, Fut>
    where
        F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.mint_id();
        WithRetryBuilder::new(self.clone(), op_id, f)
    }

    /// Creates a wait-for-condition operation that polls until a predicate
    /// is satisfied.
    ///
    /// The check function is called repeatedly with the current state, and
    /// the configured strategy decides after each check whether to
    /// complete, keep polling, or fail. Set one with
    /// [`wait_strategy`](crate::builders::WaitForConditionBuilder::wait_strategy)
    /// (a bounded [`WaitStrategy`](crate::builders::wait_for_condition::WaitStrategy)
    /// configuration) or
    /// [`wait_strategy_fn`](crate::builders::WaitForConditionBuilder::wait_strategy_fn)
    /// (a custom closure). With no strategy set, the check runs exactly
    /// once and the operation completes with that state.
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
    ) -> WaitForConditionBuilder<S, F, Fut>
    where
        F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
        S: Clone + Send + Sync + 'static,
    {
        let op_id = self.mint_id();
        WaitForConditionBuilder::new(self.clone(), op_id, initial_state, check)
    }

    /// Creates a callback token for external completion.
    ///
    /// The returned [`Callback`](crate::builders::callback::Callback) provides an ID that
    /// external systems use to complete the operation, plus a
    /// [`DurableFuture`] that resolves when the callback arrives.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize, Deserialize)]
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
        O: Send + 'static,
    {
        let op_id = self.mint_id();
        CreateCallbackBuilder::new(self.clone(), op_id)
    }

    /// Creates a wait-for-callback operation that registers and waits for
    /// an external callback in one step.
    ///
    /// The submitter closure receives the callback ID as an owned
    /// [`String`] and is responsible for delivering it to the external
    /// system that will complete it.
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
    pub fn wait_for_callback<O, F, Fut>(&self, submitter: F) -> WaitForCallbackBuilder<O, F, Fut>
    where
        F: FnOnce(StepContext, String) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
        O: DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        WaitForCallbackBuilder::new(self.clone(), op_id, submitter)
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
    pub fn map<Items, I, O, F, Fut>(&self, items: Items, f: F) -> MapBuilder<I, O, F, Fut>
    where
        Items: IntoIterator<Item = I>,
        F: Fn(DurableContext, I, usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
        I: Send + 'static,
        O: Send + 'static,
    {
        let op_id = self.mint_id();
        // The closure is stored unerased; at execution it is shared as
        // `Arc<F>` and every item produces a concrete future, so the
        // internal JoinSet holds unboxed futures. The single erasure point
        // is the builder's `.future()` / `.await`, producing one
        // DurableFuture box.
        let items: Vec<I> = items.into_iter().collect();
        MapBuilder::new(self.clone(), op_id, items, f)
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
    pub fn parallel<Branches, O>(&self, branches: Branches) -> ParallelBuilder<O>
    where
        Branches: IntoIterator<Item = Branch<O>>,
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        let op_id = self.mint_id();
        let branch_tuples: Vec<_> = branches.into_iter().map(Branch::into_parts).collect();
        ParallelBuilder::new(self.clone(), op_id, branch_tuples)
    }

    /// Joins all futures, failing fast on the first error.
    ///
    /// Returns `Vec<O>` on success, or the first `OperationError`
    /// encountered.
    ///
    /// # Empty input
    ///
    /// Called with no futures, resolves to `Ok` with an empty `Vec`,
    /// matching `futures::future::try_join_all`.
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
    /// # Empty input
    ///
    /// Called with no futures, resolves to `Ok` with an empty `Vec`,
    /// matching `futures::future::join_all`.
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
    /// If every future fails, the operation fails with
    /// [`CombinatorErrorKind::AllFailed`](crate::CombinatorErrorKind::AllFailed)
    /// preserving each future's error (see [`CombinatorError::failures`](crate::CombinatorError::failures)).
    ///
    /// # Empty input
    ///
    /// Called with no futures, fails with
    /// [`CombinatorErrorKind::EmptyInput`](crate::CombinatorErrorKind::EmptyInput) —
    /// there is no future that could succeed.
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
    /// When the first settled outcome is a failure, the operation fails
    /// with
    /// [`CombinatorErrorKind::FirstSettledFailed`](crate::CombinatorErrorKind::FirstSettledFailed)
    /// carrying the losing future's error as its source; the same variant
    /// is produced live and on replay.
    ///
    /// # Empty input
    ///
    /// Called with no futures, fails with
    /// [`CombinatorErrorKind::EmptyInput`](crate::CombinatorErrorKind::EmptyInput) —
    /// there is no future that could settle.
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
#[non_exhaustive]
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

/// Canonicalizes an operation type string to the service wire format via
/// [`aws_sdk_lambda::types::OperationType`].
///
/// Operation types reach the SDK in two spellings: the typed SDK path
/// stores `PascalCase` (`ChainedInvoke`, from `operation_type_to_string`),
/// while the inline JSON envelope path stores the raw wire value
/// (`CHAINED_INVOKE`, the `OperationType::as_str()` form). A plain
/// case-insensitive comparison misses the underscore, so both spellings
/// are converted to `SCREAMING_SNAKE` and round-tripped through
/// `OperationType` before comparison. Unknown types canonicalize to their
/// `SCREAMING_SNAKE` form, which keeps genuine mismatches detectable.
fn canonical_op_type(raw: &str) -> String {
    // PascalCase → SCREAMING_SNAKE: insert `_` at lower→upper boundaries,
    // then uppercase. Already-wire values (all caps + underscores) pass
    // through unchanged because they contain no lower→upper boundary.
    let mut screaming = String::with_capacity(raw.len() + 4);
    let mut prev_is_lower = false;
    for ch in raw.chars() {
        if ch.is_ascii_uppercase() && prev_is_lower {
            screaming.push('_');
        }
        prev_is_lower = ch.is_ascii_lowercase();
        screaming.push(ch.to_ascii_uppercase());
    }
    // Round-trip through the SDK enum so every known type lands on the
    // exact canonical wire constant (`OperationType::as_str()`).
    aws_sdk_lambda::types::OperationType::from(screaming.as_str())
        .as_str()
        .to_owned()
}

/// Compares a checkpoint record's stored identity against a claimed one.
///
/// Returns `None` when the identities match (or the record predates
/// identity recording), and `Some(expected)` — the stored identity
/// formatted for the mismatch error — when they differ. The comparisons
/// borrow the record, so the match path clones nothing; only the cold
/// mismatch path allocates. This is the shared core behind
/// `DurableContext::validate_replay_identity` and
/// `DurableContext::checkpoint_view_validated`.
fn replay_identity_mismatch(
    record: &CheckpointRecord,
    claimed_type: &str,
    claimed_sub_type: Option<&str>,
    claimed_name: Option<&str>,
) -> Option<String> {
    // A record without a stored operation type predates identity
    // recording (legacy checkpoint) — there is genuinely nothing to
    // validate against, so skip. This is the ONLY lenient path; once a
    // record carries identity, every field is compared in full.
    let expected_type = record.op_type.as_deref()?;

    let mismatch = || {
        Some(format_op_identity(
            expected_type,
            record.sub_type.as_deref(),
            record.op_name.as_deref(),
        ))
    };

    // Compare canonicalized types: the checkpoint log stores PascalCase
    // on the typed SDK path but the raw wire form (e.g.
    // `CHAINED_INVOKE`) on the inline JSON envelope path, so both sides
    // canonicalize through `OperationType` before comparison.
    if canonical_op_type(expected_type) != canonical_op_type(claimed_type) {
        return mismatch();
    }

    // Sub-type is compared as a complete Option in both directions: a
    // stored sub-type with no claimed sub-type (or the reverse) is a
    // mismatch, not a skip. Otherwise a reordered same-type operation
    // could consume the wrong checkpoint silently.
    let sub_type_matches = match (record.sub_type.as_deref(), claimed_sub_type) {
        (Some(expected), Some(claimed)) => expected.eq_ignore_ascii_case(claimed),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    if !sub_type_matches {
        return mismatch();
    }

    // Name is likewise compared as a complete Option in both directions.
    // Adding or removing `.name(...)` between runs changes the operation's
    // replay identity and must be flagged, for the same reason as above.
    //
    // Empty names are normalized to `None` on BOTH sides before the
    // comparison: the checkpoint builders deliberately omit the `Name`
    // field when the string is empty (see `build_child_update` in
    // `map_parallel`), so the record stores `None` where the claim
    // computes `Some("")` — e.g. a map `item_namer` or a parallel
    // `Branch` whose name is the empty string. Without normalization an
    // UNCHANGED handler would be rejected on resume.
    let claimed_name = claimed_name.filter(|n| !n.is_empty());
    let expected_name = record.op_name.as_deref().filter(|n| !n.is_empty());
    let name_matches = match (expected_name, claimed_name) {
        (Some(expected), Some(claimed)) => expected == claimed,
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    };
    if !name_matches {
        return mismatch();
    }

    None
}

/// Formats an operation's identity as a human-readable string for error
/// messages (e.g. `"Step/Step named \"fetch-name\""` or `"Wait/Wait"`).
fn format_op_identity(op_type: &str, sub_type: Option<&str>, name: Option<&str>) -> String {
    let mut s = String::with_capacity(64);
    s.push_str(op_type);
    if let Some(sub) = sub_type {
        s.push('/');
        s.push_str(sub);
    }
    if let Some(n) = name {
        s.push_str(" named \"");
        s.push_str(n);
        s.push('"');
    }
    s
}

#[cfg(test)]
#[allow(clippy::expect_used)] // reason: test assertions — panics are acceptable
#[allow(clippy::unwrap_used)] // reason: test assertions
mod tests {
    use super::*;
    use crate::client::{
        GetStateOutput, InMemoryExecutionClient, TestResponse, operations_to_checkpoint_log,
    };
    use crate::engine::{CheckpointLog, CheckpointStatus};
    use aws_sdk_lambda::types::Operation;
    use std::sync::Arc;

    /// Helper: builds a Step operation.
    #[allow(clippy::unwrap_used)]
    fn make_step_op(id: &str, result: &str) -> Operation {
        Operation::builder()
            .id(id)
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .step_details(
                aws_sdk_lambda::types::StepDetails::builder()
                    .result(result)
                    .build(),
            )
            .build()
            .unwrap()
    }

    /// Tests that `checkpoint_updates` paginates when the checkpoint response
    /// has a `next_marker` — calling `get_state` to fetch all remaining
    /// operations and merging them into the checkpoint log.
    #[tokio::test]
    async fn checkpoint_updates_paginates_on_marker() {
        // The full state has 3 operations (what get_state returns).
        let all_ops = vec![
            make_step_op("step-1", "\"r1\""),
            make_step_op("step-2", "\"r2\""),
            make_step_op("step-3", "\"r3\""),
        ];
        let client = Arc::new(InMemoryExecutionClient::new(all_ops));

        // Enqueue a paginated checkpoint response: returns only step-1
        // but signals more pages via next_marker.
        let page1_ops = vec![make_step_op("step-1", "\"r1\"")];
        client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
            page1_ops,
            "page-2-token".to_owned(),
        ));

        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::clone(&log),
            client.clone(),
            "initial-token".to_owned(),
        );

        // Perform a checkpoint with no updates (we just want to exercise pagination).
        let result = ctx.checkpoint_updates(Vec::new()).await;
        assert!(result.is_ok());

        // The checkpoint log should now contain ALL operations from get_state.
        assert!(log.get("step-1").is_some(), "step-1 must be in the log");
        assert!(
            log.get("step-2").is_some(),
            "step-2 must be in the log (paginated)"
        );
        assert!(
            log.get("step-3").is_some(),
            "step-3 must be in the log (paginated)"
        );

        // get_state should have been called once for pagination.
        #[allow(clippy::unwrap_used)]
        let get_state_count = *client.get_state_call_count.lock().unwrap();
        assert_eq!(
            get_state_count, 1,
            "get_state must be called for pagination"
        );
    }

    /// Mock client for the concurrent pagination/token race test: every
    /// `checkpoint` validates and rotates the token, always returns a
    /// pagination marker, and `get_state` records whether the token it was
    /// handed is still the client's CURRENT token.
    #[derive(Debug)]
    struct PaginatedTokenValidatingClient {
        current_token: std::sync::Mutex<String>,
        counter: std::sync::atomic::AtomicU32,
        stale_checkpoint_tokens: std::sync::atomic::AtomicU32,
        stale_get_state_tokens: std::sync::atomic::AtomicU32,
        get_state_calls: std::sync::atomic::AtomicU32,
    }

    impl PaginatedTokenValidatingClient {
        fn new(initial_token: &str) -> Self {
            Self {
                current_token: std::sync::Mutex::new(initial_token.to_owned()),
                counter: std::sync::atomic::AtomicU32::new(0),
                stale_checkpoint_tokens: std::sync::atomic::AtomicU32::new(0),
                stale_get_state_tokens: std::sync::atomic::AtomicU32::new(0),
                get_state_calls: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    impl ExecutionClient for PaginatedTokenValidatingClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
            _updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>,
        > {
            let submitted = checkpoint_token.to_owned();
            Box::pin(async move {
                // Widen the race window like the real network call would.
                tokio::task::yield_now().await;
                let mut current = self
                    .current_token
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *current != submitted {
                    self.stale_checkpoint_tokens
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    return Err(ClientError::non_retryable(format!(
                        "stale checkpoint token: expected {current}, got {submitted}"
                    )));
                }
                let n = self
                    .counter
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let next = format!("token-{}", n + 1);
                current.clone_from(&next);
                drop(current);
                Ok(CheckpointOutput {
                    checkpoint_token: next,
                    updated_operations: Vec::new(),
                    // Every response paginated — forces the marker-triggered
                    // get_state on every checkpoint.
                    next_marker: Some("more-pages".to_owned()),
                })
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
        {
            let submitted = checkpoint_token.to_owned();
            Box::pin(async move {
                self.get_state_calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Yield so a concurrent branch would have every chance to
                // checkpoint (and rotate the token) if the caller released
                // the token lock too early.
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                let current = self
                    .current_token
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if *current != submitted {
                    self.stale_get_state_tokens
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                drop(current);
                Ok(GetStateOutput {
                    operations: Vec::new(),
                })
            })
        }
    }

    /// Regression test (issue #5): concurrent checkpoints whose responses
    /// carry a pagination marker must perform the marker-triggered
    /// `get_state` while STILL holding the checkpoint-token lock. If the
    /// lock is dropped first, a concurrent branch checkpoints, consumes and
    /// rotates the token, and the paginated `get_state` runs with a stale
    /// token. The mock's `get_state` validates the token it receives
    /// against the client's current token.
    #[tokio::test]
    async fn concurrent_paginated_checkpoints_get_state_uses_current_token() {
        let client = Arc::new(PaginatedTokenValidatingClient::new("token-0"));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            "token-0".to_owned(),
        );

        // Race several concurrent checkpoint_updates calls, each of which
        // paginates. Every one must succeed (no stale checkpoint token) and
        // every get_state must observe the then-current token.
        let (r1, r2, r3, r4) = tokio::join!(
            ctx.checkpoint_updates(Vec::new()),
            ctx.checkpoint_updates(Vec::new()),
            ctx.checkpoint_updates(Vec::new()),
            ctx.checkpoint_updates(Vec::new()),
        );
        assert!(r1.is_ok(), "checkpoint 1 failed: {r1:?}");
        assert!(r2.is_ok(), "checkpoint 2 failed: {r2:?}");
        assert!(r3.is_ok(), "checkpoint 3 failed: {r3:?}");
        assert!(r4.is_ok(), "checkpoint 4 failed: {r4:?}");

        assert_eq!(
            client
                .get_state_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            4,
            "every paginated checkpoint must trigger a get_state"
        );
        assert_eq!(
            client
                .stale_checkpoint_tokens
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "no checkpoint may run with a stale token"
        );
        assert_eq!(
            client
                .stale_get_state_tokens
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the marker-triggered get_state must run with the current token \
             (token lock held through the paginated fetch)"
        );
    }

    /// Tests that `checkpoint_updates` does NOT call `get_state` when there
    /// is no pagination marker.
    #[tokio::test]
    async fn checkpoint_updates_no_pagination_when_no_marker() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        // Default response has no marker.

        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "initial-token".to_owned(),
        );

        let result = ctx.checkpoint_updates(Vec::new()).await;
        assert!(result.is_ok());

        #[allow(clippy::unwrap_used)]
        let get_state_count = *client.get_state_call_count.lock().unwrap();
        assert_eq!(
            get_state_count, 0,
            "get_state must NOT be called without a marker"
        );
    }

    /// Tests that the bootstrap pagination path works: when the initial
    /// state has a `NextMarker`, `get_state` is called to fetch the full
    /// operation set, and all operations appear in the checkpoint log.
    #[tokio::test]
    async fn bootstrap_pagination_fetches_all_pages() {
        // Simulate a paginated initial state: page 1 has step-1,
        // and get_state returns the full set (step-1 + step-2 + step-3).
        let all_ops = vec![
            make_step_op("step-1", "\"result-1\""),
            make_step_op("step-2", "\"result-2\""),
            make_step_op("step-3", "\"result-3\""),
        ];
        let client: Arc<dyn ExecutionClient> = Arc::new(InMemoryExecutionClient::new(all_ops));

        // The full log comes from get_state when the initial state is paginated.
        let full_state = client.get_state("arn:test", "token").await;
        assert!(full_state.is_ok());
        #[allow(clippy::unwrap_used)]
        let full_state = full_state.unwrap();

        let log = operations_to_checkpoint_log(&full_state.operations);
        assert!(log.get("step-1").is_some());
        assert!(log.get("step-2").is_some());
        assert!(log.get("step-3").is_some());

        // Verify that the results are correct.
        #[allow(clippy::unwrap_used)]
        let r2 = log.get("step-2").unwrap();
        assert_eq!(r2.result.as_deref(), Some("\"result-2\""));
        assert_eq!(r2.status, CheckpointStatus::Succeeded);
    }

    // ── Non-determinism detection tests ─────────────────────────────────

    #[test]
    fn validate_replay_identity_passes_when_types_match() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: Some("42".to_owned()),
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
            op_name: Some("my-step".to_owned()),
        };

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("my-step"));
        assert!(result.is_ok());
    }

    // ── Targeted checkpoint accessors ────────────────────────────────────

    /// Builds a context whose checkpoint log holds one record at positional
    /// ID `"1"` (keyed by its wire ID, matching the production log shape).
    fn ctx_with_record_at_1(record: CheckpointRecord) -> DurableContext {
        let wire_id = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_id, record)]));
        DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        )
    }

    #[test]
    fn checkpoint_view_validated_returns_view_on_identity_match() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: Some("42".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 2,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: true,
            callback_id: None,
            op_type: Some("Step".to_owned()),
            sub_type: Some("Step".to_owned()),
            op_name: Some("my-step".to_owned()),
        });

        let result =
            ctx.checkpoint_view_validated("1", "wire-1", "Step", Some("Step"), Some("my-step"));
        assert!(result.is_ok());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Ok above
        let view = result.unwrap();
        assert!(view.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let view = view.unwrap();
        assert_eq!(view.status, CheckpointStatus::Succeeded);
        assert_eq!(view.attempt, 2);
        assert!(view.replay_children);

        // No record at another position: Ok(None), nothing to validate.
        let live = ctx.checkpoint_view_validated("2", "wire-2", "Step", Some("Step"), None);
        assert!(matches!(live, Ok(None)));
    }

    #[test]
    fn checkpoint_view_validated_errors_on_identity_mismatch() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_name: None,
        });

        // Claimed as Wait but checkpointed as Step → the same
        // NonDeterministicExecution error `validate_replay_identity` builds.
        let result = ctx.checkpoint_view_validated("1", "wire-1", "Wait", Some("Wait"), None);
        assert!(result.is_err());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Err above
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("Step") && display.contains("Wait"),
            "expected both identities in error: {display}"
        );
    }

    /// Replay reaching a record whose status this SDK version does not
    /// recognize must fail the execution naming the raw status (issue
    /// #45) — never return the view for the operation to act on.
    #[test]
    #[allow(clippy::panic)] // reason: test assertions over non_exhaustive kinds
    fn checkpoint_view_validated_fails_execution_on_unrecognized_status() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Unknown("PAUSED".to_owned()),
            result: None,
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
            op_name: None,
        });

        assert!(
            ctx.suspension_signal().fatal_error().is_none(),
            "no fatal recorded before the replay check"
        );

        // Identity matches — the failure is the status, not the identity.
        let result = ctx.checkpoint_view_validated("1", "wire-1", "Step", Some("Step"), None);
        assert!(result.is_err());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Err above
        let err = result.unwrap_err();
        let OperationErrorKind::NonDeterministicExecution(nde) = err.kind() else {
            panic!("expected NonDeterministicExecution, got {:?}", err.kind());
        };
        let NonDeterministicExecutionErrorKind::UnrecognizedStatus(details) = nde.kind() else {
            panic!("expected UnrecognizedStatus, got {:?}", nde.kind());
        };
        assert_eq!(details.wire_id(), "wire-1");
        assert_eq!(details.status(), "PAUSED");
        let display = format!("{err:#}");
        assert!(
            display.contains("PAUSED"),
            "error must name the raw status: {display}"
        );

        // The fatal slot is written so the invocation driver fails the
        // execution even if the returned `Err` is swallowed downstream.
        let fatal = ctx
            .suspension_signal()
            .fatal_error()
            .expect("unrecognized status must record a fatal error");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        assert!(
            fatal.error_message.contains("PAUSED"),
            "fatal message must carry the raw status: {}",
            fatal.error_message
        );
    }

    /// A record with a recognized status and matching identity records no
    /// fatal and returns the view — the #45 guard fires only on `Unknown`.
    #[test]
    fn checkpoint_view_validated_known_status_records_no_fatal() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::TimedOut,
            result: None,
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
            op_name: None,
        });

        let result = ctx.checkpoint_view_validated("1", "wire-1", "Step", Some("Step"), None);
        assert!(matches!(result, Ok(Some(_))));
        assert!(ctx.suspension_signal().fatal_error().is_none());
    }

    #[test]
    fn targeted_getters_return_single_fields() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Failed,
            result: Some(r#""state""#.to_owned()),
            error_type: Some("SomeError".to_owned()),
            error_message: Some("it broke".to_owned()),
            error_data: None,
            stack_trace: None,
            attempt: 5,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: Some("cb-42".to_owned()),
            op_type: None,
            sub_type: None,
            op_name: None,
        });

        assert!(ctx.has_checkpoint_record("1"));
        assert!(!ctx.has_checkpoint_record("2"));

        assert_eq!(
            ctx.checkpoint_result_payload("1").as_deref(),
            Some(r#""state""#)
        );
        assert!(ctx.checkpoint_result_payload("2").is_none());

        assert_eq!(ctx.checkpoint_callback_id("1").as_deref(), Some("cb-42"));

        let wire = ctx.checkpoint_wire_error("1");
        let wire = wire.expect("record 1 exists");
        assert_eq!(wire.error_type(), Some("SomeError"));
        assert_eq!(wire.error_message(), Some("it broke"));
        assert!(ctx.checkpoint_wire_error("2").is_none());

        let view = ctx.checkpoint_status_view("1");
        assert!(view.is_some());
        #[allow(clippy::unwrap_used)] // reason: test assertion — verified Some above
        let view = view.unwrap();
        assert_eq!(view.status, CheckpointStatus::Failed);
        assert_eq!(view.attempt, 5);
    }

    #[test]
    fn checkpoint_terminal_replay_projects_payload_and_error_fields() {
        let ctx = ctx_with_record_at_1(CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Failed,
            result: Some(r#""summary""#.to_owned()),
            error_type: Some("BatchError".to_owned()),
            error_message: Some("batch broke".to_owned()),
            error_data: None,
            stack_trace: None,
            attempt: 3,
            invoke_result: Some("never-projected".to_owned()),
            invoke_error_type: Some("never-projected".to_owned()),
            invoke_error_message: Some("never-projected".to_owned()),
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: true,
            callback_id: Some("cb-99".to_owned()),
            op_type: Some("Context".to_owned()),
            sub_type: Some("Map".to_owned()),
            op_name: Some("batch".to_owned()),
        });

        let snapshot = ctx.checkpoint_terminal_replay("1");
        assert_eq!(
            snapshot,
            Some(TerminalReplaySnapshot {
                status: CheckpointStatus::Failed,
                replay_children: true,
                result: Some(r#""summary""#.to_owned()),
                error_message: Some("batch broke".to_owned()),
                error_type: Some("BatchError".to_owned()),
            })
        );

        // No record → no snapshot.
        assert!(ctx.checkpoint_terminal_replay("2").is_none());
    }

    #[test]
    fn validate_replay_identity_fails_on_type_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_name: None,
        };

        // Claimed as Wait but checkpointed as Step → error
        let result = ctx.validate_replay_identity(&record, "wire-1", "Wait", Some("Wait"), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("Step"),
            "expected Step in error: {display}"
        );
        assert!(
            display.contains("Wait"),
            "expected Wait in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_fails_on_subtype_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_name: None,
        };

        // Same op_type but different sub_type → error
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("WaitForCondition"), None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
    }

    #[test]
    fn validate_replay_identity_fails_on_name_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_name: Some("fetch-user".to_owned()),
        };

        // Same type but different name → error
        let result = ctx.validate_replay_identity(
            &record,
            "wire-1",
            "Step",
            Some("Step"),
            Some("fetch-order"),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let display = format!("{err:#}");
        assert!(
            display.contains("fetch-user"),
            "expected name in error: {display}"
        );
        assert!(
            display.contains("fetch-order"),
            "expected claimed name in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_skips_when_no_identity_stored() {
        // Backwards compatibility: old checkpoint data has no identity fields.
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
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
            op_type: None,
            sub_type: None,
            op_name: None,
        };

        // Even though claimed type differs, validation passes because the
        // record has no identity to compare against.
        let result = ctx.validate_replay_identity(&record, "wire-1", "Wait", Some("Wait"), None);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_replay_identity_case_insensitive_type_match() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_type: Some("STEP".to_owned()),
            sub_type: Some("STEP".to_owned()),
            op_name: None,
        };

        // Wire format uses UPPER_CASE — validation is case-insensitive.
        let result = ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), None);
        assert!(result.is_ok());
    }

    /// Regression test (issue #6): the inline JSON envelope path stores the
    /// raw wire value `CHAINED_INVOKE` (with underscore) while the SDK
    /// claims `ChainedInvoke` (`PascalCase`). A case-insensitive comparison
    /// alone rejects every non-paginated replayed invoke as
    /// non-deterministic; canonicalization must accept it.
    #[test]
    fn validate_replay_identity_inline_chained_invoke_replay_matches() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: Some("\"ok\"".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            // Exactly what parse_single_operation stores from the inline
            // envelope: the raw wire `Type` value.
            op_type: Some("CHAINED_INVOKE".to_owned()),
            sub_type: Some("ChainedInvoke".to_owned()),
            op_name: None,
        };

        let result = ctx.validate_replay_identity(
            &record,
            "wire-1",
            "ChainedInvoke",
            Some("ChainedInvoke"),
            None,
        );
        assert!(
            result.is_ok(),
            "inline CHAINED_INVOKE record must match claimed ChainedInvoke: {result:?}"
        );
    }

    /// Canonicalization must not weaken detection: a genuinely different
    /// type still mismatches after canonicalization.
    #[test]
    fn validate_replay_identity_canonicalized_types_still_mismatch() {
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let record = CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_type: Some("CHAINED_INVOKE".to_owned()),
            sub_type: Some("ChainedInvoke".to_owned()),
            op_name: None,
        };

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("ChainedInvoke"), None);
        assert!(
            result.is_err(),
            "Step claimed against CHAINED_INVOKE record must mismatch"
        );
    }

    /// Both spellings of every known operation type canonicalize to the
    /// same wire constant, and unknown types stay distinguishable.
    #[test]
    fn canonical_op_type_bridges_wire_and_pascal_spellings() {
        for (pascal, wire) in [
            ("Callback", "CALLBACK"),
            ("ChainedInvoke", "CHAINED_INVOKE"),
            ("Context", "CONTEXT"),
            ("Execution", "EXECUTION"),
            ("Step", "STEP"),
            ("Wait", "WAIT"),
        ] {
            assert_eq!(
                canonical_op_type(pascal),
                canonical_op_type(wire),
                "{pascal} and {wire} must canonicalize identically"
            );
        }
        assert_ne!(
            canonical_op_type("Step"),
            canonical_op_type("ChainedInvoke"),
            "distinct types must stay distinct"
        );
    }

    /// Builds a checkpoint record carrying full identity fields for the
    /// Some↔None conformance tests below.
    fn record_with_identity(
        op_type: Option<&str>,
        sub_type: Option<&str>,
        op_name: Option<&str>,
    ) -> CheckpointRecord {
        CheckpointRecord {
            id: "wire-1".to_owned(),
            status: CheckpointStatus::Succeeded,
            result: None,
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
            op_type: op_type.map(ToOwned::to_owned),
            sub_type: sub_type.map(ToOwned::to_owned),
            op_name: op_name.map(ToOwned::to_owned),
        }
    }

    fn test_ctx() -> DurableContext {
        DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
        )
    }

    #[test]
    fn validate_replay_identity_fails_when_stored_name_removed() {
        // Record was checkpointed with a name; the claim carries none.
        // Removing `.name(...)` changes replay identity and must be flagged,
        // otherwise a reordered unnamed same-type operation could consume
        // this checkpoint silently.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("my-step"));

        let result = ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), None);
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("my-step"),
            "expected stored name in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_fails_when_name_added() {
        // Record was checkpointed without a name (but with identity); the
        // claim now carries one. The None↔Some direction is a mismatch too.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), None);

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("new-name"));
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::NonDeterministicExecution(_)
        ));
        let display = format!("{err:#}");
        assert!(
            display.contains("new-name"),
            "expected claimed name in error: {display}"
        );
    }

    #[test]
    fn validate_replay_identity_fails_when_subtype_dropped() {
        // Sub-type is compared as a complete Option in both directions.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Context"), Some("Map"), None);

        let result = ctx.validate_replay_identity(&record, "wire-1", "Context", None, None);
        assert!(
            result.is_err(),
            "expected mismatch when stored sub-type is dropped from the claim"
        );

        // And the reverse direction: record without a sub-type, claim with one.
        let record = record_with_identity(Some("Context"), None, None);
        let result = ctx.validate_replay_identity(&record, "wire-1", "Context", Some("Map"), None);
        assert!(
            result.is_err(),
            "expected mismatch when a sub-type is added to the claim"
        );
    }

    #[test]
    fn validate_replay_identity_legacy_record_skips_name_and_subtype() {
        // A record genuinely lacking identity (no op_type) is the ONLY
        // lenient path: nothing is compared, whatever the claim carries.
        let ctx = test_ctx();
        let record = record_with_identity(None, None, None);

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("any-name"));
        assert!(result.is_ok());
    }

    #[test]
    fn validate_replay_identity_empty_claimed_name_matches_stored_none() {
        // The checkpoint builders omit `Name` when the string is empty, so
        // the record stores `None` where the claim computes `Some("")` — a
        // map `item_namer` or parallel `Branch` with an empty name. An
        // unchanged handler must NOT be rejected on resume.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Context"), Some("MapIteration"), None);

        let result = ctx.validate_replay_identity(
            &record,
            "wire-1",
            "Context",
            Some("MapIteration"),
            Some(""),
        );
        assert!(
            result.is_ok(),
            "claimed empty name must match a stored None: {result:?}"
        );
    }

    #[test]
    fn validate_replay_identity_empty_stored_name_matches_claimed_none() {
        // The reverse direction: a backend that stored an empty name string
        // must match a claim carrying no name.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some(""));

        let result = ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), None);
        assert!(
            result.is_ok(),
            "stored empty name must match a claimed None: {result:?}"
        );
    }

    #[test]
    fn validate_replay_identity_empty_name_still_mismatches_real_name() {
        // Normalization must not weaken detection: an empty claim against a
        // real stored name (and the reverse) is still a mismatch.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("real"));
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some(""));
        assert!(
            result.is_err(),
            "empty claim vs stored 'real' must mismatch"
        );

        let record = record_with_identity(Some("Step"), Some("Step"), None);
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("real"));
        assert!(
            result.is_err(),
            "claimed 'real' vs stored None must mismatch"
        );
    }

    #[test]
    fn validate_replay_identity_mismatch_records_fatal_error() {
        // A mismatch must record the execution-fatal error on the shared
        // suspension-signal slot so the invocation driver fails the execution
        // even when the returned `Err` is swallowed downstream.
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("alpha"));

        assert!(
            ctx.suspension_signal().fatal_error().is_none(),
            "no fatal recorded before validation"
        );
        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("beta"));
        assert!(result.is_err());

        let fatal = ctx
            .suspension_signal()
            .fatal_error()
            .expect("mismatch must record a fatal error");
        assert_eq!(fatal.error_type, "NonDeterministicExecutionError");
        assert!(
            fatal.error_message.contains("alpha") && fatal.error_message.contains("beta"),
            "fatal message must name both identities: {}",
            fatal.error_message
        );
    }

    #[test]
    fn validate_replay_identity_pass_records_no_fatal() {
        let ctx = test_ctx();
        let record = record_with_identity(Some("Step"), Some("Step"), Some("same"));

        let result =
            ctx.validate_replay_identity(&record, "wire-1", "Step", Some("Step"), Some("same"));
        assert!(result.is_ok());
        assert!(
            ctx.suspension_signal().fatal_error().is_none(),
            "a passing validation must not record a fatal error"
        );
    }

    // ── Debug redaction tests ───────────────────────────────────────────

    #[test]
    fn debug_output_redacts_checkpoint_token() {
        let secret_token = "super-secret-credential-value-12345";
        let log = Arc::new(CheckpointLog::empty());
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:my-fn".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            secret_token.to_owned(),
        );

        let debug_output = format!("{ctx:?}");

        // The actual token value MUST NOT appear in debug output.
        assert!(
            !debug_output.contains(secret_token),
            "checkpoint_token value leaked in Debug output: {debug_output}"
        );

        // The redacted placeholder MUST appear.
        assert!(
            debug_output.contains("<redacted>"),
            "expected '<redacted>' in Debug output: {debug_output}"
        );

        // Useful fields MUST be present.
        assert!(
            debug_output.contains("arn:aws:lambda:us-east-1:123456789012:function:my-fn"),
            "expected execution_arn in Debug output: {debug_output}"
        );
    }

    // ── Checkpoint coalescing tests ─────────────────────────────────────

    /// Builds a root context with a checkpoint-coalescing delay, backed by
    /// the in-memory client.
    fn coalescing_ctx(client: Arc<dyn ExecutionClient>, delay: Duration) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-0".to_owned(),
            Some(delay),
        )
    }

    /// Helper: builds a bare step START update with the given wire ID.
    #[allow(clippy::expect_used)]
    fn make_update(id: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(id.to_owned())
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .action(aws_sdk_lambda::types::OperationAction::Start)
            .build()
            .expect("all required OperationUpdate fields set")
    }

    /// Two concurrent checkpoint calls inside the delay window coalesce
    /// into ONE client call carrying both updates.
    #[tokio::test(start_paused = true)]
    async fn coalesced_checkpoints_batch_into_one_call() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = coalescing_ctx(client.clone(), Duration::from_millis(50));

        let (a, b) = tokio::join!(
            ctx.checkpoint_updates(vec![make_update("op-a")]),
            ctx.checkpoint_updates(vec![make_update("op-b")]),
        );
        assert!(a.is_ok(), "first coalesced caller succeeds");
        assert!(b.is_ok(), "second coalesced caller succeeds");

        #[allow(clippy::unwrap_used)]
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 1, "both updates must share one API call");

        let ids: Vec<String> = client
            .recorded_updates()
            .iter()
            .map(|u| u.id.clone())
            .collect();
        assert_eq!(
            ids,
            vec!["op-a".to_owned(), "op-b".to_owned()],
            "the single call carries both updates in join order"
        );
    }

    /// An urgent checkpoint (the callback-creation flush point) does not
    /// wait out the delay window: it flushes immediately, carrying any
    /// previously buffered updates with it.
    #[tokio::test(start_paused = true)]
    async fn urgent_checkpoint_flushes_without_waiting_the_window() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        // A window long enough that waiting it out would be visible in
        // paused time.
        let ctx = coalescing_ctx(client.clone(), Duration::from_hours(1));

        let start = tokio::time::Instant::now();
        ctx.checkpoint_updates_urgent(vec![make_update("op-urgent")])
            .await
            .expect("urgent checkpoint succeeds");

        assert_eq!(
            start.elapsed(),
            Duration::ZERO,
            "urgent flush must not wait for the coalescing window"
        );
        #[allow(clippy::unwrap_used)]
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 1);
    }

    /// `flush_pending_checkpoints` (the suspension / end-of-invocation
    /// flush point) drains a buffered batch long before its window
    /// elapses, and the buffered caller observes the flushed result.
    #[tokio::test(start_paused = true)]
    async fn flush_pending_checkpoints_drains_the_buffer() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = coalescing_ctx(client.clone(), Duration::from_hours(1));

        let start = tokio::time::Instant::now();
        let buffered = tokio::spawn({
            let ctx = ctx.clone();
            async move {
                ctx.checkpoint_updates(vec![make_update("op-buffered")])
                    .await
            }
        });
        // Let the spawned caller join the batch (well under the window).
        tokio::time::sleep(Duration::from_millis(1)).await;

        ctx.flush_pending_checkpoints()
            .await
            .expect("flush succeeds");

        let result = buffered.await.expect("buffered caller task completes");
        assert!(result.is_ok(), "buffered caller observes the flush result");
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "flush must not wait for the one-hour window"
        );
        #[allow(clippy::unwrap_used)]
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 1);
    }

    /// Without a configured delay, `checkpoint_updates` writes immediately
    /// — one call per checkpoint, exactly the pre-knob behavior.
    #[tokio::test(start_paused = true)]
    async fn no_delay_checkpoints_write_immediately() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client.clone(),
            "token-0".to_owned(),
        );

        let start = tokio::time::Instant::now();
        ctx.checkpoint_updates(vec![make_update("op-1")])
            .await
            .expect("first write succeeds");
        ctx.checkpoint_updates(vec![make_update("op-2")])
            .await
            .expect("second write succeeds");

        assert_eq!(start.elapsed(), Duration::ZERO);
        #[allow(clippy::unwrap_used)]
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(call_count, 2, "no coalescing without a configured delay");
    }

    /// A test client whose `checkpoint` calls block until released,
    /// delegating to an [`InMemoryExecutionClient`] once the gate opens.
    /// Lets tests hold a batched write "in flight" deterministically.
    #[derive(Debug)]
    struct GatedClient {
        inner: InMemoryExecutionClient,
        gate: tokio::sync::Semaphore,
        /// Snapshot of update-ID lists, one entry per checkpoint call, in
        /// call order.
        calls: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl GatedClient {
        fn new() -> Self {
            Self {
                inner: InMemoryExecutionClient::new(Vec::new()),
                gate: tokio::sync::Semaphore::new(0),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Allows one blocked (or future) checkpoint call to proceed.
        fn release_one(&self) {
            self.gate.add_permits(1);
        }

        fn call_ids(&self) -> Vec<Vec<String>> {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl ExecutionClient for GatedClient {
        fn checkpoint(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>,
        > {
            let arn = execution_arn.to_owned();
            let token = checkpoint_token.to_owned();
            Box::pin(async move {
                // Block until the test releases the gate. The permit is
                // consumed (forgotten) so each call needs its own release.
                let permit = self
                    .gate
                    .acquire()
                    .await
                    .map_err(|e| ClientError::non_retryable(format!("gate closed: {e}")))?;
                permit.forget();
                self.calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(updates.iter().map(|u| u.id.clone()).collect());
                self.inner.checkpoint(&arn, &token, updates).await
            })
        }

        fn get_state(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
        {
            self.inner.get_state(execution_arn, checkpoint_token)
        }
    }

    /// REGRESSION (flush contract): once a delay timer claims a batch and
    /// its write is in flight, `flush_pending_checkpoints` must NOT return
    /// until that write finishes — the wrapper reports
    /// PENDING/SUCCEEDED/FAILED right after the flush, and an unawaited
    /// in-flight checkpoint would cross that boundary.
    #[tokio::test(start_paused = true)]
    async fn flush_waits_for_in_flight_timer_claimed_batch() {
        let client = Arc::new(GatedClient::new());
        let ctx = coalescing_ctx(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Duration::from_millis(10),
        );

        // A buffered contributor: its delay timer fires at +10ms, claiming
        // the batch and starting a write that blocks on the gate.
        let contributor = tokio::spawn({
            let ctx = ctx.clone();
            async move {
                ctx.checkpoint_updates(vec![make_update("op-in-flight")])
                    .await
            }
        });
        // Advance past the window so the timer claims the batch and the
        // write task blocks inside the gated client.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // The end-of-invocation flush must now WAIT for that in-flight
        // write. Prove it stays pending while the gate is closed: in
        // paused time, the timeout can only fire because every task is
        // blocked on the (non-timer) gate.
        let mut flush = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.flush_pending_checkpoints().await }
        });
        let raced = tokio::time::timeout(Duration::from_mins(1), &mut flush).await;
        assert!(
            raced.is_err(),
            "flush returned while the timer-claimed batch write was still in flight — \
             the flush barrier is broken"
        );

        // Release the write; now both the contributor and the flush finish.
        client.release_one();
        flush
            .await
            .expect("flush task completes")
            .expect("flush succeeds");
        contributor
            .await
            .expect("contributor task completes")
            .expect("contributor observes the published write");
        assert_eq!(
            client.call_ids(),
            vec![vec!["op-in-flight".to_owned()]],
            "exactly the claimed batch was written, once"
        );
    }

    /// A test client that FAILS every checkpoint call retryably, holding
    /// the FIRST call at a gate so the test can order a second batch's
    /// queuing against the in-flight failing write. Calls after the first
    /// fail immediately (no gate), so a regression that performs an extra
    /// write shows up as an extra recorded call instead of a hang.
    #[derive(Debug)]
    struct GatedRetryableFailClient {
        gate: tokio::sync::Semaphore,
        first: std::sync::atomic::AtomicBool,
        /// Snapshot of update-ID lists, one entry per checkpoint call, in
        /// call order.
        calls: std::sync::Mutex<Vec<Vec<String>>>,
    }

    impl GatedRetryableFailClient {
        fn new() -> Self {
            Self {
                gate: tokio::sync::Semaphore::new(0),
                first: std::sync::atomic::AtomicBool::new(true),
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        /// Releases the gated first call so its rejection lands.
        fn release_first(&self) {
            self.gate.add_permits(1);
        }

        fn call_ids(&self) -> Vec<Vec<String>> {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }

        fn call_count(&self) -> usize {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len()
        }
    }

    impl ExecutionClient for GatedRetryableFailClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>,
        > {
            Box::pin(async move {
                self.calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(updates.iter().map(|u| u.id.clone()).collect());
                if self.first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    // Hold the first call in flight until the test
                    // releases it; the permit is consumed so a second
                    // call never blocks here.
                    if let Ok(permit) = self.gate.acquire().await {
                        permit.forget();
                    }
                }
                Err(ClientError::from_retryable(
                    "injected retryable checkpoint failure".to_owned(),
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
        {
            Box::pin(async {
                Err(ClientError::new_non_retryable(
                    "get_state not used in this test",
                ))
            })
        }
    }

    /// REGRESSION (issue #43 review): a flusher QUEUED behind a failing
    /// batch write must not perform another checkpoint call. Batch A's
    /// write is in flight (held at the client's gate) when batch B opens
    /// and queues its flusher on the writer lock; A then fails retryably.
    /// "Retryable exhaustion fails the invocation with no further
    /// writes": B's flusher must observe the failure latch under the
    /// writer lock, publish the prior error to its contributors, and
    /// retain its updates as unwritten — without calling the backend.
    /// Before the latch, B's flusher acquired the lock after A's failure
    /// and performed another write, persisting replay-visible transitions
    /// after the invocation was already doomed.
    #[tokio::test(start_paused = true)]
    async fn queued_batch_flush_after_retryable_failure_writes_nothing() {
        let client = Arc::new(GatedRetryableFailClient::new());
        let ctx = coalescing_ctx(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Duration::ZERO,
        );

        // Batch A: its zero-delay flusher claims the batch under the
        // writer lock and starts the write, which is held at the gate.
        let contributor_a = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-a")]).await }
        });
        while client.call_count() < 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // Batch B opens while A's write is in flight; its flusher queues
        // on the writer lock behind A.
        let contributor_b = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-b")]).await }
        });
        // Let B join the buffer and its flusher reach the writer lock.
        tokio::time::sleep(Duration::from_millis(5)).await;

        // A's held write now fails retryably; B's queued flusher runs
        // right after, under the same writer lock.
        client.release_first();

        let a = contributor_a.await.expect("contributor A task completes");
        let b = contributor_b.await.expect("contributor B task completes");
        let a_err = a.expect_err("batch A observes its own write failure");
        let b_err = b.expect_err("batch B observes the latched prior failure");
        assert!(a_err.is_retryable(), "A's injected failure is retryable");
        assert!(
            b_err.is_retryable(),
            "the latch republishes the prior retryable error, so B's \
             contributors route through the same fail-the-invocation path"
        );

        assert_eq!(
            client.call_ids(),
            vec![vec!["op-a".to_owned()]],
            "batch B must cause NO checkpoint call after batch A's \
             retryable failure — the channel is down and the invocation \
             is already doomed"
        );

        // Both failures are retained for the flush-point classification:
        // A's from the failed write, B's from the latch hit, each with
        // its own unwritten updates.
        let retained = ctx
            .take_retained_flush_failures()
            .await
            .expect("both failures are retained");
        let unwritten: Vec<Vec<String>> = retained
            .failures
            .iter()
            .map(|f| f.unwritten.iter().map(|u| u.id.clone()).collect())
            .collect();
        assert_eq!(
            unwritten,
            vec![vec!["op-a".to_owned()], vec!["op-b".to_owned()]],
            "each batch's updates are retained as unwritten, in order"
        );
    }

    /// A fan-out exceeding one request's operation-count limit is split
    /// into multiple ordered requests rather than one oversized call.
    #[tokio::test(start_paused = true)]
    async fn coalesced_fanout_splits_into_size_capped_requests() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let limits = BatchLimits {
            max_operations: 2,
            max_payload_bytes: usize::MAX,
        };
        let ctx = DurableContext::new_root_with_client_and_coalescer(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client.clone(),
            "token-0".to_owned(),
            CheckpointCoalescer::with_limits(Duration::from_millis(50), limits),
        );

        // Five contributors join one coalescing window.
        let results = tokio::join!(
            ctx.checkpoint_updates(vec![make_update("op-1")]),
            ctx.checkpoint_updates(vec![make_update("op-2")]),
            ctx.checkpoint_updates(vec![make_update("op-3")]),
            ctx.checkpoint_updates(vec![make_update("op-4")]),
            ctx.checkpoint_updates(vec![make_update("op-5")]),
        );
        for outcome in [results.0, results.1, results.2, results.3, results.4] {
            assert!(outcome.is_ok(), "every contributor observes success");
        }

        #[allow(clippy::unwrap_used)]
        let call_count = *client.checkpoint_call_count.lock().unwrap();
        assert_eq!(
            call_count, 3,
            "five updates under a two-op cap must go out as 2+2+1 requests"
        );
        let ids: Vec<String> = client
            .recorded_updates()
            .iter()
            .map(|u| u.id.clone())
            .collect();
        assert_eq!(
            ids,
            vec!["op-1", "op-2", "op-3", "op-4", "op-5"],
            "join order is preserved across the split requests"
        );
    }

    /// Pure batching mode (zero window): checkpoints arriving while an
    /// earlier write is in flight batch into ONE follow-up call, and no
    /// artificial delay is added anywhere.
    #[tokio::test(start_paused = true)]
    async fn batching_mode_batches_writes_behind_in_flight_call() {
        let client = Arc::new(GatedClient::new());
        let ctx = coalescing_ctx(
            Arc::clone(&client) as Arc<dyn ExecutionClient>,
            Duration::ZERO,
        );

        // First checkpoint drives immediately and blocks in the client.
        let first = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-first")]).await }
        });
        tokio::task::yield_now().await;

        // Two more checkpoints arrive while the first write is in flight:
        // they join the open batch behind the writer lock.
        let second = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-second")]).await }
        });
        let third = tokio::spawn({
            let ctx = ctx.clone();
            async move { ctx.checkpoint_updates(vec![make_update("op-third")]).await }
        });
        // Let both join and block on the writer lock / gate.
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Release both calls (first write, then the batched follow-up).
        client.release_one();
        client.release_one();

        for task in [first, second, third] {
            task.await
                .expect("contributor task completes")
                .expect("contributor observes success");
        }

        assert_eq!(
            client.call_ids(),
            vec![
                vec!["op-first".to_owned()],
                vec!["op-second".to_owned(), "op-third".to_owned()],
            ],
            "the two checkpoints that arrived during the in-flight write \
             must batch into one follow-up call"
        );
    }

    // ── Lifecycle-event ownership regression tests ──────────────────────
    //
    // The documented contract (`crate::observability`) guarantees that
    // every transition the service records emits its lifecycle event.
    // These tests pin the two ways the old contributor-owned emission
    // violated that: a contributor dropped after joining a buffered batch,
    // and a split batch whose earlier chunk persists before a later chunk
    // fails.

    /// A `MakeWriter` that captures subscriber output in a shared buffer.
    #[derive(Clone)]
    struct EventCaptureWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for EventCaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if let Ok(mut inner) = self.0.lock() {
                inner.extend_from_slice(buf);
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for EventCaptureWriter {
        type Writer = EventCaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Installs a JSON capture subscriber (all levels, flattened fields,
    /// span list included) and returns the shared buffer plus the guard.
    fn lifecycle_capture() -> (
        Arc<std::sync::Mutex<Vec<u8>>>,
        tracing::subscriber::DefaultGuard,
    ) {
        use tracing_subscriber::layer::SubscriberExt as _;
        let buffer = Arc::new(std::sync::Mutex::new(Vec::new()));
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(true)
            .with_writer(EventCaptureWriter(Arc::clone(&buffer)));
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let guard = tracing::subscriber::set_default(subscriber);
        (buffer, guard)
    }

    /// Parses the captured output into the JSON lifecycle-event lines with
    /// the given event name, returning each line's parsed JSON.
    fn captured_lifecycle_events(
        buffer: &Arc<std::sync::Mutex<Vec<u8>>>,
        event_name: &str,
    ) -> Vec<serde_json::Value> {
        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );
        output
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|v| {
                v.get("message").and_then(serde_json::Value::as_str) == Some(event_name)
                    && v.get("target").and_then(serde_json::Value::as_str)
                        == Some(crate::observability::TARGET)
            })
            .collect()
    }

    /// REGRESSION (event ownership, dropped contributor): a buffered
    /// contributor dropped after joining a batch — explicitly supported for
    /// `race`/`select_ok` losers — must not take its lifecycle events with
    /// it. The flush task persists the update, so the write path must emit
    /// the event, inside the span the contributor captured.
    ///
    /// The scenario retries a few times because `tracing`'s global
    /// max-level hint and callsite-interest caches are rebuilt whenever any
    /// concurrently running test registers or drops a subscriber, and a
    /// rebuild racing this test's `DEBUG`-level emission can drop the event
    /// before it reaches this test's (thread-local) subscriber. The
    /// persistence invariant is asserted on every attempt; a genuine
    /// emission regression fails all attempts.
    #[tokio::test(start_paused = true)]
    async fn dropped_buffered_contributor_still_emits_events_for_persisted_updates() {
        use crate::observability::event_names;

        let mut emitted = false;
        for _attempt in 0..5 {
            let (buffer, _guard) = lifecycle_capture();
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let ctx = coalescing_ctx(client.clone(), Duration::from_hours(1));

            // The contributor polls once inside its own span — far enough
            // to capture its events and join the coalescing batch — then is
            // dropped, exactly like a lost `race` branch.
            {
                let span = tracing::info_span!("contributor-span");
                let _entered = span.enter();
                let mut fut = Box::pin(ctx.checkpoint_updates(vec![make_update("op-dropped")]));
                let mut poll_cx = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(
                    Future::poll(fut.as_mut(), &mut poll_cx).is_pending(),
                    "the buffered contributor must be awaiting the batch"
                );
            } // `fut` dropped here, before any flush.

            // The end-of-invocation flush persists the batch anyway.
            ctx.flush_pending_checkpoints()
                .await
                .expect("flush persists the dropped contributor's update");

            let ids: Vec<String> = client
                .recorded_updates()
                .iter()
                .map(|u| u.id.clone())
                .collect();
            assert_eq!(
                ids,
                vec!["op-dropped".to_owned()],
                "the dropped contributor's update must still be persisted"
            );

            let events = captured_lifecycle_events(&buffer, event_names::OPERATION_STARTED);
            if events.is_empty() {
                // A concurrent subscriber rebuild dropped the event before
                // it reached this test's subscriber; retry the scenario.
                continue;
            }

            let matching: Vec<&serde_json::Value> = events
                .iter()
                .filter(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-dropped")
                })
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "the persisted transition must emit exactly one operation_started \
                 event even though its contributor was dropped; got {events:?}"
            );
            let spans = format!("{:?}", matching.first().and_then(|v| v.get("spans")));
            assert!(
                spans.contains("contributor-span"),
                "the event must carry the originating contributor's span context, \
                 got spans: {spans}"
            );
            emitted = true;
            break;
        }
        assert!(
            emitted,
            "no attempt emitted operation_started for the dropped contributor's \
             persisted update — telemetry for a persisted transition was lost"
        );
    }

    /// A client whose second (and any later) `checkpoint` call fails,
    /// delegating the first to an [`InMemoryExecutionClient`]. Lets tests
    /// persist an early batch chunk and then fail a later one.
    #[derive(Debug)]
    struct FailSecondCallClient {
        inner: InMemoryExecutionClient,
        calls: std::sync::Mutex<u32>,
    }

    impl FailSecondCallClient {
        fn new() -> Self {
            Self {
                inner: InMemoryExecutionClient::new(Vec::new()),
                calls: std::sync::Mutex::new(0),
            }
        }
    }

    impl ExecutionClient for FailSecondCallClient {
        fn checkpoint(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>,
        > {
            let arn = execution_arn.to_owned();
            let token = checkpoint_token.to_owned();
            Box::pin(async move {
                let call_index = {
                    let mut calls = self
                        .calls
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *calls += 1;
                    *calls
                };
                if call_index >= 2 {
                    return Err(ClientError::new_non_retryable(
                        "injected failure on the second chunk",
                    ));
                }
                self.inner.checkpoint(&arn, &token, updates).await
            })
        }

        fn get_state(
            &self,
            execution_arn: &str,
            checkpoint_token: &str,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
        {
            self.inner.get_state(execution_arn, checkpoint_token)
        }
    }

    /// REGRESSION (event ownership, split batch): when a sealed batch splits
    /// into several requests and an earlier request persists before a later
    /// one fails, the persisted chunk's events must be emitted — the
    /// aggregate batch error published to contributors must not suppress
    /// telemetry for transitions the service actually recorded.
    ///
    /// Retries for the same reason as
    /// [`dropped_buffered_contributor_still_emits_events_for_persisted_updates`]:
    /// concurrent tests rebuilding `tracing`'s global caches can race a
    /// `DEBUG` emission. Hard invariants assert on every attempt.
    #[tokio::test(start_paused = true)]
    async fn split_batch_emits_events_for_chunks_persisted_before_a_failure() {
        use crate::observability::event_names;

        let mut emitted = false;
        for _attempt in 0..5 {
            let (buffer, _guard) = lifecycle_capture();
            let client = Arc::new(FailSecondCallClient::new());
            let limits = BatchLimits {
                max_operations: 1,
                max_payload_bytes: usize::MAX,
            };
            let ctx = DurableContext::new_root_with_client_and_coalescer(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::empty()),
                Arc::clone(&client) as Arc<dyn ExecutionClient>,
                "token-0".to_owned(),
                CheckpointCoalescer::with_limits(Duration::from_millis(50), limits),
            );

            // Two contributors share one batch; the one-op cap splits it
            // into two requests. The first persists, the second fails.
            let (first, second) = tokio::join!(
                ctx.checkpoint_updates(vec![make_update("op-persisted")]),
                ctx.checkpoint_updates(vec![make_update("op-rejected")]),
            );
            assert!(
                first.is_err() && second.is_err(),
                "every contributor observes the aggregate batch error"
            );

            let ids: Vec<String> = client
                .inner
                .recorded_updates()
                .iter()
                .map(|u| u.id.clone())
                .collect();
            assert_eq!(
                ids,
                vec!["op-persisted".to_owned()],
                "exactly the first chunk was persisted"
            );

            let events = captured_lifecycle_events(&buffer, event_names::OPERATION_STARTED);
            if events.is_empty() {
                // A concurrent subscriber rebuild dropped the event before
                // it reached this test's subscriber; retry the scenario.
                continue;
            }

            let persisted: Vec<&serde_json::Value> = events
                .iter()
                .filter(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-persisted")
                })
                .collect();
            assert_eq!(
                persisted.len(),
                1,
                "the persisted chunk must emit its operation_started event \
                 despite the later chunk's failure; got {events:?}"
            );
            assert!(
                !events.iter().any(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-rejected")
                }),
                "the rejected chunk recorded nothing, so it must emit nothing; \
                 got {events:?}"
            );
            emitted = true;
            break;
        }
        assert!(
            emitted,
            "no attempt emitted operation_started for the persisted chunk — \
             telemetry for a persisted transition was lost"
        );
    }

    /// REGRESSION (event ownership, failed pagination hydration): the
    /// checkpoint call can persist the transitions and *then* fail while
    /// hydrating paginated state through `get_state`. The persisted
    /// transitions' events must already be emitted by then — emission
    /// happens immediately after the service accepts the write, before the
    /// fallible pagination fetch — even though the caller observes an error.
    ///
    /// Retries for the same reason as
    /// [`dropped_buffered_contributor_still_emits_events_for_persisted_updates`]:
    /// concurrent tests rebuilding `tracing`'s global caches can race a
    /// `DEBUG` emission. Hard invariants assert on every attempt.
    #[tokio::test]
    async fn persisted_checkpoint_emits_events_when_pagination_hydration_fails() {
        use crate::observability::event_names;

        let mut emitted = false;
        for _attempt in 0..5 {
            let (buffer, _guard) = lifecycle_capture();
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            // The write itself succeeds but signals more pages, and the
            // follow-up get_state fetch fails.
            client.enqueue_checkpoint_response(TestResponse::SuccessPaginated(
                Vec::new(),
                "page-2-token".to_owned(),
            ));
            client.fail_get_state("injected get_state failure");

            let ctx = DurableContext::new_root_with_client(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                Arc::new(CheckpointLog::empty()),
                Arc::clone(&client) as Arc<dyn ExecutionClient>,
                "token-0".to_owned(),
            );

            let result = ctx
                .checkpoint_updates(vec![make_update("op-hydration")])
                .await;
            assert!(
                result.is_err(),
                "the caller must observe the pagination-hydration failure"
            );
            let ids: Vec<String> = client
                .recorded_updates()
                .iter()
                .map(|u| u.id.clone())
                .collect();
            assert_eq!(
                ids,
                vec!["op-hydration".to_owned()],
                "the transition was persisted before get_state failed"
            );

            let events = captured_lifecycle_events(&buffer, event_names::OPERATION_STARTED);
            if events.is_empty() {
                // A concurrent subscriber rebuild dropped the event before
                // it reached this test's subscriber; retry the scenario.
                continue;
            }

            let matching: Vec<&serde_json::Value> = events
                .iter()
                .filter(|v| {
                    v.get(crate::tracing_layer::fields::OPERATION_ID)
                        .and_then(serde_json::Value::as_str)
                        == Some("op-hydration")
                })
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "the persisted transition must emit exactly one operation_started \
                 event even though pagination hydration failed; got {events:?}"
            );
            emitted = true;
            break;
        }
        assert!(
            emitted,
            "no attempt emitted operation_started for the persisted transition — \
             telemetry was suppressed by a failed pagination fetch"
        );
    }
}
