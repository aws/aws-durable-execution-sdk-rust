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
use crate::error::{
    ChildFnError, NonDeterministicExecutionError, NonDeterministicExecutionErrorKind,
    OperationError, OperationErrorKind, StepError, StepErrorKind,
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
    /// Cached parent wire ID — the SHA-256 hash of this context's prefix
    /// (positional ID of the parent operation). `None` for root contexts.
    parent_wire_id: Option<String>,
    /// Execution-wide default serdes, applied by an operation only when it
    /// sets no serdes of its own. Threaded in from [`Options`](crate::Options)
    /// by [`wrap`](crate::wrap); shared with every child context.
    default_serdes: Option<Arc<dyn Serdes>>,
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
                parent_wire_id: None,
                default_serdes: None,
                replay_span,
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
                parent_wire_id: None,
                default_serdes: None,
                replay_span,
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
                parent_wire_id: None,
                default_serdes: None,
                replay_span,
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
                parent_wire_id: None,
                default_serdes,
                replay_span,
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
                parent_wire_id: Some(crate::engine::compute_wire_id_public(parent_positional_id)),
                default_serdes: self.inner.default_serdes.clone(),
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
                parent_wire_id: Some(crate::engine::compute_wire_id_public(parent_positional_id)),
                default_serdes: self.inner.default_serdes.clone(),
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
                parent_wire_id: Some(parent_wire_id_override.to_owned()),
                default_serdes: self.inner.default_serdes.clone(),
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
                parent_wire_id: self.inner.parent_wire_id.clone(),
                default_serdes: self.inner.default_serdes.clone(),
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
        if let Some(record) = self.checkpoint_record(op_id.positional()) {
            self.validate_replay_identity(
                &record,
                op_id.wire(),
                claimed_type,
                claimed_sub_type,
                claimed_name,
            )?;
        }
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
    pub(crate) fn validate_replay_identity(
        &self,
        record: &CheckpointRecord,
        wire_id: &str,
        claimed_type: &str,
        claimed_sub_type: Option<&str>,
        claimed_name: Option<&str>,
    ) -> Result<(), OperationError> {
        // A record without a stored operation type predates identity
        // recording (legacy checkpoint) — there is genuinely nothing to
        // validate against, so skip. This is the ONLY lenient path; once a
        // record carries identity, every field is compared in full.
        let Some(expected_type) = record.op_type.as_deref() else {
            return Ok(());
        };

        let mismatch = || {
            let err = OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
                NonDeterministicExecutionError::from_kind(
                    NonDeterministicExecutionErrorKind::OperationMismatch {
                        wire_id: wire_id.to_owned(),
                        expected: format_op_identity(
                            expected_type,
                            record.sub_type.as_deref(),
                            record.op_name.as_deref(),
                        ),
                        actual: format_op_identity(claimed_type, claimed_sub_type, claimed_name),
                    },
                ),
            ));
            // A replay identity mismatch is execution-fatal: record it on the
            // shared slot so the invocation driver fails the execution with
            // the dedicated error even if this `Err` is swallowed on its way
            // up — stored as a rejected outcome by `join_all`, out-raced by a
            // sibling's success in `select_ok`, stringified through a
            // child-context boundary, or tolerated by a map/parallel
            // completion config.
            self.inner
                .suspension_signal
                .record_fatal("NonDeterministicExecutionError".to_owned(), err.to_string());
            err
        };

        // Compare canonicalized types: the checkpoint log stores PascalCase
        // on the typed SDK path but the raw wire form (e.g.
        // `CHAINED_INVOKE`) on the inline JSON envelope path, so both sides
        // canonicalize through `OperationType` before comparison.
        if canonical_op_type(expected_type) != canonical_op_type(claimed_type) {
            return Err(mismatch());
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
            return Err(mismatch());
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
            return Err(mismatch());
        }

        Ok(())
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

    /// Checkpoints operation updates via the execution client.
    ///
    /// Serializes all concurrent callers through a single critical section:
    /// the lock is held across the full read-token → API-call →
    /// rotate-token sequence, and — when the response carries a pagination
    /// marker — through the follow-up `get_state` fetch as well, so no
    /// concurrent branch can rotate the token out from under the paginated
    /// state read.
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
        // Erase input to `Value` at the call site (synchronous, before the
        // future body): the input type is erased past this point. Carrying a
        // `Value` rather than pre-rendered text means the custom serdes and the
        // default path share one conversion — no re-parsing needed.
        let erased_input = serde_json::to_value(&input).map_err(|e| e.to_string());
        InvokeBuilder::new(self.clone(), op_id, function_id.to_owned(), erased_input)
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
    use crate::client::{InMemoryExecutionClient, TestResponse, operations_to_checkpoint_log};
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
        ) -> std::pin::Pin<
            Box<
                dyn Future<Output = Result<crate::client::GetStateOutput, ClientError>> + Send + '_,
            >,
        > {
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
                Ok(crate::client::GetStateOutput {
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
        let display = err.to_string();
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
        let display = err.to_string();
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
            attempt: 0,
            invoke_result: Some("\"ok\"".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
        let display = err.to_string();
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
        let display = err.to_string();
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
}
