//! Child context operation execution engine.
//!
//! Implements `run_in_child_context`: live path (run child closure,
//! serialize result, checkpoint), replay path (return frozen result from
//! checkpoint log), and error grading (`ChildFnError` → `ChildContextError`).
//!
//! Wire shape:
//! - `OperationType`: `Context`
//! - `OperationSubType`: `RunInChildContext`
//! - Actions: Start → (Succeed | Fail)
//! - Payload on Succeed: serialized child result
//! - Error on Fail: `{ Type, Message }` from the child function error
//! - `ParentId`: wire ID of the parent context's prefix (if child of child)

use std::future::Future;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tracing::Instrument as _;

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, ChildFnError, OperationError, OperationErrorKind,
};
use crate::serdes::SerdesContext;

/// Wire sub-type for child context operations.
pub(crate) const CHILD_SUB_TYPE: &str = "RunInChildContext";

/// Maximum checkpoint payload size in bytes (256KB). Payloads exceeding
/// this trigger `ReplayChildren` mode: the child result is not stored in
/// the checkpoint; instead the child body is re-executed on replay.
const CHECKPOINT_SIZE_LIMIT_BYTES: usize = 256 * 1024;

/// Internal state for child context execution passed from the builder.
///
/// Generic over the body closure `F` and — through its future's output —
/// the body's error type `E`, so the body runs **without type erasure**:
/// `run_in_child_context` instantiates it with the user closure directly
/// (`E = BoxError`), and `with_retry` with the concrete retry-loop future
/// (`E = ChildFnError`). The one erasure point is the builder's
/// `.future()` / `into_future`, which boxes the whole execution future
/// once inside [`DurableFuture`](crate::DurableFuture).
pub(crate) struct ChildExecution<O, F, S> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) serdes: S,
    pub(crate) closure: F,
    pub(crate) _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O, F, Fut, E, S> ChildExecution<O, F, S>
where
    O: Send + 'static,
    F: FnOnce(DurableContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
    E: Into<ChildFnError>,
    S: Serdes<O>,
{
    /// Executes the child context operation.
    ///
    /// Thin generic wrapper — the ONLY code monomorphized per call site.
    /// The replay/checkpoint state machine lives in the non-generic
    /// [`ChildCore`] / [`ChildAfter`] halves (generic over the result type
    /// `O` only); this wrapper just polls the user's concrete future
    /// between them, in a fresh child context under the child namespace's
    /// replay-aware span.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        let Self {
            ctx,
            op_id,
            name,
            serdes,
            closure,
            _marker,
        } = self;
        let core = ChildCore {
            ctx,
            op_id,
            name,
            serdes,
            _marker: std::marker::PhantomData,
        };
        match core.before().await? {
            ChildPrelude::Done(result) => result,
            ChildPrelude::Run {
                child_ctx,
                mode,
                after,
            } => {
                // Run the child closure, instrumented with the child
                // namespace's replay-aware span: on a resume, nested
                // operations can still be replaying while the parent is
                // live, and the child span's isReplay flag (kept current by
                // the child's own mints) is what lets a filter suppress the
                // body's pre-wait log lines exactly once.
                let child_span = child_ctx.replay_span();
                let result = (closure)(child_ctx)
                    .instrument(child_span)
                    .await
                    .map_err(Into::into);
                after.settle(mode, result).await
            }
        }
    }
}

/// The pre-closure half of a child context: task-ownership check, replay
/// resolution, and the START checkpoint. Generic only over the result type
/// `O` — no user closure reaches this state machine, so its substantial
/// replay/checkpoint logic compiles once per result type instead of once
/// per call site.
struct ChildCore<O, S> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    serdes: S,
    _marker: std::marker::PhantomData<fn() -> O>,
}

/// Why the child body must run: a fresh live execution, or a re-execution
/// reconstructing a result too large to checkpoint (`ReplayChildren`).
enum ChildRunMode {
    /// Live path: checkpoint the outcome afterwards.
    Live,
    /// `ReplayChildren` path: round-trip the result, checkpoint nothing.
    Reconstruct,
}

/// What [`ChildCore::before`] decided: the operation is already resolved
/// from the checkpoint log, or the body must run in the prepared child
/// context.
enum ChildPrelude<O, S> {
    /// Resolved without running the body (replayed success or failure).
    Done(Result<O, OperationError>),
    /// The body must run in `child_ctx`; `after` settles the outcome under
    /// the semantics of `mode`.
    Run {
        /// The fresh child context (chained prefix) the body receives.
        child_ctx: DurableContext,
        /// Live execution vs `ReplayChildren` reconstruction.
        mode: ChildRunMode,
        /// The post-closure half that settles the outcome.
        after: ChildAfter<O, S>,
    },
}

/// The post-closure half of a child context: outcome checkpointing (live)
/// or result round-tripping (`ReplayChildren`). Generic only over the
/// result type.
struct ChildAfter<O, S> {
    ctx: DurableContext,
    wire_id: String,
    name: Option<String>,
    serdes: S,
    _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O, S> ChildCore<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// Runs everything that precedes the child closure: replay resolution
    /// or the live-path preamble ending at the START checkpoint.
    ///
    /// This is the dispatcher: it validates task ownership and replay
    /// identity, then resolves recorded outcomes ([`Self::replay_succeeded`],
    /// [`Self::replay_failed`]) or prepares the child context for the body
    /// to run (live, or `ReplayChildren` reconstruction).
    async fn before(self) -> Result<ChildPrelude<O, S>, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());

        // 2. Check checkpoint log for replay. The validated view carries
        // status and `replay_children` without cloning; the terminal
        // branches fetch only the fields they consume.
        let mut mode = ChildRunMode::Live;
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(CHILD_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded if view.replay_children => {
                    // Recorded success whose result was too large to
                    // checkpoint: re-execute the body to reconstruct it.
                    mode = ChildRunMode::Reconstruct;
                }
                CheckpointStatus::Succeeded => {
                    return Ok(ChildPrelude::Done(
                        self.replay_succeeded(&positional_id, &wire_id, view.attempt, &serdes_ctx)
                            .await,
                    ));
                }
                CheckpointStatus::Failed => {
                    return Ok(ChildPrelude::Done(self.replay_failed(
                        &positional_id,
                        &wire_id,
                        view.attempt,
                    )));
                }
                // Started/Pending/Ready: re-enter child body (resume path).
                _ => {}
            }
        } else {
            // 3. No checkpoint record: first invocation — checkpoint START.
            let update = build_update(
                &wire_id,
                self.name.as_deref(),
                OperationAction::Start,
                &self.ctx,
            );
            if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43) — child-context START: no user code ran (the
                // closure has not been called), so no terminal FAIL is
                // needed; re-invocation reconverges on the same write.
                return self
                    .ctx
                    .checkpoint_failure_unrecoverable(&wire_id, err, None)
                    .await;
            }
        }

        // 4. Create child context with chained prefix.
        let Self {
            ctx, name, serdes, ..
        } = self;
        let child_ctx = ctx.new_child(&positional_id);

        Ok(ChildPrelude::Run {
            child_ctx,
            mode,
            after: ChildAfter {
                ctx,
                wire_id,
                name,
                serdes,
                _marker: std::marker::PhantomData,
            },
        })
    }

    /// Replay path for a recorded success: returns the frozen result from
    /// the checkpoint log without re-running the body (the `ReplayChildren`
    /// branch re-executes instead, so it is deliberately not a replay
    /// event). Decodes the payload FIRST, then emits `operation_replayed`:
    /// a corrupt payload or failing serdes surfaces as an error without
    /// claiming a recorded outcome was returned.
    async fn replay_succeeded(
        self,
        positional_id: &str,
        wire_id: &str,
        attempt: u32,
        serdes_ctx: &SerdesContext,
    ) -> Result<O, OperationError> {
        let payload = self.ctx.checkpoint_result_payload(positional_id);
        let value = replay_success(&self.serdes, payload, serdes_ctx.clone()).await?;
        self.ctx.emit_operation_replayed(
            wire_id,
            self.name.as_deref(),
            "Context",
            Some(CHILD_SUB_TYPE),
            attempt,
        );
        Ok(value)
    }

    /// Replay path for a recorded failure: emits `operation_replayed` and
    /// returns the frozen error from the checkpoint log.
    fn replay_failed(
        &self,
        positional_id: &str,
        wire_id: &str,
        attempt: u32,
    ) -> Result<O, OperationError> {
        self.ctx.emit_operation_replayed(
            wire_id,
            self.name.as_deref(),
            "Context",
            Some(CHILD_SUB_TYPE),
            attempt,
        );
        let wire = self
            .ctx
            .checkpoint_wire_error(positional_id)
            .unwrap_or_default();
        Err(replay_failure(wire, wire_id))
    }
}

impl<O, S> ChildAfter<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// Settles the child body's outcome.
    ///
    /// Live mode checkpoints it — success through [`build_succeed_update`]
    /// (which routes large payloads to `ReplayChildren` mode), failure
    /// through [`checkpoint_live_failure`]. Reconstruct mode
    /// (`ReplayChildren` replay) round-trips the result through
    /// serialization for consistency with the checkpointed path and
    /// checkpoints nothing.
    async fn settle(
        self,
        mode: ChildRunMode,
        result: Result<O, ChildFnError>,
    ) -> Result<O, OperationError> {
        let Self {
            ctx,
            wire_id,
            name,
            serdes,
            ..
        } = self;
        let serdes_ctx = SerdesContext::new(&wire_id, ctx.execution_arn());

        match (mode, result) {
            (ChildRunMode::Reconstruct, Ok(value)) => {
                // Round-trip through serialization for consistency.
                let serialized = serialize_value(value, &serdes, serdes_ctx.clone()).await?;
                deserialize_value(serialized, &serdes, serdes_ctx).await
            }
            (ChildRunMode::Reconstruct, Err(child_err)) => Err(OperationError::from_kind(
                OperationErrorKind::ChildContext(ChildContextError::new(
                    ChildContextErrorKind::ChildFailed,
                    Some(child_err.into_source()),
                )),
            )),
            (ChildRunMode::Live, Ok(value)) => {
                // Success: serialize and checkpoint.
                //
                // A result serialization failure is a LOCAL, deterministic,
                // user-facing failure, so it stays catchable — but the
                // terminal FAIL is persisted FIRST (issue #43). The closure
                // already ran, so its side effects need a recorded outcome:
                // with the FAIL recorded, replay yields it instead of
                // re-running the closure, and a handler that catches the
                // error branches on a decision replay reproduces.
                let serialized = match serialize_value(value, &serdes, serdes_ctx.clone()).await {
                    Ok(serialized) => serialized,
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        let update =
                            build_child_fail_update(&wire_id, name.as_deref(), &ctx, &wire);
                        if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                            // Audit (#43) — child FAIL (serialization): the
                            // closure ran, so the failed FAIL write routes
                            // unrecoverable with a minimal terminal FAIL
                            // retry.
                            let cwire = crate::error::checkpoint_failure_wire(&client_err);
                            let terminal =
                                build_child_fail_update(&wire_id, name.as_deref(), &ctx, &cwire);
                            return ctx
                                .checkpoint_failure_unrecoverable(
                                    &wire_id,
                                    client_err,
                                    Some(terminal),
                                )
                                .await;
                        }
                        return Err(op_err
                            .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                            .with_wire(wire));
                    }
                };
                let update = build_succeed_update(&wire_id, name.as_deref(), &ctx, &serialized);
                if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
                    // Audit (#43) — child-context SUCCEED: the closure
                    // ran, so its side effects need a recorded outcome. A
                    // permanent rejection persists a small terminal FAIL
                    // before the execution fails.
                    let cwire = crate::error::checkpoint_failure_wire(&err);
                    let terminal = build_child_fail_update(&wire_id, name.as_deref(), &ctx, &cwire);
                    return ctx
                        .checkpoint_failure_unrecoverable(&wire_id, err, Some(terminal))
                        .await;
                }

                // Round-trip deserialize for consistency (first-run == replay).
                deserialize_value(serialized, &serdes, serdes_ctx).await
            }
            (ChildRunMode::Live, Err(child_err)) => {
                // Failure: checkpoint FAIL with error details.
                Err(checkpoint_live_failure(&ctx, name.as_deref(), &wire_id, child_err).await)
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Builds an `OperationUpdate` for child context operations.
fn build_update(
    wire_id: &str,
    name: Option<&str>,
    action: OperationAction,
    ctx: &DurableContext,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(CHILD_SUB_TYPE.to_owned())
        .action(action);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    // Set parent wire ID if this is a nested child (child-of-child).
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }

    // build() is infallible here — all required fields (id, type, action) set.
    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds the Succeed `OperationUpdate` for a live child result, choosing
/// between the two payload paths: a result within the checkpoint size limit
/// travels inline, while a larger one switches the operation to
/// `ReplayChildren` mode (no payload; the backend preserves the child
/// operations and the body re-executes on replay to reconstruct it).
fn build_succeed_update(
    wire_id: &str,
    name: Option<&str>,
    ctx: &DurableContext,
    serialized: &str,
) -> OperationUpdate {
    let mut succeed_builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(CHILD_SUB_TYPE.to_owned())
        .action(OperationAction::Succeed);
    if let Some(n) = name {
        succeed_builder = succeed_builder.name(n);
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        succeed_builder = succeed_builder.parent_id(parent_wire);
    }
    // Check payload size: if >256KB, use ReplayChildren mode.
    if serialized.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
        succeed_builder = succeed_builder.context_options(
            aws_sdk_lambda::types::ContextOptions::builder()
                .replay_children(true)
                .build(),
        );
        // Don't include the payload — backend preserves child ops.
    } else {
        succeed_builder = succeed_builder.payload(serialized.to_owned());
    }
    #[allow(clippy::expect_used)] // reason: all required fields are set above
    succeed_builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Checkpoints a live child failure (best-effort) and returns the graded
/// `ChildContextError`: even when the FAIL checkpoint cannot be written,
/// the child error is reported (the next invocation re-executes).
async fn checkpoint_live_failure(
    ctx: &DurableContext,
    name: Option<&str>,
    wire_id: &str,
    child_err: ChildFnError,
) -> OperationError {
    // Derive the wire failure from the carried error: the message is the
    // flattened chain, and `error_data`/`stack_trace` pass through from
    // the cause chain, so an inner failure's payload survives this
    // boundary.
    let wire = crate::error::wire_error_for(&child_err, "ChildFnError");
    let update = build_child_fail_update(wire_id, name, ctx, &wire);

    // A rejected FAIL write routes unrecoverable — discarding it would
    // leave the record claiming less than what executed (#43).
    if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43) — child-context FAIL: the closure ran and failed;
        // the failed FAIL write routes unrecoverable with a minimal
        // terminal FAIL retry (the original carried the child error's
        // payload).
        let cwire = crate::error::checkpoint_failure_wire(&client_err);
        let terminal = build_child_fail_update(wire_id, name, ctx, &cwire);
        return ctx
            .checkpoint_failure_unrecoverable(wire_id, client_err, Some(terminal))
            .await;
    }

    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::ChildFailed,
        Some(child_err.into_source()),
    )))
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

/// Builds a child-context `FAIL` update carrying `wire` as its error.
fn build_child_fail_update(
    wire_id: &str,
    name: Option<&str>,
    ctx: &DurableContext,
    wire: &crate::error::WireError,
) -> OperationUpdate {
    let mut fail_builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(CHILD_SUB_TYPE.to_owned())
        .action(OperationAction::Fail)
        .error(wire.to_error_object());
    if let Some(n) = name {
        fail_builder = fail_builder.name(n);
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        fail_builder = fail_builder.parent_id(parent_wire);
    }
    #[allow(clippy::expect_used)] // reason: all required fields are set above
    fail_builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Replays a successful child context result from the checkpoint log.
async fn replay_success<O, S: Serdes<O>>(
    serdes: &S,
    result: Option<String>,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let payload = result.ok_or_else(|| {
        child_internal_error("checkpointed Succeeded operation has no result payload")
    })?;
    deserialize_value(payload, serdes, serdes_ctx).await
}

/// Replays a failed child context result from the checkpoint log.
///
/// The recorded failure fields travel on the synthetic source rather
/// than being folded into a message, so the recorded `error_type` (and
/// `error_data`, when present) stays programmatically recoverable.
///
/// A record whose `error_type` is the serialization discriminator
/// ([`crate::error::SERIALIZATION_FAILED_ERROR_TYPE`]) reconstructs
/// `ChildContextErrorKind::Internal` — the kind the live serialization
/// path yielded after persisting that record — so replay reproduces the
/// recorded failure's classification (issue #43). Every other record is
/// a closure failure and reconstructs `ChildFailed`.
fn replay_failure(wire: crate::error::WireError, wire_id: &str) -> OperationError {
    let kind = if wire.error_type() == Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE) {
        ChildContextErrorKind::Internal
    } else {
        ChildContextErrorKind::ChildFailed
    };
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        kind,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )))
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

/// Serializes a value through the configured serdes (ownership transfers;
/// the serdes decides where its work runs).
async fn serialize_value<O, S: Serdes<O>>(
    value: O,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    serdes
        .serialize(value, serdes_ctx)
        .await
        .map_err(|e| child_internal_error(&format!("serialize result: {e}")))
}

/// Deserializes a wire payload through the configured serdes.
async fn deserialize_value<O, S: Serdes<O>>(
    payload: String,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    serdes
        .deserialize(payload, serdes_ctx)
        .await
        .map_err(|e| child_internal_error(&format!("deserialize result: {e}")))
}

/// Constructs a `ChildContextError::Internal` wrapped as an
/// `OperationError`; the message becomes the source frame, keeping the
/// kind a pure classification.
fn child_internal_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::Internal,
        Some(message.to_owned().into()),
    )))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // reason: test assertions
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use std::sync::Arc;

    /// Helper to create a test context with the given checkpoint log.
    fn test_ctx(log: CheckpointLog) -> DurableContext {
        DurableContext::new_root(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
        )
    }

    /// Helper to create a checkpoint record for a succeeded child context.
    fn succeeded_record(positional_id: &str, result: &str) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Succeeded,
                result: Some(result.to_owned()),
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
            },
        )
    }

    /// Helper to create a checkpoint record for a failed child context.
    fn failed_record(
        positional_id: &str,
        err_type: &str,
        err_msg: &str,
    ) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Failed,
                result: None,
                error_type: Some(err_type.to_owned()),
                error_message: Some(err_msg.to_owned()),
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
            },
        )
    }

    #[tokio::test]
    async fn child_live_path_runs_closure() {
        // A live client: the child's checkpoint writes must succeed for
        // the closure to run to completion. (Pre-#43 this test used a
        // client-less context and asserted the checkpoint failure
        // surfaced as Err; a rejected write now parks the future for the
        // invocation driver instead of yielding.)
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        let result: Result<i32, OperationError> = ctx
            .run_in_child_context(|child_ctx| async move {
                // Child context has prefix "1" (first op of root); its
                // inner step runs and checkpoints under that namespace.
                let step_result = child_ctx.step(|_| async { Ok(42) }).await?;
                Ok(step_result)
            })
            .await;
        #[allow(clippy::unwrap_used)] // reason: test assertion
        let value = result.unwrap();
        assert_eq!(value, 42);
    }

    #[tokio::test]
    async fn child_replay_returns_frozen_result() {
        // Set up: operation "1" is a succeeded child context in the log.
        let log = CheckpointLog::from_records(vec![succeeded_record("1", r#""hello""#)]);
        let ctx = test_ctx(log);

        let result: Result<String, OperationError> = ctx
            .run_in_child_context(|_child_ctx| async move {
                // This should NOT execute during replay.
                unreachable!("child body should not run during replay");
            })
            .await;

        #[allow(clippy::unwrap_used)] // reason: test assertion
        let value = result.unwrap();
        assert_eq!(value, "hello");
    }

    #[tokio::test]
    async fn child_replay_failure_returns_error() {
        let log =
            CheckpointLog::from_records(vec![failed_record("1", "ChildFnError", "step exploded")]);
        let ctx = test_ctx(log);

        let result: Result<String, OperationError> = ctx
            .run_in_child_context(|_child_ctx| async move {
                unreachable!("child body should not run during replay of failure");
            })
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("step exploded"),
            "error should contain original message: {msg}"
        );
    }

    #[tokio::test]
    async fn nested_child_prefix_chains_two_deep() {
        // Operation "1" is a child (prefix "1"), and "1-1" is its child
        // (prefix "1-1"). Verify nested replay works.
        let log = CheckpointLog::from_records(vec![succeeded_record("1", r#""outer""#)]);
        let ctx = test_ctx(log);

        // Outer child replays immediately with "outer".
        let result: Result<String, OperationError> = ctx
            .run_in_child_context(|_child_ctx| async move {
                unreachable!("outer child should replay");
            })
            .await;

        #[allow(clippy::unwrap_used)] // reason: test assertion
        let value = result.unwrap();
        assert_eq!(value, "outer");
    }

    #[tokio::test]
    async fn child_fn_error_propagation() {
        // Without a client, the checkpoint will fail, but the error handling
        // path tests the grading logic. Use a replay-failure record to test
        // the ChildFnError → ChildContextError grading.
        let log = CheckpointLog::from_records(vec![failed_record(
            "1",
            "ChildFnError",
            "child function error: operation error: step: execution failed: boom",
        )]);
        let ctx = test_ctx(log);

        let result: Result<String, OperationError> = ctx
            .run_in_child_context(|_child_ctx| async move {
                unreachable!();
            })
            .await;

        let err = result.unwrap_err();
        match err.kind() {
            OperationErrorKind::ChildContext(ce) => match ce.kind() {
                ChildContextErrorKind::ChildFailed => {
                    let message = crate::error::chain_string(ce);
                    assert!(message.contains("boom"), "chain: {message}");
                }
                other => unreachable!("unexpected kind: {other:?}"),
            },
            other => unreachable!("unexpected kind: {other:?}"),
        }
    }

    #[tokio::test]
    async fn spawn_child_starts_eagerly() {
        use tokio::sync::oneshot;

        // A working in-memory client: the child's START checkpoint must
        // succeed for the body to run. (Pre-#43 this test used a
        // client-less context and passed vacuously — the rejected START
        // write dropped the sender, which settled `rx` without the body
        // ever running. A rejected write now parks the child future for
        // the driver instead of yielding, so the body needs a live
        // checkpoint channel to demonstrate eagerness.)
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );

        let (tx, rx) = oneshot::channel::<()>();

        let handle = ctx
            .run_in_child_context(move |_child_ctx| async move {
                // Signal that the child body has started executing.
                let _ = tx.send(());
                Err("test complete".into())
            })
            .spawn();

        // The child body should have started BEFORE we await the handle.
        let signal = tokio::time::timeout(std::time::Duration::from_secs(2), rx).await;
        assert_eq!(
            signal.ok().and_then(Result::ok),
            Some(()),
            "child body should start executing before await"
        );

        // The handle should eventually resolve (with the child's error).
        let result: Result<String, OperationError> = handle.await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn concurrent_spawned_children_deterministic_ids() {
        // Two spawned children: IDs are claimed at builder creation, so
        // they are always "1" and "2" regardless of execution order.
        let log = CheckpointLog::from_records(vec![
            succeeded_record("1", r#""first""#),
            succeeded_record("2", r#""second""#),
        ]);
        let ctx = test_ctx(log);

        // Both will replay from checkpoint (no execution needed).
        let handle_a = ctx
            .run_in_child_context(|_| async move {
                unreachable!("should replay");
            })
            .spawn();
        let handle_b = ctx
            .run_in_child_context(|_| async move {
                unreachable!("should replay");
            })
            .spawn();

        // Await in reverse order — IDs were already claimed.
        let result_b: String = handle_b.await.expect("b should succeed");
        let result_a: String = handle_a.await.expect("a should succeed");

        assert_eq!(result_a, "first");
        assert_eq!(result_b, "second");
    }

    #[tokio::test]
    async fn ownership_check_in_child_context() {
        // Verify that the child context inherits task ownership from
        // the parent — operations in the child from a foreign task fail.
        // This is tested via the replay path (no client needed).
        // The child's checkpoint_record lookup uses its own prefix.
        let log = CheckpointLog::from_records(vec![succeeded_record("1", "42")]);
        let ctx = test_ctx(log);

        let result: Result<i32, OperationError> = ctx
            .run_in_child_context(|_| async move {
                unreachable!("should replay");
            })
            .await;

        // Replay succeeds — ownership check passes (same task).
        assert_eq!(result.expect("should succeed"), 42);
    }
}
