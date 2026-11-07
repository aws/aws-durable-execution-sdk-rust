//! Child context operation execution engine.
//!
//! Implements `run_in_child_context`: live path (run child closure,
//! serialize result, checkpoint), replay path (return frozen result from
//! checkpoint log), and error grading (`ChildFnError` → `ChildContextError`).
//!
//! Wire shape:
//! - `OperationType`: `Context`
//! - `OperationSubType`: the caller's subtype, or `RunInChildContext`
//! - Actions: Start → (Succeed | Fail)
//! - Payload on Succeed: serialized child result
//! - Error on Fail: `{ Type, Message }` from the child function error
//! - `ParentId`: wire ID of the parent context's prefix (if child of child)

use std::future::Future;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tracing::Instrument as _;

use crate::Serdes;
use crate::checkpoint_coalescer::ScopeClose;
use crate::context::{DurableContext, SubmittedWrite, run_body};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, ChildFnError, OperationError, OperationErrorKind,
};
use crate::plugin::UserBody;
use crate::serdes::SerdesContext;
use crate::start::StartStep;

/// Wire sub-type for child context operations.
pub(crate) const CHILD_SUB_TYPE: &str = "RunInChildContext";

/// Maximum checkpoint payload size in bytes (256KB). Payloads exceeding
/// this trigger `ReplayChildren` mode: the child result is not stored in
/// the checkpoint; instead the child body is re-executed on replay.
pub(crate) const CHECKPOINT_SIZE_LIMIT_BYTES: usize = 256 * 1024;

/// The caller's summary of an oversized child result (see
/// [`ChildBuilder::summary`](crate::builders::ChildBuilder::summary)).
pub(crate) type ChildSummary<O> = Box<dyn Fn(&O) -> String + Send + Sync>;

/// Internal state for child context execution passed from the builder.
///
/// Generic over the body closure `F` and, through its future's output,
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
    /// The resolved subtype: the caller's, or `RunInChildContext`.
    pub(crate) sub_type: crate::OperationSubType,
    pub(crate) serdes: S,
    pub(crate) closure: F,
    pub(crate) user_function: UserFunctionPolicy,
    /// The payload of an oversized result's record; `None` records `""`.
    pub(crate) summary: Option<ChildSummary<O>>,
    pub(crate) _marker: std::marker::PhantomData<fn() -> O>,
}

/// Whether a child's body is a user function. Set by the caller, never
/// inferred from the child's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UserFunctionPolicy {
    /// A frame, wraps and user-function hooks.
    Hooks,
    /// None of them: a `with_retry` attempt. In a block with its own record
    /// the outer block is the user function; a block without a child context
    /// has no user function, and its attempts run inside the caller's wraps.
    None,
}

impl<O, F, Fut, E, S> ChildExecution<O, F, S>
where
    O: Send + 'static,
    F: FnOnce(DurableContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
    E: Into<ChildFnError> + 'static,
    S: Serdes<O>,
{
    /// Executes the child context operation.
    ///
    /// Thin generic wrapper: the ONLY code monomorphized per call site.
    /// The replay/checkpoint state machine lives in the non-generic
    /// [`ChildCore`] / [`ChildAfter`] halves (generic over the result type
    /// `O` only); this wrapper just polls the user's concrete future
    /// between them, in a fresh child context under the child namespace's
    /// replay-aware span.
    ///
    /// Tests call it; the SDK builds every operation future with the start
    /// step form (see [`crate::future::DurableFuture::for_start_step`]).
    #[cfg(test)]
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        let (core, closure, user_function) = self.split();
        let started = core.start();
        Self::run(core, closure, user_function, started).await
    }

    /// Runs the child's start step now, on the caller's task, and returns
    /// the future of the rest of the operation (see [`crate::start`]).
    pub(crate) fn execute_started(
        self,
    ) -> impl Future<Output = Result<O, OperationError>> + Send + 'static {
        let (core, closure, user_function) = self.split();
        let started = core.start();
        Self::run(core, closure, user_function, started)
    }

    /// Executes the child as a virtual child context: the body runs in a
    /// context with its own operation namespace, and nothing is recorded
    /// for the child itself.
    ///
    /// The caller has already rejected a record at the child's ID (see
    /// [`DurableContext::reject_record_at_virtual_child`]). The body runs
    /// in every invocation, inline on the awaiting task, as a user
    /// function, between `on_operation_start` and `on_operation_end` with
    /// `is_virtual` true. The child writes no update, emits no
    /// `operation_replayed` event, and closes no scope when the body
    /// returns, so an operation the body started and did not await keeps
    /// running in the caller's scope.
    pub(crate) async fn execute_virtual(self) -> Result<O, OperationError> {
        let Self {
            ctx,
            op_id,
            name,
            sub_type,
            serdes,
            closure,
            ..
        } = self;
        let wire_id = op_id.wire();
        let sub_type = sub_type.as_str();
        let child_ctx = ctx.new_virtual_child(op_id.positional(), name.as_deref());
        let rec = child_info(&ctx, wire_id, sub_type, name.as_deref()).with_virtual(true);
        let info = std::sync::Arc::new(rec.clone().with_replay(child_ctx.is_replaying()));
        if let Some(plugins) = ctx.plugins() {
            crate::plugin::operation_started(plugins, &info);
        }
        let span = child_ctx.replay_span();
        let body = UserBody::Context(child_ctx.prefix().to_owned());
        let result = crate::plugin::run_user_function(
            ctx.shared_plugins(),
            info,
            body,
            span,
            move || async move { (closure)(child_ctx).await.map_err(Into::into) },
            ChildFnError::failure_wire,
        )
        .await
        .into_parts();
        // The end hooks report the replay state after the body: true when
        // an earlier invocation already got past the child.
        let ended = |status, error| {
            if let Some(plugins) = ctx.plugins() {
                let info = rec
                    .clone()
                    .with_status(status)
                    .with_error(error)
                    .with_replay(ctx.is_replaying());
                crate::plugin::operation_ended(plugins, &info);
            }
        };
        match result {
            Ok(value) => {
                let serdes_ctx = SerdesContext::new(wire_id, ctx.execution_arn());
                match serialize_value(&ctx, name.as_deref(), value, &serdes, serdes_ctx.clone())
                    .await
                {
                    Ok(serialized) => {
                        ended(crate::OperationStatus::Succeeded, None);
                        deserialize_value(&ctx, name.as_deref(), serialized, &serdes, serdes_ctx)
                            .await
                    }
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        ended(crate::OperationStatus::Failed, Some(wire));
                        Err(op_err)
                    }
                }
            }
            Err((child_err, wire)) => {
                ended(crate::OperationStatus::Failed, Some(wire));
                Err(OperationError::from_kind(OperationErrorKind::ChildContext(
                    ChildContextError::new(
                        ChildContextErrorKind::ChildFailed,
                        Some(child_err.into_source()),
                    ),
                )))
            }
        }
    }

    /// Splits the execution into its non-generic core and the closure.
    fn split(self) -> (ChildCore<O, S>, F, UserFunctionPolicy) {
        let Self {
            ctx,
            op_id,
            name,
            sub_type,
            serdes,
            closure,
            user_function,
            summary,
            _marker,
        } = self;
        let core = ChildCore {
            ctx,
            op_id,
            name,
            sub_type,
            serdes,
            summary,
            _marker: std::marker::PhantomData,
        };
        (core, closure, user_function)
    }

    /// The rest of the operation after its start step `started`.
    async fn run(
        core: ChildCore<O, S>,
        closure: F,
        user_function: UserFunctionPolicy,
        started: Result<StartStep, OperationError>,
    ) -> Result<O, OperationError> {
        match core.before(started?).await? {
            ChildPrelude::Done(result) => result,
            ChildPrelude::Run {
                child_ctx,
                mode,
                info,
                start,
                after,
            } => {
                // Run the child closure inside the child namespace's
                // replay-aware span: on a resume, nested operations can
                // still be replaying while the parent is live, and the child
                // span's isReplay flag (kept current by the child's own
                // mints) is what lets a filter suppress the body's pre-wait
                // log lines exactly once. A live body's `Runner` owns the
                // span and enters it outside its hooks and wraps on every
                // poll and drop; a reconstruction or a `with_retry` attempt
                // runs no user function, so its body is instrumented with
                // the span instead.
                //
                // The scope encloses the closure call, so code the closure
                // runs before it returns its future is checked too.
                let child_span = child_ctx.replay_span();
                let body = UserBody::Context(child_ctx.prefix().to_owned());
                // The child's scope closes when the body returns, and also
                // when this future is dropped before that, because the
                // operation that owns the child was stopped. An operation
                // the child started and did not await then writes no further
                // checkpoint, and its task stops (see `Admission`). The
                // child's own terminal update is written through the parent
                // context, whose owners do not include this child, so it is
                // admitted.
                let scope = ScopeClose::new(after.ctx.admission(), &after.wire_id);
                let result = if mode == ChildRunMode::Live
                    && user_function == UserFunctionPolicy::Hooks
                {
                    // A live body is a user function. The runner enters the
                    // span outside its hooks and wraps, so a span a wrap
                    // enters is a child of it. The scope closes inside the
                    // body future, so it is closed before the wire and the
                    // end hooks.
                    crate::plugin::run_user_function(
                        after.ctx.shared_plugins(),
                        info,
                        body,
                        child_span,
                        move || async move {
                            let result = (closure)(child_ctx).await.map_err(Into::into);
                            drop(scope);
                            result
                        },
                        ChildFnError::failure_wire,
                    )
                    .await
                    .into_parts()
                    .map_err(|(error, wire)| (error, Some(wire)))
                } else {
                    // A reconstruction, or a `with_retry` attempt: no frame
                    // and no user-function hook.
                    let result = run_body(body.into(), async move { (closure)(child_ctx).await })
                        .instrument(child_span)
                        .await
                        .map_err(|error| (error.into(), None));
                    drop(scope);
                    result
                };
                after.settle(mode, start, result).await
            }
        }
    }
}

/// The pre-closure half of a child context: task-ownership check, replay
/// resolution, and the START checkpoint. Generic only over the result type
/// `O`, no user closure reaches this state machine, so its substantial
/// replay/checkpoint logic compiles once per result type instead of once
/// per call site.
struct ChildCore<O, S> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    sub_type: crate::OperationSubType,
    serdes: S,
    summary: Option<ChildSummary<O>>,
    _marker: std::marker::PhantomData<fn() -> O>,
}

/// Why the child body must run: a fresh live execution, or a re-execution
/// reconstructing a result too large to checkpoint (`ReplayChildren`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        /// The info `on_operation_start` received, with the replay flag
        /// set to the child context's `is_replaying()` when the body
        /// starts (JS: `isReplay: childReplayMode !== ExecutionMode`); the
        /// user-function hooks receive this value.
        info: std::sync::Arc<crate::plugin::OperationInfo>,
        /// The `START` write of this cycle, still in flight. The outcome
        /// write waits for it. `None` when the record already existed.
        start: Option<SubmittedWrite>,
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
    sub_type: crate::OperationSubType,
    serdes: S,
    summary: Option<ChildSummary<O>>,
    _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O, S> ChildCore<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// The start step of the child (see [`crate::start`]): admission, task
    /// ownership, the replay identity check, and, without a record, the
    /// `START`, queued, then `on_operation_start`, then released, as JS
    /// orders them. A record that re-enters the body fires the replay
    /// start hook here.
    fn start(&self) -> Result<StartStep, OperationError> {
        let wire_id = self.op_id.wire();
        crate::start::start_step(
            &self.ctx,
            &crate::start::StartSpec {
                op_id: &self.op_id,
                operation_type: "Context",
                sub_type: self.sub_type.as_str(),
                name: self.name.as_deref(),
                order: crate::start::StartOrder::QueueThenHook,
            },
            |view| match view.status {
                CheckpointStatus::Succeeded | CheckpointStatus::Failed => {
                    crate::start::Cycle::Record
                }
                // Started/Pending/Ready: re-enter the child body.
                _ => crate::start::Cycle::Resume,
            },
            || {
                build_update(
                    wire_id,
                    self.name.as_deref(),
                    self.sub_type.as_str(),
                    OperationAction::Start,
                    &self.ctx,
                )
            },
            |plugins, is_replay| {
                let info = child_info(
                    &self.ctx,
                    wire_id,
                    self.sub_type.as_str(),
                    self.name.as_deref(),
                );
                let info = if is_replay {
                    info.with_replay(true)
                } else {
                    info.with_status(crate::OperationStatus::Started)
                        .with_replay(false)
                };
                crate::plugin::operation_started(plugins, &info);
            },
        )
    }

    /// Runs everything that follows the start step `started` and precedes
    /// the child closure: replay resolution, or the child context for the
    /// body (live, or `ReplayChildren` reconstruction).
    ///
    /// The body does not wait for a queued `START`: the outcome write does
    /// (see [`ChildAfter::settle`]), as in JS.
    async fn before(self, started: StartStep) -> Result<ChildPrelude<O, S>, OperationError> {
        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());

        let mut mode = ChildRunMode::Live;
        // The info `on_operation_start` received, when it fired: the start
        // step built it from the same fields (see `ChildCore::start`).
        let mut start_info = None;
        let start = match started {
            StartStep::Refused => return std::future::pending().await,
            StartStep::Queued(write) => Some(write),
            StartStep::Recorded(view) => match view.status {
                CheckpointStatus::Succeeded if view.replay_children => {
                    // Recorded success whose result was too large to
                    // checkpoint: re-execute the body to reconstruct it.
                    mode = ChildRunMode::Reconstruct;
                    None
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
                // The start step fired the replay start hook.
                _ => {
                    start_info = Some(
                        child_info(
                            &self.ctx,
                            &wire_id,
                            self.sub_type.as_str(),
                            self.name.as_deref(),
                        )
                        .with_replay(true),
                    );
                    None
                }
            },
        };
        if start.is_some() {
            // No record: the start step queued `START` and fired the live
            // start hook.
            start_info = Some(
                child_info(
                    &self.ctx,
                    &wire_id,
                    self.sub_type.as_str(),
                    self.name.as_deref(),
                )
                .with_status(crate::OperationStatus::Started)
                .with_replay(false),
            );
        }

        // 4. Create child context with chained prefix.
        let Self {
            ctx,
            name,
            sub_type,
            serdes,
            summary,
            ..
        } = self;
        let child_ctx = ctx.new_child(&positional_id, name.as_deref());

        // A reconstruction fires no start hook; its info is never read,
        // because only a live body reaches the runner.
        let info = start_info
            .unwrap_or_else(|| child_info(&ctx, &wire_id, sub_type.as_str(), name.as_deref()));
        // The user-function hooks see the child context's replay state at
        // the body's start, not the record-based flag of the start hook: a
        // resumed child with no inner record yet starts live.
        let info = info.with_replay(child_ctx.is_replaying());
        Ok(ChildPrelude::Run {
            child_ctx,
            mode,
            info: std::sync::Arc::new(info),
            start,
            after: ChildAfter {
                ctx,
                wire_id,
                name,
                sub_type,
                serdes,
                summary,
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
        let value = replay_success(
            &self.ctx,
            self.name.as_deref(),
            &self.serdes,
            payload,
            serdes_ctx.clone(),
        )
        .await?;
        self.ctx.emit_operation_replayed(
            wire_id,
            self.name.as_deref(),
            "Context",
            Some(self.sub_type.as_str()),
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
            Some(self.sub_type.as_str()),
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
    /// Live mode checkpoints it: success through [`build_succeed_update`]
    /// (which routes large payloads to `ReplayChildren` mode), failure
    /// through [`checkpoint_live_failure`]. Reconstruct mode
    /// (`ReplayChildren` replay) round-trips the result through
    /// serialization for consistency with the checkpointed path and
    /// checkpoints nothing.
    ///
    /// A live outcome write first waits for `start`, the cycle's `START`
    /// write. When that write fails, the invocation ends and no outcome is
    /// written.
    #[expect(clippy::too_many_lines)] // reason: the two modes and both outcomes read better in one match
    async fn settle(
        self,
        mode: ChildRunMode,
        start: Option<SubmittedWrite>,
        result: Result<O, (ChildFnError, Option<crate::error::WireError>)>,
    ) -> Result<O, OperationError> {
        let Self {
            ctx,
            wire_id,
            name,
            sub_type,
            serdes,
            summary,
            ..
        } = self;
        let sub_type = sub_type.as_str();
        let serdes_ctx = SerdesContext::new(&wire_id, ctx.execution_arn());

        match (mode, result) {
            (ChildRunMode::Reconstruct, Ok(value)) => {
                // Round-trip through serialization for consistency.
                let serialized =
                    serialize_value(&ctx, name.as_deref(), value, &serdes, serdes_ctx.clone())
                        .await?;
                deserialize_value(&ctx, name.as_deref(), serialized, &serdes, serdes_ctx).await
            }
            (ChildRunMode::Reconstruct, Err((child_err, _))) => Err(OperationError::from_kind(
                OperationErrorKind::ChildContext(ChildContextError::new(
                    ChildContextErrorKind::ChildFailed,
                    Some(child_err.into_source()),
                )),
            )),
            (ChildRunMode::Live, Ok(value)) => {
                // Success: serialize and checkpoint.
                //
                // A result serialization failure is a LOCAL, deterministic,
                // user-facing failure, so it stays catchable, but the
                // terminal FAIL is persisted FIRST (issue #43). The closure
                // already ran, so its side effects need a recorded outcome:
                // with the FAIL recorded, replay yields it instead of
                // re-running the closure, and a handler that catches the
                // error branches on a decision replay reproduces.
                let serialized = match serialize_value(
                    &ctx,
                    name.as_deref(),
                    value,
                    &serdes,
                    serdes_ctx.clone(),
                )
                .await
                {
                    Ok(serialized) => serialized,
                    Err(op_err) => {
                        ctx.await_retained_start(&wire_id, start, OperationAction::Fail)
                            .await?;
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        let update = build_child_fail_update(
                            &wire_id,
                            name.as_deref(),
                            sub_type,
                            &ctx,
                            &wire,
                        );
                        if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                            return ctx.checkpoint_failure_unrecoverable(client_err).await;
                        }
                        child_failed(&ctx, &wire_id, sub_type, name.as_deref(), &wire);
                        return Err(op_err
                            .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                            .with_wire(wire));
                    }
                };
                // A result too large to checkpoint is decoded before the
                // write, so the summary sees the value the caller gets. Its
                // decode error is routed only after the write, because the
                // body ran and its success needs a record.
                let (payload, decoded) = if serialized.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
                    let decoded = ctx
                        .hold_activity(serdes.deserialize(serialized, serdes_ctx.clone()))
                        .await;
                    let payload = match (&decoded, &summary) {
                        (Ok(value), Some(summary)) => summary(value),
                        _ => String::new(),
                    };
                    (payload, Some(decoded))
                } else {
                    (serialized, None)
                };
                ctx.await_retained_start(&wire_id, start, OperationAction::Succeed)
                    .await?;
                let update = build_succeed_update(
                    &wire_id,
                    name.as_deref(),
                    sub_type,
                    &ctx,
                    &payload,
                    decoded.is_some(),
                );
                // The end hook reports the payload the update carries: the
                // summary, or `""`, for a `ReplayChildren` result.
                let result = update.payload().map(str::to_owned);
                if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
                    return ctx.checkpoint_failure_unrecoverable(err).await;
                }
                if let Some(plugins) = ctx.plugins() {
                    let info = child_info(&ctx, &wire_id, sub_type, name.as_deref())
                        .with_status(crate::OperationStatus::Succeeded)
                        .with_result(result)
                        .with_replay(false);
                    crate::plugin::operation_ended(plugins, &info);
                }

                match decoded {
                    Some(Ok(value)) => Ok(value),
                    Some(Err(err)) => {
                        Err(deserialize_failure(&ctx, name.as_deref(), &wire_id, err).await)
                    }
                    // Round-trip deserialize for consistency (first-run ==
                    // replay).
                    None => {
                        deserialize_value(&ctx, name.as_deref(), payload, &serdes, serdes_ctx).await
                    }
                }
            }
            (ChildRunMode::Live, Err((child_err, wire))) => {
                // Failure: checkpoint FAIL with error details. A runner's
                // wire is the one its end hooks saw.
                ctx.await_retained_start(&wire_id, start, OperationAction::Fail)
                    .await?;
                Err(checkpoint_live_failure(
                    &ctx,
                    name.as_deref(),
                    sub_type,
                    &wire_id,
                    child_err,
                    wire,
                )
                .await)
            }
        }
    }
}

/// The plugin info of the child context `wire_id`, created in `ctx`.
fn child_info(
    ctx: &DurableContext,
    wire_id: &str,
    sub_type: &str,
    name: Option<&str>,
) -> crate::plugin::OperationInfo {
    crate::plugin::OperationInfo::for_operation(
        ctx,
        wire_id,
        crate::OperationKind::Context,
        sub_type,
        name,
    )
}

/// Fires `on_operation_end` for a child context whose `FAIL` with `wire`
/// was written.
fn child_failed(
    ctx: &DurableContext,
    wire_id: &str,
    sub_type: &str,
    name: Option<&str>,
    wire: &crate::error::WireError,
) {
    if let Some(plugins) = ctx.plugins() {
        let info = child_info(ctx, wire_id, sub_type, name)
            .with_status(crate::OperationStatus::Failed)
            .with_error(Some(wire.clone()))
            .with_replay(false);
        crate::plugin::operation_ended(plugins, &info);
    }
}

/// Builds an `OperationUpdate` for child context operations.
fn build_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    action: OperationAction,
    ctx: &DurableContext,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(sub_type.to_owned())
        .action(action);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    // Set parent wire ID if this is a nested child (child-of-child).
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    // build() is infallible here: all required fields (id, type, action) set.
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds the Succeed `OperationUpdate` for a live child result with
/// `payload`. An `oversized` result switches the operation to
/// `ReplayChildren` mode: `payload` is then its summary, or `""`, the
/// backend preserves the child operations, and the body re-executes on
/// replay to reconstruct the result. Within the plugin child operations
/// depth of `ctx`, a result that fits also carries `ReplayChildren`.
fn build_succeed_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    ctx: &DurableContext,
    payload: &str,
    oversized: bool,
) -> OperationUpdate {
    let mut succeed_builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(sub_type.to_owned())
        .action(OperationAction::Succeed);
    if let Some(n) = name {
        succeed_builder = succeed_builder.name(n);
    }
    if let Some(parent_wire) = ctx.parent_wire_id() {
        succeed_builder = succeed_builder.parent_id(parent_wire);
    }
    succeed_builder = succeed_builder.payload(payload.to_owned());
    // Within the plugin child operations depth, the record keeps its
    // children for later invocations.
    if oversized || ctx.preserves_children(1) {
        succeed_builder = succeed_builder.context_options(
            aws_sdk_lambda::types::ContextOptions::builder()
                .replay_children(true)
                .build(),
        );
    }
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    succeed_builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Checkpoints a live child failure and returns the graded
/// `ChildContextError`. When the `FAIL` write fails, the failure is
/// recorded where its response arrives, and this never returns (see
/// [`DurableContext::checkpoint_failure_unrecoverable`]).
async fn checkpoint_live_failure(
    ctx: &DurableContext,
    name: Option<&str>,
    sub_type: &str,
    wire_id: &str,
    child_err: ChildFnError,
    wire: Option<crate::error::WireError>,
) -> OperationError {
    // Derive the wire failure from the error the body returned: the type
    // is the body error's recorded type, the message is its recorded
    // message (see `wire_message`), and `error_data`/`stack_trace` pass through from its cause
    // chain, so an inner failure's payload survives this boundary.
    // A body without a runner (a `with_retry` attempt) has no wire yet.
    let wire = wire.unwrap_or_else(|| child_err.failure_wire());
    let update = build_child_fail_update(wire_id, name, sub_type, ctx, &wire);

    // A rejected FAIL write fails the invocation, and the child writes
    // nothing more.
    if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
        return ctx.checkpoint_failure_unrecoverable(client_err).await;
    }
    child_failed(ctx, wire_id, sub_type, name, &wire);

    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::ChildFailed,
        Some(child_err.into_source()),
    )))
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

/// Builds a child-context `FAIL` update carrying `wire` as its error.
/// `ctx` is the context the child record belongs to; within its plugin
/// child operations depth, the update carries `ReplayChildren`.
fn build_child_fail_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    ctx: &DurableContext,
    wire: &crate::error::WireError,
) -> OperationUpdate {
    let mut fail_builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(sub_type.to_owned())
        .action(OperationAction::Fail)
        .error(wire.to_error_object());
    if let Some(n) = name {
        fail_builder = fail_builder.name(n);
    }
    if let Some(parent_wire) = ctx.parent_wire_id() {
        fail_builder = fail_builder.parent_id(parent_wire);
    }
    // Within the plugin child operations depth, the record keeps its
    // children for later invocations.
    if ctx.preserves_children(1) {
        fail_builder = fail_builder.context_options(
            aws_sdk_lambda::types::ContextOptions::builder()
                .replay_children(true)
                .build(),
        );
    }
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    fail_builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Replays a successful child context result from the checkpoint log.
async fn replay_success<O, S: Serdes<O>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &S,
    result: Option<String>,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let payload = result.ok_or_else(|| {
        child_internal_error("checkpointed Succeeded operation has no result payload")
    })?;
    deserialize_value(ctx, name, payload, serdes, serdes_ctx).await
}

/// Replays a failed child context result from the checkpoint log.
///
/// The recorded failure fields travel on the synthetic source rather
/// than being folded into a message, so the recorded `error_type` (and
/// `error_data`, when present) stays programmatically recoverable.
///
/// A record whose `error_type` is the serialization discriminator
/// ([`crate::error::SERIALIZATION_FAILED_ERROR_TYPE`]) reconstructs
/// `ChildContextErrorKind::Internal`, the kind the live serialization
/// path yielded after persisting that record, so replay reproduces the
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
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `Internal`.
async fn serialize_value<O, S: Serdes<O>>(
    ctx: &DurableContext,
    name: Option<&str>,
    value: O,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx.hold_activity(serdes.serialize(value, serdes_ctx)).await {
        Ok(serialized) => Ok(serialized),
        Err(err) => {
            let e = ctx
                .route_serdes_error("serialize", &operation_id, name, err)
                .await;
            Err(child_internal_error(&format!(
                "serialize result: {}",
                crate::error::wire_message(&*e)
            )))
        }
    }
}

/// Deserializes a wire payload through the configured serdes.
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `Internal`.
async fn deserialize_value<O, S: Serdes<O>>(
    ctx: &DurableContext,
    name: Option<&str>,
    payload: String,
    serdes: &S,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx
        .hold_activity(serdes.deserialize(payload, serdes_ctx))
        .await
    {
        Ok(value) => Ok(value),
        Err(err) => Err(deserialize_failure(ctx, name, &operation_id, err).await),
    }
}

/// The error of a failed decode of the result of `operation_id`. A marked
/// error ends the invocation ([`DurableContext::route_serdes_error`]); any
/// other error becomes `Internal`.
async fn deserialize_failure(
    ctx: &DurableContext,
    name: Option<&str>,
    operation_id: &str,
    err: crate::BoxError,
) -> OperationError {
    let e = ctx
        .route_serdes_error("deserialize", operation_id, name, err)
        .await;
    child_internal_error(&format!(
        "deserialize result: {}",
        crate::error::wire_message(&*e)
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use std::future::IntoFuture;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn child_failure_keeps_typed_error_data() {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );

        let result: Result<i32, OperationError> = ctx
            .run_in_child_context(|child| async move {
                let value = child
                    .step(|_| async {
                        Err::<i32, crate::BoxError>(Box::new(
                            crate::TypedError::with_type("PaymentError", "declined")
                                .with_error_data(r#"{"code":42}"#),
                        ))
                    })
                    .retry_strategy(|_err, _attempt| crate::RetryDecision::Stop)
                    .await?;
                Ok(value)
            })
            .await;
        assert!(result.is_err(), "the child must fail");

        let updates = client.recorded_updates();
        let fail = updates
            .iter()
            .find(|u| {
                matches!(u.action(), OperationAction::Fail)
                    && matches!(u.r#type(), OperationType::Context)
            })
            .expect("the child records a FAIL update");
        let error = fail
            .error()
            .expect("the FAIL update carries an error object");
        assert_eq!(error.error_data(), Some(r#"{"code":42}"#));
        assert!(error.stack_trace().is_empty(), "{error:?}");
    }

    /// Runs `body` as a child context against a fresh in-memory client and
    /// returns the `ErrorType` and `ErrorMessage` of the child FAIL update.
    async fn child_fail_record<F, Fut>(body: F) -> (Option<String>, Option<String>)
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<i32, crate::BoxError>> + Send + 'static,
    {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        let result: Result<i32, OperationError> = ctx.run_in_child_context(body).await;
        assert!(result.is_err(), "the child must fail");
        let updates = client.recorded_updates();
        let fail = updates
            .iter()
            .find(|u| {
                matches!(u.action(), OperationAction::Fail)
                    && matches!(u.r#type(), OperationType::Context)
            })
            .expect("the child records a FAIL update");
        let error = fail
            .error()
            .expect("the FAIL update carries an error object");
        (
            error.error_type().map(str::to_owned),
            error.error_message().map(str::to_owned),
        )
    }

    /// A failed child records the type and message of the error its body
    /// returned, not of the SDK's internal carrier. A propagated
    /// operation failure records that operation's recorded message. One
    /// row per case: a plain error, a `TypedError`, a propagated step
    /// failure (untyped and typed), a propagated external callback
    /// failure, and a `TypedError` around a propagated step failure.
    #[tokio::test]
    async fn child_failure_records_body_error_type() {
        // Row 1: a plain error records the fallback type "Error".
        let (ty, msg) = child_fail_record(|_child| async { Err("plain boom".into()) }).await;
        assert_eq!(ty.as_deref(), Some("Error"));
        assert_eq!(msg.as_deref(), Some("plain boom"));

        // Row 2: a TypedError records the user's type.
        let (ty, msg) = child_fail_record(|_child| async {
            Err(crate::TypedError::with_type("PaymentError", "declined").into())
        })
        .await;
        assert_eq!(ty.as_deref(), Some("PaymentError"));
        assert_eq!(msg.as_deref(), Some("declined"));

        // Row 3: a propagated untyped step failure records the step's
        // recorded type, "Error".
        let (ty, msg) = child_fail_record(|child| async move {
            let v = child
                .step(|_| async { Err::<i32, crate::BoxError>("step boom".into()) })
                .retry_strategy(|_err, _attempt| crate::RetryDecision::Stop)
                .await?;
            Ok(v)
        })
        .await;
        assert_eq!(ty.as_deref(), Some("Error"));
        assert_eq!(msg.as_deref(), Some("step boom"));

        // Row 4: a propagated typed step failure records the user's type.
        let (ty, msg) = child_fail_record(|child| async move {
            let v = child
                .step(|_| async {
                    Err::<i32, crate::BoxError>(Box::new(crate::TypedError::with_type(
                        "PaymentError",
                        "declined",
                    )))
                })
                .retry_strategy(|_err, _attempt| crate::RetryDecision::Stop)
                .await?;
            Ok(v)
        })
        .await;
        assert_eq!(ty.as_deref(), Some("PaymentError"));
        assert_eq!(msg.as_deref(), Some("declined"));

        // Row 5: a propagated external callback failure records the type
        // the callback failure carries.
        let (ty, msg) = child_fail_record(|_child| async {
            let wire = crate::error::WireError::new(Some("VendorError"), Some("denied"));
            let err = OperationError::from_kind(OperationErrorKind::Callback(
                crate::error::CallbackError::new(
                    crate::error::CallbackErrorKind::ExternalFailure,
                    Some(crate::error::ReplayedFailure::source_from(wire.clone())),
                ),
            ))
            .with_wire(wire);
            Err(err.into())
        })
        .await;
        assert_eq!(ty.as_deref(), Some("VendorError"));
        assert_eq!(msg.as_deref(), Some("denied"));

        // Row 6: a TypedError around a propagated step failure records the
        // user's type and the step's recorded message.
        let (ty, msg) = child_fail_record(|child| async move {
            let v = child
                .step(|_| async { Err::<i32, crate::BoxError>("step boom".into()) })
                .retry_strategy(|_err, _attempt| crate::RetryDecision::Stop)
                .await
                .map_err(|e| crate::TypedError::with_type("PaymentError", e))?;
            Ok(v)
        })
        .await;
        assert_eq!(ty.as_deref(), Some("PaymentError"));
        assert_eq!(msg.as_deref(), Some("step boom"));
    }

    /// A root context over `records` whose transport is a fresh in-memory
    /// client, with the recorder's `p0` and `p1` as the invocation's
    /// plugins and no checkpoint buffer window.
    fn uf_ctx(
        hooks: &crate::plugin::recorder::UfHooks,
        records: Vec<(String, CheckpointRecord)>,
    ) -> (DurableContext, Arc<crate::client::InMemoryExecutionClient>) {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(records)),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
            None,
            0,
            Some(hooks.plugins()),
        );
        (ctx, client)
    }

    /// A record of the operation at `positional_id` with `status` and no
    /// payload.
    fn record_at(positional_id: &str, status: CheckpointStatus) -> (String, CheckpointRecord) {
        let (wire_id, mut record) = succeeded_record(positional_id, "null");
        record.status = status;
        record.result = None;
        (wire_id, record)
    }

    /// The fatal error message `ctx`'s invocation recorded.
    fn fatal_message(ctx: &DurableContext) -> String {
        ctx.coordinator()
            .fatal_error()
            .expect("fatal recorded")
            .error_message
    }

    /// Rule 8.2: a failing child, `map` item and `parallel` branch derive
    /// their wire under `ActiveBody::Outcome`, so a wait that the error's
    /// `Display` calls on the captured child context fails the execution
    /// with a context usage error naming the operation. The `Display`
    /// runs once.
    #[tokio::test]
    async fn failing_child_item_branch_display_calls_its_context() {
        use crate::builders::map_parallel::CompletionConfig;
        use crate::plugin::recorder::{Counted, UfHooks};
        let claims_wait = |dc: &Counted, child: DurableContext| -> crate::BoxError {
            dc.sharing(move || drop(child.wait(Duration::from_secs(1))))
                .boxed()
        };

        // A child.
        let (ctx, _client) = uf_ctx(&UfHooks::default(), Vec::new());
        let dc = Counted::new();
        let body_dc = dc.clone();
        let err = ctx
            .run_in_child_context(move |child| async move {
                Err::<u32, crate::BoxError>(claims_wait(&body_dc, child))
            })
            .name("c")
            .await
            .unwrap_err();
        drop(err);
        assert_eq!(dc.calls(), 1, "child");
        let message = fatal_message(&ctx);
        assert!(
            message.contains("while the SDK records the outcome of operation \"1\""),
            "{message}"
        );

        // A `map` item.
        let (ctx, _client) = uf_ctx(&UfHooks::default(), Vec::new());
        let dc = Counted::new();
        let body_dc = dc.clone();
        let batch = ctx
            .map(vec![1_u32], move |child, _item, _idx| {
                let error = claims_wait(&body_dc, child);
                async move { Err::<u32, crate::BoxError>(error) }
            })
            .name("m")
            .completion(CompletionConfig::with_tolerated_failure_count(1))
            .await_batch()
            .await;
        drop(batch);
        assert_eq!(dc.calls(), 1, "map item");
        let message = fatal_message(&ctx);
        assert!(
            message.contains("while the SDK records the outcome of operation \"1-1\""),
            "{message}"
        );

        // A `parallel` branch.
        let (ctx, _client) = uf_ctx(&UfHooks::default(), Vec::new());
        let dc = Counted::new();
        let body_dc = dc.clone();
        let branch = crate::Branch::new("b", move |child| async move {
            Err::<u32, crate::BoxError>(claims_wait(&body_dc, child))
        });
        let batch = ctx
            .parallel(vec![branch])
            .name("p")
            .completion(CompletionConfig::with_tolerated_failure_count(1))
            .await_batch()
            .await;
        drop(batch);
        assert_eq!(dc.calls(), 1, "parallel branch");
        let message = fatal_message(&ctx);
        assert!(
            message.contains("while the SDK records the outcome of operation \"1-1\""),
            "{message}"
        );
    }

    /// Rule 7.3: the user-function hooks and wraps of a child get the
    /// child context's replay flag when the body starts: false in the
    /// invocation that starts it, true in the one that resumes it over
    /// its inner wait. The wraps and the end hook get the start hook's
    /// info.
    #[tokio::test]
    async fn child_body_info_has_child_replay_flag() {
        use crate::plugin::recorder::UfHooks;
        let child = |ctx: &DurableContext| {
            ctx.run_in_child_context(|child| async move {
                child.wait(Duration::from_secs(1)).name("w").await?;
                Ok(5_u32)
            })
            .name("c")
        };

        // Invocation 1: the child suspends at its wait.
        let hooks = UfHooks::default();
        let (ctx, _client) = uf_ctx(&hooks, Vec::new());
        let signal = Arc::clone(ctx.coordinator());
        let ownership = Arc::clone(ctx.task_ownership());
        let outcome =
            crate::driver::test_support::outcome_of(ctx.clone(), ownership, signal, child(&ctx))
                .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        let start = hooks.find("c", "uf_start");
        assert!(!start.replay);
        assert!(!hooks.hooks_of("c").contains(&"uf_end"));

        // Invocation 2: the child is STARTED and its wait succeeded.
        let hooks = UfHooks::default();
        let records = vec![
            record_at("1", CheckpointStatus::Started),
            record_at("1-1", CheckpointStatus::Succeeded),
        ];
        let (ctx, _client) = uf_ctx(&hooks, records);
        assert_eq!(child(&ctx).await.unwrap(), 5);
        let start = hooks.find("c", "uf_start");
        assert!(start.replay);
        let records = hooks.of("c");
        let later: Vec<_> = records
            .iter()
            .filter(|r| matches!(r.hook, "wrap_enter" | "wrap_exit" | "uf_end"))
            .collect();
        assert!(later.iter().any(|r| r.hook == "uf_end"), "{records:?}");
        for record in &later {
            assert_eq!(
                record.info, start.info,
                "{} gets the start hook's info",
                record.hook
            );
        }
        assert_eq!(hooks.find("c", "uf_end").ok, Some(true));
    }

    /// Rule 7.3: a STARTED child with no inner record reports a replay to
    /// `on_operation_start` and none to its user-function hooks and wraps.
    #[tokio::test]
    async fn started_child_without_inner_record_replay_flags() {
        use crate::plugin::recorder::UfHooks;
        let hooks = UfHooks::default();
        let (ctx, _client) = uf_ctx(&hooks, vec![record_at("1", CheckpointStatus::Started)]);
        let value = ctx
            .run_in_child_context(|child| async move {
                Ok(child.step(|_| async { Ok(3_u32) }).name("s").await?)
            })
            .name("c")
            .await
            .unwrap();
        assert_eq!(value, 3);
        assert!(hooks.find("c", "start").replay);
        let records = hooks.of("c");
        let user_function: Vec<_> = records
            .iter()
            .filter(|r| matches!(r.hook, "uf_start" | "wrap_enter" | "wrap_exit" | "uf_end"))
            .collect();
        // One start, a wrap per poll, one end.
        let hook_names: Vec<_> = user_function.iter().map(|r| r.hook).collect();
        let wraps = hook_names
            .get(1..hook_names.len().saturating_sub(1))
            .unwrap_or_default();
        assert_eq!(hook_names.first(), Some(&"uf_start"), "{hook_names:?}");
        assert_eq!(hook_names.last(), Some(&"uf_end"), "{hook_names:?}");
        assert!(
            !wraps.is_empty()
                && wraps
                    .chunks(2)
                    .all(|pair| pair == ["wrap_enter", "wrap_exit"]),
            "{hook_names:?}"
        );
        assert!(user_function.iter().all(|r| !r.replay), "{user_function:?}");
    }

    /// Rule 7.9: a child's scope closes before its end hooks. At the end
    /// hook the scope admits nothing and the step the body spawned and did
    /// not await is aborted; the child records the body's result.
    #[tokio::test]
    async fn child_scope_closed_before_end_hooks() {
        use crate::plugin::recorder::UfHooks;
        use std::sync::atomic::{AtomicBool, Ordering};

        /// Sets its flag when dropped.
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let child_wire = crate::engine::compute_wire_id_public("1");
        let admission: Arc<std::sync::OnceLock<Arc<crate::checkpoint_coalescer::Admission>>> =
            Arc::default();
        let probe_admission = Arc::clone(&admission);
        let probe_wire = child_wire.clone();
        let hooks = UfHooks::probing(move || {
            let Some(admission) = probe_admission.get() else {
                return String::new();
            };
            format!(
                "admits={} task={}",
                admission.admits(std::slice::from_ref(&probe_wire)),
                admission.has_task_in(&probe_wire)
            )
        });
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        admission
            .set(Arc::clone(ctx.admission()))
            .expect("set once");
        let dropped = Arc::new(AtomicBool::new(false));
        let guard_flag = Arc::clone(&dropped);
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            crate::driver::test_support::drive_handler(ctx, None, move |ctx| async move {
                let value = ctx
                    .run_in_child_context(move |child| async move {
                        // The step task holds the only references.
                        let guard = Arc::new(Dropped(guard_flag));
                        let pending = child
                            .step(move |_| async move {
                                let _guard = guard;
                                std::future::pending::<()>().await;
                                Ok(0_u32)
                            })
                            .name("pending")
                            .spawn();
                        drop(pending);
                        Ok(5_u32)
                    })
                    .name("c")
                    .await
                    .map_err(crate::driver::test_support::wire_err)?;
                Ok(value.to_string())
            }),
        )
        .await
        .expect("the invocation ends within 5 s")
        .expect("no panic");
        assert_eq!(
            outcome,
            crate::driver::InvocationOutcome::Complete("5".to_owned())
        );
        assert_eq!(hooks.find("c", "uf_start").probe, "admits=true task=false");
        assert_eq!(
            hooks.find("c", "wrap_enter").probe,
            "admits=true task=false"
        );
        assert_eq!(
            hooks.find("c", "wrap_exit").probe,
            "admits=false task=false"
        );
        assert_eq!(hooks.find("c", "uf_end").probe, "admits=false task=false");
        let succeed = client
            .recorded_updates()
            .into_iter()
            .find(|u| u.name() == Some("c") && u.action() == &OperationAction::Succeed)
            .expect("the child's SUCCEED");
        assert_eq!(succeed.payload(), Some("5"));
        tokio::time::timeout(Duration::from_secs(5), async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the step's task stopped");
        assert!(
            client
                .recorded_updates()
                .iter()
                .all(|u| u.name() != Some("pending") || u.action() == &OperationAction::Start),
            "the stopped step records no outcome"
        );
    }

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
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
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
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
            },
        )
    }

    /// A fresh root context over an in-memory client, and the client.
    fn live_ctx() -> (DurableContext, Arc<crate::client::InMemoryExecutionClient>) {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        (ctx, client)
    }

    fn checkout() -> crate::OperationSubType {
        crate::OperationSubType::new("Checkout").expect("a valid subtype")
    }

    /// The actions of the updates named `name`, after asserting that each
    /// of them carries the subtype `Checkout`.
    fn checkout_actions(updates: &[OperationUpdate], name: &str) -> Vec<OperationAction> {
        updates
            .iter()
            .filter(|u| u.name() == Some(name))
            .map(|u| {
                assert_eq!(u.sub_type(), Some("Checkout"), "{u:?}");
                u.action().clone()
            })
            .collect()
    }

    /// A serdes whose `serialize` returns a plain, unmarked error.
    struct PlainSerializeFailure;

    impl<T: serde::de::DeserializeOwned + Send + 'static> Serdes<T> for PlainSerializeFailure {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            _value: T,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            Err("cannot encode".into())
        }
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            wire: String,
            _context: SerdesContext,
        ) -> Result<T, crate::BoxError> {
            Ok(serde_json::from_str(&wire)?)
        }
    }

    /// Every update of a succeeded child, of a child whose body fails and
    /// of a child whose result fails to serialize carries the custom
    /// subtype.
    #[tokio::test]
    async fn custom_sub_type_on_every_child_update() {
        let (ctx, client) = live_ctx();
        let ok: Result<u32, OperationError> = ctx
            .run_in_child_context(|_child| async { Ok(1_u32) })
            .name("ok")
            .sub_type(checkout())
            .await;
        assert_eq!(ok.ok(), Some(1));
        let body_fails: Result<u32, OperationError> = ctx
            .run_in_child_context(|_child| async { Err("declined".into()) })
            .name("body-fails")
            .sub_type(checkout())
            .await;
        assert!(body_fails.is_err());
        let serdes_fails: Result<u32, OperationError> = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .name("serdes-fails")
            .sub_type(checkout())
            .serdes(PlainSerializeFailure)
            .await;
        assert!(serdes_fails.is_err());

        let updates = client.recorded_updates();
        assert_eq!(
            checkout_actions(&updates, "ok"),
            [OperationAction::Start, OperationAction::Succeed]
        );
        assert_eq!(
            checkout_actions(&updates, "body-fails"),
            [OperationAction::Start, OperationAction::Fail]
        );
        assert_eq!(
            checkout_actions(&updates, "serdes-fails"),
            [OperationAction::Start, OperationAction::Fail]
        );
    }

    /// Whether `update` carries `ContextOptions` with `ReplayChildren`.
    fn has_replay_children(update: &OperationUpdate) -> Option<bool> {
        update
            .context_options()
            .and_then(aws_sdk_lambda::types::ContextOptions::replay_children)
    }

    /// The update of the operation named `name` with `action`.
    fn named_update(
        updates: &[OperationUpdate],
        name: &str,
        action: &OperationAction,
    ) -> OperationUpdate {
        updates
            .iter()
            .find(|u| u.name() == Some(name) && u.action() == action)
            .cloned()
            .unwrap_or_else(|| unreachable!("no {action:?} update for {name}"))
    }

    /// Within the depth, a child's SUCCEED keeps its payload and carries
    /// `ReplayChildren`.
    #[tokio::test]
    async fn preserved_child_keeps_payload_and_flag() {
        let (ctx, client) = DurableContext::test_context_with_depth(1);
        let value: String = ctx
            .run_in_child_context(|_child| async { Ok("x".to_owned()) })
            .name("c")
            .await
            .expect("child succeeds");
        assert_eq!(value, "x");
        let succeed = named_update(&client.recorded_updates(), "c", &OperationAction::Succeed);
        assert_eq!(has_replay_children(&succeed), Some(true));
        assert_eq!(succeed.payload(), Some("\"x\""));
    }

    /// Within the depth, a child's FAIL carries `ReplayChildren`; at depth 0
    /// it carries no `ContextOptions`.
    #[tokio::test]
    async fn preserved_child_failure_has_flag() {
        for (depth, flag) in [(1, Some(true)), (0, None)] {
            let (ctx, client) = DurableContext::test_context_with_depth(depth);
            let result: Result<u32, OperationError> = ctx
                .run_in_child_context(|_child| async { Err("declined".into()) })
                .name("c")
                .await;
            assert!(result.is_err());
            let fail = named_update(&client.recorded_updates(), "c", &OperationAction::Fail);
            assert_eq!(has_replay_children(&fail), flag, "depth {depth}");
        }
    }

    /// A child inside a top-level child sits at level 2: depth 1 flags only
    /// the outer record.
    #[tokio::test]
    async fn nested_child_is_flagged_only_within_depth() {
        let (ctx, client) = DurableContext::test_context_with_depth(1);
        let value: u32 = ctx
            .run_in_child_context(|child| async move {
                Ok(child
                    .run_in_child_context(|_inner| async { Ok(7_u32) })
                    .name("inner")
                    .await?)
            })
            .name("outer")
            .await
            .expect("children succeed");
        assert_eq!(value, 7);
        let updates = client.recorded_updates();
        let outer = named_update(&updates, "outer", &OperationAction::Succeed);
        let inner = named_update(&updates, "inner", &OperationAction::Succeed);
        assert_eq!(has_replay_children(&outer), Some(true));
        assert_eq!(has_replay_children(&inner), None);
        assert_eq!(inner.payload(), Some("7"));
    }

    /// A child whose result fails to serialize writes its FAIL through the
    /// same builder, so it carries the flag within the depth.
    #[tokio::test]
    async fn preserved_child_serialization_failure_has_flag() {
        let (ctx, client) = DurableContext::test_context_with_depth(1);
        let result: Result<u32, OperationError> = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .name("c")
            .serdes(PlainSerializeFailure)
            .await;
        assert!(result.is_err());
        let fail = named_update(&client.recorded_updates(), "c", &OperationAction::Fail);
        assert_eq!(has_replay_children(&fail), Some(true));
    }

    /// A result too large to checkpoint writes a `ReplayChildren` SUCCEED
    /// that carries the custom subtype.
    #[tokio::test]
    async fn custom_sub_type_on_replay_children_succeed() {
        let (ctx, client) = live_ctx();
        let value: Result<String, OperationError> = ctx
            .run_in_child_context(|_child| async { Ok("x".repeat(300 * 1024)) })
            .name("big")
            .sub_type(checkout())
            .await;
        assert_eq!(value.map(|v| v.len()).ok(), Some(300 * 1024));

        let updates = client.recorded_updates();
        let succeed = updates
            .iter()
            .find(|u| matches!(u.action(), OperationAction::Succeed))
            .expect("the SUCCEED update is recorded");
        assert_eq!(
            succeed
                .context_options()
                .and_then(aws_sdk_lambda::types::ContextOptions::replay_children),
            Some(true)
        );
        assert_eq!(succeed.payload(), Some(""));
        assert_eq!(succeed.sub_type(), Some("Checkout"));
    }

    /// Runs `child` with the START write accepted and the next write
    /// rejected, and returns the recorded updates.
    async fn reject_second_write<F>(
        ctx: DurableContext,
        client: &crate::client::InMemoryExecutionClient,
        child: F,
    ) -> Vec<OperationUpdate>
    where
        F: IntoFuture + Send + 'static,
        F::IntoFuture: Send + 'static,
    {
        use crate::client::TestResponse;

        client.enqueue_checkpoint_response(TestResponse::Success(vec![]));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError(
            "payload rejected".to_owned(),
        ));
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            child,
        )
        .await;
        assert!(
            matches!(outcome, crate::driver::InvocationOutcome::Failed { .. }),
            "a rejected write fails the execution: {outcome:?}"
        );
        client.recorded_updates()
    }

    /// The write the service rejects carries the custom subtype, and no
    /// terminal FAIL follows it: a rejected SUCCEED, a rejected FAIL of a
    /// body failure, and a rejected FAIL of a serialization failure. The
    /// child's record stays at its accepted START.
    #[tokio::test]
    async fn custom_sub_type_on_rejected_write_and_no_terminal_fail() {
        // A rejected SUCCEED.
        let (ctx, client) = live_ctx();
        let child = ctx
            .run_in_child_context(|_child| async { Ok(1_u32) })
            .name("c")
            .sub_type(checkout());
        let updates = reject_second_write(ctx, &client, child).await;
        assert_eq!(
            checkout_actions(&updates, "c"),
            [OperationAction::Start, OperationAction::Succeed]
        );

        // A rejected FAIL of a body failure.
        let (ctx, client) = live_ctx();
        let child = ctx
            .run_in_child_context(|_child| async { Err::<u32, crate::BoxError>("no".into()) })
            .name("c")
            .sub_type(checkout());
        let updates = reject_second_write(ctx, &client, child).await;
        assert_eq!(
            checkout_actions(&updates, "c"),
            [OperationAction::Start, OperationAction::Fail]
        );

        // A rejected FAIL of a serialization failure.
        let (ctx, client) = live_ctx();
        let child = ctx
            .run_in_child_context(|_child| async { Ok(1_u32) })
            .name("c")
            .sub_type(checkout())
            .serdes(PlainSerializeFailure);
        let updates = reject_second_write(ctx, &client, child).await;
        assert_eq!(
            checkout_actions(&updates, "c"),
            [OperationAction::Start, OperationAction::Fail]
        );
    }

    /// A record with the custom subtype replays its success and its
    /// failure when the child sets the same subtype.
    #[tokio::test]
    async fn custom_sub_type_replays_recorded_outcomes() {
        let (wire_id, mut record) = succeeded_record("1", "7");
        record.op_type = Some("CONTEXT".to_owned());
        record.sub_type = Some("Checkout".to_owned());
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let value: Result<u32, OperationError> = ctx
            .run_in_child_context(|_child| async { unreachable!("replayed") })
            .sub_type(checkout())
            .await;
        assert_eq!(value.ok(), Some(7));

        let (wire_id, mut record) = failed_record("1", "Error", "no");
        record.op_type = Some("CONTEXT".to_owned());
        record.sub_type = Some("Checkout".to_owned());
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let value: Result<u32, OperationError> = ctx
            .run_in_child_context(|_child| async { unreachable!("replayed") })
            .sub_type(checkout())
            .await;
        assert!(value.is_err());
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

        let value = result.unwrap();
        assert_eq!(value, "hello");
    }

    /// A serdes whose every call fails with a marked error.
    struct MarkedFailure;

    impl<T: Send + 'static> Serdes<T> for MarkedFailure {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            _value: T,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            Err(crate::serdes::RetryableSerdesError::new("injected failure").into())
        }
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            _wire: String,
            _context: SerdesContext,
        ) -> Result<T, crate::BoxError> {
            Err(crate::serdes::RetryableSerdesError::new("injected failure").into())
        }
    }

    #[tokio::test]
    async fn child_reconstruct_marked_failure_faults_invocation() {
        // A ReplayChildren child: the recorded success carries no payload,
        // so the body runs again and its result takes the round trip.
        let (wire_id, mut record) = succeeded_record("1", "");
        record.result = None;
        record.replay_children = true;
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let signal = Arc::clone(ctx.coordinator());

        let child = ctx
            .run_in_child_context(|_child_ctx| async move { Ok(7_i32) })
            .name("rebuild")
            .serdes(MarkedFailure);
        crate::driver::test_support::expect_serdes_fault(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            child,
            "serdes serialize failed for operation \"rebuild\"",
        )
        .await;
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
        // client-less context and passed vacuously: the rejected START
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
        let signal = tokio::time::timeout(Duration::from_secs(2), rx).await;
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

        // Await in reverse order: IDs were already claimed.
        let result_b: String = handle_b.await.expect("b should succeed");
        let result_a: String = handle_a.await.expect("a should succeed");

        assert_eq!(result_a, "first");
        assert_eq!(result_b, "second");
    }

    #[tokio::test]
    async fn ownership_check_in_child_context() {
        // Verify that the child context inherits task ownership from
        // the parent: operations in the child from a foreign task fail.
        // This is tested via the replay path (no client needed).
        // The child's checkpoint_record lookup uses its own prefix.
        let log = CheckpointLog::from_records(vec![succeeded_record("1", "42")]);
        let ctx = test_ctx(log);

        let result: Result<i32, OperationError> = ctx
            .run_in_child_context(|_| async move {
                unreachable!("should replay");
            })
            .await;

        // Replay succeeds: ownership check passes (same task).
        assert_eq!(result.expect("should succeed"), 42);
    }

    /// A child whose result is the 300 KiB string, named `big`, with the
    /// summary `len=<length>` that counts its calls in `calls`.
    async fn run_big_child(
        ctx: &DurableContext,
        value: String,
        calls: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Result<String, OperationError> {
        let calls = Arc::clone(calls);
        ctx.run_in_child_context(move |_child| async move { Ok(value) })
            .name("big")
            .summary(move |s: &String| {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                format!("len={}", s.len())
            })
            .await
    }

    /// An oversized child records its summary as the `ReplayChildren`
    /// payload, and the summary runs once.
    #[tokio::test]
    async fn child_summary_written_when_oversized() {
        let (ctx, client) = live_ctx();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let value = run_big_child(&ctx, "x".repeat(300 * 1024), &calls)
            .await
            .expect("child succeeds");
        assert_eq!(value.len(), 300 * 1024);
        let succeed = named_update(&client.recorded_updates(), "big", &OperationAction::Succeed);
        assert_eq!(has_replay_children(&succeed), Some(true));
        assert_eq!(succeed.payload(), Some("len=307200"));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// A result that fits is recorded as before, and the summary is not
    /// called.
    #[tokio::test]
    async fn child_summary_not_called_under_limit() {
        let (ctx, client) = live_ctx();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let value = run_big_child(&ctx, "small".to_owned(), &calls)
            .await
            .expect("child succeeds");
        assert_eq!(value, "small");
        let succeed = named_update(&client.recorded_updates(), "big", &OperationAction::Succeed);
        assert_eq!(succeed.payload(), Some("\"small\""));
        assert!(succeed.context_options().is_none(), "{succeed:?}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// An oversized child without a summary records the payload `""`.
    #[tokio::test]
    async fn oversized_child_without_summary_writes_empty_payload() {
        let (ctx, client) = live_ctx();
        let value: String = ctx
            .run_in_child_context(|_child| async { Ok("x".repeat(300 * 1024)) })
            .name("big")
            .await
            .expect("child succeeds");
        assert_eq!(value.len(), 300 * 1024);
        let succeed = named_update(&client.recorded_updates(), "big", &OperationAction::Succeed);
        assert_eq!(has_replay_children(&succeed), Some(true));
        assert_eq!(succeed.payload(), Some(""));
    }

    /// A serdes that encodes a `String` as JSON and appends `!` to every
    /// decoded value, counting its decodes in `decodes`.
    struct SuffixOnDecode {
        decodes: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Serdes<String> for SuffixOnDecode {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: String,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            Ok(serde_json::to_string(&value)?)
        }
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            wire: String,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            self.decodes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let value: String = serde_json::from_str(&wire)?;
            Ok(format!("{value}!"))
        }
    }

    /// An oversized child decodes its serialized result once, passes the
    /// decoded value to the summary, and returns that same value.
    #[tokio::test]
    async fn oversized_child_summary_sees_decoded_value_with_one_decode() {
        let (ctx, client) = live_ctx();
        let decodes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let value: String = ctx
            .run_in_child_context(|_child| async { Ok("x".repeat(300 * 1024)) })
            .name("big")
            .serdes(SuffixOnDecode {
                decodes: Arc::clone(&decodes),
            })
            .summary(|s: &String| format!("len={} last={:?}", s.len(), s.chars().last()))
            .await
            .expect("child succeeds");
        assert_eq!(value.len(), 300 * 1024 + 1);
        assert!(value.ends_with("x!"), "the caller gets the decoded value");
        let succeed = named_update(&client.recorded_updates(), "big", &OperationAction::Succeed);
        assert_eq!(has_replay_children(&succeed), Some(true));
        assert_eq!(succeed.payload(), Some("len=307201 last=Some('!')"));
        assert_eq!(decodes.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// A serdes that encodes JSON and fails every decode, with a marked
    /// error when `marked` is set.
    struct DecodeFailure {
        marked: bool,
    }

    impl<T: serde::Serialize + Send + 'static> Serdes<T> for DecodeFailure {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: T,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            Ok(serde_json::to_string(&value)?)
        }
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            _wire: String,
            _context: SerdesContext,
        ) -> Result<T, crate::BoxError> {
            if self.marked {
                Err(crate::serdes::RetryableSerdesError::new("injected decode failure").into())
            } else {
                Err("injected decode failure".into())
            }
        }
    }

    /// An oversized result whose decode fails records the success with the
    /// payload `""`, without calling the summary, and then returns the
    /// decode error (a) or ends the invocation for a marked error (b).
    #[tokio::test]
    async fn oversized_child_decode_failure_records_success_then_routes() {
        let child =
            |ctx: &DurableContext, marked: bool, calls: &Arc<std::sync::atomic::AtomicUsize>| {
                let calls = Arc::clone(calls);
                ctx.run_in_child_context(|_child| async { Ok("x".repeat(300 * 1024)) })
                    .name("big")
                    .serdes(DecodeFailure { marked })
                    .summary(move |s: &String| {
                        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        format!("len={}", s.len())
                    })
            };
        let assert_recorded = |client: &crate::client::InMemoryExecutionClient| {
            let succeed =
                named_update(&client.recorded_updates(), "big", &OperationAction::Succeed);
            assert_eq!(has_replay_children(&succeed), Some(true));
            assert_eq!(succeed.payload(), Some(""));
        };

        // (a) An unmarked error is the deserialize helper's error.
        let (ctx, client) = live_ctx();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let err = child(&ctx, false, &calls)
            .await
            .expect_err("the decode fails");
        assert!(
            matches!(
                err.kind(),
                OperationErrorKind::ChildContext(c) if matches!(c.kind(), ChildContextErrorKind::Internal)
            ),
            "{err:?}"
        );
        let chain = crate::error::chain_string(&err);
        assert!(
            chain.contains("deserialize result: injected decode failure"),
            "{chain}"
        );
        assert_recorded(&client);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        // (b) A marked error ends the invocation, with the success recorded.
        let (ctx, client) = live_ctx();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let signal = Arc::clone(ctx.coordinator());
        crate::driver::test_support::expect_serdes_fault(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            child(&ctx, true, &calls),
            "serdes deserialize failed for operation \"big\"",
        )
        .await;
        assert_recorded(&client);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    /// A serdes whose `serialize` encodes JSON and whose `deserialize`
    /// fails, with a [`RetryableSerdesError`](crate::serdes::RetryableSerdesError)
    /// when `marked`.
    struct DeserializeFailure {
        marked: bool,
    }

    impl<T: serde::Serialize + Send + 'static> Serdes<T> for DeserializeFailure {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: T,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            Ok(serde_json::to_string(&value)?)
        }
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            _wire: String,
            _context: SerdesContext,
        ) -> Result<T, crate::BoxError> {
            if self.marked {
                Err(crate::serdes::RetryableSerdesError::new("injected decode failure").into())
            } else {
                Err("cannot decode".into())
            }
        }
    }

    /// The error a typed serdes fails with: the type `PaymentError`, the
    /// message `declined` and the data `d`.
    fn typed_declined() -> crate::BoxError {
        Box::new(crate::TypedError::with_type("PaymentError", "declined").with_error_data("d"))
    }

    /// A JSON serdes whose `serialize` (when `on_serialize`) or
    /// `deserialize` (otherwise) fails, unmarked, with [`typed_declined`].
    struct TypedSerdesFailure {
        on_serialize: bool,
    }

    impl<T> Serdes<T> for TypedSerdesFailure
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: T,
            _context: SerdesContext,
        ) -> Result<String, crate::BoxError> {
            if self.on_serialize {
                Err(typed_declined())
            } else {
                Ok(serde_json::to_string(&value)?)
            }
        }
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            wire: String,
            _context: SerdesContext,
        ) -> Result<T, crate::BoxError> {
            if self.on_serialize {
                Ok(serde_json::from_str(&wire)?)
            } else {
                Err(typed_declined())
            }
        }
    }

    /// The text of the source of a child context error.
    fn child_source_text(err: &OperationError) -> Option<String> {
        let OperationErrorKind::ChildContext(child) = err.kind() else {
            unreachable!("expected a child context error, got {:?}", err.kind());
        };
        std::error::Error::source(child).map(ToString::to_string)
    }

    /// The recorded message of a typed serialize failure of a child result.
    const TYPED_SERIALIZE_MESSAGE: &str =
        "operation error: child_context: internal error: serialize result: declined";

    /// A typed permanent serialize failure of a child result records the
    /// wrapped message without the type name, under the serialization
    /// type, with no data and no stack. Live and replay both give
    /// `Internal` with that record.
    #[tokio::test]
    async fn typed_serialize_failure_records_the_wrapped_message() {
        let expected = crate::error::WireError::new(
            Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE),
            Some(TYPED_SERIALIZE_MESSAGE),
        );
        let (ctx, client) = live_ctx();
        let live = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .serdes(TypedSerdesFailure { on_serialize: true })
            .await
            .expect_err("serialize fails");
        assert!(is_child_internal(&live), "{live:?}");
        assert_eq!(
            child_source_text(&live).as_deref(),
            Some("serialize result: declined")
        );
        assert_eq!(live.wire(), Some(&expected));
        let updates = client.recorded_updates();
        let error = updates
            .iter()
            .find(|u| u.action() == &OperationAction::Fail)
            .and_then(OperationUpdate::error)
            .expect("the child records a FAIL with an error");
        assert_eq!(
            error.error_type(),
            Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE)
        );
        assert_eq!(error.error_message(), Some(TYPED_SERIALIZE_MESSAGE));
        assert_eq!(error.error_data(), None);
        assert!(error.stack_trace().is_empty(), "{error:?}");

        let ctx = test_ctx(CheckpointLog::from_records(vec![failed_record(
            "1",
            crate::error::SERIALIZATION_FAILED_ERROR_TYPE,
            TYPED_SERIALIZE_MESSAGE,
        )]));
        let replayed = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .serdes(TypedSerdesFailure { on_serialize: true })
            .await
            .expect_err("the recorded failure replays");
        assert!(is_child_internal(&replayed), "{replayed:?}");
        assert_eq!(replayed.wire(), Some(&expected));
    }

    /// A typed permanent deserialize failure of a child result gives an
    /// `Internal` error whose recorded message omits the type name, with
    /// the child's type and no data or stack, live and on replay of the
    /// recorded success.
    #[tokio::test]
    async fn typed_deserialize_failure_records_the_wrapped_message() {
        let expected = crate::error::WireError::new(
            Some("ChildContextError"),
            Some("operation error: child_context: internal error: deserialize result: declined"),
        );
        let (ctx, client) = live_ctx();
        let live = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .serdes(TypedSerdesFailure {
                on_serialize: false,
            })
            .await
            .expect_err("deserialize fails");
        assert!(is_child_internal(&live), "{live:?}");
        assert_eq!(live.wire(), None);
        assert_eq!(
            child_source_text(&live).as_deref(),
            Some("deserialize result: declined")
        );
        assert_eq!(crate::error::wire_error_for(&live, "Error"), expected);
        assert!(
            client
                .recorded_updates()
                .iter()
                .any(|u| u.action() == &OperationAction::Succeed),
            "the child records its success"
        );

        let ctx = test_ctx(CheckpointLog::from_records(vec![succeeded_record(
            "1", "3",
        )]));
        let replayed = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .serdes(TypedSerdesFailure {
                on_serialize: false,
            })
            .await
            .expect_err("the recorded success fails to decode");
        assert!(is_child_internal(&replayed), "{replayed:?}");
        assert_eq!(replayed.wire(), None);
        assert_eq!(
            child_source_text(&replayed).as_deref(),
            Some("deserialize result: declined")
        );
        assert_eq!(crate::error::wire_error_for(&replayed, "Error"), expected);
    }

    /// Whether `err` is a child context error of kind `Internal`.
    fn is_child_internal(err: &OperationError) -> bool {
        matches!(
            err.kind(),
            OperationErrorKind::ChildContext(child)
                if matches!(child.kind(), ChildContextErrorKind::Internal)
        )
    }

    /// An unmarked serialize failure of a virtual child returns the serialize
    /// helper's error, writes nothing, and ends the child with `Failed`.
    #[tokio::test]
    async fn virtual_child_unmarked_serialize_failure() {
        let hooks = crate::plugin::recorder::UfHooks::default();
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        let err = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .name("v")
            .virtual_context()
            .serdes(PlainSerializeFailure)
            .await
            .expect_err("serialize fails");
        assert!(is_child_internal(&err), "{err:?}");
        let message = format!("{err:#}");
        assert!(message.contains("serialize result: "), "{message}");
        assert!(message.contains("cannot encode"), "{message}");
        assert_eq!(client.recorded_updates(), [] as [OperationUpdate; 0]);
        let end = hooks.find("v", "end");
        assert_eq!(end.status, Some(crate::OperationStatus::Failed));
        assert!(end.is_virtual);
    }

    /// A marked serialize failure of a virtual child ends the invocation
    /// and fires no end hook.
    #[tokio::test]
    async fn virtual_child_marked_serialize_failure_ends_invocation() {
        let hooks = crate::plugin::recorder::UfHooks::default();
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        let child = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .name("v")
            .virtual_context()
            .serdes(MarkedFailure);
        crate::driver::test_support::expect_serdes_fault(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            Arc::clone(ctx.coordinator()),
            child,
            "serdes serialize failed for operation \"v\"",
        )
        .await;
        assert!(hooks.hooks_of("v").contains(&"start"));
        assert!(!hooks.hooks_of("v").contains(&"end"));
        assert_eq!(client.recorded_updates(), [] as [OperationUpdate; 0]);
    }

    /// An unmarked deserialize failure of a virtual child returns the
    /// deserialize helper's error after the `Succeeded` end hook; a marked
    /// one ends the invocation.
    #[tokio::test]
    async fn virtual_child_deserialize_failures() {
        let hooks = crate::plugin::recorder::UfHooks::default();
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        let err = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .name("v")
            .virtual_context()
            .serdes(DeserializeFailure { marked: false })
            .await
            .expect_err("deserialize fails");
        assert!(is_child_internal(&err), "{err:?}");
        let message = format!("{err:#}");
        assert!(message.contains("deserialize result: "), "{message}");
        assert!(message.contains("cannot decode"), "{message}");
        assert_eq!(
            hooks.find("v", "end").status,
            Some(crate::OperationStatus::Succeeded)
        );
        assert_eq!(client.recorded_updates(), [] as [OperationUpdate; 0]);

        let (ctx, _client) = uf_ctx(&crate::plugin::recorder::UfHooks::default(), Vec::new());
        let child = ctx
            .run_in_child_context(|_child| async { Ok(3_u32) })
            .name("v")
            .virtual_context()
            .serdes(DeserializeFailure { marked: true });
        crate::driver::test_support::expect_serdes_fault(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            Arc::clone(ctx.coordinator()),
            child,
            "serdes deserialize failed for operation \"v\"",
        )
        .await;
    }

    /// A virtual child is not abandoned when its body returns: a step the
    /// body spawned and did not await keeps running and records its
    /// outcome, and the virtual child records nothing.
    #[tokio::test]
    async fn virtual_child_is_not_abandoned() {
        let (ctx, client) = live_ctx();
        let slot: Arc<std::sync::Mutex<Option<crate::DurableFuture<u32>>>> = Arc::default();
        let notify = Arc::new(tokio::sync::Notify::new());
        let (body_slot, body_notify, body_client) =
            (Arc::clone(&slot), Arc::clone(&notify), Arc::clone(&client));
        let value = ctx
            .run_in_child_context(move |child| async move {
                let handle = child
                    .step(move |_| async move {
                        body_notify.notified().await;
                        Ok(7_u32)
                    })
                    .name("s")
                    .spawn();
                *body_slot.lock().expect("lock") = Some(handle);
                let started = |client: &crate::client::InMemoryExecutionClient| {
                    client
                        .recorded_updates()
                        .iter()
                        .any(|u| u.name() == Some("s") && u.action() == &OperationAction::Start)
                };
                let mut yields = 0;
                while !started(&body_client) {
                    assert!(yields < 1_000, "the step's START is recorded");
                    yields += 1;
                    tokio::task::yield_now().await;
                }
                Ok(1_u32)
            })
            .name("v")
            .virtual_context()
            .await
            .expect("the child succeeds");
        assert_eq!(value, 1);
        notify.notify_one();
        let handle = slot.lock().expect("lock").take().expect("the handle");
        let step = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("the step settles within 1 s")
            .expect("the step succeeds");
        assert_eq!(step, 7);
        let updates = client.recorded_updates();
        let actions: Vec<_> = updates
            .iter()
            .filter(|u| u.name() == Some("s"))
            .map(|u| u.action().clone())
            .collect();
        assert_eq!(actions, [OperationAction::Start, OperationAction::Succeed]);
        assert!(updates.iter().all(|u| u.name() != Some("v")), "{updates:?}");
    }
}
