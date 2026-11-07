//! Wait-for-condition operation execution engine.
//!
//! Implements the live path (evaluate check fn, checkpoint state per attempt,
//! apply wait strategy → wait for the delay, complete or exhaust) and replay
//! path
//! (frozen terminal state; LOUD error on state deserialization failure: never
//! silent reset to initial state).
//!
//! ## Key invariants
//!
//! - The SDK surfaces a `WaitForConditionError::SerializationFailed` error
//!   when checkpointed state fails to deserialize with an unmarked serdes
//!   error. A serdes error marked with `RetryableSerdesError` ends the
//!   invocation instead. Neither falls back to `initial_state`.
//! - [`WaitDecision`] is the type returned by the wait strategy.
//! - Wait strategy exhaustion raises `WaitForConditionError::MaxChecksExceeded`,
//!   live and on replay: the terminal record carries a dedicated wire
//!   discriminator, so the replayed error keeps the same kind and check count.
//! - The first check receives the initial state after a serdes round trip,
//!   as a resumed check receives the recorded state.
//! - A check error records its own type, and the live path classifies the
//!   record with the function replay uses, so both yield the same kind.

use std::future::Future;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

use crate::context::{DurableContext, StepContext, SubmittedWrite};
use crate::engine::CheckpointStatus;
use crate::engine::OperationId;
use crate::error::{
    OperationError, OperationErrorKind, WaitForConditionError, WaitForConditionErrorKind,
};
use crate::plugin::UserBody;
use crate::serdes::SerdesContext;
use crate::{BoxError, Serdes};

/// Wire sub-type for wait-for-condition operations.
pub(crate) const WFC_SUB_TYPE: &str = "WaitForCondition";

/// Decision returned by a wait strategy function.
///
/// Tells the engine whether to continue polling (with a delay), stop
/// (condition met), or fail (exhaustion).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::wait_for_condition::WaitDecision;
/// use std::time::Duration;
///
/// // Condition met: stop polling.
/// let done = WaitDecision::complete();
///
/// // Not met: poll again after 5 seconds.
/// let cont = WaitDecision::continue_with(Duration::from_secs(5));
///
/// // Exhausted: fail the operation.
/// let exhausted = WaitDecision::exhausted("max attempts exceeded");
/// # drop(done); drop(cont); drop(exhausted);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WaitDecision {
    /// Condition met: return the final state.
    Complete,
    /// Condition not met: wait for `delay`, then re-check.
    Continue {
        /// Duration to wait before the next check attempt.
        delay: Duration,
    },
    /// Strategy exhaustion: fail the operation with an error.
    Exhausted {
        /// Reason for exhaustion (e.g., "max attempts exceeded").
        reason: String,
    },
}

impl WaitDecision {
    /// Creates a `Complete` decision (condition met).
    #[must_use]
    pub fn complete() -> Self {
        Self::Complete
    }

    /// Creates a `Continue` decision with the specified delay.
    #[must_use]
    pub fn continue_with(delay: Duration) -> Self {
        Self::Continue { delay }
    }

    /// Creates an `Exhausted` decision (strategy failure).
    #[must_use]
    pub fn exhausted(reason: impl Into<String>) -> Self {
        Self::Exhausted {
            reason: reason.into(),
        }
    }
}

/// Type alias for a boxed wait strategy function.
///
/// Receives the current (deserialized) state and the 1-based attempt number,
/// and returns a [`WaitDecision`].
///
/// Crate-internal: the boxing is an implementation detail. The public setter
/// [`WaitForConditionBuilder::wait_strategy_fn`](crate::WaitForConditionBuilder::wait_strategy_fn)
/// takes a generic closure and boxes it here.
pub(crate) type WaitStrategyFn<S> = Box<dyn Fn(S, u32) -> WaitDecision + Send + Sync>;

/// Internal state for wait-for-condition execution.
///
/// Generic over the check closure `F` (and, through `F`'s output, its
/// future), so each poll produces a concrete future with no per-check box.
/// The one erasure point is the builder's `.future()` / `into_future`,
/// which boxes the whole execution future once inside
/// [`DurableFuture`](crate::DurableFuture).
pub(crate) struct WaitForConditionExecution<S, F, SD> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) initial_state: S,
    pub(crate) wait_strategy: WaitStrategyFn<S>,
    pub(crate) serdes: SD,
    pub(crate) check: F,
}

impl<S, F, Fut, SD> WaitForConditionExecution<S, F, SD>
where
    S: Clone + Send + Sync + 'static,
    F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
    SD: Serdes<S>,
{
    /// Executes the wait-for-condition operation.
    ///
    /// Thin generic wrapper: the ONLY code monomorphized per call site.
    /// The checkpoint-state-strategy machine lives in the non-generic
    /// [`WfcCore`] (generic over the state type `S` only); this wrapper
    /// just polls the user's concrete check future between its halves, one
    /// cycle per check, and waits for the retry timer between cycles.
    pub(crate) async fn execute(self) -> Result<S, OperationError> {
        let Self {
            ctx,
            op_id,
            name,
            initial_state,
            wait_strategy,
            serdes,
            check,
        } = self;
        let core = WfcCore {
            ctx,
            op_id,
            name,
            initial_state,
            wait_strategy,
            serdes,
        };
        let mut resumed = false;
        loop {
            match core.before(resumed).await? {
                WfcPrelude::Done(result) => return result,
                WfcPrelude::Wait => {}
                WfcPrelude::Run {
                    attempt,
                    state,
                    start,
                    info,
                } => {
                    // The check runs as a `Step` body: an operation claimed
                    // inside it fails the execution with a context usage
                    // error. It holds an activity permit while it runs.
                    let body = UserBody::Step(core.op_id.positional().to_owned());
                    let step_ctx = StepContext::new(attempt);
                    // The check runs inside a `durable_operation` span, as a
                    // live step body does. A live check is never replay.
                    let span = crate::tracing_layer::operation_span(
                        core.ctx.execution_arn(),
                        &core.ctx.lambda_context().request_id,
                        core.ctx.lambda_context().tenant_id.as_deref(),
                        core.op_id.wire(),
                        core.name.as_deref(),
                        attempt,
                        false,
                    );
                    // The check is a user function: its hooks and wrap run
                    // through the runner; the permit stays here. The wire
                    // the end hooks see is the one the FAIL records and the
                    // returned error carries: the type a `TypedError` names,
                    // else `Error`, as #135 records a check failure.
                    let check = &check;
                    let ran = core
                        .ctx
                        .hold_activity(crate::plugin::run_user_function(
                            core.ctx.shared_plugins(),
                            info,
                            body,
                            span,
                            move || (check)(step_ctx, state),
                            |error: &BoxError| crate::error::wire_error_for(&**error, "Error"),
                        ))
                        .await;
                    core.await_start(start).await?;
                    match core.settle(attempt, ran).await? {
                        WfcSettled::Done(value) => return Ok(value),
                        WfcSettled::Retry => {}
                    }
                }
            }
            // The record is PENDING until its retry timer fires. Wait until
            // a checkpoint response carries the next status, then handle
            // the record as a new invocation would: READY runs the next
            // check cycle, a terminal status returns its outcome.
            core.ctx
                .wait_for_record(core.op_id.wire(), |status| {
                    *status != CheckpointStatus::Pending
                })
                .await;
            resumed = true;
        }
    }
}

/// The pre-check half of `wait_for_condition`: task-ownership check, replay
/// resolution, carried-state derivation, and the START checkpoint. Generic
/// only over the state type `S`, no user closure reaches this state
/// machine, so its substantial replay/checkpoint logic compiles once per
/// state type instead of once per call site.
struct WfcCore<S, SD> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    initial_state: S,
    wait_strategy: WaitStrategyFn<S>,
    serdes: SD,
}

/// What [`WfcCore::before`] decided: the operation is already resolved from
/// the checkpoint log, its retry timer has not fired, or one check-attempt
/// cycle must run.
enum WfcPrelude<S> {
    /// Resolved without running the check (replay).
    Done(Result<S, OperationError>),
    /// The record is `PENDING`: the retry timer has not fired yet.
    Wait,
    /// The check must run at `attempt` against the carried `state`.
    Run {
        /// 1-based attempt number for this check cycle.
        attempt: u32,
        /// The carried state the check receives.
        state: S,
        /// The in-flight `START` write, awaited after the check returns and
        /// before the outcome is written. `None` when START is already
        /// recorded. A check that panics drops it, and the write continues.
        start: Option<SubmittedWrite>,
        /// The check's info, as `on_attempt_start` saw it.
        info: std::sync::Arc<crate::plugin::OperationInfo>,
    },
}

/// How [`WfcCore::settle`] ended one check cycle.
enum WfcSettled<S> {
    /// The operation completed with this state.
    Done(S),
    /// A `RETRY` was checkpointed: the next cycle waits for its timer.
    Retry,
}

impl<S, SD> WfcCore<S, SD>
where
    S: Clone + Send + Sync + 'static,
    SD: Serdes<S>,
{
    /// Runs everything that precedes the check closure: replay path, or
    /// the live-path preamble (carried-state derivation + START checkpoint).
    #[expect(clippy::too_many_lines)] // reason: replay-status protocol is a single logical unit; splitting would obscure it
    ///
    /// `resumed` is true when the record changed during this invocation: a
    /// terminal record then returns its outcome without an
    /// `operation_replayed` event.
    async fn before(&self, resumed: bool) -> Result<WfcPrelude<S>, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());

        // 2. Check checkpoint log for replay / resume status. The validated
        // view covers the non-terminal branches without cloning.
        let mut already_started = false;
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Step",
            Some(WFC_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Terminal success: deserialize the final state.
                    // CRITICAL: LOUD error on deserialization failure.
                    // Never fall back to initial_state (Python #574 / JS #754).
                    // Decode FIRST, then emit `operation_replayed`: a corrupt
                    // payload or failing serdes surfaces as an error without
                    // claiming a recorded outcome was returned.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_terminal_success(
                        &self.ctx,
                        self.name.as_deref(),
                        &self.serdes,
                        payload,
                        serdes_ctx.clone(),
                    )
                    .await?;
                    if !resumed {
                        self.ctx.emit_operation_replayed(
                            &wire_id,
                            self.name.as_deref(),
                            "Step",
                            Some(WFC_SUB_TYPE),
                            view.attempt,
                        );
                    }
                    return Ok(WfcPrelude::Done(Ok(value)));
                }
                CheckpointStatus::Failed => {
                    // Terminal failure: reconstruct error from checkpoint.
                    if !resumed {
                        self.ctx.emit_operation_replayed(
                            &wire_id,
                            self.name.as_deref(),
                            "Step",
                            Some(WFC_SUB_TYPE),
                            view.attempt,
                        );
                    }
                    let wire = self
                        .ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Ok(WfcPrelude::Done(Err(replay_terminal_failure(
                        wire,
                        &wire_id,
                        view.attempt,
                    ))));
                }
                CheckpointStatus::Pending => {
                    // Retry timer hasn't fired yet: wait for it.
                    return Ok(WfcPrelude::Wait);
                }
                CheckpointStatus::Started => {
                    // This invocation already checkpointed START: skip it.
                    already_started = true;
                }
                CheckpointStatus::Ready
                | CheckpointStatus::Cancelled
                | CheckpointStatus::TimedOut
                | CheckpointStatus::Stopped => {
                    // Only statusStarted skips START: other statuses
                    // need a fresh StepStarted for the per-attempt protocol.
                }
                CheckpointStatus::Unknown(ref raw) => {
                    // Unreachable in production: `checkpoint_view_validated`
                    // already failed the execution (issue #45). Kept as a
                    // typed arm so a future bypass cannot fall through to a
                    // fresh attempt against a possibly-terminal record.
                    return Err(self.ctx.unrecognized_status_error(&wire_id, raw));
                }
            }
        }

        // 3. Live execution: run one attempt cycle.
        // Derive attempt from checkpoint record.
        let attempt = self.ctx.get_attempt(&self.op_id).saturating_add(1);

        // Determine current state from the checkpoint record BEFORE
        // checkpointing START. The START checkpoint response is merged back
        // into the checkpoint log, and a fresh attempt's START carries no
        // prior result; reading the state after START would overwrite the
        // carried state with initial_state and prevent the condition from
        // ever advancing. The carried state is a property of the prior
        // attempt's checkpoint, independent of this attempt's START.
        let current_state =
            if let Some(payload) = self.ctx.checkpoint_result_payload(&positional_id) {
                // On deserialization failure, surface the error loudly and
                // never silently fall back to initial_state.
                deserialize_state_str(
                    &self.ctx,
                    self.name.as_deref(),
                    &self.serdes,
                    payload,
                    serdes_ctx.clone(),
                )
                .await?
            } else {
                // No recorded state: round-trip the initial state through
                // the serdes, so the first check sees the shape a check
                // that resumes from a checkpoint sees, as JS does. The
                // round trip runs before START, so a failure writes nothing:
                // a marked failure ends the invocation, and any other one
                // returns `SerializationFailed` without running the check.
                let serialized = serialize_state(
                    &self.ctx,
                    self.name.as_deref(),
                    &self.serdes,
                    self.initial_state.clone(),
                    serdes_ctx.clone(),
                )
                .await?;
                deserialize_state_str(
                    &self.ctx,
                    self.name.as_deref(),
                    &self.serdes,
                    serialized,
                    serdes_ctx.clone(),
                )
                .await?
            };

        // A plugin sees the start of every check. It is a replay from the
        // second check on, and on re-entry into a `STARTED` record. The
        // hooks fire while the START write is in flight, as the check does
        // not wait for it either.
        //
        // Checkpoint START if not already started. START is reserved while
        // `on_operation_start` runs, and released when its dispatch returns,
        // so the hook returns before the request is sent, as in JS (see
        // `crate::start`). The check runs while START is in flight; the
        // write is awaited after the check returns and before the outcome
        // write (see `WfcPrelude::Run`).
        let (start, info) = if already_started {
            let info = match self.ctx.plugins() {
                Some(plugins) => {
                    crate::plugin::attempt_started(plugins, self.info(), attempt, true)
                }
                None => std::sync::Arc::new(self.info()),
            };
            (None, info)
        } else {
            let start_update = build_wfc_update(
                &wire_id,
                self.name.as_deref(),
                self.ctx.parent_wire_id(),
                OperationAction::Start,
                None,
                None,
            );
            let reserved = self.ctx.reserve_checkpoint_updates(vec![start_update]);
            let plugins = self.ctx.plugins();
            if let Some(plugins) = plugins {
                let info = self
                    .info()
                    .with_status(crate::OperationStatus::Started)
                    .with_replay(attempt > 1);
                crate::plugin::operation_started(plugins, &info);
            }
            let start = reserved.release();
            // The attempt's info, as `on_attempt_start` saw it, is the one
            // the user-function hooks and wraps of the check receive.
            let info = match plugins {
                Some(plugins) => crate::plugin::attempt_start_only(plugins, self.info(), attempt),
                None => std::sync::Arc::new(self.info()),
            };
            (Some(start), info)
        };

        Ok(WfcPrelude::Run {
            attempt,
            state: current_state,
            start,
            info,
        })
    }

    /// Awaits the `START` write a check cycle submitted before its check
    /// ran, once the check has returned.
    ///
    /// Audit (#43): a failed write fails the invocation, and no outcome is
    /// written. No terminal `FAIL` is needed: the check may run again in
    /// the next invocation, which re-attempts the same write.
    async fn await_start(&self, start: Option<SubmittedWrite>) -> Result<(), OperationError> {
        if let Some(start) = start
            && let Err(err) = start.await
        {
            return self.ctx.checkpoint_failure_unrecoverable(err).await;
        }
        Ok(())
    }

    /// Settles one check cycle: serializes the new state, consults the wait
    /// strategy, and runs the Succeed/Retry/Fail checkpoint protocol.
    #[expect(clippy::too_many_lines)] // reason: the settle protocol (serialize, strategy, checkpoint) is a single logical unit; splitting would obscure it
    async fn settle(
        &self,
        attempt: u32,
        ran: crate::plugin::Ran<S>,
    ) -> Result<WfcSettled<S>, OperationError> {
        let wire_id = self.op_id.wire();
        let serdes_ctx = SerdesContext::new(wire_id, self.ctx.execution_arn());
        match ran.into_parts() {
            Ok(new_state) => {
                // Serialize the new state (ownership transfers to the
                // serdes; the round trip below reconstructs it).
                //
                // A state serialization (or round-trip) failure is a
                // LOCAL, deterministic, user-facing failure, so it stays
                // catchable, but the terminal FAIL is persisted FIRST
                // (issue #43). The check already ran, so its side effects
                // need a recorded outcome: with the FAIL recorded, replay
                // yields it instead of re-running the check.
                let round_trip = async {
                    let serialized = serialize_state(
                        &self.ctx,
                        self.name.as_deref(),
                        &self.serdes,
                        new_state,
                        serdes_ctx.clone(),
                    )
                    .await?;
                    // Round-trip through serdes for consistency.
                    let deserialized: S = deserialize_state_str(
                        &self.ctx,
                        self.name.as_deref(),
                        &self.serdes,
                        serialized.clone(),
                        serdes_ctx,
                    )
                    .await?;
                    Ok::<_, OperationError>((serialized, deserialized))
                };
                let (serialized, deserialized) = match round_trip.await {
                    Ok(pair) => pair,
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        let update = build_wfc_fail_update(
                            wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            &wire,
                        );
                        if let Err(client_err) = self.ctx.checkpoint_updates(vec![update]).await {
                            return self.ctx.checkpoint_failure_unrecoverable(client_err).await;
                        }
                        return Err(op_err
                            .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
                            .with_wire(wire));
                    }
                };

                // Consult the wait strategy.
                let decision = (self.wait_strategy)(deserialized.clone(), attempt);

                match decision {
                    WaitDecision::Complete => {
                        // Condition met: checkpoint terminal SUCCEED.
                        let update = build_wfc_update(
                            wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            OperationAction::Succeed,
                            Some(&serialized),
                            None,
                        );
                        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                            return self.ctx.checkpoint_failure_unrecoverable(err).await;
                        }
                        self.check_ended(
                            attempt,
                            crate::OperationStatus::Succeeded,
                            Some(serialized),
                            None,
                        );

                        Ok(WfcSettled::Done(deserialized))
                    }
                    WaitDecision::Continue { delay } => {
                        // Not met: checkpoint RETRY with state + delay, then wait.
                        #[expect(clippy::cast_possible_truncation)] // reason: delay clamped to i32
                        let delay_secs = (delay.as_secs_f64().ceil() as i64)
                            .clamp(1, i64::from(i32::MAX))
                            as i32;
                        let update = build_wfc_update(
                            wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            OperationAction::Retry,
                            Some(&serialized),
                            Some(delay_secs),
                        );
                        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                            return self.ctx.checkpoint_failure_unrecoverable(err).await;
                        }
                        self.check_ended(
                            attempt,
                            crate::OperationStatus::Pending,
                            Some(serialized),
                            None,
                        );

                        Ok(WfcSettled::Retry)
                    }
                    WaitDecision::Exhausted { reason } => {
                        // Strategy exhaustion: checkpoint FAIL, raise
                        // WaitForConditionError (Python #530 fix). The
                        // record's type is the fixed exhaustion
                        // discriminator so replay reconstructs the same
                        // `MaxChecksExceeded` classification (see
                        // `replay_terminal_failure`).
                        let wire = crate::error::WireError::new(
                            Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE),
                            Some(reason.as_str()),
                        );
                        let update = build_wfc_fail_update(
                            wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            &wire,
                        );
                        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                            return self.ctx.checkpoint_failure_unrecoverable(err).await;
                        }
                        self.check_ended(
                            attempt,
                            crate::OperationStatus::Failed,
                            None,
                            Some(&wire),
                        );

                        Err(wfc_op_error(WaitForConditionErrorKind::MaxChecksExceeded(
                            crate::error::MaxChecksExceeded::new(attempt),
                        ))
                        .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
                        .with_wire(wire))
                    }
                }
            }
            Err((check_err, wire)) => {
                // Check function error: checkpoint FAIL immediately (no retry
                // for check errors).
                // The runner derived the wire record from the check error
                // once, and the end hooks saw it: a `TypedError` in its
                // chain names the type, and an untyped error records
                // `Error`, as JS and a step do.
                let update = build_wfc_fail_update(
                    wire_id,
                    self.name.as_deref(),
                    self.ctx.parent_wire_id(),
                    &wire,
                );
                if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                    return self.ctx.checkpoint_failure_unrecoverable(err).await;
                }
                self.check_ended(attempt, crate::OperationStatus::Failed, None, Some(&wire));

                // The live kind comes from the recorded type, with the
                // function replay uses, so a check error named like an SDK
                // type has the same kind live and on replay.
                Err(wfc_failure_error(wire, wire_id, attempt, check_err))
            }
        }
    }

    /// The plugin info of this operation.
    fn info(&self) -> crate::plugin::OperationInfo {
        crate::plugin::OperationInfo::for_operation(
            &self.ctx,
            self.op_id.wire(),
            crate::OperationKind::Step,
            WFC_SUB_TYPE,
            self.name.as_deref(),
        )
    }

    /// Fires the attempt-end hooks of check `attempt`, right after the
    /// write of `status` returned `Ok`. The operation info carries what the
    /// SDK wrote: the state `result`, or the error `error`. A
    /// check that wrote `RETRY` (`Pending`) ended without an error, so its
    /// attempt succeeded, as in JS, and the operation continues.
    fn check_ended(
        &self,
        attempt: u32,
        status: crate::OperationStatus,
        result: Option<String>,
        error: Option<&crate::error::WireError>,
    ) {
        let Some(plugins) = self.ctx.plugins() else {
            return;
        };
        let outcome = if error.is_some() {
            crate::plugin::AttemptOutcome::Failed
        } else {
            crate::plugin::AttemptOutcome::Succeeded
        };
        crate::plugin::attempt_ended(
            plugins,
            self.info()
                .with_status(status)
                .with_result(result)
                .with_error(error.cloned()),
            crate::plugin::AttemptEnd {
                attempt,
                outcome,
                error: error.cloned(),
                ends_operation: status != crate::OperationStatus::Pending,
            },
        );
    }
}

// Update builders

/// Replays a terminal success from the checkpoint log.
/// CRITICAL: NEVER falls back to `initial_state` (Python #574 / JS #754 fix).
async fn replay_terminal_success<S, SD: Serdes<S>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &SD,
    result: Option<String>,
    serdes_ctx: SerdesContext,
) -> Result<S, OperationError> {
    let payload = result.unwrap_or_else(|| "null".to_owned());
    deserialize_state_str(ctx, name, serdes, payload, serdes_ctx).await
}

/// Replays a terminal failure from the checkpoint log.
///
/// The recorded failure fields travel on the synthetic source rather
/// than being folded into a message, so `kind()` stays meaningful after
/// a replay.
///
/// The wire `error_type` discriminates the classification so replay
/// reproduces what the live path yielded after persisting the record
/// (issue #43):
///
/// - [`crate::error::SERIALIZATION_FAILED_ERROR_TYPE`] reconstructs
///   `WaitForConditionErrorKind::SerializationFailed`.
/// - [`crate::error::WFC_EXHAUSTED_ERROR_TYPE`] reconstructs
///   `WaitForConditionErrorKind::MaxChecksExceeded`, deriving the check
///   count from `recorded_attempt`, the checkpoint record's completed
///   retry count, exactly as the live path derived its 1-based attempt
///   number (`recorded + 1`), so `MaxChecksExceeded::checks()` reports
///   the same value live and replayed. Like the live exhaustion error,
///   the replayed one carries no foreign source: the kind is the whole
///   story, and the recorded reason stays readable on the attached wire
///   record.
/// - Anything else is a check failure.
///
/// The live check-failure path classifies its record with the same
/// function ([`classify_wfc_failure`]).
fn replay_terminal_failure(
    wire: crate::error::WireError,
    wire_id: &str,
    recorded_attempt: u32,
) -> OperationError {
    let source = crate::error::ReplayedFailure::source_from(wire.clone());
    wfc_failure_error(wire, wire_id, recorded_attempt.saturating_add(1), source)
}

/// The error kind a `wait_for_condition` FAIL record yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WfcFailureClass {
    /// The wait strategy ran out of checks.
    MaxChecksExceeded,
    /// The state failed to serialize or deserialize.
    SerializationFailed,
    /// The check returned an error.
    CheckFailed,
}

/// Classifies a FAIL record by its recorded error type. Replay and the live
/// check-failure path both use it, so they yield the same kind.
fn classify_wfc_failure(wire_type: Option<&str>) -> WfcFailureClass {
    match wire_type {
        Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE) => WfcFailureClass::MaxChecksExceeded,
        Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE) => WfcFailureClass::SerializationFailed,
        _ => WfcFailureClass::CheckFailed,
    }
}

/// Builds the error a FAIL record `wire` yields, classified by
/// [`classify_wfc_failure`]. `checks` is the 1-based number of the check
/// that wrote the record. A `MaxChecksExceeded` error carries no source:
/// the kind is the whole story, and the recorded reason stays readable on
/// the attached wire record. Every other kind carries `source`.
fn wfc_failure_error(
    wire: crate::error::WireError,
    wire_id: &str,
    checks: u32,
    source: crate::error::Source,
) -> OperationError {
    let err = match classify_wfc_failure(wire.error_type()) {
        WfcFailureClass::MaxChecksExceeded => {
            wfc_op_error(WaitForConditionErrorKind::MaxChecksExceeded(
                crate::error::MaxChecksExceeded::new(checks),
            ))
        }
        WfcFailureClass::SerializationFailed => {
            wfc_op_error_with_source(WaitForConditionErrorKind::SerializationFailed, Some(source))
        }
        WfcFailureClass::CheckFailed => {
            wfc_op_error_with_source(WaitForConditionErrorKind::CheckFailed, Some(source))
        }
    };
    err.with_operation(wire_id, CheckpointStatus::Failed.wire_str())
        .with_wire(wire)
}

/// Serializes state through the configured serdes (ownership transfers;
/// the serdes decides where its work runs).
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `SerializationFailed`.
async fn serialize_state<S, SD: Serdes<S>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &SD,
    value: S,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx.hold_activity(serdes.serialize(value, serdes_ctx)).await {
        Ok(serialized) => Ok(serialized),
        Err(err) => {
            let e = ctx
                .route_serdes_error("serialize", &operation_id, name, err)
                .await;
            Err(wfc_op_error_with_source(
                WaitForConditionErrorKind::SerializationFailed,
                Some(e),
            ))
        }
    }
}

/// Deserializes state from a string payload.
/// LOUD error on failure, never silently resets (Python #574 fix).
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `SerializationFailed`.
async fn deserialize_state_str<S, SD: Serdes<S>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &SD,
    payload: String,
    serdes_ctx: SerdesContext,
) -> Result<S, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx
        .hold_activity(serdes.deserialize(payload, serdes_ctx))
        .await
    {
        Ok(value) => Ok(value),
        Err(err) => {
            let e = ctx
                .route_serdes_error("deserialize", &operation_id, name, err)
                .await;
            Err(wfc_op_error_with_source(
                WaitForConditionErrorKind::SerializationFailed,
                Some(crate::error::ContextualError::source_from(
                    "state deserialization failed",
                    e,
                )),
            ))
        }
    }
}

// OperationUpdate builders

fn build_wfc_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    action: OperationAction,
    payload: Option<&str>,
    delay_secs: Option<i32>,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(WFC_SUB_TYPE)
        .action(action);

    if let Some(n) = name {
        builder = builder.name(n);
    }
    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }
    if let Some(p) = payload {
        builder = builder.payload(p);
    }
    if let Some(d) = delay_secs {
        builder = builder.step_options(
            aws_sdk_lambda::types::StepOptions::builder()
                .next_attempt_delay_seconds(d)
                .build(),
        );
    }

    #[expect(clippy::expect_used)] // reason: all required fields (id, type, action) are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_wfc_fail_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    error: &crate::error::WireError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(WFC_SUB_TYPE)
        .action(OperationAction::Fail)
        .error(error.to_error_object());

    if let Some(n) = name {
        builder = builder.name(n);
    }
    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[expect(clippy::expect_used)] // reason: all required fields (id, type, action) are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

// Error helpers

fn wfc_op_error(kind: WaitForConditionErrorKind) -> OperationError {
    wfc_op_error_with_source(kind, None)
}

fn wfc_op_error_with_source(
    kind: WaitForConditionErrorKind,
    source: Option<crate::error::Source>,
) -> OperationError {
    OperationError::from_kind(OperationErrorKind::WaitForCondition(
        WaitForConditionError::new(kind, source),
    ))
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions with descriptive messages
mod tests {
    use super::*;
    use crate::client::InMemoryExecutionClient;
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord};
    use std::sync::Arc;

    fn make_ctx_with_client() -> DurableContext {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        )
    }

    fn make_ctx_with_log(records: Vec<(String, CheckpointRecord)>) -> DurableContext {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::from_records(records));
        DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        )
    }

    /// REGRESSION TEST: carried state must survive the START checkpoint on
    /// re-invoke; the START-response merge must not reset it to `initial_state`.
    ///
    /// On re-invocation after a `Continue` (RETRY), the checkpoint log carries
    /// the prior attempt's state. The engine checkpoints a fresh START for the
    /// new attempt, and the checkpoint response is merged back into the log. A
    /// fresh attempt's START response carries no prior result, so reading the
    /// current state AFTER the START checkpoint would overwrite the carried
    /// state with `initial_state`. That reset stalls the condition forever
    /// (the check re-runs from initial every attempt and never advances), which
    /// manifests as an unbounded ~1/second retry loop that never completes.
    ///
    /// The check function records the state it receives; it MUST be the carried
    /// state (1), never `initial_state` (0).
    #[tokio::test]
    async fn regression_carried_state_survives_start_checkpoint_merge() {
        use aws_sdk_lambda::types::{Operation, OperationStatus, OperationType, StepDetails};
        use std::sync::Mutex as StdMutex;

        // Re-invoke: prior attempt checkpointed RETRY with state "1"; the
        // retry timer has fired so the record arrives with status Ready.
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Ready,
            result: Some("1".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 1,
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
        };

        // The START checkpoint response echoes the operation as a fresh
        // STARTED attempt with NO result: the shape that clobbers the
        // carried state when merged back into the log.
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let start_response_op = Operation::builder()
            .id(wire_key.clone())
            .r#type(OperationType::Step)
            .status(OperationStatus::Started)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
            .step_details(StepDetails::builder().attempt(1).build())
            .build()
            .expect("all required Operation fields set");
        client.enqueue_checkpoint_response(crate::client::TestResponse::Success(vec![
            start_response_op,
        ]));

        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id(); // mints "1"

        let seen = Arc::new(StdMutex::new(None::<i32>));
        let seen_check = Arc::clone(&seen);

        let signal = Arc::clone(ctx.coordinator());
        let ownership = Arc::clone(ctx.task_ownership());
        let root = ctx.clone();
        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state: 0i32,
            wait_strategy: Box::new(|state: i32, _attempt| {
                if state >= 3 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            }),
            serdes: crate::serdes::JsonSerdes,
            check: move |_ctx, state: i32| {
                *seen_check
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(state);
                async move { Ok::<i32, BoxError>(state + 1) }
            },
        };

        // Continues (state 1 -> 2, still < 3) and suspends (parks). Drive
        // through the driver so it terminates as Pending; the point of the
        // test is the state the check FN observed, not the result.
        let outcome =
            crate::driver::test_support::outcome_of(root, ownership, signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        let observed = seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .expect("check function must run on the Ready re-invoke path");
        assert_eq!(
            observed, 1,
            "check must receive the carried state (1) from the pre-START \
             checkpoint record, not initial_state (0); a START-merge reset \
             would stall the condition in an unbounded retry loop"
        );
    }

    /// ISSUE #46 REPRODUCER: a failed Retry checkpoint must not change
    /// state referenced by the previously accepted checkpoint.
    ///
    /// The prior attempt's accepted checkpoint references a
    /// `FileSystemSerdes` payload file. The new attempt serializes fresh
    /// state and its Retry checkpoint FAILS. Replaying the accepted
    /// envelope must still read the original bytes: the failed attempt's
    /// write must have gone to a new file, never over the committed one.
    #[tokio::test]
    async fn failed_retry_checkpoint_does_not_mutate_prior_filesystem_state() {
        use crate::client::TestResponse;
        use crate::serdes::{FileSystemSerdes, Serdes, SerdesContext};
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp = std::env::temp_dir().join(format!(
            "wfc_failed_retry_checkpoint_{}_{}",
            std::process::id(),
            unique
        ));
        let filesystem_serdes = FileSystemSerdes::new(tmp.to_string_lossy().into_owned());

        let wire_key = crate::engine::compute_wire_id_public("1");
        let serdes_ctx = SerdesContext::new(&wire_key, "arn:test");

        // Model an accepted Retry checkpoint from the prior attempt. Keep its
        // exact envelope so the assertion below observes what replay would see.
        let accepted_envelope = filesystem_serdes
            .serialize(1_i32, serdes_ctx.clone())
            .await
            .unwrap();
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Ready,
            result: Some(accepted_envelope.clone()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 1,
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
        };

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::Success(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError(
            "injected Retry checkpoint failure".to_owned(),
        ));

        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state: 0i32,
            wait_strategy: Box::new(|_state: i32, _attempt| {
                WaitDecision::continue_with(Duration::from_secs(1))
            }),
            serdes: filesystem_serdes.clone(),
            check: |_ctx, state: i32| async move { Ok::<i32, BoxError>(state + 1) },
        };

        // The injected RETRY-write rejection routes through the
        // unrecoverable path (issue #43): the operation future parks for
        // the invocation driver instead of yielding a catchable error, and
        // the fatal slot records the checkpoint failure.
        let signal = Arc::clone(exec.ctx.coordinator());
        let parked = tokio::time::timeout(Duration::from_millis(200), exec.execute()).await;
        assert!(
            parked.is_err(),
            "a rejected RETRY checkpoint must park the future for the \
             driver, not resolve"
        );
        let fatal = signal
            .fatal_error()
            .expect("the checkpoint failure records an execution-fatal error");
        assert!(
            fatal
                .error_message
                .contains("injected Retry checkpoint failure"),
            "unexpected fatal: {fatal:?}"
        );

        let restored: i32 = filesystem_serdes
            .deserialize(accepted_envelope, serdes_ctx)
            .await
            .unwrap();
        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            restored, 1,
            "a failed Retry checkpoint must not mutate the state referenced \
             by the prior accepted checkpoint"
        );
    }

    /// Test: condition met on first try (immediate completion).
    #[tokio::test]
    async fn condition_met_first_try() {
        let ctx = make_ctx_with_client();
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: Some("immediate".to_owned()),
            initial_state: 10i32,
            wait_strategy: Box::new(|state: i32, _attempt| {
                if state >= 5 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            }),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, state: i32| async move { Ok::<i32, BoxError>(state) },
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 10);
    }

    /// Test: condition met after N attempts (state evolution).
    #[tokio::test]
    async fn condition_met_after_n_state_evolution() {
        // Simulate being re-invoked after one continue: the checkpoint log
        // has a Ready record with result "1" and attempt 1.
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Ready,
            result: Some("1".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 1,
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
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id(); // mints "1"
        let signal = Arc::clone(ctx.coordinator());
        let ownership = Arc::clone(ctx.task_ownership());
        let root = ctx.clone();

        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state: 0i32,
            wait_strategy: Box::new(|state: i32, _attempt| {
                if state >= 3 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            }),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, state: i32| async move { Ok::<i32, BoxError>(state + 1) },
        };

        // State was 1 (from checkpoint), check fn produces 2, strategy says
        // continue (< 3), so it checkpoints RETRY and suspends (parks) rather
        // than surfacing a fabricated error to the caller.
        let outcome =
            crate::driver::test_support::outcome_of(root, ownership, signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    /// Test: wait strategy exhaustion produces `WaitForConditionError`.
    #[tokio::test]
    async fn exhaustion_produces_error() {
        let ctx = make_ctx_with_client();
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state: 0i32,
            wait_strategy: Box::new(|_state: i32, attempt| {
                // Always exhaust on attempt 1.
                if attempt >= 1 {
                    WaitDecision::exhausted("max attempts exceeded")
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            }),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, state: i32| async move { Ok::<i32, BoxError>(state + 1) },
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err.kind() {
            OperationErrorKind::WaitForCondition(wfc_err) => match wfc_err.kind() {
                WaitForConditionErrorKind::MaxChecksExceeded(details) => {
                    assert_eq!(details.checks(), 1);
                }
                other => panic!("expected MaxChecksExceeded, got: {other:?}"),
            },
            other => panic!("expected WaitForCondition error, got: {other:?}"),
        }
        // The SDK builds the exhaustion record with `WireError::new`.
        // `WireError::new` carries no stack frames. So the recorded
        // stack trace is empty.
        let wire = err.wire().unwrap();
        assert_eq!(
            wire.error_type(),
            Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE)
        );
        assert_eq!(wire.error_message(), Some("max attempts exceeded"));
        assert!(
            wire.stack_trace().is_empty(),
            "the SDK captures no stack trace for an exhaustion"
        );
    }

    /// REGRESSION TEST: corrupt checkpointed state → loud `WaitForConditionError`.
    ///
    /// When checkpointed state fails to deserialize, the SDK surfaces
    /// `WaitForConditionError::SerializationFailed`. It never silently
    /// resets to `initial_state`.
    #[tokio::test]
    async fn regression_python574_corrupt_state_loud_error_not_silent_reset() {
        // Simulate a checkpoint with corrupt/incompatible state data.
        // The state type is i32 but the checkpointed result is "not-a-number".
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Succeeded,
            result: Some("\"not-a-number\"".to_owned()),
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
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: Some("regression".to_owned()),
            initial_state: 0,
            wait_strategy: Box::new(|_state: i32, _attempt| WaitDecision::complete()),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, _state: i32| async move {
                #[expect(unreachable_code)] // reason: type anchor for the diverging body
                {
                    panic!("check must NOT execute during replay");
                    Ok::<i32, BoxError>(0)
                }
            },
        };

        let result = exec.execute().await;
        // MUST be an error, never silently returns initial_state (0).
        assert!(
            result.is_err(),
            "corrupt state MUST produce an error, not silently reset to initial_state"
        );
        let err = result.unwrap_err();
        match err.kind() {
            OperationErrorKind::WaitForCondition(wfc_err) => match wfc_err.kind() {
                WaitForConditionErrorKind::SerializationFailed => {
                    let message = crate::error::chain_string(wfc_err);
                    assert!(
                        message.contains("state deserialization failed"),
                        "error chain should indicate state deserialization failure: {message}"
                    );
                }
                other => panic!("expected SerializationFailed, got: {other:?}"),
            },
            other => panic!("expected WaitForCondition error, got: {other:?}"),
        }
    }

    /// A root context over `records` whose transport is a fresh in-memory
    /// client, with the recorder's `p0` and `p1` as the invocation's
    /// plugins and no checkpoint buffer window.
    fn uf_ctx(
        hooks: &crate::plugin::recorder::UfHooks,
        records: Vec<(String, CheckpointRecord)>,
    ) -> (DurableContext, Arc<InMemoryExecutionClient>) {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
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

    /// The error object of the `FAIL` update `client` recorded, as
    /// `(type, message)`.
    fn recorded_fail(client: &InMemoryExecutionClient) -> (Option<String>, Option<String>) {
        let updates = client.recorded_updates();
        let fail = updates
            .iter()
            .find(|u| u.action() == &OperationAction::Fail)
            .expect("a FAIL update");
        let error = fail.error().expect("the FAIL carries an error");
        (
            error.error_type().map(str::to_owned),
            error.error_message().map(str::to_owned),
        )
    }

    /// The user-function records of one live check, in `p0`'s order.
    const LIVE_CHECK: [&str; 6] = [
        "attempt_start",
        "uf_start",
        "wrap_enter",
        "wrap_exit",
        "uf_end",
        "attempt_end",
    ];

    /// Rule 11's check row. A check that returns `Err` ends with the wire
    /// the FAIL stores. A check
    /// exhausted by `max_attempts(2)` ends `Ok` at both checks, and the
    /// FAIL records `WaitForCondition.MaxChecksExceeded`. The in-memory
    /// client returns no record, so the end infos carry what the SDK
    /// wrote.
    #[tokio::test]
    async fn condition_check_error_and_exhaustion() {
        use crate::builders::wait_for_condition::WaitStrategy;
        use crate::plugin::recorder::UfHooks;

        // Run 1: the check returns `Err`.
        let hooks = UfHooks::default();
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        let err = ctx
            .wait_for_condition(
                |_ctx, _state: u32| async { Err::<u32, BoxError>("bad".into()) },
                0_u32,
            )
            .wait_strategy_fn(|_state, _attempt| WaitDecision::complete())
            .name("cond")
            .await
            .unwrap_err();
        assert!(err.wire().is_some());
        let mut expected = vec!["start"];
        expected.extend(LIVE_CHECK);
        expected.push("end");
        assert_eq!(hooks.hooks_of("cond"), expected);
        let end = hooks.find("cond", "uf_end");
        assert_eq!(end.ok, Some(false));
        let wire = end.wire.expect("an Err outcome");
        // The check's wire is the one the FAIL records and the returned
        // error carries: `wire_error_for(err, "Error")`, the record #135
        // set, so an untyped error records `Error`.
        assert_eq!(wire.error_type(), Some("Error"));
        assert_eq!(wire.error_message(), Some("bad"));
        assert_eq!(err.wire(), Some(&wire));
        assert_eq!(
            recorded_fail(&client),
            (
                wire.error_type().map(str::to_owned),
                wire.error_message().map(str::to_owned)
            )
        );
        let attempt_end = hooks.find("cond", "attempt_end");
        assert_eq!(
            attempt_end
                .wire
                .as_ref()
                .and_then(crate::error::WireError::error_message),
            Some("bad")
        );
        assert_eq!(attempt_end.status, Some(crate::OperationStatus::Failed));
        assert_eq!(attempt_end.result, None);

        // Run 2: `max_attempts(2)`, never done. Check 1 ends `Ok` and
        // retries; check 2, in the next invocation, ends `Ok` and exhausts.
        let strategy = || {
            WaitStrategy::builder(|_state: &u32| false)
                .max_attempts(2)
                .build()
        };
        let hooks = UfHooks::default();
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        let cond = ctx
            .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0_u32)
            .wait_strategy(strategy())
            .name("cond");
        let signal = Arc::clone(ctx.coordinator());
        let ownership = Arc::clone(ctx.task_ownership());
        let outcome =
            crate::driver::test_support::outcome_of(ctx.clone(), ownership, signal, cond).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        let mut expected = vec!["start"];
        expected.extend(LIVE_CHECK);
        assert_eq!(hooks.hooks_of("cond"), expected);
        assert_eq!(hooks.find("cond", "uf_end").ok, Some(true));
        assert!(
            client
                .recorded_updates()
                .iter()
                .any(|u| u.action() == &OperationAction::Retry)
        );

        let hooks = UfHooks::default();
        let record = wfc_record(CheckpointStatus::Ready, Some("1"), None, 1);
        let (ctx, client) = uf_ctx(&hooks, vec![record]);
        let err = ctx
            .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0_u32)
            .wait_strategy(strategy())
            .name("cond")
            .await
            .unwrap_err();
        assert!(err.wire().is_some());
        let mut expected = vec!["start"];
        expected.extend(LIVE_CHECK);
        expected.push("end");
        assert_eq!(hooks.hooks_of("cond"), expected);
        let end = hooks.find("cond", "uf_end");
        assert_eq!((end.ok, end.info_attempt), (Some(true), Some(2)));
        let (error_type, _) = recorded_fail(&client);
        assert_eq!(
            error_type.as_deref(),
            Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE)
        );
    }

    /// Rule 11's check row with a `TypedError`: the type it names is the one
    /// the end hook, the FAIL and the returned error all carry.
    #[tokio::test]
    async fn condition_check_typed_error_reaches_the_end_hook() {
        use crate::plugin::recorder::UfHooks;

        let hooks = UfHooks::default();
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        let err = ctx
            .wait_for_condition(
                |_ctx, _state: u32| async {
                    Err::<u32, BoxError>(Box::new(crate::TypedError::with_type(
                        "Throttled",
                        "slow down",
                    )))
                },
                0_u32,
            )
            .wait_strategy_fn(|_state, _attempt| WaitDecision::complete())
            .name("cond")
            .await
            .unwrap_err();
        let wire = hooks.find("cond", "uf_end").wire.expect("an Err outcome");
        assert_eq!(wire.error_type(), Some("Throttled"));
        assert_eq!(err.wire(), Some(&wire));
        assert_eq!(
            recorded_fail(&client),
            (
                Some("Throttled".to_owned()),
                wire.error_message().map(str::to_owned)
            )
        );
    }

    /// A check is a user function: one start hook and one end hook, for
    /// attempt 1 with `Ok`, before the SUCCEED is written.
    #[tokio::test]
    async fn condition_check_is_a_user_function() {
        let client_cell: Arc<std::sync::OnceLock<Arc<InMemoryExecutionClient>>> = Arc::default();
        let probe_client = Arc::clone(&client_cell);
        let hooks = crate::plugin::recorder::UfHooks::probing(move || {
            let succeeded = probe_client.get().is_some_and(|client| {
                client
                    .recorded_updates()
                    .iter()
                    .any(|u| u.action() == &OperationAction::Succeed)
            });
            succeeded.to_string()
        });
        let (ctx, client) = uf_ctx(&hooks, Vec::new());
        client_cell.set(Arc::clone(&client)).expect("set once");
        let state = ctx
            .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0_u32)
            .wait_strategy_fn(|state: u32, _attempt| {
                if state >= 1 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })
            .name("cond")
            .await
            .unwrap();
        assert_eq!(state, 1);
        let records = hooks.of("cond");
        let starts: Vec<_> = records.iter().filter(|r| r.hook == "uf_start").collect();
        let ends: Vec<_> = records.iter().filter(|r| r.hook == "uf_end").collect();
        assert_eq!((starts.len(), ends.len()), (1, 1), "{records:?}");
        let end = ends.first().expect("one end");
        assert_eq!((end.info_attempt, end.ok), (Some(1), Some(true)));
        assert_eq!(end.probe, "false", "the end hook ran before the SUCCEED");
        assert!(
            client
                .recorded_updates()
                .iter()
                .any(|u| u.action() == &OperationAction::Succeed)
        );
    }

    /// Test: check function error immediately fails the operation.
    #[tokio::test]
    async fn check_function_error_fails_immediately() {
        let ctx = make_ctx_with_client();
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: Box::new(|_state: i32, _attempt| WaitDecision::complete()),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, _state: i32| async {
                Err::<i32, BoxError>("check function failed".into())
            },
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(err_msg.contains("check function failed"), "got: {err_msg}");
    }

    /// Test: replay frozen success returns stored result.
    #[tokio::test]
    async fn replay_frozen_success() {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Succeeded,
            result: Some("42".to_owned()),
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 3,
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
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, _state: i32| async move {
                #[expect(unreachable_code)] // reason: type anchor for the diverging body
                {
                    panic!("check must NOT execute during replay");
                    Ok::<i32, BoxError>(0)
                }
            },
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 42);
    }

    /// Test: replay frozen failure returns error.
    #[tokio::test]
    async fn replay_frozen_failure() {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Failed,
            result: None,
            error_type: Some("WaitForConditionError".to_owned()),
            error_message: Some("max attempts exceeded".to_owned()),
            error_data: None,
            stack_trace: None,
            attempt: 3,
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
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, _state: i32| async move {
                #[expect(unreachable_code)] // reason: type anchor for the diverging body
                {
                    panic!("check must NOT execute during replay");
                    Ok::<i32, BoxError>(0)
                }
            },
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(err_msg.contains("max attempts exceeded"), "got: {err_msg}");
    }

    /// REGRESSION TEST: a replayed exhaustion record keeps its
    /// `MaxChecksExceeded` classification.
    ///
    /// The live path persists the fixed exhaustion discriminator on the
    /// terminal FAIL; replay must reconstruct the same kind, not the
    /// generic `CheckFailed`, with `checks()` derived from the record's
    /// completed-retry count (`attempt + 1`), matching what the live path
    /// reported.
    #[tokio::test]
    async fn replay_exhaustion_keeps_max_checks_exceeded_classification() {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Failed,
            result: None,
            error_type: Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE.to_owned()),
            error_message: Some("max checks exceeded: 3".to_owned()),
            error_data: None,
            stack_trace: None,
            // Two completed retries: the outcome came from attempt 3.
            attempt: 2,
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
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, _state: i32| async move {
                #[expect(unreachable_code)] // reason: type anchor for the diverging body
                {
                    panic!("check must NOT execute during replay");
                    Ok::<i32, BoxError>(0)
                }
            },
        };

        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::WaitForCondition(wfc_err) => match wfc_err.kind() {
                WaitForConditionErrorKind::MaxChecksExceeded(details) => {
                    assert_eq!(
                        details.checks(),
                        3,
                        "replay must derive checks() from the recorded attempt count"
                    );
                }
                other => panic!("expected MaxChecksExceeded on replay, got: {other:?}"),
            },
            other => panic!("expected WaitForCondition error, got: {other:?}"),
        }
        // The recorded reason stays readable on the attached wire record.
        let wire = err.wire().unwrap();
        assert_eq!(
            wire.error_type(),
            Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE)
        );
        assert_eq!(wire.error_message(), Some("max checks exceeded: 3"));
    }

    /// Test: spawn delegates to blessed task.
    #[tokio::test]
    async fn spawn_executes_on_blessed_task() {
        let ctx = make_ctx_with_client();

        let handle = ctx
            .wait_for_condition(|_ctx, state: i32| async move { Ok(state) }, 5)
            .wait_strategy_fn(|_state: i32, _attempt| WaitDecision::complete())
            .spawn();

        let result = handle.await;
        assert_eq!(result.unwrap(), 5);
    }

    // Initial-state round trip and check-failure classification.

    /// A `Serdes<String>` whose `serialize` upper-cases the string and whose
    /// `deserialize` is the identity.
    struct UpperSerdes;

    impl Serdes<String> for UpperSerdes {
        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: String,
            _context: SerdesContext,
        ) -> Result<String, BoxError> {
            Ok(value.to_uppercase())
        }

        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            wire: String,
            _context: SerdesContext,
        ) -> Result<String, BoxError> {
            Ok(wire)
        }
    }

    /// Which half of [`FailingSerdes`] fails, and how.
    #[derive(Clone, Copy)]
    enum Failure {
        Serialize,
        Deserialize,
    }

    /// A `Serdes<String>` that fails `failure`'s half, with a plain error
    /// or, when `retryable`, a `RetryableSerdesError`. The other half is
    /// the identity.
    struct FailingSerdes {
        failure: Failure,
        retryable: bool,
    }

    impl FailingSerdes {
        fn error(&self) -> BoxError {
            if self.retryable {
                crate::serdes::RetryableSerdesError::new("transient").into()
            } else {
                "plain serdes failure".into()
            }
        }
    }

    impl Serdes<String> for FailingSerdes {
        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(
            &self,
            value: String,
            _context: SerdesContext,
        ) -> Result<String, BoxError> {
            match self.failure {
                Failure::Serialize => Err(self.error()),
                Failure::Deserialize => Ok(value),
            }
        }

        // reason: a test serdes that completes at once, in the async-fn form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(
            &self,
            wire: String,
            _context: SerdesContext,
        ) -> Result<String, BoxError> {
            match self.failure {
                Failure::Serialize => Ok(wire),
                Failure::Deserialize => Err(self.error()),
            }
        }
    }

    /// A record of operation "1" with `status`, `result` and `error_type`.
    fn wfc_record(
        status: CheckpointStatus,
        result: Option<&str>,
        error_type: Option<&str>,
        attempt: u32,
    ) -> (String, CheckpointRecord) {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status,
            result: result.map(str::to_owned),
            error_type: error_type.map(str::to_owned),
            error_message: error_type.map(|_| "recorded".to_owned()),
            error_data: None,
            stack_trace: None,
            attempt,
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
        };
        (wire_key, record)
    }

    /// A root context over `records` whose transport is `client`.
    fn ctx_with(
        client: &Arc<InMemoryExecutionClient>,
        records: Vec<(String, CheckpointRecord)>,
    ) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(records)),
            Arc::clone(client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// Runs one `wait_for_condition` whose strategy completes after the
    /// first check, and whose check records the state it receives and
    /// returns it.
    async fn first_check_receives<S, SD>(
        records: Vec<(String, CheckpointRecord)>,
        initial_state: S,
        serdes: SD,
    ) -> S
    where
        S: Clone + Send + Sync + std::fmt::Debug + 'static,
        SD: Serdes<S>,
    {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = ctx_with(&client, records);
        let op_id = ctx.mint_id();
        let seen = Arc::new(std::sync::Mutex::new(None::<S>));
        let seen_check = Arc::clone(&seen);
        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state,
            wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
            serdes,
            check: move |_ctx, state: S| {
                let seen = Arc::clone(&seen_check);
                async move {
                    *seen.lock().unwrap() = Some(state.clone());
                    Ok::<S, BoxError>(state)
                }
            },
        };
        exec.execute().await.expect("the operation succeeds");
        seen.lock().unwrap().clone().expect("the check ran")
    }

    /// C2 and C3: the round trip is the identity for `JsonSerdes`, and a
    /// recorded state wins over the initial state.
    #[tokio::test]
    async fn initial_state_round_trip_cases() {
        // C2: JsonSerdes, 5, no recorded state.
        assert_eq!(
            first_check_receives(Vec::new(), 5_i32, crate::serdes::JsonSerdes).await,
            5
        );
        // C3: UpperSerdes, "a", recorded state "B".
        let record = wfc_record(CheckpointStatus::Ready, Some("B"), None, 1);
        assert_eq!(
            first_check_receives(vec![record], "a".to_owned(), UpperSerdes).await,
            "B"
        );
        // The C3 control: with no recorded state, UpperSerdes shows "A".
        assert_eq!(
            first_check_receives(Vec::new(), "a".to_owned(), UpperSerdes).await,
            "A"
        );
    }

    /// An execution of initial state "a" over `serdes` whose check sets
    /// `called`, with its context and transport.
    fn flagged_exec(
        serdes: FailingSerdes,
        called: &Arc<std::sync::atomic::AtomicBool>,
    ) -> (
        DurableContext,
        Arc<InMemoryExecutionClient>,
        impl Future<Output = Result<String, OperationError>> + Send + 'static,
    ) {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = ctx_with(&client, Vec::new());
        let op_id = ctx.mint_id();
        let called = Arc::clone(called);
        let exec = WaitForConditionExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            initial_state: "a".to_owned(),
            wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
            serdes,
            check: move |_ctx, state: String| {
                called.store(true, std::sync::atomic::Ordering::SeqCst);
                async move { Ok::<String, BoxError>(state) }
            },
        };
        (ctx, client, exec.execute())
    }

    /// C4 and C4b: an unmarked serialize or deserialize failure of the
    /// initial state returns `SerializationFailed`, does not call the check
    /// and writes no update.
    #[tokio::test]
    async fn initial_state_round_trip_failure_writes_nothing() {
        for failure in [Failure::Serialize, Failure::Deserialize] {
            let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let serdes = FailingSerdes {
                failure,
                retryable: false,
            };
            let (_ctx, client, fut) = flagged_exec(serdes, &called);
            let err = fut.await.unwrap_err();
            match err.kind() {
                OperationErrorKind::WaitForCondition(wfc_err) => assert!(
                    matches!(
                        wfc_err.kind(),
                        WaitForConditionErrorKind::SerializationFailed
                    ),
                    "expected SerializationFailed, got {:?}",
                    wfc_err.kind()
                ),
                other => panic!("expected WaitForCondition error, got: {other:?}"),
            }
            assert!(!called.load(std::sync::atomic::Ordering::SeqCst));
            assert!(
                client.recorded_updates().is_empty(),
                "no update is written: {:?}",
                client.recorded_updates()
            );
        }
    }

    /// C4a and C4c: a marked serialize or deserialize failure of the
    /// initial state ends the invocation, does not call the check and
    /// writes no update.
    #[tokio::test]
    async fn initial_state_retryable_round_trip_failure_ends_invocation() {
        for (failure, needle) in [
            (Failure::Serialize, "serdes serialize failed"),
            (Failure::Deserialize, "serdes deserialize failed"),
        ] {
            let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let serdes = FailingSerdes {
                failure,
                retryable: true,
            };
            let (ctx, client, fut) = flagged_exec(serdes, &called);
            crate::driver::test_support::expect_serdes_fault(
                ctx.clone(),
                Arc::clone(ctx.task_ownership()),
                Arc::clone(ctx.coordinator()),
                fut,
                needle,
            )
            .await;
            assert!(!called.load(std::sync::atomic::Ordering::SeqCst));
            assert!(
                client.recorded_updates().is_empty(),
                "no update is written: {:?}",
                client.recorded_updates()
            );
        }
    }

    /// The kind of a `wait_for_condition` error, with the `checks()` of a
    /// `MaxChecksExceeded`.
    fn wfc_kind(err: &OperationError) -> (&'static str, Option<u32>) {
        let OperationErrorKind::WaitForCondition(wfc_err) = err.kind() else {
            panic!("expected WaitForCondition error, got: {:?}", err.kind());
        };
        match wfc_err.kind() {
            WaitForConditionErrorKind::MaxChecksExceeded(details) => {
                ("MaxChecksExceeded", Some(details.checks()))
            }
            WaitForConditionErrorKind::SerializationFailed => ("SerializationFailed", None),
            WaitForConditionErrorKind::CheckFailed => ("CheckFailed", None),
        }
    }

    /// The source chain of a `wait_for_condition` error, rendered.
    fn wfc_source(err: &OperationError) -> Option<String> {
        let OperationErrorKind::WaitForCondition(wfc_err) = err.kind() else {
            panic!("expected WaitForCondition error, got: {:?}", err.kind());
        };
        std::error::Error::source(wfc_err).map(crate::error::chain_string)
    }

    /// C5 to C8: a check failure records through `wire_error_for` with the
    /// fallback `Error`, and the live kind equals the replay kind.
    #[tokio::test]
    async fn check_failure_records_typed_error_type() {
        type MakeErr = fn() -> BoxError;
        let cases: [(MakeErr, &str, &str, &str); 4] = [
            // C5: a plain error records `Error`.
            (
                || "check function failed".into(),
                "Error",
                "CheckFailed",
                "check function failed",
            ),
            // C6: a `TypedError` records its type.
            (
                || Box::new(crate::TypedError::with_type("Throttled", "slow down")),
                "Throttled",
                "CheckFailed",
                "slow down",
            ),
            // C7: the exhaustion type gives `MaxChecksExceeded`.
            (
                || {
                    Box::new(crate::TypedError::with_type(
                        "WaitForCondition.MaxChecksExceeded",
                        "x",
                    ))
                },
                "WaitForCondition.MaxChecksExceeded",
                "MaxChecksExceeded",
                "x",
            ),
            // C8: the serialization type gives `SerializationFailed`.
            (
                || Box::new(crate::TypedError::with_type("SerializationError", "x")),
                "SerializationError",
                "SerializationFailed",
                "x",
            ),
        ];
        for (make_err, recorded_type, kind, message) in cases {
            // Live.
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let ctx = ctx_with(&client, Vec::new());
            let op_id = ctx.mint_id();
            let exec = WaitForConditionExecution::<i32, _, _> {
                ctx,
                op_id,
                name: None,
                initial_state: 0,
                wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
                serdes: crate::serdes::JsonSerdes,
                check: move |_ctx, _state: i32| async move { Err::<i32, BoxError>(make_err()) },
            };
            let live = exec.execute().await.unwrap_err();
            let (live_kind, live_checks) = wfc_kind(&live);
            assert_eq!(live_kind, kind, "live kind for {recorded_type}");
            assert_eq!(
                live.wire().and_then(crate::error::WireError::error_type),
                Some(recorded_type)
            );
            let fail = client
                .recorded_updates()
                .into_iter()
                .find(|u| u.action() == &OperationAction::Fail)
                .expect("a FAIL update");
            assert_eq!(
                fail.error().and_then(|e| e.error_type()),
                Some(recorded_type),
                "the FAIL record's type"
            );
            let live_source = wfc_source(&live);
            if kind == "MaxChecksExceeded" {
                assert_eq!(live_source, None, "MaxChecksExceeded carries no source");
            } else {
                let source = live_source.expect("the check error is the source");
                assert!(source.contains(message), "source {source:?}");
            }

            // Replay of the record the live check wrote: first check, so
            // no completed retry.
            let record = wfc_record(CheckpointStatus::Failed, None, Some(recorded_type), 0);
            let ctx = make_ctx_with_log(vec![record]);
            let op_id = ctx.mint_id();
            let exec = WaitForConditionExecution::<i32, _, _> {
                ctx,
                op_id,
                name: None,
                initial_state: 0,
                wait_strategy: Box::new(|_state, _attempt| WaitDecision::complete()),
                serdes: crate::serdes::JsonSerdes,
                check: |_ctx, _state: i32| async move {
                    #[expect(unreachable_code)] // reason: type anchor for the diverging body
                    {
                        panic!("check must NOT execute during replay");
                        Ok::<i32, BoxError>(0)
                    }
                },
            };
            let replayed = exec.execute().await.unwrap_err();
            let (replay_kind, replay_checks) = wfc_kind(&replayed);
            assert_eq!(replay_kind, kind, "replay kind for {recorded_type}");
            assert_eq!(live_checks, replay_checks, "checks() for {recorded_type}");
            assert_eq!(
                replayed
                    .wire()
                    .and_then(crate::error::WireError::error_type),
                Some(recorded_type)
            );
        }
    }

    /// Each recorded type maps to its class, and every other type, or none,
    /// is a check failure.
    #[test]
    fn classify_wfc_failure_maps_each_type() {
        assert_eq!(
            classify_wfc_failure(Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE)),
            WfcFailureClass::MaxChecksExceeded
        );
        assert_eq!(
            classify_wfc_failure(Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE)),
            WfcFailureClass::SerializationFailed
        );
        assert_eq!(
            classify_wfc_failure(Some("Error")),
            WfcFailureClass::CheckFailed
        );
        assert_eq!(classify_wfc_failure(None), WfcFailureClass::CheckFailed);
    }
}

#[cfg(test)]
mod coordinator_tests {
    use crate::BoxError;
    use crate::builders::wait_for_condition::WaitDecision;
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{
        ScriptedClient, drive_handler, named, operation_like, scripted_ctx, wire_err,
    };
    use aws_sdk_lambda::types::{OperationAction, OperationStatus, StepDetails};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn wfc_resumed_in_invocation_fires_start() {
        use crate::driver::test_support::{RefreshClient, poll_once};
        use crate::plugin::AttemptOutcome::Succeeded;
        // Answers the RETRY of `cond` with PENDING and the state it
        // carries, and the first refresh with `cond` READY. The SUCCEED
        // gets no record.
        let client = Arc::new(RefreshClient::new(|req| {
            let record =
                |status: OperationStatus, update: &aws_sdk_lambda::types::OperationUpdate| {
                    let mut details = StepDetails::builder().attempt(1);
                    if let Some(payload) = update.payload() {
                        details = details.result(payload);
                    }
                    operation_like(update, status)
                        .step_details(details.build())
                        .build()
                        .expect("all required fields set")
                };
            let mut ops = Vec::new();
            if let Some(retry) = named(req.updates, "cond", &OperationAction::Retry) {
                ops.push(record(OperationStatus::Pending, retry));
            }
            if req.empty_number == 1
                && let Some(retry) = named(req.history, "cond", &OperationAction::Retry)
            {
                ops.push(record(OperationStatus::Ready, retry));
            }
            Ok(ops)
        }));
        let hooks = crate::plugin::test_support::OpHooks::default();
        let ctx = crate::DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            None,
            0,
            Some(hooks.plugins()),
        );
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let mut cond = std::pin::pin!(
                ctx.wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0_u32)
                    .wait_strategy_fn(|state: u32, _attempt| {
                        if state >= 2 {
                            WaitDecision::complete()
                        } else {
                            WaitDecision::continue_with(Duration::from_secs(1))
                        }
                    })
                    .name("cond")
                    .spawn()
            );
            poll_once(cond.as_mut()).await;
            ctx.step(|_| async {
                tokio::time::sleep(Duration::from_secs(3)).await;
                Ok(1_u32)
            })
            .name("s")
            .await
            .map_err(wire_err)?;
            let state = cond.await.map_err(wire_err)?;
            Ok(state.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("2".to_owned()));
        let records = hooks.of("cond");
        let shapes: Vec<_> = records
            .iter()
            .map(crate::plugin::test_support::OpRecord::shape)
            .collect();
        assert_eq!(
            shapes,
            [
                ("start", false, None, None),
                ("attempt_start", false, Some(1), None),
                ("attempt_end", false, Some(1), Some(Succeeded)),
                ("start", true, None, None),
                ("attempt_start", false, Some(2), None),
                ("attempt_end", false, Some(2), Some(Succeeded)),
                ("end", false, None, None),
            ]
        );
        // `on_operation_start` keeps main's info, the record's attempt;
        // only the attempt's info carries the current attempt.
        let starts = records.iter().filter(|r| r.hook.ends_with("start"));
        let info_attempts: Vec<_> = starts.map(|r| r.info_attempt).collect();
        assert_eq!(info_attempts, [None, Some(1), Some(1), Some(2)]);
        // The record carries the state 1; the SDK wrote 2.
        let results: Vec<_> = records.iter().map(|r| r.result.as_deref()).collect();
        assert_eq!(results.get(5..), Some(&[Some("2"), Some("2")][..]));
        let statuses: Vec<_> = records.iter().map(|r| r.status).collect();
        assert_eq!(
            statuses,
            [
                Some(crate::OperationStatus::Started),
                Some(crate::OperationStatus::Started),
                Some(crate::OperationStatus::Pending),
                Some(crate::OperationStatus::Started),
                Some(crate::OperationStatus::Started),
                Some(crate::OperationStatus::Succeeded),
                Some(crate::OperationStatus::Succeeded),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn condition_check_resumes_on_sibling_response() {
        let (events, _guard) = crate::driver::test_support::EventMessages::install();
        let client = Arc::new(ScriptedClient::new(|updates, history| {
            if named(updates, "s", &OperationAction::Succeed).is_none() {
                return Vec::new();
            }
            named(history, "cond", &OperationAction::Start)
                .map(|start| {
                    operation_like(start, OperationStatus::Ready)
                        .step_details(StepDetails::builder().attempt(1).result("1").build())
                        .build()
                        .expect("all required fields set")
                })
                .into_iter()
                .collect()
        }));
        let ctx = scripted_ctx(&client);
        let checks = Arc::new(AtomicU32::new(0));
        let check_count = Arc::clone(&checks);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cond = ctx
                .wait_for_condition(
                    move |_ctx, state: i32| {
                        let checks = Arc::clone(&check_count);
                        async move {
                            checks.fetch_add(1, Ordering::SeqCst);
                            Ok::<i32, BoxError>(state + 1)
                        }
                    },
                    0,
                )
                .wait_strategy_fn(|state: i32, _attempt| {
                    if state >= 2 {
                        WaitDecision::complete()
                    } else {
                        WaitDecision::continue_with(Duration::from_secs(1))
                    }
                })
                .name("cond")
                .spawn();
            ctx.step(move |_| async move {
                // The READY arrives only after the retry was recorded.
                for _ in 0..1000 {
                    if body_client.received("cond", &OperationAction::Retry) {
                        return Ok(1_u32);
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                Err("the condition never retried".into())
            })
            .name("s")
            .await
            .map_err(wire_err)?;
            let state = cond.await.map_err(wire_err)?;
            Ok(state.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("2".to_owned()));
        assert_eq!(checks.load(Ordering::SeqCst), 2, "the check ran twice");
        assert_eq!(
            events.replayed(),
            0,
            "a resumed operation emits no replay event"
        );
    }

    /// A record that a sibling's response completes while the condition
    /// waits returns its recorded outcome, without an `operation_replayed`
    /// event and without another check.
    #[tokio::test(start_paused = true)]
    async fn resumed_terminal_record_returns_without_replay_event() {
        let (events, _guard) = crate::driver::test_support::EventMessages::install();
        let client = Arc::new(ScriptedClient::new(|updates, history| {
            if named(updates, "s", &OperationAction::Succeed).is_none() {
                return Vec::new();
            }
            named(history, "cond", &OperationAction::Start)
                .map(|start| {
                    operation_like(start, OperationStatus::Succeeded)
                        .step_details(StepDetails::builder().attempt(1).result("5").build())
                        .build()
                        .expect("all required fields set")
                })
                .into_iter()
                .collect()
        }));
        let ctx = scripted_ctx(&client);
        let checks = Arc::new(AtomicU32::new(0));
        let check_count = Arc::clone(&checks);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cond = ctx
                .wait_for_condition(
                    move |_ctx, state: i32| {
                        let checks = Arc::clone(&check_count);
                        async move {
                            checks.fetch_add(1, Ordering::SeqCst);
                            Ok::<i32, BoxError>(state + 1)
                        }
                    },
                    0,
                )
                .wait_strategy_fn(|_state: i32, _attempt| {
                    WaitDecision::continue_with(Duration::from_secs(1))
                })
                .name("cond")
                .spawn();
            ctx.step(move |_| async move {
                for _ in 0..1000 {
                    if body_client.received("cond", &OperationAction::Retry) {
                        return Ok(1_u32);
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                Err("the condition never retried".into())
            })
            .name("s")
            .await
            .map_err(wire_err)?;
            let state = cond.await.map_err(wire_err)?;
            Ok(state.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("5".to_owned()));
        assert_eq!(checks.load(Ordering::SeqCst), 1, "the check ran once");
        assert_eq!(
            events.replayed(),
            0,
            "a resumed record emits no replay event"
        );
    }
}

/// When a condition check runs relative to its `START` write, under paused
/// tokio time, with a client that answers each checkpoint call 500 ms after
/// it receives it.
#[cfg(test)]
mod start_write_tests {
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{LatencyClient, drive_handler, latency_ctx, wire_err};
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn condition_check_starts_before_start_response() {
        let client = Arc::new(LatencyClient::new(Duration::from_millis(500)));
        let ctx = latency_ctx(&client, None);
        let started = Arc::new(Mutex::new(None));
        let check_started = Arc::clone(&started);
        let clock = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let state = ctx
                .wait_for_condition(
                    move |_, state: u32| {
                        let started = Arc::clone(&check_started);
                        let clock = Arc::clone(&clock);
                        async move {
                            *started.lock().expect("lock") = Some(clock.elapsed());
                            Ok(state + 1)
                        }
                    },
                    0,
                )
                .wait_strategy_fn(|_state, _attempt| {
                    crate::builders::wait_for_condition::WaitDecision::complete()
                })
                .name("c")
                .await
                .map_err(wire_err)?;
            Ok(state.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("1".to_owned()));
        assert_eq!(
            *started.lock().expect("lock"),
            Some(Duration::ZERO),
            "the check starts before START is answered"
        );
        let start = client.request_with(&OperationAction::Start).expect("START");
        let succeed = client
            .request_with(&OperationAction::Succeed)
            .expect("SUCCEED");
        assert_eq!(start.received, Duration::ZERO);
        assert_eq!(
            succeed.received,
            Duration::from_millis(500),
            "SUCCEED waits for START"
        );
    }
}
