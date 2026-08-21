//! Wait-for-condition operation execution engine.
//!
//! Implements the live path (evaluate check fn, checkpoint state per attempt,
//! apply wait strategy → delay-suspend or complete or exhaust) and replay path
//! (frozen terminal state; LOUD error on state deserialization failure — never
//! silent reset to initial state).
//!
//! ## Key invariants
//!
//! - The SDK surfaces a `WaitForConditionError::SerializationFailed` error
//!   when checkpointed state fails to deserialize. It never silently resets
//!   to `initial_state`.
//! - [`WaitDecision`] is the type returned by the wait strategy.
//! - Wait strategy exhaustion raises `WaitForConditionError::MaxChecksExceeded`,
//!   live and on replay: the terminal record carries a dedicated wire
//!   discriminator, so the replayed error keeps the same kind and check count.

use std::future::Future;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

use crate::context::{DurableContext, StepContext};
use crate::engine::CheckpointStatus;
use crate::engine::OperationId;
use crate::error::{
    OperationError, OperationErrorKind, WaitForConditionError, WaitForConditionErrorKind,
};
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
/// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitDecision;
/// use std::time::Duration;
///
/// // Condition met — stop polling.
/// let done = WaitDecision::complete();
///
/// // Not met — poll again after 5 seconds.
/// let cont = WaitDecision::continue_with(Duration::from_secs(5));
///
/// // Exhausted — fail the operation.
/// let exhausted = WaitDecision::exhausted("max attempts exceeded");
/// # drop(done); drop(cont); drop(exhausted);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WaitDecision {
    /// Condition met — return the final state.
    Complete,
    /// Condition not met — suspend for `delay` then re-check.
    Continue {
        /// Duration to wait before the next check attempt.
        delay: Duration,
    },
    /// Strategy exhaustion — fail the operation with an error.
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
    pub(crate) wait_strategy: Option<WaitStrategyFn<S>>,
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
    /// Thin generic wrapper — the ONLY code monomorphized per call site.
    /// The checkpoint-state-strategy machine lives in the non-generic
    /// [`WfcCore`] / [`WfcAfter`] halves (generic over the state type `S`
    /// only); this wrapper just polls the user's concrete check future
    /// between them.
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
        match core.before().await? {
            WfcPrelude::Done(result) => result,
            WfcPrelude::Run {
                attempt,
                state,
                after,
            } => {
                // Execute the check function.
                let step_ctx = StepContext::new(attempt);
                let check_result = (check)(step_ctx, state).await;
                after.settle(attempt, check_result).await
            }
        }
    }
}

/// The pre-check half of `wait_for_condition`: task-ownership check, replay
/// resolution, carried-state derivation, and the START checkpoint. Generic
/// only over the state type `S` — no user closure reaches this state
/// machine, so its substantial replay/checkpoint logic compiles once per
/// state type instead of once per call site.
struct WfcCore<S, SD> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    initial_state: S,
    wait_strategy: Option<WaitStrategyFn<S>>,
    serdes: SD,
}

/// What [`WfcCore::before`] decided: the operation is already resolved from
/// the checkpoint log (or suspended), or one check-attempt cycle must run.
enum WfcPrelude<S, SD> {
    /// Resolved without running the check (replay or suspension).
    Done(Result<S, OperationError>),
    /// The check must run at `attempt` against the carried `state`; `after`
    /// settles the decision cycle.
    Run {
        /// 1-based attempt number for this check cycle.
        attempt: u32,
        /// The carried state the check receives.
        state: S,
        /// The post-check half that runs the strategy/decision protocol.
        after: WfcAfter<S, SD>,
    },
}

/// The post-check half of `wait_for_condition`: state serialization, wait
/// strategy consultation, and the Succeed/Retry/Fail checkpoint protocol.
/// Generic only over the state type.
struct WfcAfter<S, SD> {
    ctx: DurableContext,
    wire_id: String,
    name: Option<String>,
    wait_strategy: Option<WaitStrategyFn<S>>,
    serdes: SD,
}

impl<S, SD> WfcCore<S, SD>
where
    S: Clone + Send + Sync + 'static,
    SD: Serdes<S>,
{
    /// Runs everything that precedes the check closure: replay path, or
    /// the live-path preamble (carried-state derivation + START checkpoint).
    #[expect(clippy::too_many_lines)] // reason: replay-status protocol is a single logical unit; splitting would obscure it
    async fn before(self) -> Result<WfcPrelude<S, SD>, OperationError> {
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
                    let value =
                        replay_terminal_success(&self.serdes, payload, serdes_ctx.clone()).await?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Step",
                        Some(WFC_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(WfcPrelude::Done(Ok(value)));
                }
                CheckpointStatus::Failed => {
                    // Terminal failure: reconstruct error from checkpoint.
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Step",
                        Some(WFC_SUB_TYPE),
                        view.attempt,
                    );
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
                    // Retry timer hasn't fired yet — suspend.
                    return Ok(WfcPrelude::Done(self.ctx.suspend_now().await));
                }
                CheckpointStatus::Started => {
                    // This invocation already checkpointed START — skip it.
                    already_started = true;
                }
                CheckpointStatus::Ready
                | CheckpointStatus::Cancelled
                | CheckpointStatus::TimedOut
                | CheckpointStatus::Stopped => {
                    // Only statusStarted skips START — other statuses
                    // need a fresh StepStarted for the per-attempt protocol.
                }
                CheckpointStatus::Unknown(ref raw) => {
                    // Unreachable in production — `checkpoint_view_validated`
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
                deserialize_state_str(&self.serdes, payload, serdes_ctx.clone()).await?
            } else {
                self.initial_state.clone()
            };

        // Checkpoint START if not already started.
        if !already_started {
            let start_update = build_wfc_update(
                &wire_id,
                self.name.as_deref(),
                self.ctx.parent_wire_id(),
                OperationAction::Start,
                None,
                None,
            );
            if let Err(err) = self.ctx.checkpoint_updates(vec![start_update]).await {
                // Audit (#43) — wait-for-condition START: no user code ran
                // for this attempt, so no terminal FAIL is needed;
                // re-invocation reconverges on the same write.
                return self
                    .ctx
                    .checkpoint_failure_unrecoverable(&wire_id, err, None)
                    .await;
            }
        }

        let Self {
            ctx,
            name,
            wait_strategy,
            serdes,
            ..
        } = self;

        Ok(WfcPrelude::Run {
            attempt,
            state: current_state,
            after: WfcAfter {
                ctx,
                wire_id,
                name,
                wait_strategy,
                serdes,
            },
        })
    }
}

impl<S, SD> WfcAfter<S, SD>
where
    S: Clone + Send + Sync + 'static,
    SD: Serdes<S>,
{
    /// Settles one check cycle: serializes the new state, consults the wait
    /// strategy, and runs the Succeed/Retry/Fail checkpoint protocol.
    #[expect(clippy::too_many_lines)] // reason: the settle protocol (serialize, strategy, checkpoint) is a single logical unit; splitting would obscure it
    async fn settle(
        self,
        attempt: u32,
        check_result: Result<S, BoxError>,
    ) -> Result<S, OperationError> {
        let serdes_ctx = SerdesContext::new(&self.wire_id, self.ctx.execution_arn());
        match check_result {
            Ok(new_state) => {
                // Serialize the new state (ownership transfers to the
                // serdes; the round trip below reconstructs it).
                //
                // A state serialization (or round-trip) failure is a
                // LOCAL, deterministic, user-facing failure, so it stays
                // catchable — but the terminal FAIL is persisted FIRST
                // (issue #43). The check already ran, so its side effects
                // need a recorded outcome: with the FAIL recorded, replay
                // yields it instead of re-running the check.
                let round_trip = async {
                    let serialized =
                        serialize_state(&self.serdes, new_state, serdes_ctx.clone()).await?;
                    // Round-trip through serdes for consistency.
                    let deserialized: S =
                        deserialize_state_str(&self.serdes, serialized.clone(), serdes_ctx).await?;
                    Ok::<_, OperationError>((serialized, deserialized))
                };
                let (serialized, deserialized) = match round_trip.await {
                    Ok(pair) => pair,
                    Err(op_err) => {
                        let wire = crate::error::serialization_failure_wire(&op_err);
                        let update = build_wfc_fail_update(
                            &self.wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            &wire,
                        );
                        if let Err(client_err) = self.ctx.checkpoint_updates(vec![update]).await {
                            // Audit (#43) — wfc FAIL (serialization): the
                            // check ran; the failed FAIL write routes
                            // unrecoverable with a minimal terminal FAIL
                            // retry.
                            return self.checkpoint_unrecoverable(client_err).await;
                        }
                        return Err(op_err
                            .with_operation(&self.wire_id, CheckpointStatus::Failed.wire_str())
                            .with_wire(wire));
                    }
                };

                // Consult the wait strategy.
                let decision = if let Some(strategy) = &self.wait_strategy {
                    strategy(deserialized.clone(), attempt)
                } else {
                    // Documented no-strategy behavior: the check runs exactly
                    // once and the operation completes with that state (see
                    // `WaitForConditionBuilder` docs).
                    WaitDecision::Complete
                };

                match decision {
                    WaitDecision::Complete => {
                        // Condition met: checkpoint terminal SUCCEED.
                        let update = build_wfc_update(
                            &self.wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            OperationAction::Succeed,
                            Some(&serialized),
                            None,
                        );
                        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                            // Audit (#43) — wfc SUCCEED: the check ran, so
                            // its side effects need a recorded outcome. A
                            // permanent rejection persists a small
                            // terminal FAIL before the execution fails.
                            return self.checkpoint_unrecoverable(err).await;
                        }

                        Ok(deserialized)
                    }
                    WaitDecision::Continue { delay } => {
                        // Not met: checkpoint RETRY with state + delay, then suspend.
                        #[expect(clippy::cast_possible_truncation)] // reason: delay clamped to i32
                        let delay_secs = (delay.as_secs_f64().ceil() as i64)
                            .clamp(1, i64::from(i32::MAX))
                            as i32;
                        let update = build_wfc_update(
                            &self.wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            OperationAction::Retry,
                            Some(&serialized),
                            Some(delay_secs),
                        );
                        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                            // Audit (#43) — wfc RETRY: the check ran; a
                            // retry the service never recorded would not
                            // replay, so a permanent rejection persists a
                            // small terminal FAIL before the execution
                            // fails.
                            return self.checkpoint_unrecoverable(err).await;
                        }

                        self.ctx.suspend_now().await
                    }
                    WaitDecision::Exhausted { reason } => {
                        // Strategy exhaustion: checkpoint FAIL, raise
                        // WaitForConditionError (Python #530 fix). The
                        // record's type is the fixed exhaustion
                        // discriminator so replay reconstructs the same
                        // `MaxChecksExceeded` classification (see
                        // `replay_terminal_failure`).
                        let wire = crate::error::wire_error_manual(
                            crate::error::WFC_EXHAUSTED_ERROR_TYPE,
                            reason.as_str(),
                        );
                        let update = build_wfc_fail_update(
                            &self.wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            &wire,
                        );
                        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                            // Audit (#43) — wfc FAIL (strategy exhausted):
                            // the check ran; the failed FAIL write routes
                            // unrecoverable with a minimal terminal FAIL
                            // retry.
                            return self.checkpoint_unrecoverable(err).await;
                        }

                        Err(wfc_op_error(WaitForConditionErrorKind::MaxChecksExceeded(
                            crate::error::MaxChecksExceeded::new(attempt),
                        ))
                        .with_operation(&self.wire_id, CheckpointStatus::Failed.wire_str())
                        .with_wire(wire))
                    }
                }
            }
            Err(check_err) => {
                // Check function error: checkpoint FAIL immediately (no retry
                // for check errors).
                // The wire record is derived from the check error itself:
                // flattened message, `error_data`/`stack_trace` pass-through.
                let wire = crate::error::wire_error_with_type(&*check_err, "WaitForConditionError");
                let update = build_wfc_fail_update(
                    &self.wire_id,
                    self.name.as_deref(),
                    self.ctx.parent_wire_id(),
                    &wire,
                );
                if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
                    // Audit (#43) — wfc FAIL (check error): the check ran
                    // and failed; the failed FAIL write routes
                    // unrecoverable with a minimal terminal FAIL retry
                    // (the original carried the check error's payload).
                    return self.checkpoint_unrecoverable(err).await;
                }

                Err(wfc_op_error_with_source(
                    WaitForConditionErrorKind::CheckFailed,
                    Some(check_err),
                )
                .with_operation(&self.wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire))
            }
        }
    }

    /// Routes a rejected outcome write through the unrecoverable path
    /// (issue #43), never returning. The check ran at every settle site,
    /// so each supplies a small checkpoint-failure-derived terminal `FAIL`
    /// — the record persisted before the execution dies when the
    /// rejection is permanent.
    async fn checkpoint_unrecoverable<T>(&self, err: crate::client::ClientError) -> T {
        let cwire = crate::error::checkpoint_failure_wire(&err);
        let terminal = build_wfc_fail_update(
            &self.wire_id,
            self.name.as_deref(),
            self.ctx.parent_wire_id(),
            &cwire,
        );
        self.ctx
            .checkpoint_failure_unrecoverable(&self.wire_id, err, Some(terminal))
            .await
    }
}

// ── Update builders ─────────────────────────────────────────────────────

/// Replays a terminal success from the checkpoint log.
/// CRITICAL: NEVER falls back to `initial_state` (Python #574 / JS #754 fix).
async fn replay_terminal_success<S, SD: Serdes<S>>(
    serdes: &SD,
    result: Option<String>,
    serdes_ctx: SerdesContext,
) -> Result<S, OperationError> {
    let payload = result.unwrap_or_else(|| "null".to_owned());
    deserialize_state_str(serdes, payload, serdes_ctx).await
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
///   count from `recorded_attempt` — the checkpoint record's completed
///   retry count — exactly as the live path derived its 1-based attempt
///   number (`recorded + 1`), so `MaxChecksExceeded::checks()` reports
///   the same value live and replayed. Like the live exhaustion error,
///   the replayed one carries no foreign source: the kind is the whole
///   story, and the recorded reason stays readable on the attached wire
///   record.
/// - Anything else is a check failure.
fn replay_terminal_failure(
    wire: crate::error::WireError,
    wire_id: &str,
    recorded_attempt: u32,
) -> OperationError {
    if wire.error_type() == Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE) {
        return wfc_op_error(WaitForConditionErrorKind::MaxChecksExceeded(
            crate::error::MaxChecksExceeded::new(recorded_attempt.saturating_add(1)),
        ))
        .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
        .with_wire(wire);
    }
    let kind = if wire.error_type() == Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE) {
        WaitForConditionErrorKind::SerializationFailed
    } else {
        WaitForConditionErrorKind::CheckFailed
    };
    wfc_op_error_with_source(
        kind,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

/// Serializes state through the configured serdes (ownership transfers;
/// the serdes decides where its work runs).
async fn serialize_state<S, SD: Serdes<S>>(
    serdes: &SD,
    value: S,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    serdes.serialize(value, serdes_ctx).await.map_err(|e| {
        wfc_op_error_with_source(WaitForConditionErrorKind::SerializationFailed, Some(e))
    })
}

/// Deserializes state from a string payload.
/// LOUD error on failure — never silently resets (Python #574 fix).
async fn deserialize_state_str<S, SD: Serdes<S>>(
    serdes: &SD,
    payload: String,
    serdes_ctx: SerdesContext,
) -> Result<S, OperationError> {
    serdes.deserialize(payload, serdes_ctx).await.map_err(|e| {
        wfc_op_error_with_source(
            WaitForConditionErrorKind::SerializationFailed,
            Some(crate::error::ContextualError::source_from(
                "state deserialization failed",
                e,
            )),
        )
    })
}

// ── OperationUpdate builders ────────────────────────────────────────────

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

// ── Error helpers ───────────────────────────────────────────────────────

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
            op_name: None,
        };

        // The START checkpoint response echoes the operation as a fresh
        // STARTED attempt with NO result — the shape that clobbers the
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

        let signal = Arc::clone(ctx.suspension_signal());
        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state: 0i32,
            wait_strategy: Some(Box::new(|state: i32, _attempt| {
                if state >= 3 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })),
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
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
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
    /// envelope must still read the original bytes — the failed attempt's
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
            op_name: None,
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
            wait_strategy: Some(Box::new(|_state: i32, _attempt| {
                WaitDecision::continue_with(Duration::from_secs(1))
            })),
            serdes: filesystem_serdes.clone(),
            check: |_ctx, state: i32| async move { Ok::<i32, BoxError>(state + 1) },
        };

        // The injected RETRY-write rejection routes through the
        // unrecoverable path (issue #43): the operation future parks for
        // the invocation driver instead of yielding a catchable error, and
        // the fatal slot records the checkpoint failure.
        let signal = Arc::clone(exec.ctx.suspension_signal());
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
            wait_strategy: Some(Box::new(|state: i32, _attempt| {
                if state >= 5 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })),
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
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id(); // mints "1"
        let signal = Arc::clone(ctx.suspension_signal());

        let exec = WaitForConditionExecution {
            ctx,
            op_id,
            name: None,
            initial_state: 0i32,
            wait_strategy: Some(Box::new(|state: i32, _attempt| {
                if state >= 3 {
                    WaitDecision::complete()
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })),
            serdes: crate::serdes::JsonSerdes,
            check: |_ctx, state: i32| async move { Ok::<i32, BoxError>(state + 1) },
        };

        // State was 1 (from checkpoint), check fn produces 2, strategy says
        // continue (< 3), so it checkpoints RETRY and suspends (parks) rather
        // than surfacing a fabricated error to the caller.
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
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
            wait_strategy: Some(Box::new(|_state: i32, attempt| {
                // Always exhaust on attempt 1.
                if attempt >= 1 {
                    WaitDecision::exhausted("max attempts exceeded")
                } else {
                    WaitDecision::continue_with(Duration::from_secs(1))
                }
            })),
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
        // The exhaustion record is constructed at a fixed-discriminator
        // site with no escaping error to walk — the stack is captured at
        // the construction site rather than left empty.
        let wire = err.wire().unwrap();
        assert_eq!(
            wire.error_type(),
            Some(crate::error::WFC_EXHAUSTED_ERROR_TYPE)
        );
        assert_eq!(wire.error_message(), Some("max attempts exceeded"));
        assert!(
            !wire.stack_trace().is_empty(),
            "exhaustion failure record must carry a stack trace"
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
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: Some("regression".to_owned()),
            initial_state: 0,
            wait_strategy: Some(Box::new(|_state: i32, _attempt| WaitDecision::complete())),
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
        // MUST be an error — never silently returns initial_state (0).
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
            wait_strategy: Some(Box::new(|_state: i32, _attempt| WaitDecision::complete())),
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
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: None,
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
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: None,
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
    /// terminal FAIL; replay must reconstruct the same kind — not the
    /// generic `CheckFailed` — with `checks()` derived from the record's
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
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32, _, _> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: None,
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
}
