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
//! - Wait strategy exhaustion raises `WaitForConditionError::MaxChecksExceeded`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::context::{DurableContext, StepContext};
use crate::engine::CheckpointStatus;
use crate::engine::OperationId;
use crate::error::{
    OperationError, OperationErrorKind, WaitForConditionError, WaitForConditionErrorKind,
};
use crate::{BoxError, Serdes, SerdesContext};

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
/// use aws_durable_execution_sdk_rust::WaitDecision;
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
pub(crate) struct WaitForConditionExecution<S> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) initial_state: S,
    pub(crate) wait_strategy: Option<WaitStrategyFn<S>>,
    pub(crate) serdes: Option<Arc<dyn Serdes>>,
    #[allow(clippy::type_complexity)] // reason: boxed Fn closure is inherently complex
    pub(crate) check: Box<
        dyn Fn(StepContext, S) -> Pin<Box<dyn Future<Output = Result<S, BoxError>> + Send>>
            + Send
            + Sync,
    >,
}

impl<S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static> WaitForConditionExecution<S> {
    /// Executes the wait-for-condition operation.
    #[allow(clippy::too_many_lines)] // reason: checkpoint-state-strategy flow is a single logical unit; splitting would obscure the protocol
    pub(crate) async fn execute(self) -> Result<S, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());

        // 2. Check checkpoint log for replay / resume status.
        let mut already_started = false;
        if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            // Non-determinism detection: verify the record's identity matches.
            self.ctx.validate_replay_identity(
                &record,
                &wire_id,
                "Step",
                Some(WFC_SUB_TYPE),
                self.name.as_deref(),
            )?;
            match &record.status {
                CheckpointStatus::Succeeded => {
                    // Terminal success: deserialize the final state.
                    // CRITICAL: LOUD error on deserialization failure.
                    // Never fall back to initial_state (Python #574 / JS #754).
                    return replay_terminal_success(
                        self.serdes.as_ref().or_else(|| self.ctx.default_serdes()),
                        record.result.as_ref(),
                        &serdes_ctx,
                    )
                    .await;
                }
                CheckpointStatus::Failed => {
                    // Terminal failure: reconstruct error from checkpoint.
                    return Err(replay_terminal_failure(
                        record.error_type.as_deref(),
                        record.error_message.as_deref(),
                    ));
                }
                CheckpointStatus::Pending => {
                    // Retry timer hasn't fired yet — suspend.
                    return self.ctx.suspend_now().await;
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
        let current_state = if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            if record.result.is_some() {
                // On deserialization failure, surface the error loudly and
                // never silently fall back to initial_state.
                deserialize_state(
                    self.serdes.as_ref().or_else(|| self.ctx.default_serdes()),
                    record.result.as_ref(),
                    &serdes_ctx,
                )
                .await?
            } else {
                self.initial_state.clone()
            }
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
            self.ctx
                .checkpoint_updates(vec![start_update])
                .await
                .map_err(|e| wfc_client_error(&e))?;
        }

        // Execute the check function.
        let step_ctx = StepContext::new(attempt);
        let check_result = (self.check)(step_ctx, current_state).await;

        match check_result {
            Ok(new_state) => {
                // Serialize the new state.
                let serialized = serialize_state(
                    self.serdes.as_ref().or_else(|| self.ctx.default_serdes()),
                    &new_state,
                    &serdes_ctx,
                )
                .await?;

                // Round-trip through serdes for consistency.
                let deserialized: S = deserialize_state_str(
                    self.serdes.as_ref().or_else(|| self.ctx.default_serdes()),
                    &serialized,
                    &serdes_ctx,
                )
                .await?;

                // Consult the wait strategy.
                let decision = if let Some(strategy) = &self.wait_strategy {
                    strategy(deserialized.clone(), attempt)
                } else {
                    // Default: if no strategy provided, complete immediately.
                    WaitDecision::Complete
                };

                match decision {
                    WaitDecision::Complete => {
                        // Condition met: checkpoint terminal SUCCEED.
                        let update = build_wfc_update(
                            &wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            OperationAction::Succeed,
                            Some(&serialized),
                            None,
                        );
                        self.ctx
                            .checkpoint_updates(vec![update])
                            .await
                            .map_err(|e| wfc_client_error(&e))?;

                        Ok(deserialized)
                    }
                    WaitDecision::Continue { delay } => {
                        // Not met: checkpoint RETRY with state + delay, then suspend.
                        #[allow(clippy::cast_possible_truncation)] // reason: delay clamped to i32
                        #[allow(clippy::cast_sign_loss)]
                        // reason: ceil is non-negative
                        let delay_secs = (delay.as_secs_f64().ceil() as i64)
                            .clamp(1, i64::from(i32::MAX))
                            as i32;
                        let update = build_wfc_update(
                            &wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            OperationAction::Retry,
                            Some(&serialized),
                            Some(delay_secs),
                        );
                        self.ctx
                            .checkpoint_updates(vec![update])
                            .await
                            .map_err(|e| wfc_client_error(&e))?;

                        self.ctx.suspend_now().await
                    }
                    WaitDecision::Exhausted { reason } => {
                        // Strategy exhaustion: checkpoint FAIL, raise
                        // WaitForConditionError (Python #530 fix).
                        let update = build_wfc_fail_update(
                            &wire_id,
                            self.name.as_deref(),
                            self.ctx.parent_wire_id(),
                            &reason,
                        );
                        self.ctx
                            .checkpoint_updates(vec![update])
                            .await
                            .map_err(|e| wfc_client_error(&e))?;

                        Err(wfc_op_error(WaitForConditionErrorKind::MaxChecksExceeded {
                            checks: attempt,
                        }))
                    }
                }
            }
            Err(check_err) => {
                // Check function error: checkpoint FAIL immediately (no retry
                // for check errors).
                let update = build_wfc_fail_update(
                    &wire_id,
                    self.name.as_deref(),
                    self.ctx.parent_wire_id(),
                    &check_err.to_string(),
                );
                self.ctx
                    .checkpoint_updates(vec![update])
                    .await
                    .map_err(|e| wfc_client_error(&e))?;

                Err(wfc_op_error(WaitForConditionErrorKind::CheckFailed {
                    message: check_err.to_string(),
                }))
            }
        }
    }
}

// ── Update builders ─────────────────────────────────────────────────────

/// Replays a terminal success from the checkpoint log.
/// CRITICAL: NEVER falls back to `initial_state` (Python #574 / JS #754 fix).
async fn replay_terminal_success<S: DeserializeOwned>(
    serdes: Option<&Arc<dyn Serdes>>,
    result: Option<&String>,
    serdes_ctx: &SerdesContext,
) -> Result<S, OperationError> {
    let payload = result.map_or("null", String::as_str);
    deserialize_state_str(serdes, payload, serdes_ctx).await
}

/// Replays a terminal failure from the checkpoint log.
fn replay_terminal_failure(
    error_type: Option<&str>,
    error_message: Option<&str>,
) -> OperationError {
    let msg = match (error_type, error_message) {
        (Some(t), Some(m)) => format!("{t}: {m}"),
        (None, Some(m)) => m.to_owned(),
        (Some(t), None) => t.to_owned(),
        (None, None) => "unknown error".to_owned(),
    };
    wfc_op_error(WaitForConditionErrorKind::CheckFailed { message: msg })
}

/// Serializes state using the configured serdes or default JSON.
fn serialize_state<'a, S: Serialize>(
    serdes: Option<&'a Arc<dyn Serdes>>,
    value: &S,
    serdes_ctx: &'a SerdesContext,
) -> impl Future<Output = Result<String, OperationError>> + Send + 'a {
    // Phase 1 (sync): consume the `&S` borrow now, so the returned future
    // holds no `&S` across its await (which would force `S: Sync`). No
    // custom serdes renders straight to the wire; a custom serdes receives
    // the state erased to `serde_json::Value` — the same shape every other
    // operation path provides.
    let prepared = crate::serdes::prepare_value(serdes, value);
    // Phase 2 (async): a custom serdes may block (e.g. filesystem I/O), so
    // the call runs off the async runtime.
    async move {
        prepared
            .map_err(|e| {
                wfc_op_error(WaitForConditionErrorKind::SerializationFailed {
                    message: e.to_string(),
                })
            })?
            .into_wire(serdes_ctx)
            .await
            .map_err(|e| {
                wfc_op_error(WaitForConditionErrorKind::SerializationFailed {
                    message: e.to_string(),
                })
            })
    }
}

/// Deserializes state from checkpoint result (Option<&String>).
/// LOUD error on failure — never silently resets (Python #574 fix).
async fn deserialize_state<S: DeserializeOwned>(
    serdes: Option<&Arc<dyn Serdes>>,
    result: Option<&String>,
    serdes_ctx: &SerdesContext,
) -> Result<S, OperationError> {
    let payload = result.map_or("null", String::as_str);
    deserialize_state_str(serdes, payload, serdes_ctx).await
}

/// Deserializes state from a string payload.
async fn deserialize_state_str<S: DeserializeOwned>(
    serdes: Option<&Arc<dyn Serdes>>,
    payload: &str,
    serdes_ctx: &SerdesContext,
) -> Result<S, OperationError> {
    let Some(s) = serdes else {
        return serde_json::from_str(payload).map_err(|e| {
            wfc_op_error(WaitForConditionErrorKind::SerializationFailed {
                message: format!("state deserialization failed: {e}"),
            })
        });
    };
    // Custom serdes may block (e.g. filesystem I/O): run off the runtime.
    let json_value = crate::serdes::deserialize_off_runtime(s, payload.to_owned(), serdes_ctx)
        .await
        .map_err(|e| {
            wfc_op_error(WaitForConditionErrorKind::SerializationFailed {
                message: format!("state deserialization failed: {e}"),
            })
        })?;
    serde_json::from_value(json_value).map_err(|e| {
        wfc_op_error(WaitForConditionErrorKind::SerializationFailed {
            message: format!("state deserialization failed: {e}"),
        })
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

    #[allow(clippy::expect_used)] // reason: all required fields (id, type, action) are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_wfc_fail_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    error_message: &str,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(WFC_SUB_TYPE)
        .action(OperationAction::Fail)
        .error(
            aws_sdk_lambda::types::ErrorObject::builder()
                .error_type("WaitForConditionError")
                .error_message(error_message)
                .build(),
        );

    if let Some(n) = name {
        builder = builder.name(n);
    }
    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[allow(clippy::expect_used)] // reason: all required fields (id, type, action) are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

// ── Error helpers ───────────────────────────────────────────────────────

fn wfc_op_error(kind: WaitForConditionErrorKind) -> OperationError {
    OperationError::from_kind(OperationErrorKind::WaitForCondition(
        WaitForConditionError::from_kind(kind),
    ))
}

fn wfc_client_error(err: &crate::client::ClientError) -> OperationError {
    wfc_op_error(WaitForConditionErrorKind::CheckFailed {
        message: err.to_string(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::panic)] // reason: test assertions with descriptive messages
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
            attempt: 1,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
        #[allow(clippy::expect_used)] // reason: test fixture — all required fields set
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
            serdes: None,
            check: Box::new(move |_ctx, state| {
                *seen_check
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(state);
                Box::pin(async move { Ok(state + 1) })
            }),
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
            serdes: None,
            check: Box::new(|_ctx, state| Box::pin(async move { Ok(state) })),
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
            attempt: 1,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
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
            serdes: None,
            check: Box::new(|_ctx, state| Box::pin(async move { Ok(state + 1) })),
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
            serdes: None,
            check: Box::new(|_ctx, state| Box::pin(async move { Ok(state + 1) })),
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err.kind() {
            OperationErrorKind::WaitForCondition(wfc_err) => match wfc_err.kind() {
                WaitForConditionErrorKind::MaxChecksExceeded { checks } => {
                    assert_eq!(*checks, 1);
                }
                other => panic!("expected MaxChecksExceeded, got: {other:?}"),
            },
            other => panic!("expected WaitForCondition error, got: {other:?}"),
        }
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
            attempt: 2,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32> {
            ctx,
            op_id,
            name: Some("regression".to_owned()),
            initial_state: 0,
            wait_strategy: Some(Box::new(|_state: i32, _attempt| WaitDecision::complete())),
            serdes: None,
            check: Box::new(|_ctx, _state| {
                Box::pin(async move { panic!("check must NOT execute during replay") })
            }),
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
                WaitForConditionErrorKind::SerializationFailed { message } => {
                    assert!(
                        message.contains("state deserialization failed"),
                        "error message should indicate state deserialization failure: {message}"
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

        let exec = WaitForConditionExecution::<i32> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: Some(Box::new(|_state: i32, _attempt| WaitDecision::complete())),
            serdes: None,
            check: Box::new(|_ctx, _state| Box::pin(async { Err("check function failed".into()) })),
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
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
            attempt: 3,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: None,
            serdes: None,
            check: Box::new(|_ctx, _state| {
                Box::pin(async { panic!("check must NOT execute during replay") })
            }),
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
            attempt: 3,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let ctx = make_ctx_with_log(vec![(wire_key, record)]);
        let op_id = ctx.mint_id();

        let exec = WaitForConditionExecution::<i32> {
            ctx,
            op_id,
            name: None,
            initial_state: 0,
            wait_strategy: None,
            serdes: None,
            check: Box::new(|_ctx, _state| {
                Box::pin(async { panic!("check must NOT execute during replay") })
            }),
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max attempts exceeded"), "got: {err_msg}");
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
