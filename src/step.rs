//! Step operation execution engine.
//!
//! Implements the live path (run closure, serialize, checkpoint), replay path
//! (return frozen result), and retry strategy (checkpoint-suspend for delays).
//!
//! Retry delays use checkpoint-suspend rather than in-process sleep:
//! a RETRY action with `NextAttemptDelaySeconds` is checkpointed, then the
//! function suspends; the backend owns the timer.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::Instrument;

use crate::client::ClientError;
use crate::context::{DurableContext, StepContext};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{OperationError, OperationErrorKind, StepError, StepErrorKind};
use crate::tracing_layer;
use crate::{BoxError, RetryDecision, RetryStrategy, Serdes, SerdesContext};

/// The default retry strategy: 6 total attempts, 5s initial delay, 60s max
/// delay, 2x backoff rate, FULL jitter.
///
/// Matches the standard `ExponentialBackoff` defaults.
pub(crate) fn default_retry_strategy() -> RetryStrategy {
    Box::new(|_err: &StepError, attempt: u32| {
        const MAX_ATTEMPTS: u32 = 6;
        const INITIAL_DELAY_SECS: f64 = 5.0;
        const MAX_DELAY_SECS: f64 = 60.0;
        const BACKOFF_RATE: f64 = 2.0;

        if attempt >= MAX_ATTEMPTS {
            return RetryDecision::Stop;
        }

        // Exponential backoff: initial * rate^(attempt-1), capped at max.
        // attempt is 1-based: first failure is attempt=1.
        #[allow(clippy::cast_possible_truncation)] // reason: attempt is small (≤6)
        let exponent = (i32::try_from(attempt).unwrap_or(1)) - 1;
        let base = (INITIAL_DELAY_SECS * BACKOFF_RATE.powi(exponent)).min(MAX_DELAY_SECS);

        // FULL jitter: random in [0, base] (the default jitter strategy).
        let jittered = rand_full_jitter(base);

        // Round to whole seconds, minimum 1.
        #[allow(clippy::cast_possible_truncation)] // reason: result is ≤ 60
        #[allow(clippy::cast_sign_loss)] // reason: jittered ≥ 0
        let delay_secs = jittered.round().max(1.0) as u64;
        RetryDecision::Retry {
            delay: Duration::from_secs(delay_secs),
        }
    })
}

/// Full jitter: returns a value in `[0, max_secs]`.
///
/// Uses time + thread-id + counter hashing for determinism-safe randomness
/// (no rand crate dependency).
fn rand_full_jitter(max_secs: f64) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    // Mix in a counter to avoid same result on rapid calls.
    COUNTER
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        .hash(&mut hasher);
    let bits = hasher.finish();
    // Map u64 to [0.0, 1.0)
    #[allow(clippy::cast_precision_loss)] // reason: approximation is fine for jitter
    let fraction = (bits as f64) / (u64::MAX as f64);
    fraction * max_secs
}

/// Execution semantics for a step, controlling behavior on interrupted replay.
///
/// When a step is interrupted mid-execution (e.g., a Lambda timeout or crash),
/// the checkpoint log records a `Started` status without an outcome. On the
/// next invocation, the SDK must decide whether to re-execute the step body.
///
/// This is a **client-side only** configuration — it is not sent on the wire.
/// The checkpoint `Start` action is identical regardless of semantics; the
/// difference is purely in the replay decision when `Started` status is
/// encountered.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::StepSemantics;
///
/// // Default: re-execute on replay if interrupted.
/// let default = StepSemantics::AtLeastOncePerRetry;
///
/// // Idempotency-sensitive: treat interrupted as failed, consult retry strategy.
/// let at_most_once = StepSemantics::AtMostOncePerRetry;
/// # drop(default);
/// # drop(at_most_once);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum StepSemantics {
    /// Re-execute the step body on replay if the previous attempt was
    /// interrupted (status `Started` with no outcome).
    ///
    /// This is the default behavior and matches the semantics of most
    /// idempotent operations. The step may execute more than once per retry
    /// attempt if the process is interrupted between `Start` checkpoint and
    /// the outcome checkpoint.
    #[default]
    AtLeastOncePerRetry,

    /// Do **not** re-execute the step body if the previous attempt was
    /// interrupted. Instead, treat the interruption as a failure and
    /// consult the retry strategy to decide whether to schedule a new
    /// attempt or fail permanently.
    ///
    /// Use this for non-idempotent operations (e.g., charging a credit
    /// card) where re-execution could produce duplicate side effects.
    AtMostOncePerRetry,
}

/// Wire sub-type for step operations.
pub(crate) const STEP_SUB_TYPE: &str = "Step";

/// Internal state for step execution passed from the builder.
pub(crate) struct StepExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) retry_strategy: Option<RetryStrategy>,
    pub(crate) serdes: Option<Arc<dyn Serdes>>,
    pub(crate) semantics: StepSemantics,
    #[allow(clippy::type_complexity)] // reason: boxed future factory is inherently complex
    pub(crate) closure: Box<
        dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<O, BoxError>> + Send>> + Send,
    >,
}

impl<O: Serialize + DeserializeOwned + Send + 'static> StepExecution<O> {
    /// Executes the step operation: replay path or live path with retry.
    #[allow(clippy::too_many_lines)] // reason: validation adds lines but splitting would obscure flow
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // 2. Check checkpoint log for replay.
        let mut already_started = false;
        if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            // Non-determinism detection: verify the record's identity matches.
            self.ctx.validate_replay_identity(
                &record,
                &wire_id,
                "Step",
                Some(STEP_SUB_TYPE),
                self.name.as_deref(),
            )?;
            match &record.status {
                CheckpointStatus::Succeeded => {
                    let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());
                    return replay_success(
                        self.serdes.as_ref().or_else(|| self.ctx.default_serdes()),
                        record.result.as_ref(),
                        &serdes_ctx,
                    )
                    .await;
                }
                CheckpointStatus::Failed => {
                    return Err(replay_failure(
                        record.error_type.as_deref(),
                        record.error_message.as_deref(),
                    ));
                }
                CheckpointStatus::Pending => {
                    // Retry timer hasn't fired yet — suspend.
                    return self.ctx.suspend_now().await;
                }
                CheckpointStatus::Started => {
                    // Already started — behavior depends on semantics.
                    if self.semantics == StepSemantics::AtMostOncePerRetry {
                        // The previous attempt was interrupted before
                        // recording an outcome. Do NOT re-execute; treat as
                        // a step-interrupted failure and consult the retry
                        // strategy to decide FAIL or RETRY.
                        let attempt = self.ctx.get_attempt(&self.op_id).saturating_add(1);
                        let interrupted_err: BoxError =
                            "step interrupted (AtMostOncePerRetry)".into();
                        return handle_failure::<O>(
                            &self.ctx,
                            &wire_id,
                            self.name.as_deref(),
                            self.retry_strategy.as_ref(),
                            interrupted_err,
                            attempt,
                        )
                        .await;
                    }
                    // AtLeastOncePerRetry (default): skip re-checkpointing
                    // START but re-execute the body.
                    already_started = true;
                }
                CheckpointStatus::Ready
                | CheckpointStatus::Cancelled
                | CheckpointStatus::TimedOut
                | CheckpointStatus::Stopped => {
                    // Fall through to live execution.
                }
            }
        }

        // 3. Live execution path.
        // Derive the current attempt from checkpoint log: if there's a
        // recorded operation with step details, attempt = recorded + 1.
        let attempt = self.ctx.get_attempt(&self.op_id).saturating_add(1);

        // Destructure self to avoid partial-move issues with the FnOnce closure.
        let ctx = self.ctx;
        let name = self.name;
        let retry_strategy = self.retry_strategy;
        let serdes = self.serdes;
        let closure = self.closure;

        // Checkpoint START (skip if step was already in Started state).
        if !already_started {
            let start_update = build_start_update(&wire_id, name.as_deref(), ctx.parent_wire_id());
            ctx.checkpoint_updates(vec![start_update])
                .await
                .map_err(|e| client_error_to_op_error(&e))?;
        }

        // Execute the step body inside a tracing span carrying the
        // structured-log field contract.
        let is_replay = false; // Live execution is never replay.
        let span = tracing_layer::operation_span(
            ctx.execution_arn(),
            &ctx.lambda_context().request_id,
            &wire_id,
            attempt,
            is_replay,
        );
        let step_ctx = StepContext::new(attempt);
        let result = async { (closure)(step_ctx).await }.instrument(span).await;

        match result {
            Ok(value) => {
                handle_success::<O>(
                    &ctx,
                    &wire_id,
                    name.as_deref(),
                    serdes.as_ref().or_else(|| ctx.default_serdes()),
                    value,
                )
                .await
            }
            Err(err) => {
                handle_failure::<O>(
                    &ctx,
                    &wire_id,
                    name.as_deref(),
                    retry_strategy.as_ref(),
                    err,
                    attempt,
                )
                .await
            }
        }
    }
}

// ── Free functions for post-closure operations ──────────────────────────

/// Handles a successful step execution: serialize, checkpoint, return.
async fn handle_success<O: Serialize + DeserializeOwned>(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    serdes: Option<&Arc<dyn Serdes>>,
    value: O,
) -> Result<O, OperationError> {
    let serdes_ctx = SerdesContext::new(wire_id, ctx.execution_arn());

    // Serialize the result.
    let serialized = serialize_value(serdes, &value, &serdes_ctx).await?;

    // Checkpoint SUCCEED with payload.
    let update = build_succeed_update(wire_id, name, ctx.parent_wire_id(), &serialized);
    ctx.checkpoint_updates(vec![update])
        .await
        .map_err(|e| client_error_to_op_error(&e))?;

    // Return deserialized from the serialized form (round-trip parity).
    deserialize_result(serdes, &serialized, &serdes_ctx).await
}

/// Handles a failed step: consult retry strategy, checkpoint accordingly.
async fn handle_failure<O: Serialize + DeserializeOwned>(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    retry_strategy: Option<&RetryStrategy>,
    err: BoxError,
    attempt: u32,
) -> Result<O, OperationError> {
    let step_err = StepError::from_kind(StepErrorKind::ExecutionFailed {
        message: err.to_string(),
    });

    // Consult the retry strategy.
    let decision = if let Some(strategy) = retry_strategy {
        strategy(&step_err, attempt)
    } else {
        default_retry_strategy()(&step_err, attempt)
    };

    match decision {
        RetryDecision::Retry { delay } => {
            // Checkpoint RETRY with delay. Round fractional delays UP so the
            // retry never fires earlier than requested (1.9s -> 2s), enforce
            // a 1-second minimum, and clamp to i32::MAX.
            let whole_secs = delay
                .as_secs()
                .saturating_add(u64::from(delay.subsec_nanos() > 0));
            let delay_secs = i32::try_from(whole_secs.max(1)).unwrap_or(i32::MAX);
            let update = build_retry_update(wire_id, name, ctx.parent_wire_id(), &err, delay_secs);
            ctx.checkpoint_updates(vec![update])
                .await
                .map_err(|e| client_error_to_op_error(&e))?;

            // Suspend — the backend owns the retry timer.
            ctx.suspend_now().await
        }
        RetryDecision::Stop => {
            // Checkpoint FAIL (permanent).
            let update = build_fail_update(wire_id, name, ctx.parent_wire_id(), &err);
            ctx.checkpoint_updates(vec![update])
                .await
                .map_err(|e| client_error_to_op_error(&e))?;

            Err(OperationError::from_kind(OperationErrorKind::Step(
                StepError::from_kind(StepErrorKind::RetriesExhausted {
                    attempts: attempt,
                    last_error: err.to_string(),
                }),
            )))
        }
    }
}

// ── Update builders ─────────────────────────────────────────────────────

fn build_start_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(STEP_SUB_TYPE)
        .action(OperationAction::Start);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    // build() is infallible here — all required fields (id, type, action) set.
    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_succeed_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    payload: &str,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(STEP_SUB_TYPE)
        .action(OperationAction::Succeed)
        .payload(payload);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_retry_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    err: &BoxError,
    delay_secs: i32,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(STEP_SUB_TYPE)
        .action(OperationAction::Retry)
        .error(
            aws_sdk_lambda::types::ErrorObject::builder()
                .error_type(error_type_name(&**err))
                .error_message(err.to_string())
                .build(),
        )
        .step_options(
            aws_sdk_lambda::types::StepOptions::builder()
                .next_attempt_delay_seconds(delay_secs)
                .build(),
        );

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_fail_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    err: &BoxError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(STEP_SUB_TYPE)
        .action(OperationAction::Fail)
        .error(
            aws_sdk_lambda::types::ErrorObject::builder()
                .error_type(error_type_name(&**err))
                .error_message(err.to_string())
                .build(),
        );

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

// ── Serialization helpers ───────────────────────────────────────────────

fn serialize_value<'a, O: Serialize>(
    serdes: Option<&'a Arc<dyn Serdes>>,
    value: &O,
    serdes_ctx: &'a SerdesContext,
) -> impl Future<Output = Result<String, OperationError>> + Send + 'a {
    // Phase 1 (sync): consume the `&O` borrow now, so the returned future
    // holds no `&O` across its await (which would force `O: Sync` on every
    // operation future). No custom serdes renders straight to the wire; a
    // custom serdes receives the value erased to `serde_json::Value` — the
    // same shape every other operation path provides.
    let prepared = crate::serdes::prepare_value(serdes, value);
    // Phase 2 (async): a custom serdes may block (e.g. filesystem I/O), so
    // the call runs off the async runtime.
    async move {
        prepared
            .map_err(|e| step_serialization_error(&e))?
            .into_wire(serdes_ctx)
            .await
            .map_err(|e| step_serialization_error(&*e))
    }
}

async fn deserialize_result<O: DeserializeOwned>(
    serdes: Option<&Arc<dyn Serdes>>,
    serialized: &str,
    serdes_ctx: &SerdesContext,
) -> Result<O, OperationError> {
    let Some(s) = serdes else {
        return serde_json::from_str(serialized).map_err(|e| step_serialization_error(&e));
    };
    // Custom serdes may block (e.g. filesystem I/O): run off the runtime.
    let json_value = crate::serdes::deserialize_off_runtime(s, serialized.to_owned(), serdes_ctx)
        .await
        .map_err(|e| step_serialization_error(&*e))?;
    serde_json::from_value(json_value).map_err(|e| step_serialization_error(&e))
}

/// Wraps any error as a step `SerializationFailed` operation error.
fn step_serialization_error<E: std::fmt::Display + ?Sized>(e: &E) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
        StepErrorKind::SerializationFailed {
            message: e.to_string(),
        },
    )))
}

async fn replay_success<O: DeserializeOwned>(
    serdes: Option<&Arc<dyn Serdes>>,
    result: Option<&String>,
    serdes_ctx: &SerdesContext,
) -> Result<O, OperationError> {
    let payload = result.map_or("null", String::as_str);
    deserialize_result(serdes, payload, serdes_ctx).await
}

fn replay_failure(error_type: Option<&str>, error_message: Option<&str>) -> OperationError {
    let msg = match (error_type, error_message) {
        (Some(t), Some(m)) => format!("{t}: {m}"),
        (None, Some(m)) => m.to_owned(),
        (Some(t), None) => t.to_owned(),
        (None, None) => "unknown error".to_owned(),
    };
    OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
        StepErrorKind::ExecutionFailed { message: msg },
    )))
}

fn client_error_to_op_error(err: &ClientError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
        StepErrorKind::ExecutionFailed {
            message: err.to_string(),
        },
    )))
}

/// Derives the wire error type name from a boxed error.
///
/// Uses the concrete type name as the error type name.
fn error_type_name(err: &(dyn std::error::Error + Send + Sync)) -> String {
    let debug = format!("{err:?}");
    // Heuristic: first capitalized word as type name.
    if let Some(first) = debug
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .find(|s| !s.is_empty() && s.starts_with(char::is_uppercase))
    {
        return first.to_owned();
    }
    "Error".to_owned()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
#[allow(clippy::type_complexity)] // reason: boxed future factories in test setup
#[allow(clippy::panic)] // reason: test assertions with descriptive messages
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use std::sync::Arc;

    // ── Replay tests ────────────────────────────────────────────────────

    #[tokio::test]
    async fn replay_success_deserializes_json() {
        let payload = "42".to_owned();
        let ctx = SerdesContext::new("op-1", "arn:test");
        let result: Result<i32, OperationError> = replay_success(None, Some(&payload), &ctx).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn replay_success_null_returns_unit() {
        let ctx = SerdesContext::new("op-1", "arn:test");
        let result: Result<(), OperationError> = replay_success(None, None, &ctx).await;
        assert!(result.is_ok());
    }

    #[test]
    fn replay_failure_formats_error_with_type_and_message() {
        let err = replay_failure(Some("TypeError"), Some("bad input"));
        let msg = err.to_string();
        assert!(msg.contains("TypeError"), "got: {msg}");
        assert!(msg.contains("bad input"), "got: {msg}");
    }

    #[test]
    fn replay_failure_message_only() {
        let err = replay_failure(None, Some("something failed"));
        let msg = err.to_string();
        assert!(msg.contains("something failed"), "got: {msg}");
    }

    #[test]
    fn replay_failure_unknown_when_empty() {
        let err = replay_failure(None, None);
        let msg = err.to_string();
        assert!(msg.contains("unknown error"), "got: {msg}");
    }

    // ── Serialization tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn serialize_deserialize_round_trip() {
        let ctx = SerdesContext::new("op-1", "arn:test");
        let serialized = serialize_value::<String>(None, &"hello".to_owned(), &ctx)
            .await
            .unwrap();
        let deserialized: String = deserialize_result(None, &serialized, &ctx).await.unwrap();
        assert_eq!(deserialized, "hello");
    }

    #[tokio::test]
    async fn serialize_with_custom_serdes_uppercases() {
        struct Upper;
        impl std::fmt::Debug for Upper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Upper")
            }
        }
        impl Serdes for Upper {
            fn serialize(
                &self,
                value: &serde_json::Value,
                _context: &SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(value.to_string().to_uppercase())
            }
            fn deserialize(
                &self,
                data: &str,
                _context: &SerdesContext,
            ) -> Result<serde_json::Value, BoxError> {
                Ok(serde_json::from_str(data)?)
            }
        }
        let serdes: Arc<dyn Serdes> = Arc::new(Upper);
        let ctx = SerdesContext::new("op-1", "arn:test");
        let result = serialize_value(Some(&serdes), &"hello".to_owned(), &ctx)
            .await
            .unwrap();
        assert_eq!(result, "\"HELLO\"");
    }

    // ── Default retry strategy tests ────────────────────────────────────

    #[test]
    fn default_retry_stops_at_max_attempts() {
        let strategy = default_retry_strategy();
        let err = StepError::from_kind(StepErrorKind::ExecutionFailed {
            message: "fail".to_owned(),
        });
        // Attempt 6 should stop (max_attempts = 6).
        let decision = strategy(&err, 6);
        assert_eq!(decision, RetryDecision::Stop);
    }

    #[test]
    fn default_retry_retries_below_max() {
        let strategy = default_retry_strategy();
        let err = StepError::from_kind(StepErrorKind::ExecutionFailed {
            message: "fail".to_owned(),
        });
        let decision = strategy(&err, 1);
        match decision {
            RetryDecision::Retry { delay } => {
                // With full jitter on 5s base, delay should be 1-5s.
                assert!(delay.as_secs() >= 1);
                assert!(delay.as_secs() <= 5);
            }
            RetryDecision::Stop => panic!("expected retry for attempt 1"),
        }
    }

    #[test]
    fn default_retry_delay_grows_with_attempts() {
        // Attempt 5: base = 5 * 2^4 = 80, capped at 60.
        // With full jitter: [0, 60] → rounded: [1, 60].
        let strategy = default_retry_strategy();
        let err = StepError::from_kind(StepErrorKind::ExecutionFailed {
            message: "fail".to_owned(),
        });
        let decision = strategy(&err, 5);
        match decision {
            RetryDecision::Retry { delay } => {
                // Max possible is 60s.
                assert!(delay.as_secs() <= 60);
                assert!(delay.as_secs() >= 1);
            }
            RetryDecision::Stop => panic!("expected retry for attempt 5"),
        }
    }

    // ── Error type name extraction ──────────────────────────────────────

    #[test]
    fn error_type_name_extracts_first_capitalized() {
        #[derive(Debug)]
        struct TransientError;
        impl std::fmt::Display for TransientError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "transient error")
            }
        }
        impl std::error::Error for TransientError {}
        let name = error_type_name(&TransientError);
        assert_eq!(name, "TransientError");
    }

    #[test]
    fn error_type_name_fallback_to_error() {
        // An error whose debug starts with lowercase.
        let err: Box<dyn std::error::Error + Send + Sync> = "lowercase".into();
        let name = error_type_name(&*err);
        assert_eq!(name, "Error");
    }

    // ── Live execution tests (with mock client) ─────────────────────────

    #[tokio::test]
    async fn step_live_success_checkpoints_and_returns() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();
        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(|_| Box::pin(async { Ok(42) }));

        let exec = StepExecution {
            ctx,
            op_id,
            name: Some("test-step".to_owned()),
            retry_strategy: None,
            serdes: None,
            semantics: StepSemantics::default(),
            closure,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn default_serdes_applies_when_step_sets_none() {
        use crate::client::InMemoryExecutionClient;

        // A default serdes that uppercases on serialize and lowercases on
        // deserialize. Threaded in via Options -> context default_serdes.
        struct Upper;
        impl std::fmt::Debug for Upper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Upper")
            }
        }
        impl Serdes for Upper {
            fn serialize(
                &self,
                value: &serde_json::Value,
                _context: &SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(value.to_string().to_uppercase())
            }
            fn deserialize(
                &self,
                data: &str,
                _context: &SerdesContext,
            ) -> Result<serde_json::Value, BoxError> {
                Ok(serde_json::from_str(&data.to_lowercase())?)
            }
        }

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "token0".to_owned(),
            Some(Arc::new(Upper)),
        );
        let op_id = ctx.mint_id();
        let closure: Box<
            dyn FnOnce(
                    StepContext,
                )
                    -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>>
                + Send,
        > = Box::new(|_| Box::pin(async { Ok("hello".to_owned()) }));

        let exec = StepExecution {
            ctx,
            op_id,
            name: Some("greet".to_owned()),
            retry_strategy: None,
            serdes: None, // no per-op serdes -> must fall back to the default
            semantics: StepSemantics::default(),
            closure,
        };

        // The step returns the round-tripped value.
        let result = exec.execute().await;
        assert_eq!(result.unwrap(), "hello");

        // The Succeed checkpoint payload was serialized by the default serdes
        // (uppercased). Without default_serdes wiring the payload would be
        // the plain "\"hello\"".
        let updates = client.recorded_updates();
        let succeed = updates
            .iter()
            .find(|u| matches!(u.action(), OperationAction::Succeed))
            .expect("a Succeed update must be recorded");
        assert_eq!(
            succeed.payload(),
            Some("\"HELLO\""),
            "default serdes must serialize the step result when the step sets none"
        );
    }

    #[tokio::test]
    async fn step_replay_returns_frozen_result_without_re_executing() {
        use crate::engine::{CheckpointLog, CheckpointRecord};
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        // Build a checkpoint log with a succeeded record at position "1".
        // The log is keyed by wire ID (hash of positional "1"), matching
        // how parse_inline_operations stores records from the backend.
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Succeeded,
            result: Some("99".to_owned()),
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id(); // This mints "1"

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(move |_| {
            executed_clone.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(0) })
        });

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: None,
            semantics: StepSemantics::default(),
            closure,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 99);
        assert!(
            !executed.load(Ordering::SeqCst),
            "closure should NOT execute during replay"
        );
    }

    #[tokio::test]
    async fn step_replay_failure_returns_error_without_re_executing() {
        use crate::engine::{CheckpointLog, CheckpointRecord};
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Failed,
            result: None,
            error_type: Some("CustomError".to_owned()),
            error_message: Some("it broke".to_owned()),
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let closure: Box<
            dyn FnOnce(
                    StepContext,
                )
                    -> Pin<Box<dyn Future<Output = Result<String, BoxError>> + Send>>
                + Send,
        > = Box::new(move |_| {
            executed_clone.store(true, Ordering::SeqCst);
            Box::pin(async { Ok("should not run".to_owned()) })
        });

        let exec = StepExecution::<String> {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: None,
            semantics: StepSemantics::default(),
            closure,
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("CustomError"), "got: {err_msg}");
        assert!(err_msg.contains("it broke"), "got: {err_msg}");
        assert!(
            !executed.load(Ordering::SeqCst),
            "closure should NOT execute during replay"
        );
    }

    #[tokio::test]
    async fn step_retry_then_stop_on_exhaustion() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        // A strategy that stops immediately (max_attempts = 1).
        let no_retry: RetryStrategy =
            Box::new(|_err: &StepError, _attempt: u32| RetryDecision::Stop);

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(|_| Box::pin(async { Err("always fails".into()) }));

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: Some(no_retry),
            serdes: None,
            semantics: StepSemantics::default(),
            closure,
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("retries exhausted") || err_msg.contains("always fails"),
            "got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn step_retry_schedules_and_suspends() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        // Strategy that always retries with 1s delay.
        let always_retry: RetryStrategy =
            Box::new(|_err: &StepError, _attempt: u32| RetryDecision::Retry {
                delay: Duration::from_secs(1),
            });

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(|_| Box::pin(async { Err("transient".into()) }));

        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            retry_strategy: Some(always_retry),
            serdes: None,
            semantics: StepSemantics::default(),
            closure,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        // Retry timer is backend-owned: the operation suspends (parks) rather
        // than surfacing a fabricated error to the caller.
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn step_retry_fractional_delay_rounds_up() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        // Strategy that retries with a fractional 1.9s delay. Truncation
        // would checkpoint 1s and retry EARLIER than requested; ceiling
        // semantics must checkpoint 2s.
        let fractional_retry: RetryStrategy =
            Box::new(|_err: &StepError, _attempt: u32| RetryDecision::Retry {
                delay: Duration::from_millis(1900),
            });

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(|_| Box::pin(async { Err("transient".into()) }));

        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            retry_strategy: Some(fractional_retry),
            serdes: None,
            semantics: StepSemantics::default(),
            closure,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        // The RETRY update must carry the rounded-up delay.
        let updates = client.recorded_updates();
        let retry = updates
            .iter()
            .find(|u| matches!(u.action(), OperationAction::Retry))
            .expect("a Retry update must be recorded");
        let delay = retry
            .step_options()
            .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds)
            .expect("Retry update must carry NextAttemptDelaySeconds");
        assert_eq!(delay, 2, "1.9s must round UP to 2s, not truncate to 1s");
    }

    #[tokio::test]
    async fn step_spawn_executes_on_blessed_task() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );

        // Use ctx.step() which wires up the closure properly.
        let handle = ctx.step(|_| async { Ok(7i32) }).name("spawned").spawn();
        let result = handle.await;
        assert_eq!(result.unwrap(), 7);
    }

    #[tokio::test]
    async fn step_ownership_rejection_from_foreign_task() {
        use crate::client::InMemoryExecutionClient;

        // Must create the context inside a spawned task where try_id()
        // returns Some — #[tokio::test] root runs in block_on with no task ID.
        #[allow(clippy::unwrap_used)] // reason: test code
        let result = tokio::spawn(async {
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let log = Arc::new(CheckpointLog::empty());
            let ctx = DurableContext::new_root_with_client(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                log,
                client,
                "token0".to_owned(),
            );

            // Spawn a DIFFERENT (non-blessed) task and try executing a step.
            let ctx_clone = ctx.clone();
            let handle = tokio::spawn(async move {
                let op_id = ctx_clone.mint_id();
                let closure: Box<
                    dyn FnOnce(
                            StepContext,
                        )
                            -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                        + Send,
                > = Box::new(|_| Box::pin(async { Ok(1) }));

                let exec = StepExecution {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    retry_strategy: None,
                    serdes: None,
                    semantics: StepSemantics::default(),
                    closure,
                };
                exec.execute().await
            });

            handle.await.unwrap()
        })
        .await;

        #[allow(clippy::unwrap_used)] // reason: test code
        let inner_result = result.unwrap();
        assert!(inner_result.is_err());
        #[allow(clippy::unwrap_used)] // reason: test — verified Err above
        let err_msg = inner_result.unwrap_err().to_string();
        assert!(
            err_msg.contains("task") || err_msg.contains("ownership"),
            "expected ownership error, got: {err_msg}"
        );
    }

    // ── AtMostOncePerRetry tests ────────────────────────────────────────

    #[tokio::test]
    async fn step_at_most_once_replay_started_no_retry_fails_permanently() {
        use crate::engine::{CheckpointLog, CheckpointRecord};
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        // Build a checkpoint log with a Started record (simulating an
        // interrupted previous attempt).
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Started,
            result: None,
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));

        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        // Strategy that NEVER retries (stops immediately).
        let no_retry: RetryStrategy =
            Box::new(|_err: &StepError, _attempt: u32| RetryDecision::Stop);

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(move |_| {
            executed_clone.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(42) })
        });

        let exec = StepExecution {
            ctx,
            op_id,
            name: Some("at-most-once-step".to_owned()),
            retry_strategy: Some(no_retry),
            serdes: None,
            semantics: StepSemantics::AtMostOncePerRetry,
            closure,
        };

        let result = exec.execute().await;
        assert!(
            result.is_err(),
            "expected failure for interrupted + no retry"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("retries exhausted") || err_msg.contains("interrupted"),
            "got: {err_msg}"
        );
        assert!(
            !executed.load(Ordering::SeqCst),
            "closure should NOT execute when AtMostOncePerRetry and replay=Started"
        );
    }

    #[tokio::test]
    async fn step_at_most_once_replay_started_with_retry_suspends() {
        use crate::engine::{CheckpointLog, CheckpointRecord};
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Started,
            result: None,
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));

        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        // Strategy that always retries with 1s delay.
        let always_retry: RetryStrategy =
            Box::new(|_err: &StepError, _attempt: u32| RetryDecision::Retry {
                delay: Duration::from_secs(1),
            });

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(move |_| {
            executed_clone.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(42) })
        });

        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            retry_strategy: Some(always_retry),
            serdes: None,
            semantics: StepSemantics::AtMostOncePerRetry,
            closure,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        // Retry timer is backend-owned: the operation suspends (parks) rather
        // than surfacing a fabricated retry-schedule error to the caller.
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        assert!(
            !executed.load(Ordering::SeqCst),
            "closure should NOT execute when AtMostOncePerRetry and replay=Started"
        );
    }

    #[tokio::test]
    async fn step_at_least_once_replay_started_re_executes() {
        // Verify that the DEFAULT semantics still re-execute on Started replay.
        use crate::engine::{CheckpointLog, CheckpointRecord};
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = Arc::clone(&executed);

        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Started,
            result: None,
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));

        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        let closure: Box<
            dyn FnOnce(StepContext) -> Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
                + Send,
        > = Box::new(move |_| {
            executed_clone.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(77) })
        });

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: None,
            semantics: StepSemantics::AtLeastOncePerRetry,
            closure,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 77);
        assert!(
            executed.load(Ordering::SeqCst),
            "closure SHOULD execute under AtLeastOncePerRetry with Started replay"
        );
    }
}
