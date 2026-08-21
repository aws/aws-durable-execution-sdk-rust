//! Step operation execution engine.
//!
//! Implements the live path (run closure, serialize, checkpoint), replay path
//! (return frozen result), and retry strategy (checkpoint-suspend for delays).
//!
//! Retry delays use checkpoint-suspend rather than in-process sleep:
//! a RETRY action with `NextAttemptDelaySeconds` is checkpointed, then the
//! function suspends; the backend owns the timer.

use std::future::Future;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tracing::Instrument;

use crate::builders::{JitterStrategy, RetryStrategyConfig};
use crate::context::{DurableContext, StepContext};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{OperationError, OperationErrorKind, StepError, StepErrorKind};
use crate::serdes::SerdesContext;
use crate::tracing_layer;
use crate::{BoxError, RetryDecision, RetryStrategy, Serdes};

/// The default retry strategy: 6 total attempts, 5s initial delay, 60s max
/// delay, 2x backoff rate, FULL jitter.
///
/// Matches the standard `ExponentialBackoff` defaults. The constants live in
/// [`RetryStrategyConfig::default`]; this is one call so they are defined in
/// exactly one place.
pub(crate) fn default_retry_strategy() -> RetryStrategy {
    RetryStrategyConfig::default().into_retry_strategy()
}

impl RetryStrategyConfig {
    /// Computes the retry decision for a 1-based failed attempt number.
    ///
    /// Exponential backoff: `initial_delay * backoff_rate^(attempt - 1)`,
    /// capped at `max_delay`, jittered per the configured
    /// [`JitterStrategy`], and quantized to whole seconds with a
    /// one-second minimum (see [`quantize_delay_secs`] for the
    /// per-strategy rounding). Stops once `attempt` reaches
    /// `max_attempts`.
    pub(crate) fn decide(&self, attempt: u32) -> RetryDecision {
        if attempt >= self.max_attempts() {
            return RetryDecision::Stop;
        }

        // Exponential backoff: initial * rate^(attempt-1), capped at max.
        // attempt is 1-based: first failure is attempt=1.
        let exponent = (i32::try_from(attempt).unwrap_or(1)) - 1;
        let base = (self.initial_delay().as_secs_f64() * self.backoff_rate().powi(exponent))
            .min(self.max_delay().as_secs_f64());

        let jittered = match self.jitter() {
            JitterStrategy::None => base,
            // Half jitter: base/2 plus random in [0, base/2] => [base/2, base].
            JitterStrategy::Half => base / 2.0 + rand_full_jitter(base / 2.0),
            // Full jitter: random in [0, base].
            JitterStrategy::Full => rand_full_jitter(base),
        };

        RetryDecision::Retry {
            delay: Duration::from_secs(quantize_delay_secs(jittered, self.jitter())),
        }
    }

    /// Converts this configuration into the boxed retry-strategy closure
    /// the step engine consumes. The decision ignores the error value:
    /// delay shaping depends only on the attempt number.
    pub(crate) fn into_retry_strategy(self) -> RetryStrategy {
        Box::new(move |_err: &StepError, attempt: u32| self.decide(attempt))
    }
}

/// Quantizes a fractional delay in seconds to whole seconds, minimum 1.
///
/// The rounding rule depends on the jitter strategy:
///
/// - [`JitterStrategy::Full`] rounds to the **nearest** whole second. This
///   is the SDK's legacy quantization, preserved so
///   [`RetryStrategyConfig::default`] reproduces the historical
///   full-jitter delay distribution exactly.
/// - [`JitterStrategy::None`] rounds **up**: a deterministic configured
///   delay must never fire earlier than requested, matching the wait and
///   retry-checkpoint behavior.
/// - [`JitterStrategy::Half`] rounds **up**: the documented
///   `[base / 2, base]` lower bound survives quantization only under a
///   ceiling (nearest-rounding could dip up to half a second below it).
pub(crate) fn quantize_delay_secs(jittered: f64, jitter: JitterStrategy) -> u64 {
    let quantized = match jitter {
        JitterStrategy::Full => jittered.round(),
        JitterStrategy::None | JitterStrategy::Half => jittered.ceil(),
    };
    #[expect(clippy::cast_possible_truncation)] // reason: delays are far below u64::MAX seconds
    #[expect(clippy::cast_sign_loss)] // reason: quantized ≥ 0
    {
        quantized.max(1.0) as u64
    }
}

/// Full jitter: returns a value in `[0, max_secs]`.
///
/// Uses time + thread-id + counter hashing for determinism-safe randomness
/// (no rand crate dependency).
pub(crate) fn rand_full_jitter(max_secs: f64) -> f64 {
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
    #[expect(clippy::cast_precision_loss)] // reason: approximation is fine for jitter
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
///
/// Generic over the user's closure `F` (and, through `F`'s output, its
/// future): the closure is stored and run **without type erasure**. The one
/// erasure point for a step is `.future()` / `into_future` on the builder,
/// which boxes the whole execution future once inside
/// [`DurableFuture`](crate::DurableFuture).
pub(crate) struct StepExecution<O, F, S> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) retry_strategy: Option<RetryStrategy>,
    pub(crate) serdes: S,
    pub(crate) semantics: StepSemantics,
    pub(crate) closure: F,
    pub(crate) _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O, F, Fut, S> StepExecution<O, F, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// Executes the step operation: replay path or live path with retry.
    ///
    /// Thin generic wrapper — the ONLY code monomorphized per step call
    /// site. The replay/checkpoint state machine lives in the non-generic
    /// [`StepCore`] / [`StepLive`] halves (generic over the result type `O`
    /// only); this wrapper just polls the user's concrete future between
    /// them.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        let Self {
            ctx,
            op_id,
            name,
            retry_strategy,
            serdes,
            semantics,
            closure,
            _marker,
        } = self;
        let core = StepCore {
            ctx,
            op_id,
            name,
            retry_strategy,
            serdes,
            semantics,
            _marker: std::marker::PhantomData,
        };
        match core.before().await? {
            StepPrelude::Done(result) => result,
            StepPrelude::Run {
                attempt,
                span,
                live,
            } => {
                // Execute the step body inside a tracing span carrying the
                // structured-log field contract.
                let step_ctx = StepContext::new(attempt);
                let result = async { (closure)(step_ctx).await }.instrument(span).await;
                live.settle(attempt, result).await
            }
        }
    }
}

/// The pre-closure half of a step: task-ownership check, replay resolution,
/// attempt derivation, and the START checkpoint. Generic only over the
/// result type `O` — no user closure reaches this state machine, so its
/// substantial replay/checkpoint logic compiles once per result type
/// instead of once per step call site.
struct StepCore<O, S> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: S,
    semantics: StepSemantics,
    _marker: std::marker::PhantomData<fn() -> O>,
}

/// What [`StepCore::before`] decided: the step is already resolved from the
/// checkpoint log (or suspended), or the body must run live.
enum StepPrelude<O, S> {
    /// Resolved without running the body (replay, suspension, or an
    /// `AtMostOncePerRetry` interruption verdict).
    Done(Result<O, OperationError>),
    /// The body must run at `attempt`, instrumented with `span`; `live`
    /// settles the outcome afterwards.
    Run {
        /// 1-based attempt number for this live execution.
        attempt: u32,
        /// The operation span the body runs under.
        span: tracing::Span,
        /// The post-closure half that checkpoints the outcome.
        live: StepLive<O, S>,
    },
}

/// The post-closure half of a step: outcome checkpointing (success payload
/// or retry-strategy failure handling). Generic only over the result type.
struct StepLive<O, S> {
    ctx: DurableContext,
    wire_id: String,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: S,
    _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O, S> StepCore<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// Runs everything that precedes the user closure: replay path, or the
    /// live-path preamble ending at the START checkpoint.
    #[expect(clippy::too_many_lines)] // reason: validation adds lines but splitting would obscure flow
    async fn before(self) -> Result<StepPrelude<O, S>, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // 2. Check checkpoint log for replay. The validated view covers the
        // non-terminal branches without cloning; the terminal branches fetch
        // only the one field they consume.
        let mut already_started = false;
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Step",
            Some(STEP_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Decode the recorded payload FIRST: `operation_replayed`
                    // promises a recorded terminal outcome was returned, so a
                    // corrupt payload or failing serdes surfaces as an error
                    // without the event.
                    let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_success(&self.serdes, payload, serdes_ctx).await?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Step",
                        Some(STEP_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(StepPrelude::Done(Ok(value)));
                }
                CheckpointStatus::Failed => {
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Step",
                        Some(STEP_SUB_TYPE),
                        view.attempt,
                    );
                    let wire = self
                        .ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Ok(StepPrelude::Done(Err(replay_failure(wire, &wire_id))));
                }
                CheckpointStatus::Pending => {
                    // Retry timer hasn't fired yet — suspend.
                    return Ok(StepPrelude::Done(self.ctx.suspend_now().await));
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
                        return Ok(StepPrelude::Done(
                            handle_failure::<O>(
                                &self.ctx,
                                &wire_id,
                                self.name.as_deref(),
                                self.retry_strategy.as_ref(),
                                interrupted_err,
                                attempt,
                            )
                            .await,
                        ));
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
                CheckpointStatus::Unknown(ref raw) => {
                    // Unreachable in production — `checkpoint_view_validated`
                    // already failed the execution (issue #45). Kept as a
                    // typed arm so a future bypass cannot fall through to
                    // live execution and re-run a possibly-terminal step.
                    return Err(self.ctx.unrecognized_status_error(&wire_id, raw));
                }
            }
        }

        // 3. Live execution path.
        // Derive the current attempt from checkpoint log: if there's a
        // recorded operation with step details, attempt = recorded + 1.
        let attempt = self.ctx.get_attempt(&self.op_id).saturating_add(1);

        // Destructure self so the live half owns exactly what it needs.
        let Self {
            ctx,
            name,
            retry_strategy,
            serdes,
            ..
        } = self;

        // Checkpoint START (skip if step was already in Started state).
        if !already_started {
            let start_update = build_start_update(&wire_id, name.as_deref(), ctx.parent_wire_id());
            if let Err(err) = ctx.checkpoint_updates(vec![start_update]).await {
                // Audit (#43) — step START: no user code has run for this
                // attempt, so there is no side effect needing a recorded
                // outcome. No terminal FAIL: the invocation dies, the
                // service re-invokes, and replay reaches this point and
                // attempts the same write, losing only one invocation.
                return ctx
                    .checkpoint_failure_unrecoverable(&wire_id, err, None)
                    .await;
            }
        }

        // The body runs inside a tracing span carrying the structured-log
        // field contract.
        let is_replay = false; // Live execution is never replay.
        let span = tracing_layer::operation_span(
            ctx.execution_arn(),
            &ctx.lambda_context().request_id,
            &wire_id,
            attempt,
            is_replay,
        );

        Ok(StepPrelude::Run {
            attempt,
            span,
            live: StepLive {
                ctx,
                wire_id,
                name,
                retry_strategy,
                serdes,
                _marker: std::marker::PhantomData,
            },
        })
    }
}

impl<O, S> StepLive<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// Settles a live step outcome: checkpoints success or consults the
    /// retry strategy on failure.
    async fn settle(self, attempt: u32, result: Result<O, BoxError>) -> Result<O, OperationError> {
        match result {
            Ok(value) => {
                handle_success(
                    &self.ctx,
                    &self.wire_id,
                    self.name.as_deref(),
                    &self.serdes,
                    value,
                )
                .await
            }
            Err(err) => {
                handle_failure::<O>(
                    &self.ctx,
                    &self.wire_id,
                    self.name.as_deref(),
                    self.retry_strategy.as_ref(),
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
///
/// The value round-trips through the configured wire format — ownership
/// transfers to `serialize`, and the returned value is what `deserialize`
/// produced from the stored wire string (round-trip parity).
async fn handle_success<O, S: Serdes<O>>(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    serdes: &S,
    value: O,
) -> Result<O, OperationError> {
    let serdes_ctx = SerdesContext::new(wire_id, ctx.execution_arn());

    // Serialize the result (ownership transfers to the serdes).
    //
    // A serialization failure is a LOCAL, deterministic, user-facing
    // failure, so it stays catchable — but the terminal FAIL is persisted
    // FIRST (issue #43). With the failure recorded, replay yields the
    // recorded FAIL instead of re-running the body: the body executes
    // exactly once, and a handler that catches the error branches on a
    // decision replay reproduces.
    let serialized = match serialize_value(serdes, value, serdes_ctx.clone()).await {
        Ok(serialized) => serialized,
        Err(op_err) => {
            // The fixed serialization wire type is the replay
            // discriminator: `replay_failure` matches it to reconstruct
            // `StepErrorKind::SerializationFailed`, so a handler that
            // branches on the kind takes the same path live and replayed.
            let wire = crate::error::serialization_failure_wire(&op_err);
            let update = build_fail_update(wire_id, name, ctx.parent_wire_id(), &wire);
            if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43) — step FAIL (serialization): the body ran,
                // so the failed FAIL write routes unrecoverable with a
                // minimal retry — the record just attempted was already
                // small, but a checkpoint-failure-derived one is the
                // uniform terminal shape.
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal = build_fail_update(wire_id, name, ctx.parent_wire_id(), &cwire);
                return ctx
                    .checkpoint_failure_unrecoverable(wire_id, client_err, Some(terminal))
                    .await;
            }
            return Err(op_err
                .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire));
        }
    };

    // Checkpoint SUCCEED with payload.
    let update = build_succeed_update(wire_id, name, ctx.parent_wire_id(), &serialized);
    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43) — step SUCCEED: the body ran and its side effects
        // need a recorded outcome, so a permanent rejection persists a
        // small terminal FAIL (it goes through on a channel that rejected
        // only the payload) before the execution fails. Yielding the
        // failure instead would let the handler branch on a decision no
        // checkpoint records, and re-invoking without a terminal record
        // would re-run the body once per lap until the execution timeout.
        let cwire = crate::error::checkpoint_failure_wire(&err);
        let terminal = build_fail_update(wire_id, name, ctx.parent_wire_id(), &cwire);
        return ctx
            .checkpoint_failure_unrecoverable(wire_id, err, Some(terminal))
            .await;
    }

    // Return deserialized from the serialized form (round-trip parity).
    deserialize_result(serdes, serialized, serdes_ctx).await
}

/// Handles a failed step: consult retry strategy, checkpoint accordingly.
async fn handle_failure<O>(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    retry_strategy: Option<&RetryStrategy>,
    err: BoxError,
    attempt: u32,
) -> Result<O, OperationError> {
    // Derive the wire failure record from the escaping error before it is
    // moved into the step error: message flattening, `error_type`
    // pass-through, `error_data` chain walk, and stack capture all happen
    // here, at the single wire-derivation site.
    let wire = crate::error::wire_error_for(&*err, STEP_FALLBACK_ERROR_TYPE);
    let step_err = StepError::new(StepErrorKind::ExecutionFailed, Some(err));

    // Consult the retry strategy. The strategy sees the live escaping
    // error through the step error's `source()`.
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
            let update = build_retry_update(wire_id, name, ctx.parent_wire_id(), &wire, delay_secs);
            if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43) — step RETRY: the body ran (and failed), so
                // its side effects need a recorded outcome. A permanent
                // rejection persists a small terminal FAIL before the
                // execution fails; recorded retries replay, but a retry
                // the service never recorded would not.
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal = build_fail_update(wire_id, name, ctx.parent_wire_id(), &cwire);
                return ctx
                    .checkpoint_failure_unrecoverable(wire_id, client_err, Some(terminal))
                    .await;
            }

            // Suspend — the backend owns the retry timer.
            ctx.suspend_now().await
        }
        RetryDecision::Stop => {
            // Checkpoint FAIL (permanent).
            let update = build_fail_update(wire_id, name, ctx.parent_wire_id(), &wire);
            if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43) — step FAIL: the body ran, so the failed
                // FAIL write routes unrecoverable with a minimal terminal
                // FAIL retry (the original may have been rejected for its
                // error payload's size; the checkpoint-failure-derived
                // record is a few hundred bytes).
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal = build_fail_update(wire_id, name, ctx.parent_wire_id(), &cwire);
                return ctx
                    .checkpoint_failure_unrecoverable(wire_id, client_err, Some(terminal))
                    .await;
            }

            // The escaping error stays reachable through `source()`; the
            // attempt count is the kind's structural fact.
            Err(
                OperationError::from_kind(OperationErrorKind::Step(StepError::new(
                    StepErrorKind::RetriesExhausted(crate::error::RetriesExhausted::new(attempt)),
                    step_err.into_source(),
                )))
                .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire),
            )
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
    #[expect(clippy::expect_used)] // reason: all required fields are set above
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

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_retry_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    error: &crate::error::WireError,
    delay_secs: i32,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(STEP_SUB_TYPE)
        .action(OperationAction::Retry)
        .error(error.to_error_object())
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

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_fail_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    error: &crate::error::WireError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(STEP_SUB_TYPE)
        .action(OperationAction::Fail)
        .error(error.to_error_object());

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

// ── Serialization helpers ───────────────────────────────────────────────

/// Serializes a step value through the configured serdes.
///
/// The serdes decides where its work runs (inline for `JsonSerdes`, one
/// blocking task for `FileSystemSerdes`); the SDK awaits the returned
/// future directly. Taking `value` by ownership keeps the future `Send`
/// without requiring `O: Sync`.
async fn serialize_value<O, S: Serdes<O>>(
    serdes: &S,
    value: O,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    serdes
        .serialize(value, serdes_ctx)
        .await
        .map_err(step_serialization_error)
}

/// Deserializes a step wire string through the configured serdes.
async fn deserialize_result<O, S: Serdes<O>>(
    serdes: &S,
    serialized: String,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    serdes
        .deserialize(serialized, serdes_ctx)
        .await
        .map_err(step_serialization_error)
}

/// Wraps a serdes error as a step `SerializationFailed` operation error,
/// carrying the error itself as the source.
fn step_serialization_error(e: BoxError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Step(StepError::new(
        StepErrorKind::SerializationFailed,
        Some(e),
    )))
}

async fn replay_success<O, S: Serdes<O>>(
    serdes: &S,
    result: Option<String>,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let payload = result.unwrap_or_else(|| "null".to_owned());
    deserialize_result(serdes, payload, serdes_ctx).await
}

/// Rebuilds a step error from the failure fields of a replayed record.
///
/// The recorded wire fields travel on the synthetic source (a
/// [`crate::error::ReplayedFailure`]) rather than being folded into a
/// message, so `kind()` remains meaningful after a replay and the
/// recorded `error_type` stays programmatically recoverable through a
/// `source()` downcast.
///
/// A record whose `error_type` is the serialization discriminator
/// ([`crate::error::SERIALIZATION_FAILED_ERROR_TYPE`]) reconstructs
/// `StepErrorKind::SerializationFailed` — the kind the live path yielded
/// after persisting that record — so replay reproduces the recorded
/// failure's classification, not just its message (issue #43).
fn replay_failure(wire: crate::error::WireError, wire_id: &str) -> OperationError {
    let kind = if wire.error_type() == Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE) {
        StepErrorKind::SerializationFailed
    } else {
        StepErrorKind::ExecutionFailed
    };
    OperationError::from_kind(OperationErrorKind::Step(StepError::new(
        kind,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )))
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

/// The wire `ErrorType` recorded for a step failure whose escaping error
/// carries no structured identity.
///
/// A step body returns `Result<O, BoxError>`, so the concrete error type
/// is erased *by the caller* before the SDK ever sees it — `?` boxes the
/// error at the user's call site, and Rust offers no runtime name for a
/// `dyn Error`. Rather than guessing one from a `Debug` rendering, the
/// SDK records this explicit generic name. An error that *does* carry
/// structured identity — an [`OperationError`], or a
/// [`crate::error::ReplayedFailure`] — records its registry name or its
/// original recorded type instead (see [`crate::error::wire_error_for`]).
const STEP_FALLBACK_ERROR_TYPE: &str = "Error";

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions with descriptive messages
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
        let result: Result<i32, OperationError> =
            replay_success(&crate::serdes::JsonSerdes, Some(payload), ctx).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn replay_success_null_returns_unit() {
        let ctx = SerdesContext::new("op-1", "arn:test");
        let result: Result<(), OperationError> =
            replay_success(&crate::serdes::JsonSerdes, None, ctx).await;
        assert!(result.is_ok());
    }

    #[test]
    fn replay_failure_keeps_kind_and_wire_fields_apart() {
        let wire = crate::error::WireError::new(Some("TypeError"), Some("bad input"));
        let err = replay_failure(wire, "wire-1");
        // kind() is meaningful after a replay.
        let OperationErrorKind::Step(step_err) = err.kind() else {
            unreachable!("replay builds a step error");
        };
        assert!(matches!(step_err.kind(), StepErrorKind::ExecutionFailed));
        // The type is NOT folded into the message: the frame stays clean...
        assert_eq!(err.to_string(), "operation error: step");
        // ...and the recorded fields are programmatically recoverable.
        let source = std::error::Error::source(step_err).expect("synthetic source");
        let replayed = source
            .downcast_ref::<crate::error::ReplayedFailure>()
            .expect("replay source is a ReplayedFailure");
        assert_eq!(replayed.error_type(), Some("TypeError"));
        assert_eq!(replayed.error_message(), Some("bad input"));
        // The wire record, operation id, and status are reachable.
        assert_eq!(err.operation_id(), Some("wire-1"));
        assert_eq!(err.status(), Some("FAILED"));
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            Some("TypeError")
        );
        // The chain renders the recorded message via the alternate form —
        // without folding the type into the text.
        let chain = format!("{err:#}");
        assert!(chain.contains("bad input"), "got: {chain}");
        assert!(
            !chain.contains("TypeError"),
            "type must not fold into text: {chain}"
        );
    }

    #[test]
    fn replay_failure_empty_record_displays_unknown() {
        let err = replay_failure(crate::error::WireError::default(), "wire-1");
        let chain = format!("{err:#}");
        assert!(chain.contains("unknown error"), "got: {chain}");
    }

    // ── Serialization tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn serialize_deserialize_round_trip() {
        let ctx = SerdesContext::new("op-1", "arn:test");
        let serialized =
            serialize_value(&crate::serdes::JsonSerdes, "hello".to_owned(), ctx.clone())
                .await
                .unwrap();
        let deserialized: String = deserialize_result(&crate::serdes::JsonSerdes, serialized, ctx)
            .await
            .unwrap();
        assert_eq!(deserialized, "hello");
    }

    #[tokio::test]
    async fn serialize_with_custom_serdes_uppercases() {
        struct Upper;
        impl Serdes<String> for Upper {
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(serde_json::to_string(&value)?.to_uppercase())
            }
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(serde_json::from_str(&wire)?)
            }
        }
        let ctx = SerdesContext::new("op-1", "arn:test");
        let result = serialize_value(&Upper, "hello".to_owned(), ctx)
            .await
            .unwrap();
        assert_eq!(result, "\"HELLO\"");
    }

    // ── Default retry strategy tests ────────────────────────────────────

    #[test]
    fn default_retry_stops_at_max_attempts() {
        let strategy = default_retry_strategy();
        let err = StepError::new(StepErrorKind::ExecutionFailed, Some("fail".into()));
        // Attempt 6 should stop (max_attempts = 6).
        let decision = strategy(&err, 6);
        assert_eq!(decision, RetryDecision::Stop);
    }

    #[test]
    fn default_retry_retries_below_max() {
        let strategy = default_retry_strategy();
        let err = StepError::new(StepErrorKind::ExecutionFailed, Some("fail".into()));
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
        let err = StepError::new(StepErrorKind::ExecutionFailed, Some("fail".into()));
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
        let closure = |_: StepContext| async { Ok::<i32, BoxError>(42) };

        let exec = StepExecution {
            ctx,
            op_id,
            name: Some("test-step".to_owned()),
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn per_operation_serdes_serializes_step_result() {
        use crate::client::InMemoryExecutionClient;

        // A per-operation serdes that uppercases on serialize and lowercases
        // on deserialize. The execution-wide serdes slot was removed with
        // the generic trait; per-operation configuration is the only path.
        struct Upper;
        impl Serdes<String> for Upper {
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(serde_json::to_string(&value)?.to_uppercase())
            }
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(serde_json::from_str(&wire.to_lowercase())?)
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
            None,
        );
        let op_id = ctx.mint_id();
        let closure = |_: StepContext| async { Ok::<String, BoxError>("hello".to_owned()) };

        let exec = StepExecution {
            ctx,
            op_id,
            name: Some("greet".to_owned()),
            retry_strategy: None,
            serdes: Upper,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
        };

        // The step returns the round-tripped value.
        let result = exec.execute().await;
        assert_eq!(result.unwrap(), "hello");

        // The Succeed checkpoint payload was serialized by the configured
        // serdes (uppercased). Without the serdes the payload would be the
        // plain "\"hello\"".
        let updates = client.recorded_updates();
        let succeed = updates
            .iter()
            .find(|u| matches!(u.action(), OperationAction::Succeed))
            .expect("a Succeed update must be recorded");
        assert_eq!(
            succeed.payload(),
            Some("\"HELLO\""),
            "the per-operation serdes must serialize the step result"
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id(); // This mints "1"

        let closure = move |_: StepContext| {
            executed_clone.store(true, Ordering::SeqCst);
            async { Ok::<i32, BoxError>(0) }
        };

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
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
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let closure = move |_: StepContext| {
            executed_clone.store(true, Ordering::SeqCst);
            async { Ok::<String, BoxError>("should not run".to_owned()) }
        };

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{err:#}");
        assert!(err_msg.contains("it broke"), "got: {err_msg}");
        // The recorded type is wire data, not display text.
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            Some("CustomError")
        );
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

        let closure = |_: StepContext| async { Err::<i32, BoxError>("always fails".into()) };

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: Some(no_retry),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("retries exhausted") || err_msg.contains("always fails"),
            "got: {err_msg}"
        );
    }

    /// Acceptance (issue #41): on the live path, an operation failure
    /// exposes the caller's concrete error type through a `source()`
    /// downcast — the escaping error is carried, not stringified.
    #[tokio::test]
    async fn step_live_failure_source_downcasts_to_concrete_user_type() {
        use crate::client::InMemoryExecutionClient;

        #[derive(Debug)]
        struct PaymentDeclined {
            code: u16,
        }
        impl std::fmt::Display for PaymentDeclined {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "payment declined (code {})", self.code)
            }
        }
        impl std::error::Error for PaymentDeclined {}

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

        let no_retry: RetryStrategy =
            Box::new(|_err: &StepError, _attempt: u32| RetryDecision::Stop);
        let closure = |_: StepContext| async {
            Err::<i32, BoxError>(Box::new(PaymentDeclined { code: 402 }))
        };

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: Some(no_retry),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
        };

        let err = exec.execute().await.expect_err("step must fail");

        // Walk source() to the caller's concrete error and downcast it.
        let mut source: Option<&(dyn std::error::Error + 'static)> =
            std::error::Error::source(&err);
        let mut found = None;
        while let Some(e) = source {
            if let Some(declined) = e.downcast_ref::<PaymentDeclined>() {
                found = Some(declined);
                break;
            }
            source = e.source();
        }
        let declined = found.expect("caller's concrete error type must be reachable via source()");
        assert_eq!(declined.code, 402);

        // The wire failure record is reachable from the error too.
        assert_eq!(err.status(), Some("FAILED"));
        let wire = err.wire().expect("live failure carries its wire record");
        assert_eq!(wire.error_type(), Some("Error"));
        assert!(
            wire.error_message()
                .is_some_and(|m| m.contains("payment declined (code 402)")),
            "wire message flattens the chain: {wire:?}"
        );
        assert!(
            !wire.stack_trace().is_empty(),
            "a live failure captures a stack trace"
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

        let closure = |_: StepContext| async { Err::<i32, BoxError>("transient".into()) };

        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            retry_strategy: Some(always_retry),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
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

        let closure = |_: StepContext| async { Err::<i32, BoxError>("transient".into()) };

        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            retry_strategy: Some(fractional_retry),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            closure,
            _marker: std::marker::PhantomData,
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
                let closure = |_: StepContext| async { Ok::<i32, BoxError>(1) };

                let exec = StepExecution {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    retry_strategy: None,
                    serdes: crate::serdes::JsonSerdes,
                    semantics: StepSemantics::default(),
                    closure,
                    _marker: std::marker::PhantomData,
                };
                exec.execute().await
            });

            handle.await.unwrap()
        })
        .await;

        let inner_result = result.unwrap();
        assert!(inner_result.is_err());
        let err_msg = format!("{:#}", inner_result.unwrap_err());
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

        let closure = move |_: StepContext| {
            executed_clone.store(true, Ordering::SeqCst);
            async { Ok::<i32, BoxError>(42) }
        };

        let exec = StepExecution {
            ctx,
            op_id,
            name: Some("at-most-once-step".to_owned()),
            retry_strategy: Some(no_retry),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::AtMostOncePerRetry,
            closure,
            _marker: std::marker::PhantomData,
        };

        let result = exec.execute().await;
        assert!(
            result.is_err(),
            "expected failure for interrupted + no retry"
        );
        let err_msg = format!("{:#}", result.unwrap_err());
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

        let closure = move |_: StepContext| {
            executed_clone.store(true, Ordering::SeqCst);
            async { Ok::<i32, BoxError>(42) }
        };

        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            retry_strategy: Some(always_retry),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::AtMostOncePerRetry,
            closure,
            _marker: std::marker::PhantomData,
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

        let closure = move |_: StepContext| {
            executed_clone.store(true, Ordering::SeqCst);
            async { Ok::<i32, BoxError>(77) }
        };

        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::AtLeastOncePerRetry,
            closure,
            _marker: std::marker::PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), 77);
        assert!(
            executed.load(Ordering::SeqCst),
            "closure SHOULD execute under AtLeastOncePerRetry with Started replay"
        );
    }

    // ── RetryStrategyConfig delay shaping ───────────────────────────────

    use crate::builders::{JitterStrategy, RetryStrategyConfig};

    fn sample_error() -> StepError {
        StepError::new(StepErrorKind::ExecutionFailed, Some("boom".into()))
    }

    /// Extracts the retry delay in whole seconds, panicking on `Stop`.
    fn retry_secs(decision: &RetryDecision) -> u64 {
        match decision {
            RetryDecision::Retry { delay } => delay.as_secs(),
            RetryDecision::Stop => panic!("expected Retry, got Stop"),
        }
    }

    /// The default config with jitter disabled reproduces the documented
    /// constants attempt by attempt: 5s initial delay doubling to a 60s
    /// cap, stopping at the 6th attempt. This pins the deterministic
    /// backbone that `default_retry_strategy` jitters over.
    #[test]
    fn default_config_without_jitter_matches_documented_constants() {
        let config = RetryStrategyConfig::builder()
            .jitter(JitterStrategy::None)
            .build();

        // initial 5s, rate 2.0: 5, 10, 20, 40, then capped at 60.
        let expected = [5u64, 10, 20, 40, 60];
        for (attempt, expected_secs) in (1u32..).zip(expected) {
            assert_eq!(
                retry_secs(&config.decide(attempt)),
                expected_secs,
                "attempt {attempt}"
            );
        }
        assert_eq!(config.decide(6), RetryDecision::Stop);
        assert_eq!(config.decide(7), RetryDecision::Stop);
    }

    /// `default_retry_strategy` agrees with `RetryStrategyConfig::default`
    /// attempt by attempt, modulo jitter: it stops at exactly the same
    /// attempts, and each retry delay falls within the full-jitter envelope
    /// `[1, base]` where `base` is the no-jitter delay for that attempt.
    #[test]
    fn default_retry_strategy_matches_default_config_modulo_jitter() {
        let strategy = default_retry_strategy();
        let no_jitter = RetryStrategyConfig::builder()
            .jitter(JitterStrategy::None)
            .build();
        let err = sample_error();

        for attempt in 1u32..=5 {
            // Sample repeatedly: full jitter randomizes within [1, base].
            let base = retry_secs(&no_jitter.decide(attempt));
            for _ in 0..50 {
                let secs = retry_secs(&strategy(&err, attempt));
                assert!(
                    (1..=base).contains(&secs),
                    "attempt {attempt}: delay {secs}s outside full-jitter envelope [1, {base}]"
                );
            }
        }
        // Both stop at the same attempt boundary (6 total attempts).
        assert_eq!(strategy(&err, 6), RetryDecision::Stop);
        assert_eq!(no_jitter.decide(6), RetryDecision::Stop);
        assert_eq!(strategy(&err, 7), RetryDecision::Stop);
    }

    /// Half jitter keeps every sampled delay within `[base / 2, base]`.
    #[test]
    fn half_jitter_delays_stay_within_bounds() {
        let config = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_secs(8))
            .max_delay(Duration::from_secs(64))
            .backoff_rate(2.0)
            .jitter(JitterStrategy::Half)
            .build();

        for attempt in 1u32..=3 {
            // base: 8, 16, 32 — half-jitter bounds [4, 8], [8, 16], [16, 32].
            let base = 8u64 << (attempt - 1);
            for _ in 0..100 {
                let secs = retry_secs(&config.decide(attempt));
                assert!(
                    (base / 2..=base).contains(&secs),
                    "attempt {attempt}: delay {secs}s outside half-jitter bounds \
                     [{}, {base}]",
                    base / 2
                );
            }
        }
    }

    /// The computed delay is capped at `max_delay` and never drops below
    /// one second, whatever the configuration.
    #[test]
    fn delays_are_capped_and_have_one_second_floor() {
        // Sub-second initial delay rounds up to the 1s floor.
        let tiny = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_millis(1))
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&tiny.decide(1)), 1);

        // A huge backoff rate hits the cap immediately.
        let capped = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_secs(30))
            .max_delay(Duration::from_secs(45))
            .backoff_rate(100.0)
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&capped.decide(1)), 30);
        assert_eq!(retry_secs(&capped.decide(2)), 45);
    }

    /// `max_attempts(1)` never retries: the first failure already reaches
    /// the attempt budget.
    #[test]
    fn max_attempts_one_never_retries() {
        let config = RetryStrategyConfig::builder().max_attempts(1).build();
        assert_eq!(config.decide(1), RetryDecision::Stop);
    }

    /// Non-jittered fractional delays round UP, never down: a 1.1s
    /// configured delay must schedule 2s, not 1s. Nearest-rounding would
    /// truncate 1.1s to 1s and retry EARLIER than configured, conflicting
    /// with the wait and retry-checkpoint behavior, which both ceil
    /// fractional delays.
    #[test]
    fn fractional_delays_round_up_not_to_nearest() {
        let config = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_millis(1100))
            .jitter(JitterStrategy::None)
            .build();
        // 1.1s must become 2s (ceil), not 1s (round-to-nearest).
        assert_eq!(retry_secs(&config.decide(1)), 2);

        // Backoff-computed fractions ceil too: 1.1 * 2 = 2.2s → 3s.
        assert_eq!(retry_secs(&config.decide(2)), 3);

        // Whole-second delays are unchanged by the ceiling: the default
        // 5s/10s/... schedule is preserved.
        let whole = RetryStrategyConfig::builder()
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&whole.decide(1)), 5);
        assert_eq!(retry_secs(&whole.decide(2)), 10);
    }

    /// Regression for the default full-jitter quantization: full jitter
    /// keeps the SDK's legacy nearest-integer rounding
    /// (`round().max(1.0)`), because issue #12 requires
    /// `RetryStrategyConfig::default()` to reproduce the pre-config
    /// behavior exactly — including how fractional jitter samples map to
    /// whole seconds. A ceiling here would shift the delay distribution
    /// for nearly every fractional sample.
    #[test]
    fn default_full_jitter_keeps_legacy_nearest_rounding() {
        // Full jitter: nearest rounding, exactly as before issue #12.
        assert_eq!(quantize_delay_secs(4.4, JitterStrategy::Full), 4);
        assert_eq!(quantize_delay_secs(4.5, JitterStrategy::Full), 5);
        assert_eq!(quantize_delay_secs(59.9, JitterStrategy::Full), 60);
        // Samples near zero still respect the one-second floor.
        assert_eq!(quantize_delay_secs(0.2, JitterStrategy::Full), 1);

        // Non-jittered fractional delays ceil: a configured delay never
        // fires earlier than requested.
        assert_eq!(quantize_delay_secs(4.4, JitterStrategy::None), 5);
        assert_eq!(quantize_delay_secs(4.0, JitterStrategy::None), 4);

        // Half jitter ceils so the documented [base / 2, base] lower
        // bound survives quantization.
        assert_eq!(quantize_delay_secs(4.4, JitterStrategy::Half), 5);
    }
}
