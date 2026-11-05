//! Step operation execution engine.
//!
//! Implements the live path (run closure, serialize, checkpoint), replay path
//! (return frozen result), and retry strategy (backend-owned retry timer).
//!
//! Retry delays use the backend's timer rather than an in-process sleep:
//! a RETRY action with `NextAttemptDelaySeconds` is checkpointed, then the
//! step waits for its record to leave `PENDING`. When the record turns
//! `READY` while other work runs, the next attempt runs in the same
//! invocation; otherwise the invocation suspends.

use std::future::Future;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use tracing::Instrument;

use crate::builders::{Backoff, JitterStrategy, RetryStrategyConfig};
use crate::context::{
    ActiveBody, DurableContext, StepContext, SubmittedWrite, run_body, run_body_sync,
};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{OperationError, OperationErrorKind, StepError, StepErrorKind};
use crate::serdes::SerdesContext;
use crate::start::StartStep;
use crate::tracing_layer;
use crate::{BoxError, OperationSubType, RetryDecision, RetryStrategy, Serdes};

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
    /// Computes the retry decision for `err` at a 1-based failed attempt
    /// number.
    ///
    /// The checks run in the JS order: the attempt cap first (the error
    /// filter is not called once `attempt` reaches `max_attempts`), then
    /// the error filter set with
    /// [`retry_if`](crate::builders::RetryStrategyConfigBuilder::retry_if),
    /// then the delay schedule.
    ///
    /// The base delay follows the [`Backoff`]:
    /// `initial_delay * rate^(attempt - 1)` for an exponential backoff, and
    /// `initial_delay + increment * (attempt - 1)` for a linear one. Both are
    /// capped at `max_delay`, jittered per the configured
    /// [`JitterStrategy`], and rounded to the nearest whole second with a
    /// one-second minimum for every jitter mode (see [`quantize_delay`]).
    /// Stops once `attempt` reaches `max_attempts`.
    pub(crate) fn decide(&self, err: &StepError, attempt: u32) -> RetryDecision {
        if attempt >= self.max_attempts() {
            return RetryDecision::Stop;
        }
        if let Some(predicate) = self.retry_predicate()
            && !predicate(err)
        {
            return RetryDecision::Stop;
        }

        // attempt is 1-based: first failure is attempt=1.
        let initial = self.initial_delay().as_secs_f64();
        let base = match self.backoff() {
            // Exponential backoff: initial * rate^(attempt-1).
            Backoff::Exponential { rate } => {
                let exponent = (i32::try_from(attempt).unwrap_or(1)) - 1;
                initial * rate.powi(exponent)
            }
            // Linear backoff: initial + increment * (attempt-1).
            Backoff::Linear { increment } => {
                initial + increment.as_secs_f64() * f64::from(attempt.saturating_sub(1))
            }
        }
        .min(self.max_delay().as_secs_f64());

        let jittered = match self.jitter() {
            JitterStrategy::None => base,
            // Half jitter: base/2 plus random in [0, base/2] => [base/2, base].
            JitterStrategy::Half => base / 2.0 + rand_full_jitter(base / 2.0),
            // Full jitter: random in [0, base].
            JitterStrategy::Full => rand_full_jitter(base),
        };

        RetryDecision::Retry {
            delay: quantize_delay(jittered),
        }
    }

    /// Converts this configuration into the boxed retry-strategy closure
    /// the step engine consumes. The error reaches only the error filter;
    /// the delay depends only on the attempt number.
    pub(crate) fn into_retry_strategy(self) -> RetryStrategy {
        Box::new(move |err: &StepError, attempt: u32| self.decide(err, attempt))
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

/// Rounds a delay in seconds to the nearest whole second, minimum 1.
/// A non-finite input returns 1 second.
///
/// Both strategy paths ([`RetryStrategyConfig`] and the wait strategy of
/// `wait_for_condition`) quantize a computed delay with this function, for
/// every jitter mode. A delay a caller gives directly (a custom closure or
/// `ctx.wait`) rounds up instead, so it never ends early.
pub(crate) fn quantize_delay(secs: f64) -> Duration {
    Duration::try_from_secs_f64(secs.round().max(1.0)).unwrap_or(Duration::from_secs(1))
}

/// Execution semantics for a step, controlling behavior on interrupted replay.
///
/// When a step is interrupted mid-execution (e.g., a Lambda timeout or crash),
/// the checkpoint log records a `Started` status without an outcome. On the
/// next invocation, the SDK must decide whether to re-execute the step body.
///
/// This is a **client-side only** configuration: it is not sent on the wire.
/// The checkpoint `Start` action is identical regardless of semantics; the
/// difference is purely in the replay decision when `Started` status is
/// encountered.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::StepSemantics;
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
    ///
    /// The body starts without waiting for the `START` checkpoint. The
    /// outcome is written only after `START` is acknowledged; if `START`
    /// fails, the outcome is not written and the invocation fails.
    #[default]
    AtLeastOncePerRetry,

    /// Do **not** re-execute the step body if the previous attempt was
    /// interrupted. Instead, treat the interruption as a failure and
    /// consult the retry strategy to decide whether to schedule a new
    /// attempt or fail permanently.
    ///
    /// The retry strategy sees [`StepErrorKind::Interrupted`], and the
    /// RETRY or FAIL record carries the error type `StepInterruptedError`.
    /// When the strategy stops, the step returns that kind, live and on
    /// replay.
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
    /// The resolved subtype: the builder's `sub_type`, or `Step`.
    pub(crate) sub_type: OperationSubType,
    /// The name the plugin hooks report instead of `name`: the submitter
    /// step of a named `wait_for_callback` reports `<name>-submitter`.
    /// `None` reports `name`.
    pub(crate) plugin_name: Option<String>,
    pub(crate) closure: F,
    pub(crate) _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O, F, Fut, S> StepExecution<O, F, S>
where
    O: Send + 'static,
    F: FnOnce(StepContext) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// Executes the step operation: one loop over its record.
    ///
    /// Thin generic wrapper: the ONLY code monomorphized per step call
    /// site. The replay/checkpoint state machine lives in the non-generic
    /// [`StepCore`] (generic over the result type `O` only); this wrapper
    /// polls the user's concrete future between its halves, one attempt
    /// per cycle, and waits for the retry timer between cycles.
    ///
    /// Each attempt calls a fresh clone of the closure. The original stays
    /// here for the next attempt, so its captures never move into an
    /// attempt.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        let (core, closure) = self.split();
        Self::run(core, closure, None).await
    }

    /// Runs the step's start step now, on the caller's task, and returns
    /// the future of the rest of the step (see [`crate::start`]).
    /// `.spawn()` uses it, so the `START` is queued before it returns.
    pub(crate) fn execute_started(
        self,
    ) -> impl Future<Output = Result<O, OperationError>> + Send + 'static {
        let (core, closure) = self.split();
        let first = core.start();
        Self::run(core, closure, Some(first))
    }

    /// Splits the execution into its non-generic core and the closure.
    fn split(self) -> (StepCore<O, S>, F) {
        let Self {
            ctx,
            op_id,
            name,
            retry_strategy,
            serdes,
            semantics,
            sub_type,
            plugin_name,
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
            sub_type,
            plugin_name,
            _marker: std::marker::PhantomData,
        };
        (core, closure)
    }

    /// The step loop. `first` is the result of the first cycle's start
    /// step when it already ran; `None` runs it in the first cycle.
    async fn run(
        core: StepCore<O, S>,
        closure: F,
        mut first: Option<Result<StartStep, OperationError>>,
    ) -> Result<O, OperationError> {
        // True once the step has waited for its record in this invocation.
        // A terminal record found after that wait changed during this
        // invocation, so it is a continuation, not a replay.
        let mut resumed = false;
        loop {
            match core.before(resumed, first.take()).await? {
                StepPrelude::Done(result) => return result,
                StepPrelude::Wait => {}
                StepPrelude::Run {
                    attempt,
                    span,
                    start,
                } => {
                    // The step body runs as a `Step` body: an operation
                    // claimed inside it fails the execution with a context
                    // usage error. It holds an activity permit from its call
                    // until it returns, so the invocation does not suspend
                    // while it runs.
                    let body = ActiveBody::Step(core.op_id.positional().to_owned());
                    let step_ctx = StepContext::new(attempt);
                    // The clone runs the captures' `Clone` implementations,
                    // which are user code, so it runs under the same body.
                    let attempt_closure = run_body_sync(body.clone(), || closure.clone());
                    let result = core
                        .ctx
                        .hold_activity(
                            run_body(body, async move { (attempt_closure)(step_ctx).await })
                                .instrument(span),
                        )
                        .await;
                    core.await_start(start).await?;
                    match core.settle(attempt, result).await? {
                        StepSettled::Done(value) => return Ok(value),
                        StepSettled::Retry => {}
                    }
                }
            }
            // The record is PENDING until its retry timer fires. Wait until
            // a checkpoint response carries READY or a terminal status, as
            // the Go SDK does, then handle the record as a new invocation
            // would: READY runs the next attempt, a terminal status returns
            // its outcome. The service never answers a RETRY with STARTED,
            // so a STARTED record (a response that did not carry the
            // RETRY's result) keeps waiting rather than starting another
            // attempt at once. When no other SDK work runs, the invocation
            // suspends here instead.
            core.ctx
                .wait_for_record(core.op_id.wire(), |status| {
                    *status == CheckpointStatus::Ready || status.is_terminal()
                })
                .await;
            resumed = true;
        }
    }
}

/// The non-generic half of a step: task-ownership check, replay
/// resolution, attempt derivation, the START checkpoint, and outcome
/// checkpointing. Generic only over the result type `O`, no user closure
/// reaches this state machine, so its substantial replay/checkpoint logic
/// compiles once per result type instead of once per step call site.
struct StepCore<O, S> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    retry_strategy: Option<RetryStrategy>,
    serdes: S,
    semantics: StepSemantics,
    sub_type: OperationSubType,
    /// The name the plugin hooks report instead of `name`, if any.
    plugin_name: Option<String>,
    _marker: std::marker::PhantomData<fn() -> O>,
}

/// What [`StepCore::before`] decided for one cycle of the step loop.
enum StepPrelude<O> {
    /// Resolved without running the body: a recorded terminal outcome, or
    /// an `AtMostOncePerRetry` interruption the retry strategy stopped.
    Done(Result<O, OperationError>),
    /// The record is `PENDING`, or a `RETRY` was just recorded for an
    /// interrupted `AtMostOncePerRetry` attempt: wait for the retry timer.
    Wait,
    /// The body must run at `attempt`, instrumented with `span`.
    Run {
        /// 1-based attempt number for this live execution.
        attempt: u32,
        /// The operation span the body runs under.
        span: tracing::Span,
        /// The in-flight `START` write of an at-least-once attempt, awaited
        /// after the body returns and before the outcome is written. `None`
        /// when START was awaited before the body, or was already recorded.
        /// A body that panics drops it, and the write continues.
        start: Option<SubmittedWrite>,
    },
}

/// How [`StepCore::settle`] ended one attempt.
enum StepSettled<O> {
    /// The step succeeded with this value.
    Done(O),
    /// A `RETRY` was checkpointed: the next cycle waits for its timer.
    Retry,
}

impl<O, S> StepCore<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// Emits `operation_replayed` for a recorded terminal outcome, unless
    /// the record changed during this invocation (`resumed`).
    fn emit_replayed(&self, resumed: bool, attempt: u32) {
        if !resumed {
            self.ctx.emit_operation_replayed(
                self.op_id.wire(),
                self.name.as_deref(),
                "Step",
                Some(self.sub_type.as_str()),
                attempt,
            );
        }
    }

    /// The START update of this step.
    fn start_update(&self) -> OperationUpdate {
        build_start_update(
            self.op_id.wire(),
            self.name.as_deref(),
            self.sub_type.as_str(),
            self.ctx.parent_wire_id(),
        )
    }

    /// Runs everything that precedes the user closure in one cycle: replay
    /// resolution, or the live-path preamble ending at the START
    /// checkpoint.
    ///
    /// `resumed` is true when the step waited for its record in this
    /// invocation. A terminal record then changed during this invocation,
    /// so it returns its outcome without an `operation_replayed` event.
    ///
    /// `first` is the result of this cycle's start step when the caller
    /// already ran it (see [`StepExecution::execute_started`]).
    async fn before(
        &self,
        resumed: bool,
        first: Option<Result<StartStep, OperationError>>,
    ) -> Result<StepPrelude<O>, OperationError> {
        // 1. The start step: admission, task ownership, the replay identity
        // check, and the START of a live attempt (see `crate::start`).
        let started = match first {
            Some(started) => started,
            None => self.start(),
        }?;

        let wire_id = self.op_id.wire();

        // 2. A record that queues no START: replay its outcome, wait for
        // its retry timer, or re-enter its interrupted attempt.
        self.after_start(resumed, started, wire_id).await
    }

    /// The start step of one cycle (see [`crate::start`]).
    fn start(&self) -> Result<StartStep, OperationError> {
        crate::start::start_step(
            &self.ctx,
            &crate::start::StartSpec {
                op_id: &self.op_id,
                operation_type: "Step",
                sub_type: self.sub_type.as_str(),
                name: self.name.as_deref(),
                order: match self.semantics {
                    StepSemantics::AtLeastOncePerRetry => crate::start::StartOrder::QueueThenHook,
                    StepSemantics::AtMostOncePerRetry => crate::start::StartOrder::AwaitResponse,
                },
            },
            |view| self.cycle(view),
            || self.start_update(),
            |plugins, is_replay| self.operation_started(plugins, is_replay),
        )
    }

    /// The rest of one cycle after its start step `started`.
    async fn after_start(
        &self,
        resumed: bool,
        started: StartStep,
        wire_id: &str,
    ) -> Result<StepPrelude<O>, OperationError> {
        let start = match started {
            StartStep::Refused => return std::future::pending().await,
            StartStep::Queued(write) => Some(write),
            // AtLeastOncePerRetry (default): skip re-checkpointing START but
            // re-execute the body. The start step fired the replay start
            // hook.
            StartStep::Recorded(view)
                if view.status == CheckpointStatus::Started
                    && self.semantics == StepSemantics::AtLeastOncePerRetry =>
            {
                None
            }
            StartStep::Recorded(view) => return self.recorded(resumed, &view).await,
        };

        // 3. Live execution path.
        // Derive the current attempt from checkpoint log: if there's a
        // recorded operation with step details, attempt = recorded + 1.
        let attempt = self.ctx.get_attempt(&self.op_id).saturating_add(1);

        // An at-least-once attempt runs its body while START is in flight
        // and awaits it before the outcome write (see `StepPrelude::Run`).
        // An at-most-once body must not run without a recorded START, so
        // that attempt awaits START here, then fires its start hooks.
        let start = match start {
            None => None,
            Some(start) if self.semantics == StepSemantics::AtLeastOncePerRetry => Some(start),
            Some(start) => {
                if let Err(err) = start.await {
                    // Audit (#43): step START: no user code has run for this
                    // attempt, so there is no side effect needing a recorded
                    // outcome. No terminal FAIL: the invocation dies, the
                    // service re-invokes, and replay reaches this point and
                    // attempts the same write, losing only one invocation.
                    return self
                        .ctx
                        .checkpoint_failure_unrecoverable(wire_id, err, None)
                        .await;
                }
                if let Some(plugins) = self.ctx.plugins() {
                    self.operation_started(plugins, attempt > 1);
                }
                None
            }
        };

        // The body runs inside a tracing span carrying the structured-log
        // field contract.
        let is_replay = false; // Live execution is never replay.
        let span = tracing_layer::operation_span(
            self.ctx.execution_arn(),
            &self.ctx.lambda_context().request_id,
            self.ctx.lambda_context().tenant_id.as_deref(),
            wire_id,
            self.name.as_deref(),
            attempt,
            is_replay,
        );

        self.attempt_started(attempt);

        Ok(StepPrelude::Run {
            attempt,
            span,
            start,
        })
    }

    /// Handles a record for which this cycle neither queues a `START` nor
    /// re-enters an at-least-once attempt: it replays a terminal outcome,
    /// waits for a retry timer, or fails an interrupted at-most-once
    /// attempt.
    async fn recorded(
        &self,
        resumed: bool,
        view: &crate::engine::CheckpointStatusView,
    ) -> Result<StepPrelude<O>, OperationError> {
        let positional_id = self.op_id.positional();
        let wire_id = self.op_id.wire();
        match view.status {
            CheckpointStatus::Succeeded => {
                // Decode the recorded payload FIRST: `operation_replayed`
                // promises a recorded terminal outcome was returned, so a
                // corrupt payload or failing serdes surfaces as an error
                // without the event.
                let serdes_ctx = SerdesContext::new(wire_id, self.ctx.execution_arn());
                let payload = self.ctx.checkpoint_result_payload(positional_id);
                let value = replay_success(
                    &self.ctx,
                    self.name.as_deref(),
                    &self.serdes,
                    payload,
                    serdes_ctx,
                )
                .await?;
                self.emit_replayed(resumed, view.attempt);
                Ok(StepPrelude::Done(Ok(value)))
            }
            CheckpointStatus::Failed => {
                self.emit_replayed(resumed, view.attempt);
                let wire = self
                    .ctx
                    .checkpoint_wire_error(positional_id)
                    .unwrap_or_default();
                Ok(StepPrelude::Done(Err(replay_failure(wire, wire_id))))
            }
            CheckpointStatus::Started => {
                // AtMostOncePerRetry: the previous attempt was interrupted
                // before recording an outcome. Do NOT re-execute; treat as
                // a step-interrupted failure and consult the retry
                // strategy to decide FAIL or RETRY.
                let attempt = self.ctx.get_attempt(&self.op_id).saturating_add(1);
                handle_failure(
                    &self.ctx,
                    wire_id,
                    self.name.as_deref(),
                    self.sub_type.as_str(),
                    self.plugin_name.as_deref(),
                    self.retry_strategy.as_ref(),
                    StepErrorKind::Interrupted,
                    STEP_INTERRUPTED_MESSAGE.into(),
                    attempt,
                )
                .await?;
                Ok(StepPrelude::Wait)
            }
            CheckpointStatus::Unknown(ref raw) => {
                // Unreachable in production: `checkpoint_view_validated`
                // already failed the execution (issue #45). Kept as a typed
                // arm so a future bypass cannot fall through to live
                // execution and re-run a possibly-terminal step.
                Err(self.ctx.unrecognized_status_error(wire_id, raw))
            }
            // PENDING: the retry timer hasn't fired yet, so wait for it.
            // `cycle` starts a new attempt for the other statuses of this
            // arm, so only PENDING reaches it.
            CheckpointStatus::Pending
            | CheckpointStatus::Ready
            | CheckpointStatus::Cancelled
            | CheckpointStatus::TimedOut
            | CheckpointStatus::Stopped => Ok(StepPrelude::Wait),
        }
    }

    /// What the start step does with this step's record (see
    /// [`step_cycle`]).
    fn cycle(&self, view: &crate::engine::CheckpointStatusView) -> crate::start::Cycle {
        step_cycle(self.semantics, view)
    }

    /// Fires `on_operation_start` of this step with the status `Started`.
    fn operation_started(&self, plugins: &crate::plugin::InvocationPlugins, is_replay: bool) {
        let info = step_info(
            &self.ctx,
            self.op_id.wire(),
            self.sub_type.as_str(),
            self.name.as_deref(),
            self.plugin_name.as_deref(),
        )
        .with_status(crate::OperationStatus::Started)
        .with_replay(is_replay);
        crate::plugin::operation_started(plugins, &info);
    }

    /// Fires `on_attempt_start` of the live attempt `attempt`, after
    /// `on_operation_start` fired: in the start step for an at-least-once
    /// attempt, or after the START response for an at-most-once one.
    fn attempt_started(&self, attempt: u32) {
        if let Some(plugins) = self.ctx.plugins() {
            crate::plugin::attempt_start_only(
                plugins,
                step_info(
                    &self.ctx,
                    self.op_id.wire(),
                    self.sub_type.as_str(),
                    self.name.as_deref(),
                    self.plugin_name.as_deref(),
                ),
                attempt,
            );
        }
    }

    /// Awaits the `START` write an at-least-once attempt submitted before
    /// its body ran, once the body has returned.
    ///
    /// A failed write fails the invocation, and no outcome is written. No
    /// terminal `FAIL` is needed: at-least-once semantics allow the body to
    /// run again in the next invocation.
    async fn await_start(&self, start: Option<SubmittedWrite>) -> Result<(), OperationError> {
        if let Some(start) = start
            && let Err(err) = start.await
        {
            return self
                .ctx
                .checkpoint_failure_unrecoverable(self.op_id.wire(), err, None)
                .await;
        }
        Ok(())
    }

    /// Settles one attempt: checkpoints success or consults the retry
    /// strategy on failure.
    async fn settle(
        &self,
        attempt: u32,
        result: Result<O, BoxError>,
    ) -> Result<StepSettled<O>, OperationError> {
        let wire_id = self.op_id.wire();
        match result {
            Ok(value) => handle_success(
                &self.ctx,
                wire_id,
                self.name.as_deref(),
                self.sub_type.as_str(),
                self.plugin_name.as_deref(),
                &self.serdes,
                value,
                attempt,
            )
            .await
            .map(StepSettled::Done),
            Err(err) => {
                handle_failure(
                    &self.ctx,
                    wire_id,
                    self.name.as_deref(),
                    self.sub_type.as_str(),
                    self.plugin_name.as_deref(),
                    self.retry_strategy.as_ref(),
                    StepErrorKind::ExecutionFailed,
                    err,
                    attempt,
                )
                .await?;
                Ok(StepSettled::Retry)
            }
        }
    }
}

// Free functions for post-closure operations

/// What the start step does with a step's record under `semantics`: a
/// `READY`, `CANCELLED`, `TIMED_OUT` or `STOPPED` record begins a new
/// attempt, which is a replay from the second attempt on. A `STARTED`
/// record under `AtLeastOncePerRetry` re-enters its attempt, and every
/// other record queues no `START`.
pub(crate) fn step_cycle(
    semantics: StepSemantics,
    view: &crate::engine::CheckpointStatusView,
) -> crate::start::Cycle {
    use crate::start::Cycle;
    match view.status {
        CheckpointStatus::Ready
        | CheckpointStatus::Cancelled
        | CheckpointStatus::TimedOut
        | CheckpointStatus::Stopped => Cycle::Start {
            is_replay: view.attempt > 0,
        },
        CheckpointStatus::Started if semantics == StepSemantics::AtLeastOncePerRetry => {
            Cycle::Resume
        }
        _ => Cycle::Record,
    }
}

/// The plugin info of the step `wire_id`, created in `ctx`. A
/// `plugin_name` replaces the name the info reports.
fn step_info(
    ctx: &DurableContext,
    wire_id: &str,
    sub_type: &str,
    name: Option<&str>,
    plugin_name: Option<&str>,
) -> crate::plugin::OperationInfo {
    let info = crate::plugin::OperationInfo::for_operation(
        ctx,
        wire_id,
        crate::OperationKind::Step,
        sub_type,
        name,
    );
    match plugin_name {
        Some(plugin_name) => info.with_name(Some(plugin_name.to_owned())),
        None => info,
    }
}

/// Handles a successful step execution: serialize, checkpoint, return.
///
/// The value round-trips through the configured wire format: ownership
/// transfers to `serialize`, and the returned value is what `deserialize`
/// produced from the stored wire string (round-trip parity).
#[expect(clippy::too_many_arguments)] // reason: the plugin name joins the inputs the end hooks need
async fn handle_success<O, S: Serdes<O>>(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    plugin_name: Option<&str>,
    serdes: &S,
    value: O,
    attempt: u32,
) -> Result<O, OperationError> {
    let serdes_ctx = SerdesContext::new(wire_id, ctx.execution_arn());

    // Serialize the result (ownership transfers to the serdes).
    //
    // A serialization failure is a LOCAL, deterministic, user-facing
    // failure, so it stays catchable, but the terminal FAIL is persisted
    // FIRST (issue #43). With the failure recorded, replay yields the
    // recorded FAIL instead of re-running the body: the body executes
    // exactly once, and a handler that catches the error branches on a
    // decision replay reproduces.
    let serialized = match serialize_value(ctx, name, serdes, value, serdes_ctx.clone()).await {
        Ok(serialized) => serialized,
        Err(op_err) => {
            // The fixed serialization wire type is the replay
            // discriminator: `replay_failure` matches it to reconstruct
            // `StepErrorKind::SerializationFailed`, so a handler that
            // branches on the kind takes the same path live and replayed.
            let wire = crate::error::serialization_failure_wire(&op_err);
            let update = build_fail_update(wire_id, name, sub_type, ctx.parent_wire_id(), &wire);
            if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43): step FAIL (serialization): the body ran,
                // so the failed FAIL write routes unrecoverable with a
                // minimal retry: the record just attempted was already
                // small, but a checkpoint-failure-derived one is the
                // uniform terminal shape.
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal =
                    build_fail_update(wire_id, name, sub_type, ctx.parent_wire_id(), &cwire);
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
    let update = build_succeed_update(wire_id, name, sub_type, ctx.parent_wire_id(), &serialized);
    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43): step SUCCEED: the body ran and its side effects
        // need a recorded outcome, so a permanent rejection persists a
        // small terminal FAIL (it goes through on a channel that rejected
        // only the payload) before the execution fails. Yielding the
        // failure instead would let the handler branch on a decision no
        // checkpoint records, and re-invoking without a terminal record
        // would re-run the body once per lap until the execution timeout.
        let cwire = crate::error::checkpoint_failure_wire(&err);
        let terminal = build_fail_update(wire_id, name, sub_type, ctx.parent_wire_id(), &cwire);
        return ctx
            .checkpoint_failure_unrecoverable(wire_id, err, Some(terminal))
            .await;
    }

    if let Some(plugins) = ctx.plugins() {
        crate::plugin::attempt_ended(
            plugins,
            step_info(ctx, wire_id, sub_type, name, plugin_name)
                .with_status(crate::OperationStatus::Succeeded)
                .with_result(Some(serialized.clone())),
            crate::plugin::AttemptEnd {
                attempt,
                outcome: crate::plugin::AttemptOutcome::Succeeded,
                error: None,
                ends_operation: true,
            },
        );
    }

    // Return deserialized from the serialized form (round-trip parity).
    deserialize_result(ctx, name, serdes, serialized, serdes_ctx).await
}

/// Handles a failed step: consult retry strategy, checkpoint accordingly.
///
/// `kind` is `StepErrorKind::ExecutionFailed` when the body returned
/// `err`, and `StepErrorKind::Interrupted` when an `AtMostOncePerRetry`
/// attempt started but recorded no outcome. The retry strategy sees
/// `kind`, and the error returned when the strategy stops carries it.
///
/// Returns `Ok(())` once a `RETRY` is recorded: the caller then waits for
/// the retry timer. Returns the step error once a `FAIL` is recorded.
#[expect(clippy::too_many_arguments)] // reason: the subtype joins the inputs every outcome update needs
async fn handle_failure(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    plugin_name: Option<&str>,
    retry_strategy: Option<&RetryStrategy>,
    kind: StepErrorKind,
    err: BoxError,
    attempt: u32,
) -> Result<(), OperationError> {
    // Derive the wire failure record from the escaping error before it is
    // moved into the step error: message flattening, `error_type`
    // pass-through, and the `error_data` chain walk all happen here, at
    // the single wire-derivation site. An interrupted attempt has no
    // typed error, so its fallback type is the one the JS SDK records.
    let fallback = if matches!(kind, StepErrorKind::Interrupted) {
        STEP_INTERRUPTED_ERROR_TYPE
    } else {
        STEP_FALLBACK_ERROR_TYPE
    };
    let wire = crate::error::wire_error_for(&*err, fallback);
    // A plugin sees no hook for an interrupted attempt, as JS: no hook saw
    // that attempt start in this invocation.
    let plugins = ctx
        .plugins()
        .filter(|_| !matches!(kind, StepErrorKind::Interrupted));
    let step_err = StepError::new(kind, Some(err));
    // Fires the attempt-end hooks once the outcome write returned `Ok`.
    let attempt_ended = |status: crate::OperationStatus, ends_operation: bool| {
        if let Some(plugins) = plugins {
            crate::plugin::attempt_ended(
                plugins,
                step_info(ctx, wire_id, sub_type, name, plugin_name)
                    .with_status(status)
                    .with_error(Some(wire.clone())),
                crate::plugin::AttemptEnd {
                    attempt,
                    outcome: crate::plugin::AttemptOutcome::Failed,
                    error: Some(wire.clone()),
                    ends_operation,
                },
            );
        }
    };

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
            let update = build_retry_update(
                wire_id,
                name,
                sub_type,
                ctx.parent_wire_id(),
                &wire,
                delay_secs,
            );
            if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43): step RETRY: the body ran (and failed), so
                // its side effects need a recorded outcome. A permanent
                // rejection persists a small terminal FAIL before the
                // execution fails; recorded retries replay, but a retry
                // the service never recorded would not.
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal =
                    build_fail_update(wire_id, name, sub_type, ctx.parent_wire_id(), &cwire);
                return ctx
                    .checkpoint_failure_unrecoverable(wire_id, client_err, Some(terminal))
                    .await;
            }
            attempt_ended(crate::OperationStatus::Pending, false);
            Ok(())
        }
        RetryDecision::Stop => {
            // Checkpoint FAIL (permanent).
            let update = build_fail_update(wire_id, name, sub_type, ctx.parent_wire_id(), &wire);
            if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                // Audit (#43): step FAIL: the body ran, so the failed
                // FAIL write routes unrecoverable with a minimal terminal
                // FAIL retry (the original may have been rejected for its
                // error payload's size; the checkpoint-failure-derived
                // record is a few hundred bytes).
                let cwire = crate::error::checkpoint_failure_wire(&client_err);
                let terminal =
                    build_fail_update(wire_id, name, sub_type, ctx.parent_wire_id(), &cwire);
                return ctx
                    .checkpoint_failure_unrecoverable(wire_id, client_err, Some(terminal))
                    .await;
            }
            attempt_ended(crate::OperationStatus::Failed, true);

            // The step returns the error the strategy saw, as the JS SDK
            // does: the kind stays the one the strategy saw, and the
            // escaping error stays reachable through `source()`.
            Err(
                OperationError::from_kind(OperationErrorKind::Step(step_err))
                    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
                    .with_wire(wire),
            )
        }
    }
}

// Update builders

fn build_start_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    parent_wire_id: Option<&str>,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(sub_type)
        .action(OperationAction::Start);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    // build() is infallible here: all required fields (id, type, action) set.
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

fn build_succeed_update(
    wire_id: &str,
    name: Option<&str>,
    sub_type: &str,
    parent_wire_id: Option<&str>,
    payload: &str,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(sub_type)
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
    sub_type: &str,
    parent_wire_id: Option<&str>,
    error: &crate::error::WireError,
    delay_secs: i32,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(sub_type)
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
    sub_type: &str,
    parent_wire_id: Option<&str>,
    error: &crate::error::WireError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Step)
        .sub_type(sub_type)
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

// Serialization helpers

/// Serializes a step value through the configured serdes.
///
/// The serdes decides where its work runs (inline for `JsonSerdes`, one
/// blocking task for `FileSystemSerdes`); the SDK awaits the returned
/// future directly. Taking `value` by ownership keeps the future `Send`
/// without requiring `O: Sync`.
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `SerializationFailed`.
async fn serialize_value<O, S: Serdes<O>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &S,
    value: O,
    serdes_ctx: SerdesContext,
) -> Result<String, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx.hold_activity(serdes.serialize(value, serdes_ctx)).await {
        Ok(serialized) => Ok(serialized),
        Err(err) => Err(step_serialization_error(
            ctx.route_serdes_error("serialize", &operation_id, name, err)
                .await,
        )),
    }
}

/// Deserializes a step wire string through the configured serdes.
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `SerializationFailed`.
async fn deserialize_result<O, S: Serdes<O>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &S,
    serialized: String,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx
        .hold_activity(serdes.deserialize(serialized, serdes_ctx))
        .await
    {
        Ok(value) => Ok(value),
        Err(err) => Err(step_serialization_error(
            ctx.route_serdes_error("deserialize", &operation_id, name, err)
                .await,
        )),
    }
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
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &S,
    result: Option<String>,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let payload = result.unwrap_or_else(|| "null".to_owned());
    deserialize_result(ctx, name, serdes, payload, serdes_ctx).await
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
/// `StepErrorKind::SerializationFailed`, the kind the live path yielded
/// after persisting that record (issue #43). A record whose `error_type`
/// is [`STEP_INTERRUPTED_ERROR_TYPE`] reconstructs
/// `StepErrorKind::Interrupted`, the kind the live path yielded after the
/// retry strategy stopped an interrupted `AtMostOncePerRetry` attempt.
/// Every other record is an
/// ordinary terminal step failure, which the live path yields as
/// `StepErrorKind::ExecutionFailed` after its retry strategy stopped, so
/// replay reconstructs the SAME kind: a handler that branches on the
/// kind takes the same path live and replayed.
fn replay_failure(wire: crate::error::WireError, wire_id: &str) -> OperationError {
    let kind = match wire.error_type() {
        Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE) => StepErrorKind::SerializationFailed,
        Some(STEP_INTERRUPTED_ERROR_TYPE) => StepErrorKind::Interrupted,
        _ => StepErrorKind::ExecutionFailed,
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
/// is erased *by the caller* before the SDK ever sees it: `?` boxes the
/// error at the user's call site, and Rust offers no runtime name for a
/// `dyn Error`. Rather than guessing one from a `Debug` rendering, the
/// SDK records this explicit generic name. An error that *does* carry
/// structured identity: an [`OperationError`], or a
/// [`crate::error::ReplayedFailure`]: records its registry name or its
/// original recorded type instead (see [`crate::error::wire_error_for`]).
const STEP_FALLBACK_ERROR_TYPE: &str = "Error";

/// The wire `ErrorType` recorded for an `AtMostOncePerRetry` attempt that
/// started but recorded no outcome. It is the name the JS SDK records.
const STEP_INTERRUPTED_ERROR_TYPE: &str = "StepInterruptedError";

/// The wire `ErrorMessage` recorded for an interrupted `AtMostOncePerRetry`
/// attempt. It is the message the JS SDK records.
const STEP_INTERRUPTED_MESSAGE: &str = "The step execution process was initiated but failed to reach completion due to an interruption.";

#[cfg(test)]
#[expect(clippy::panic)]
// reason: test assertions with descriptive messages
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use std::sync::Arc;

    // Replay tests

    #[tokio::test]
    async fn replay_success_deserializes_json() {
        let payload = "42".to_owned();
        let ctx = SerdesContext::new("op-1", "arn:test");
        let (dctx, _client) = context_over(Vec::new());
        let result: Result<i32, OperationError> =
            replay_success(&dctx, None, &crate::serdes::JsonSerdes, Some(payload), ctx).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn replay_success_null_returns_unit() {
        let ctx = SerdesContext::new("op-1", "arn:test");
        let (dctx, _client) = context_over(Vec::new());
        let result: Result<(), OperationError> =
            replay_success(&dctx, None, &crate::serdes::JsonSerdes, None, ctx).await;
        assert!(result.is_ok());
    }

    #[test]
    fn replay_failure_keeps_kind_and_wire_fields_apart() {
        let wire = crate::error::WireError::new(Some("TypeError"), Some("bad input"));
        let err = replay_failure(wire, "wire-1");
        // kind() is meaningful after a replay: an ordinary terminal step
        // failure replays as ExecutionFailed, the kind the live path
        // yielded.
        let OperationErrorKind::Step(step_err) = err.kind() else {
            unreachable!("replay builds a step error");
        };
        let StepErrorKind::ExecutionFailed = step_err.kind() else {
            unreachable!("ordinary failures replay as ExecutionFailed");
        };
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
        // The chain renders the recorded message via the alternate form:
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

    #[test]
    fn replay_failure_serialization_discriminator_keeps_its_kind() {
        let wire = crate::error::WireError::new(
            Some(crate::error::SERIALIZATION_FAILED_ERROR_TYPE),
            Some("bad payload"),
        );
        let err = replay_failure(wire, "wire-1");
        let OperationErrorKind::Step(step_err) = err.kind() else {
            unreachable!("replay builds a step error");
        };
        assert!(matches!(
            step_err.kind(),
            StepErrorKind::SerializationFailed
        ));
    }

    // Serialization tests

    #[tokio::test]
    async fn serialize_deserialize_round_trip() {
        let ctx = SerdesContext::new("op-1", "arn:test");
        let (dctx, _client) = context_over(Vec::new());
        let serialized = serialize_value(
            &dctx,
            None,
            &crate::serdes::JsonSerdes,
            "hello".to_owned(),
            ctx.clone(),
        )
        .await
        .unwrap();
        let deserialized: String =
            deserialize_result(&dctx, None, &crate::serdes::JsonSerdes, serialized, ctx)
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
        let (dctx, _client) = context_over(Vec::new());
        let result = serialize_value(&dctx, None, &Upper, "hello".to_owned(), ctx)
            .await
            .unwrap();
        assert_eq!(result, "\"HELLO\"");
    }

    // Default retry strategy tests

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

    // Live execution tests (with mock client)

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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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

    /// A root context on the in-memory client, with `hooks` as the
    /// invocation's plugin, and the checkpoint buffer window `window`.
    fn context_with_plugin(
        hooks: &crate::plugin::test_support::OpHooks,
        window: Option<Duration>,
    ) -> (DurableContext, Arc<crate::client::InMemoryExecutionClient>) {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
            window,
            Some(hooks.plugins()),
        );
        (ctx, client)
    }

    #[tokio::test]
    async fn hooks_run_outside_the_token_lock() {
        for window in [Some(Duration::ZERO), None] {
            let lock: Arc<std::sync::OnceLock<Arc<tokio::sync::Mutex<String>>>> = Arc::default();
            let probe_lock = Arc::clone(&lock);
            let hooks = crate::plugin::test_support::OpHooks::probing(move || {
                let free = probe_lock
                    .get()
                    .is_some_and(|mutex| mutex.try_lock().is_ok());
                free.to_string()
            });
            let (ctx, _client) = context_with_plugin(&hooks, window);
            lock.set(ctx.checkpoint_token_mutex()).expect("set once");
            let value = ctx.step(|_| async { Ok(1_u32) }).name("a").await.unwrap();
            assert_eq!(value, 1);
            let probes: Vec<String> = hooks.records().into_iter().map(|r| r.probe).collect();
            assert_eq!(probes, ["true"; 4], "window {window:?}");
        }
    }

    #[tokio::test]
    async fn hooks_run_on_the_operation_task() {
        let hooks = crate::plugin::test_support::OpHooks::probing(|| {
            format!("{:?}", tokio::task::try_id())
        });
        let (ctx, _client) = context_with_plugin(&hooks, Some(Duration::ZERO));
        let body_task = Arc::new(std::sync::Mutex::new(None));
        let seen = Arc::clone(&body_task);
        let value = ctx
            .step(move |_| async move {
                *seen.lock().unwrap() = tokio::task::try_id();
                Ok(1_u32)
            })
            .name("a")
            .spawn()
            .await
            .unwrap();
        assert_eq!(value, 1);
        let body = body_task.lock().unwrap().expect("the body ran on a task");
        let records = hooks.records();
        let probe_of = |hook: &str| {
            records
                .iter()
                .find(|r| r.hook == hook)
                .map(|r| r.probe.clone())
                .expect("the hook ran")
        };
        let expected = format!("{:?}", Some(body));
        assert_eq!(probe_of("attempt_end"), expected);
        assert_eq!(probe_of("end"), expected);
    }

    #[tokio::test]
    async fn no_plugins_keeps_checkpoint_updates() {
        let (ctx, client) = context_over(Vec::new());
        assert!(ctx.plugins().is_none());
        let value = ctx.step(|_| async { Ok(1_u32) }).name("a").await.unwrap();
        assert_eq!(value, 1);
        let updates = client.recorded_updates();
        let wire_id = crate::engine::compute_wire_id_public("1");
        let fields: Vec<_> = updates
            .iter()
            .map(|u| {
                (
                    u.id().to_owned(),
                    u.r#type().clone(),
                    u.sub_type().map(str::to_owned),
                    u.name().map(str::to_owned),
                    u.action().clone(),
                    u.payload().map(str::to_owned),
                    u.parent_id().map(str::to_owned),
                    u.error().is_some(),
                )
            })
            .collect();
        let step = |action, payload: Option<&str>| {
            (
                wire_id.clone(),
                OperationType::Step,
                Some(STEP_SUB_TYPE.to_owned()),
                Some("a".to_owned()),
                action,
                payload.map(str::to_owned),
                None,
                false,
            )
        };
        assert_eq!(
            fields,
            [
                step(OperationAction::Start, None),
                step(OperationAction::Succeed, Some("1")),
            ]
        );
    }

    /// The SDK's in-memory client returns no record, so the end infos of a
    /// failed step carry the error the SDK wrote.
    #[tokio::test]
    async fn failed_step_end_infos_carry_the_written_error() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let (ctx, _client) = context_with_plugin(&hooks, None);
        let err = ctx
            .step(|_| async { Err::<u32, BoxError>("boom".into()) })
            .name("a")
            .retry_strategy(|_err, _attempt| RetryDecision::Stop)
            .await
            .unwrap_err();
        assert!(err.wire().is_some());
        let records = hooks.records();
        let ends: Vec<_> = records
            .iter()
            .filter(|r| r.hook == "attempt_end" || r.hook == "end")
            .collect();
        assert_eq!(ends.len(), 2);
        let attempt_end = ends.first().expect("an attempt end");
        assert_eq!(attempt_end.error.as_deref(), Some("boom"));
        assert_eq!(attempt_end.status, Some(crate::OperationStatus::Failed));
    }

    /// Builds a step record for the first operation of a root context.
    fn step_record(
        status: CheckpointStatus,
        attempt: u32,
        error: Option<&crate::error::WireError>,
    ) -> (String, crate::engine::CheckpointRecord) {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = crate::engine::CheckpointRecord {
            id: wire_key.clone(),
            status,
            result: None,
            error_type: error
                .and_then(crate::error::WireError::error_type)
                .map(str::to_owned),
            error_message: error
                .and_then(crate::error::WireError::error_message)
                .map(str::to_owned),
            error_data: error
                .and_then(crate::error::WireError::error_data)
                .map(str::to_owned),
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

    /// Builds a root context over `records` that writes to a fresh
    /// in-memory client.
    fn context_over(
        records: Vec<(String, crate::engine::CheckpointRecord)>,
    ) -> (DurableContext, Arc<crate::client::InMemoryExecutionClient>) {
        let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(records)),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token0".to_owned(),
        );
        (ctx, client)
    }

    /// The `FAIL` update a test context recorded, as a wire record.
    fn recorded_fail(client: &crate::client::InMemoryExecutionClient) -> crate::error::WireError {
        let updates = client.recorded_updates();
        let fail = updates
            .iter()
            .find(|u| matches!(u.action(), OperationAction::Fail))
            .expect("a FAIL update must be recorded");
        let error = fail
            .error()
            .expect("the FAIL update carries an error object");
        assert!(
            error.stack_trace().is_empty(),
            "the SDK captures no stack trace: {error:?}"
        );
        crate::error::WireError::new(error.error_type(), error.error_message())
            .with_error_data(error.error_data())
    }

    /// A step with subtype `Payment` that fails attempt 1 with a 1 s retry
    /// and succeeds attempt 2.
    fn payment_step(ctx: DurableContext) -> impl Future<Output = Result<i32, OperationError>> {
        let op_id = ctx.mint_id();
        StepExecution {
            ctx,
            op_id,
            name: Some("pay".to_owned()),
            retry_strategy: Some(Box::new(|_err: &StepError, attempt: u32| {
                if attempt >= 2 {
                    RetryDecision::Stop
                } else {
                    RetryDecision::Retry {
                        delay: Duration::from_secs(1),
                    }
                }
            })),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            sub_type: OperationSubType::new("Payment").expect("a valid subtype"),
            plugin_name: None,
            closure: |step_ctx: StepContext| async move {
                if step_ctx.attempt() == 1 {
                    Err::<i32, BoxError>("transient".into())
                } else {
                    Ok(7)
                }
            },
            _marker: std::marker::PhantomData,
        }
        .execute()
    }

    /// A custom subtype is the `SubType` of every update the step writes:
    /// START and RETRY on attempt 1, START and SUCCEED on attempt 2. The
    /// in-memory client never moves the record to `READY`, so attempt 2
    /// runs in a second invocation over a `READY` record.
    #[tokio::test]
    async fn custom_sub_type_on_every_step_update() {
        // Attempt 1: the body fails, the strategy retries, and the step
        // suspends.
        let (ctx, first) = context_over(Vec::new());
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            payment_step(ctx),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        // Attempt 2 over the READY record: the body succeeds.
        let (id, mut record) = step_record(CheckpointStatus::Ready, 1, None);
        record.sub_type = Some("Payment".to_owned());
        let (ctx, second) = context_over(vec![(id, record)]);
        assert_eq!(payment_step(ctx).await.expect("attempt 2 succeeds"), 7);

        let updates: Vec<OperationUpdate> = first
            .recorded_updates()
            .into_iter()
            .chain(second.recorded_updates())
            .collect();
        let actions: Vec<OperationAction> = updates.iter().map(|u| u.action().clone()).collect();
        assert_eq!(
            actions,
            [
                OperationAction::Start,
                OperationAction::Retry,
                OperationAction::Start,
                OperationAction::Succeed
            ]
        );
        for update in &updates {
            assert_eq!(update.sub_type(), Some("Payment"), "{update:?}");
        }
    }

    /// The terminal FAIL the SDK writes after the service rejects the
    /// SUCCEED write carries the custom subtype.
    #[tokio::test]
    async fn custom_sub_type_on_terminal_fail_after_rejected_write() {
        use crate::client::TestResponse;

        let (ctx, client) = context_over(Vec::new());
        client.enqueue_checkpoint_response(TestResponse::Success(vec![]));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError(
            "payload rejected".to_owned(),
        ));
        let op_id = ctx.mint_id();
        let exec = StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("pay".to_owned()),
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            sub_type: OperationSubType::new("Payment").expect("a valid subtype"),
            plugin_name: None,
            closure: |_: StepContext| async { Ok::<i32, BoxError>(7) },
            _marker: std::marker::PhantomData,
        };
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
        assert!(
            matches!(outcome, crate::driver::InvocationOutcome::Failed { .. }),
            "a rejected SUCCEED fails the execution: {outcome:?}"
        );

        let updates = client.recorded_updates();
        let actions: Vec<OperationAction> = updates.iter().map(|u| u.action().clone()).collect();
        assert_eq!(
            actions,
            [
                OperationAction::Start,
                OperationAction::Succeed,
                OperationAction::Fail
            ]
        );
        let terminal = updates.last().expect("the terminal FAIL is recorded");
        assert_eq!(terminal.sub_type(), Some("Payment"));
    }

    /// A step body that fails with a typed error carrying error data.
    async fn typed_failing_body(_: StepContext) -> Result<i32, BoxError> {
        Err(Box::new(
            crate::error::TypedError::with_type("PaymentError", "declined")
                .with_error_data(r#"{"code":42}"#),
        ))
    }

    #[tokio::test]
    async fn step_failure_records_typed_error_data() {
        // Live: the FAIL update carries the type and the data.
        let (ctx, client) = context_over(Vec::new());
        let op_id = ctx.mint_id();
        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: Some(Box::new(|_err: &StepError, _attempt: u32| {
                RetryDecision::Stop
            })),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure: typed_failing_body,
            _marker: std::marker::PhantomData,
        };
        let live = exec.execute().await.expect_err("the step must fail");
        let recorded = recorded_fail(&client);
        assert_eq!(recorded.error_type(), Some("PaymentError"));
        assert_eq!(recorded.error_data(), Some(r#"{"code":42}"#));
        assert_eq!(
            live.wire().and_then(crate::error::WireError::error_data),
            Some(r#"{"code":42}"#)
        );

        // Replay: the recorded data is reachable through `wire()`.
        let replayed_wire = crate::error::WireError::new(Some("PaymentError"), Some("declined"))
            .with_error_data(Some(r#"{"code":42}"#));
        let (ctx, _client) = context_over(vec![step_record(
            CheckpointStatus::Failed,
            0,
            Some(&replayed_wire),
        )]);
        let op_id = ctx.mint_id();
        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::default(),
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure: typed_failing_body,
            _marker: std::marker::PhantomData,
        };
        let err = exec
            .execute()
            .await
            .expect_err("the replayed step must fail");
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_data),
            Some(r#"{"code":42}"#)
        );
    }

    #[tokio::test]
    async fn step_stop_returns_execution_failed_live_and_replay() {
        fn retry_once() -> RetryStrategy {
            Box::new(|_err: &StepError, attempt: u32| {
                if attempt >= 2 {
                    RetryDecision::Stop
                } else {
                    RetryDecision::Retry {
                        delay: Duration::from_secs(1),
                    }
                }
            })
        }
        fn execution(ctx: DurableContext) -> impl Future<Output = Result<i32, OperationError>> {
            let op_id = ctx.mint_id();
            StepExecution {
                ctx,
                op_id,
                name: None,
                retry_strategy: Some(retry_once()),
                serdes: crate::serdes::JsonSerdes,
                semantics: StepSemantics::default(),
                sub_type: OperationSubType::STEP,
                plugin_name: None,
                closure: |_: StepContext| async { Err::<i32, BoxError>("boom".into()) },
                _marker: std::marker::PhantomData,
            }
            .execute()
        }
        fn assert_execution_failed(err: &OperationError) {
            let OperationErrorKind::Step(step_err) = err.kind() else {
                unreachable!("a step builds a step error: {err:?}");
            };
            assert!(
                matches!(step_err.kind(), StepErrorKind::ExecutionFailed),
                "unexpected kind: {:?}",
                step_err.kind()
            );
            let rendered = format!("{err:#}");
            assert!(rendered.ends_with("boom"), "got: {rendered}");
            assert!(!rendered.contains("retries exhausted"), "got: {rendered}");
        }

        // Attempt 1: the strategy retries, so the step suspends.
        let (ctx, client) = context_over(Vec::new());
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            execution(ctx),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        assert!(
            client
                .recorded_updates()
                .iter()
                .any(|u| matches!(u.action(), OperationAction::Retry)),
            "attempt 1 records a RETRY"
        );

        // Attempt 2 (one completed retry): the strategy stops.
        let (ctx, client) = context_over(vec![step_record(CheckpointStatus::Ready, 1, None)]);
        let live = execution(ctx).await.expect_err("the step must fail");
        assert_execution_failed(&live);
        assert_eq!(
            format!("{live:#}"),
            "operation error: step: execution failed: boom"
        );

        // Replay of the recorded FAIL: the same kind and the same chain end.
        let recorded = recorded_fail(&client);
        let (ctx, _client) = context_over(vec![step_record(
            CheckpointStatus::Failed,
            1,
            Some(&recorded),
        )]);
        let replayed = execution(ctx)
            .await
            .expect_err("the replayed step must fail");
        assert_execution_failed(&replayed);
    }

    /// Acceptance (issue #41): on the live path, an operation failure
    /// exposes the caller's concrete error type through a `source()`
    /// downcast: the escaping error is carried, not stringified.
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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
            wire.stack_trace().is_empty(),
            "the SDK captures no stack trace for a live failure"
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure,
            _marker: std::marker::PhantomData,
        };

        let signal = Arc::clone(ctx.coordinator());
        // Retry timer is backend-owned: the operation suspends (parks) rather
        // than surfacing a fabricated error to the caller.
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure,
            _marker: std::marker::PhantomData,
        };

        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
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
        // returns Some: #[tokio::test] root runs in block_on with no task ID.
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
                    sub_type: OperationSubType::STEP,
                    plugin_name: None,
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

    // AtMostOncePerRetry tests

    /// A strategy that records the display of each kind it sees and
    /// returns `decision`.
    fn kind_recording_strategy(
        decision: RetryDecision,
    ) -> (RetryStrategy, Arc<std::sync::Mutex<Vec<String>>>) {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_clone = Arc::clone(&seen);
        let strategy: RetryStrategy = Box::new(move |err: &StepError, _attempt: u32| {
            seen_clone
                .lock()
                .expect("kind log lock")
                .push(err.kind().to_string());
            decision.clone()
        });
        (strategy, seen)
    }

    /// The error object of the first update a test context recorded with
    /// `action`, as a wire record.
    fn recorded_error_for(
        client: &crate::client::InMemoryExecutionClient,
        action: &OperationAction,
    ) -> crate::error::WireError {
        let updates = client.recorded_updates();
        let update = updates
            .iter()
            .find(|u| u.action() == action)
            .unwrap_or_else(|| panic!("a {action:?} update must be recorded: {updates:?}"));
        let error = update.error().expect("the update carries an error object");
        crate::error::WireError::new(error.error_type(), error.error_message())
    }

    /// An `AtMostOncePerRetry` step over a STARTED record, whose body
    /// sets `executed` if it runs.
    fn at_most_once_over_started(
        ctx: DurableContext,
        retry_strategy: Option<RetryStrategy>,
        executed: Arc<std::sync::atomic::AtomicBool>,
    ) -> StepExecution<
        i32,
        impl FnOnce(StepContext) -> std::future::Ready<Result<i32, BoxError>> + Clone,
        crate::serdes::JsonSerdes,
    > {
        let op_id = ctx.mint_id();
        StepExecution {
            ctx,
            op_id,
            name: Some("at-most-once-step".to_owned()),
            retry_strategy,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::AtMostOncePerRetry,
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure: move |_: StepContext| {
                executed.store(true, std::sync::atomic::Ordering::SeqCst);
                std::future::ready(Ok::<i32, BoxError>(42))
            },
            _marker: std::marker::PhantomData,
        }
    }

    #[tokio::test]
    async fn step_at_most_once_replay_started_no_retry_fails_permanently() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let (ctx, client) = context_over(vec![step_record(CheckpointStatus::Started, 0, None)]);
        let (strategy, seen) = kind_recording_strategy(RetryDecision::Stop);
        let exec = at_most_once_over_started(ctx, Some(strategy), Arc::clone(&executed));

        let err = exec
            .execute()
            .await
            .expect_err("expected failure for interrupted + no retry");

        assert_eq!(*seen.lock().expect("kind log lock"), ["step interrupted"]);
        let fail = recorded_error_for(&client, &OperationAction::Fail);
        assert_eq!(fail.error_type(), Some(STEP_INTERRUPTED_ERROR_TYPE));
        assert_eq!(fail.error_message(), Some(STEP_INTERRUPTED_MESSAGE));
        let OperationErrorKind::Step(step_err) = err.kind() else {
            panic!("expected a step error, got {err:?}");
        };
        assert!(
            matches!(step_err.kind(), StepErrorKind::Interrupted),
            "got {step_err:?}"
        );
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            Some(STEP_INTERRUPTED_ERROR_TYPE)
        );
        assert!(
            !executed.load(Ordering::SeqCst),
            "closure should NOT execute when AtMostOncePerRetry and replay=Started"
        );
    }

    #[tokio::test]
    async fn step_at_most_once_replay_started_with_retry_suspends() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let executed = Arc::new(AtomicBool::new(false));
        let (ctx, client) = context_over(vec![step_record(CheckpointStatus::Started, 0, None)]);
        let (strategy, seen) = kind_recording_strategy(RetryDecision::Retry {
            delay: Duration::from_secs(5),
        });
        let signal = Arc::clone(ctx.coordinator());
        let ownership = Arc::clone(ctx.task_ownership());
        let root = ctx.clone();
        let exec = at_most_once_over_started(ctx, Some(strategy), Arc::clone(&executed));

        // Retry timer is backend-owned: the operation suspends (parks) rather
        // than surfacing a fabricated retry-schedule error to the caller.
        let outcome =
            crate::driver::test_support::outcome_of(root, ownership, signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        assert_eq!(*seen.lock().expect("kind log lock"), ["step interrupted"]);
        let retry = recorded_error_for(&client, &OperationAction::Retry);
        assert_eq!(retry.error_type(), Some(STEP_INTERRUPTED_ERROR_TYPE));
        assert_eq!(retry.error_message(), Some(STEP_INTERRUPTED_MESSAGE));
        assert!(
            !executed.load(Ordering::SeqCst),
            "closure should NOT execute when AtMostOncePerRetry and replay=Started"
        );
    }

    #[tokio::test]
    async fn step_at_most_once_interrupted_default_strategy_retries() {
        use std::sync::atomic::AtomicBool;

        let executed = Arc::new(AtomicBool::new(false));
        let (ctx, client) = context_over(vec![step_record(CheckpointStatus::Started, 0, None)]);
        let signal = Arc::clone(ctx.coordinator());
        let ownership = Arc::clone(ctx.task_ownership());
        let root = ctx.clone();
        let exec = at_most_once_over_started(ctx, None, executed);

        let outcome =
            crate::driver::test_support::outcome_of(root, ownership, signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        let retry = recorded_error_for(&client, &OperationAction::Retry);
        assert_eq!(retry.error_type(), Some(STEP_INTERRUPTED_ERROR_TYPE));
    }

    #[tokio::test]
    async fn step_body_error_strategy_sees_execution_failed() {
        let (ctx, client) = context_over(Vec::new());
        let op_id = ctx.mint_id();
        let (strategy, seen) = kind_recording_strategy(RetryDecision::Stop);
        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: Some(strategy),
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::AtMostOncePerRetry,
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure: |_: StepContext| async { Err::<i32, BoxError>("boom".into()) },
            _marker: std::marker::PhantomData,
        };

        let err = exec.execute().await.expect_err("the step must fail");

        assert_eq!(*seen.lock().expect("kind log lock"), ["execution failed"]);
        let fail = recorded_error_for(&client, &OperationAction::Fail);
        assert_eq!(fail.error_type(), Some(STEP_FALLBACK_ERROR_TYPE));
        let OperationErrorKind::Step(step_err) = err.kind() else {
            panic!("expected a step error, got {err:?}");
        };
        assert!(
            matches!(step_err.kind(), StepErrorKind::ExecutionFailed),
            "got {step_err:?}"
        );
    }

    #[tokio::test]
    async fn step_interrupted_fail_record_replays_as_interrupted() {
        let wire = crate::error::WireError::new(
            Some(STEP_INTERRUPTED_ERROR_TYPE),
            Some(STEP_INTERRUPTED_MESSAGE),
        );
        let (ctx, _client) =
            context_over(vec![step_record(CheckpointStatus::Failed, 1, Some(&wire))]);
        let op_id = ctx.mint_id();
        let exec = StepExecution {
            ctx,
            op_id,
            name: None,
            retry_strategy: None,
            serdes: crate::serdes::JsonSerdes,
            semantics: StepSemantics::AtMostOncePerRetry,
            sub_type: OperationSubType::STEP,
            plugin_name: None,
            closure: |_: StepContext| async { Ok::<i32, BoxError>(1) },
            _marker: std::marker::PhantomData,
        };

        let err = exec
            .execute()
            .await
            .expect_err("the replayed step must fail");

        let OperationErrorKind::Step(step_err) = err.kind() else {
            panic!("expected a step error, got {err:?}");
        };
        assert!(
            matches!(step_err.kind(), StepErrorKind::Interrupted),
            "got {step_err:?}"
        );
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            Some(STEP_INTERRUPTED_ERROR_TYPE)
        );
        let replayed = std::error::Error::source(step_err)
            .and_then(|s| s.downcast_ref::<crate::error::ReplayedFailure>())
            .expect("replay source is a ReplayedFailure");
        assert_eq!(replayed.error_type(), Some(STEP_INTERRUPTED_ERROR_TYPE));
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
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
            sub_type: OperationSubType::STEP,
            plugin_name: None,
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

    // RetryStrategyConfig delay shaping

    use crate::builders::{Backoff, JitterStrategy, RetryStrategyConfig};

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
            .max_attempts(6)
            .max_delay(Duration::from_mins(1))
            .jitter(JitterStrategy::None)
            .build();

        // initial 5s, rate 2.0: 5, 10, 20, 40, then capped at 60.
        let expected = [5u64, 10, 20, 40, 60];
        for (attempt, expected_secs) in (1u32..).zip(expected) {
            assert_eq!(
                retry_secs(&config.decide(&sample_error(), attempt)),
                expected_secs,
                "attempt {attempt}"
            );
        }
        assert_eq!(config.decide(&sample_error(), 6), RetryDecision::Stop);
        assert_eq!(config.decide(&sample_error(), 7), RetryDecision::Stop);
    }

    /// `default_retry_strategy` agrees with `RetryStrategyConfig::default`
    /// attempt by attempt, modulo jitter: it stops at exactly the same
    /// attempts, and each retry delay falls within the full-jitter envelope
    /// `[1, base]` where `base` is the no-jitter delay for that attempt.
    #[test]
    fn default_retry_strategy_matches_default_config_modulo_jitter() {
        let strategy = default_retry_strategy();
        let no_jitter = RetryStrategyConfig::builder()
            .max_attempts(6)
            .max_delay(Duration::from_mins(1))
            .jitter(JitterStrategy::None)
            .build();
        let err = sample_error();

        for attempt in 1u32..=5 {
            // Sample repeatedly: full jitter randomizes within [1, base].
            let base = retry_secs(&no_jitter.decide(&sample_error(), attempt));
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
        assert_eq!(no_jitter.decide(&sample_error(), 6), RetryDecision::Stop);
        assert_eq!(strategy(&err, 7), RetryDecision::Stop);
    }

    /// Half jitter keeps every sampled delay within `[base / 2, base]`.
    #[test]
    fn half_jitter_delays_stay_within_bounds() {
        let config = RetryStrategyConfig::builder()
            .max_attempts(6)
            .initial_delay(Duration::from_secs(8))
            .max_delay(Duration::from_secs(64))
            .backoff_rate(2.0)
            .jitter(JitterStrategy::Half)
            .build();

        for attempt in 1u32..=3 {
            // base: 8, 16, 32: half-jitter bounds [4, 8], [8, 16], [16, 32].
            let base = 8u64 << (attempt - 1);
            for _ in 0..100 {
                let secs = retry_secs(&config.decide(&sample_error(), attempt));
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
        // A sub-second initial delay rises to the 1s floor.
        let tiny = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_millis(1))
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&tiny.decide(&sample_error(), 1)), 1);

        // A huge backoff rate hits the cap immediately.
        let capped = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_secs(30))
            .max_delay(Duration::from_secs(45))
            .backoff_rate(100.0)
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&capped.decide(&sample_error(), 1)), 30);
        assert_eq!(retry_secs(&capped.decide(&sample_error(), 2)), 45);
    }

    /// `max_attempts(1)` never retries: the first failure already reaches
    /// the attempt budget.
    #[test]
    fn max_attempts_one_never_retries() {
        let config = RetryStrategyConfig::builder().max_attempts(1).build();
        assert_eq!(config.decide(&sample_error(), 1), RetryDecision::Stop);
    }

    /// Every row of the `quantize_delay` case table: round to nearest,
    /// one-second floor, and 1 second for a non-finite input. The rule is
    /// the JS rule `Math.max(1, Math.round(x))`.
    #[test]
    fn quantize_delay_rounds_to_nearest_with_one_second_floor() {
        let cases = [
            (0.0, 1u64),
            (0.4, 1),
            (1.1, 1),
            (2.25, 2),
            (2.5, 3),
            (4.4, 4),
            (5.0, 5),
            (f64::NAN, 1),
            (f64::INFINITY, 1),
        ];
        for (input, expected) in cases {
            assert_eq!(
                quantize_delay(input),
                Duration::from_secs(expected),
                "input {input}"
            );
        }
    }

    /// Non-jittered fractional delays round to the nearest whole second,
    /// as in JS and Go: 1.1 s schedules 1 s, 2.2 s schedules 2 s, 2.25 s
    /// schedules 2 s, and 2.5 s schedules 3 s.
    #[test]
    fn fractional_delays_round_to_nearest() {
        let config = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_millis(1100))
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&config.decide(&sample_error(), 1)), 1);
        // 1.1 * 2 = 2.2 s.
        assert_eq!(retry_secs(&config.decide(&sample_error(), 2)), 2);

        // 1 * 1.5^2 = 2.25 s.
        let slow = RetryStrategyConfig::builder()
            .max_attempts(6)
            .initial_delay(Duration::from_secs(1))
            .backoff_rate(1.5)
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&slow.decide(&sample_error(), 3)), 2);

        let half = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_millis(2500))
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(retry_secs(&half.decide(&sample_error(), 1)), 3);
    }

    /// Every row of the linear backoff case table, with no jitter. The
    /// JS formula is `min(initialDelay + increment * (attempt - 1),
    /// maxDelay)`, rounded to the nearest second with a 1 s floor. Row G
    /// pins the exponential default schedule, unchanged.
    #[test]
    fn linear_backoff_cases() {
        let linear = |initial: Duration, increment: Duration| {
            RetryStrategyConfig::builder()
                .initial_delay(initial)
                .backoff(Backoff::Linear { increment })
                .jitter(JitterStrategy::None)
        };
        let delays = |config: &RetryStrategyConfig, attempts: std::ops::RangeInclusive<u32>| {
            attempts
                .map(|attempt| retry_secs(&config.decide(&sample_error(), attempt)))
                .collect::<Vec<_>>()
        };

        // A: the preset's 1, 2, 3, 4, 5 s.
        let preset = RetryStrategyConfig::linear();
        assert_eq!(delays(&preset, 1..=5), [1, 2, 3, 4, 5], "case A");
        // B: the sixth failure is final.
        assert_eq!(
            preset.decide(&sample_error(), 6),
            RetryDecision::Stop,
            "case B"
        );

        // C: 1, 11, then 21 capped at 15.
        let capped = linear(Duration::from_secs(1), Duration::from_secs(10))
            .max_delay(Duration::from_secs(15))
            .max_attempts(6)
            .build();
        assert_eq!(delays(&capped, 1..=3), [1, 11, 15], "case C");

        // D: 1, 1.25, 1.5, 1.75 round to nearest: 1, 1, 2, 2.
        let fractional = linear(Duration::from_secs(1), Duration::from_millis(250))
            .max_attempts(6)
            .build();
        assert_eq!(delays(&fractional, 1..=4), [1, 1, 2, 2], "case D");

        // E: a zero delay rises to the 1 s floor.
        let zero = linear(Duration::ZERO, Duration::ZERO)
            .max_attempts(3)
            .build();
        assert_eq!(delays(&zero, 1..=1), [1], "case E");

        // F: no_retry stops on the first failure.
        assert_eq!(
            RetryStrategyConfig::no_retry().decide(&sample_error(), 1),
            RetryDecision::Stop,
            "case F"
        );

        // G: the default exponential schedule without jitter, unchanged.
        let exponential = RetryStrategyConfig::builder()
            .max_attempts(6)
            .max_delay(Duration::from_mins(1))
            .jitter(JitterStrategy::None)
            .build();
        assert_eq!(delays(&exponential, 1..=5), [5, 10, 20, 40, 60], "case G");
    }

    /// Half jitter on a linear backoff keeps every sampled delay within
    /// `[base / 2, base]`, with base 8, 16 and 24 s.
    #[test]
    fn linear_half_jitter_stays_within_bounds() {
        let config = RetryStrategyConfig::builder()
            .initial_delay(Duration::from_secs(8))
            .backoff(Backoff::Linear {
                increment: Duration::from_secs(8),
            })
            .jitter(JitterStrategy::Half)
            .build();

        for (attempt, base) in [(1u32, 8u64), (2, 16), (3, 24)] {
            for _ in 0..100 {
                let secs = retry_secs(&config.decide(&sample_error(), attempt));
                assert!(
                    (base / 2..=base).contains(&secs),
                    "attempt {attempt}: delay {secs}s outside half-jitter bounds \
                     [{}, {base}]",
                    base / 2
                );
            }
        }
    }

    /// A test error type for the `retry_if` case table.
    #[derive(Debug)]
    struct Transient;

    impl std::fmt::Display for Transient {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("transient")
        }
    }

    impl std::error::Error for Transient {}

    fn is_transient(err: &StepError) -> bool {
        std::error::Error::source(err)
            .and_then(|e| e.downcast_ref::<Transient>())
            .is_some()
    }

    fn mentions_timeout(err: &StepError) -> bool {
        std::error::Error::source(err)
            .map(ToString::to_string)
            .is_some_and(|message| message.contains("timeout"))
    }

    fn body_error(source: crate::error::Source) -> StepError {
        StepError::new(StepErrorKind::ExecutionFailed, Some(source))
    }

    /// Every row of the `retry_if` case table: the cap is checked first
    /// (the predicate is not called at the cap), then the predicate, then
    /// the unchanged delay schedule. Matching on the source's type
    /// (`instanceof` in JS), on its message (`includes` in JS), and on
    /// both with `||` (the OR of the two JS lists).
    #[test]
    fn retry_if_filter_cases() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        type Predicate = fn(&StepError) -> bool;
        /// (case, predicate, error, attempt, expected delay secs, predicate called)
        type Case = (
            &'static str,
            Option<Predicate>,
            StepError,
            u32,
            Option<u64>,
            bool,
        );
        let either: Predicate = |err| is_transient(err) || mentions_timeout(err);
        let boom = || body_error("boom".into());
        let transient = || body_error(Box::new(Transient));
        let timeout = || body_error("read timeout".into());

        let cases: Vec<Case> = vec![
            ("A", None, boom(), 1, Some(1), false),
            ("B", Some(|_| false), boom(), 1, None, true),
            ("C", Some(|_| true), boom(), 2, Some(2), true),
            ("D", Some(|_| true), boom(), 3, None, false),
            ("E", Some(is_transient), transient(), 1, Some(1), true),
            ("F", Some(is_transient), boom(), 1, None, true),
            ("G", Some(mentions_timeout), timeout(), 1, Some(1), true),
            ("H", Some(either), boom(), 1, None, true),
            ("I", Some(either), timeout(), 1, Some(1), true),
        ];

        for (case, predicate, err, attempt, expected, called) in cases {
            let calls = Arc::new(AtomicU32::new(0));
            let mut builder = RetryStrategyConfig::builder()
                .max_attempts(3)
                .initial_delay(Duration::from_secs(1))
                .jitter(JitterStrategy::None);
            if let Some(predicate) = predicate {
                let calls = Arc::clone(&calls);
                builder = builder.retry_if(move |err| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    predicate(err)
                });
            }
            let config = builder.build();

            let decision = config.decide(&err, attempt);
            let expected = expected.map_or(RetryDecision::Stop, |secs| RetryDecision::Retry {
                delay: Duration::from_secs(secs),
            });
            assert_eq!(decision, expected, "case {case}");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                u32::from(called),
                "case {case}: predicate calls"
            );
        }
    }

    /// The predicate sees the strategy's error kind, so a filter can tell
    /// an interrupted at-most-once attempt from a body error.
    #[test]
    fn retry_if_sees_the_error_kind() {
        let config = RetryStrategyConfig::builder()
            .max_attempts(3)
            .initial_delay(Duration::from_secs(1))
            .jitter(JitterStrategy::None)
            .retry_if(|err| matches!(err.kind(), StepErrorKind::ExecutionFailed))
            .build();
        let interrupted = StepError::new(StepErrorKind::Interrupted, None);
        assert_eq!(config.decide(&interrupted, 1), RetryDecision::Stop);
        assert_eq!(
            config.decide(&sample_error(), 1),
            RetryDecision::Retry {
                delay: Duration::from_secs(1)
            }
        );
    }

    /// `into_retry_strategy` passes the error to the filter.
    #[test]
    fn into_retry_strategy_applies_retry_if() {
        let strategy = RetryStrategyConfig::builder()
            .retry_if(|_| false)
            .build()
            .into_retry_strategy();
        assert_eq!(strategy(&sample_error(), 1), RetryDecision::Stop);
    }
}

#[cfg(test)]
mod coordinator_tests {
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{
        RefreshClient, RefreshRequest, assert_near, drive_handler, named, operation_like,
        poll_once, refresh_ctx, wire_err,
    };
    use aws_sdk_lambda::types::{
        ErrorObject, Operation, OperationAction, OperationStatus, OperationType, OperationUpdate,
        StepDetails,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    /// A step record of `update` with `status` and `attempt`, and, when
    /// `next_in` is `Some`, a next attempt `next_in` from now.
    fn step_record(
        req: &RefreshRequest<'_>,
        update: &OperationUpdate,
        status: OperationStatus,
        next_in: Option<Duration>,
    ) -> Operation {
        let mut details = StepDetails::builder().attempt(1);
        if let Some(next_in) = next_in {
            details = details.next_attempt_timestamp(req.timestamp_in(next_in));
        }
        operation_like(update, status)
            .step_details(details.build())
            .build()
            .expect("all required fields set")
    }

    /// Answers a RETRY of `name` with `PENDING` and a next attempt 1 s
    /// ahead, and the first refresh with `name` `READY`.
    fn retry_client(name: &'static str) -> RefreshClient {
        RefreshClient::new(move |req| {
            let mut ops = Vec::new();
            if let Some(retry) = named(req.updates, name, &OperationAction::Retry) {
                ops.push(step_record(
                    req,
                    retry,
                    OperationStatus::Pending,
                    Some(secs(1)),
                ));
            }
            if req.empty_number == 1
                && let Some(retry) = named(req.history, name, &OperationAction::Retry)
            {
                ops.push(step_record(req, retry, OperationStatus::Ready, None));
            }
            Ok(ops)
        })
    }

    /// Retries attempt 1 after 1 s and stops at attempt 2.
    fn retry_once(_err: &crate::StepError, attempt: u32) -> crate::RetryDecision {
        if attempt < 2 {
            crate::RetryDecision::Retry { delay: secs(1) }
        } else {
            crate::RetryDecision::Stop
        }
    }

    /// A step that sleeps 3 s of tokio time.
    async fn sleeping_step(ctx: &crate::DurableContext) -> Result<u32, crate::error::WireError> {
        ctx.step(|_| async {
            tokio::time::sleep(secs(3)).await;
            Ok(1_u32)
        })
        .name("s")
        .await
        .map_err(wire_err)
    }

    #[tokio::test(start_paused = true)]
    async fn step_retry_runs_next_attempt_in_invocation() {
        let client = Arc::new(retry_client("flaky"));
        let ctx = refresh_ctx(&client);
        let start = tokio::time::Instant::now();
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let body_attempts = Arc::clone(&attempts);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut flaky = std::pin::pin!(
                ctx.step(move |step_ctx| {
                    let attempts = Arc::clone(&body_attempts);
                    async move {
                        attempts
                            .lock()
                            .expect("lock")
                            .push((step_ctx.attempt(), start.elapsed()));
                        if step_ctx.attempt() < 2 {
                            return Err::<u32, crate::BoxError>("transient".into());
                        }
                        Ok(7)
                    }
                })
                .name("flaky")
                .retry_strategy(retry_once)
                .spawn()
            );
            poll_once(flaky.as_mut()).await;
            sleeping_step(&ctx).await?;
            let value = flaky.await.map_err(wire_err)?;
            Ok(value.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("7".to_owned()));
        let attempts = attempts.lock().expect("lock").clone();
        let numbers: Vec<u32> = attempts.iter().map(|(n, _)| *n).collect();
        assert_eq!(numbers, vec![1, 2]);
        assert_near(attempts.get(1).expect("attempt 2 ran").1, secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn step_retry_in_invocation_fires_start() {
        use crate::plugin::AttemptOutcome::{Failed, Succeeded};
        let client = Arc::new(retry_client("flaky"));
        let hooks = crate::plugin::test_support::OpHooks::default();
        let ctx = crate::DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            None,
            Some(hooks.plugins()),
        );
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut flaky = std::pin::pin!(
                ctx.step(|step_ctx| async move {
                    if step_ctx.attempt() < 2 {
                        return Err::<u32, crate::BoxError>("transient".into());
                    }
                    Ok(7)
                })
                .name("flaky")
                .retry_strategy(retry_once)
                .spawn()
            );
            poll_once(flaky.as_mut()).await;
            sleeping_step(&ctx).await?;
            let value = flaky.await.map_err(wire_err)?;
            Ok(value.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("7".to_owned()));
        let records = hooks.of("flaky");
        let shapes: Vec<_> = records
            .iter()
            .map(crate::plugin::test_support::OpRecord::shape)
            .collect();
        assert_eq!(
            shapes,
            [
                ("start", false, None, None),
                ("attempt_start", false, Some(1), None),
                ("attempt_end", false, Some(1), Some(Failed)),
                ("start", true, None, None),
                ("attempt_start", false, Some(2), None),
                ("attempt_end", false, Some(2), Some(Succeeded)),
                ("end", false, None, None),
            ]
        );
        let error = records.get(2).and_then(|r| r.error.as_deref());
        assert_eq!(error, Some("transient"));
        // The client returns no record for the SUCCEED, so the result is
        // the one the SDK wrote.
        let result = records.get(5).and_then(|r| r.result.as_deref());
        assert_eq!(result, Some("7"));
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
    async fn each_attempt_starts_from_original_captures() {
        let client = Arc::new(retry_client("vec"));
        let ctx = refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let v = vec![0_u32];
            let mut step = std::pin::pin!(
                ctx.step(move |step_ctx| async move {
                    let mut v = v;
                    v.push(step_ctx.attempt());
                    if step_ctx.attempt() < 2 {
                        return Err::<Vec<u32>, crate::BoxError>("transient".into());
                    }
                    Ok(v)
                })
                .name("vec")
                .retry_strategy(retry_once)
                .spawn()
            );
            poll_once(step.as_mut()).await;
            sleeping_step(&ctx).await?;
            let v = step.await.map_err(wire_err)?;
            Ok(format!("{v:?}"))
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("[0, 2]".to_owned()));
    }

    #[tokio::test(start_paused = true)]
    async fn arc_capture_is_shared_across_attempts_in_invocation() {
        let client = Arc::new(retry_client("arc"));
        let ctx = refresh_ctx(&client);
        let counter = Arc::new(AtomicUsize::new(0));
        let handler_counter = Arc::clone(&counter);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut step = std::pin::pin!(
                ctx.step(move |step_ctx| async move {
                    handler_counter.fetch_add(1, Ordering::SeqCst);
                    if step_ctx.attempt() < 2 {
                        return Err::<u32, crate::BoxError>("transient".into());
                    }
                    Ok(1)
                })
                .name("arc")
                .retry_strategy(retry_once)
                .spawn()
            );
            poll_once(step.as_mut()).await;
            sleeping_step(&ctx).await?;
            step.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn lone_step_retry_suspends() {
        let client = Arc::new(retry_client("lone"));
        let ctx = refresh_ctx(&client);
        let runs = Arc::new(AtomicUsize::new(0));
        let body_runs = Arc::clone(&runs);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            ctx.step(move |_| async move {
                body_runs.fetch_add(1, Ordering::SeqCst);
                Err::<u32, crate::BoxError>("transient".into())
            })
            .name("lone")
            .retry_strategy(retry_once)
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(runs.load(Ordering::SeqCst), 1, "the body ran once");
        assert!(client.empty_calls().is_empty(), "no refresh was sent");
    }

    #[tokio::test(start_paused = true)]
    async fn replayed_pending_step_runs_next_attempt_in_invocation() {
        let client = Arc::new(RefreshClient::new(|req| {
            let mut ops = Vec::new();
            if req.empty_number == 1 {
                ops.push(
                    operation_like(&flaky_update(), OperationStatus::Ready)
                        .step_details(StepDetails::builder().attempt(1).build())
                        .build()
                        .expect("all required fields set"),
                );
            }
            Ok(ops)
        }));
        let ctx = pending_flaky_ctx(&client);
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let body_attempts = Arc::clone(&attempts);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut flaky = std::pin::pin!(
                ctx.step(move |step_ctx| {
                    let attempts = Arc::clone(&body_attempts);
                    async move {
                        attempts.lock().expect("lock").push(step_ctx.attempt());
                        Ok::<u32, crate::BoxError>(7)
                    }
                })
                .name("flaky")
                .retry_strategy(retry_once)
                .spawn()
            );
            poll_once(flaky.as_mut()).await;
            sleeping_step(&ctx).await?;
            let value = flaky.await.map_err(wire_err)?;
            Ok(value.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("7".to_owned()));
        assert_eq!(*attempts.lock().expect("lock"), vec![2]);
    }

    /// The root context over a log that holds the step `flaky` `PENDING`
    /// after attempt 1, with its next attempt 1 s ahead.
    fn pending_flaky_ctx(client: &Arc<RefreshClient>) -> crate::DurableContext {
        let pending = operation_like(&flaky_update(), OperationStatus::Pending)
            .step_details(
                StepDetails::builder()
                    .attempt(1)
                    .next_attempt_timestamp(aws_smithy_types::DateTime::from(
                        std::time::SystemTime::now() + secs(1),
                    ))
                    .build(),
            )
            .build()
            .expect("all required fields set");
        crate::DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::client::operations_to_checkpoint_log(&[pending])),
            Arc::clone(client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
        )
    }

    /// A waiting step whose record turns terminal returns the recorded
    /// outcome in the invocation, without running the body and without an
    /// `operation_replayed` event.
    #[tokio::test(start_paused = true)]
    async fn pending_step_returns_terminal_record_in_invocation() {
        let (events, _guard) = crate::driver::test_support::EventMessages::install();
        let client = Arc::new(RefreshClient::new(|req| {
            let mut ops = Vec::new();
            if req.empty_number == 1 {
                ops.push(
                    operation_like(&flaky_update(), OperationStatus::Succeeded)
                        .step_details(StepDetails::builder().attempt(2).result("5").build())
                        .build()
                        .expect("all required fields set"),
                );
            }
            Ok(ops)
        }));
        let ctx = pending_flaky_ctx(&client);
        let runs = Arc::new(AtomicUsize::new(0));
        let body_runs = Arc::clone(&runs);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut flaky = std::pin::pin!(
                ctx.step(move |_| async move {
                    body_runs.fetch_add(1, Ordering::SeqCst);
                    Ok::<u32, crate::BoxError>(7)
                })
                .name("flaky")
                .spawn()
            );
            poll_once(flaky.as_mut()).await;
            sleeping_step(&ctx).await?;
            let value = flaky.await.map_err(wire_err)?;
            Ok(value.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("5".to_owned()));
        assert_eq!(runs.load(Ordering::SeqCst), 0, "the body never ran");
        assert_eq!(
            events.replayed(),
            0,
            "a resumed record emits no replay event"
        );
    }

    /// A waiting step whose record turns FAILED returns the recorded error
    /// in the invocation, without running the body and without an
    /// `operation_replayed` event.
    #[tokio::test(start_paused = true)]
    async fn pending_step_returns_failed_record_in_invocation() {
        let (events, _guard) = crate::driver::test_support::EventMessages::install();
        let client = Arc::new(RefreshClient::new(|req| {
            let mut ops = Vec::new();
            if req.empty_number == 1 {
                ops.push(
                    operation_like(&flaky_update(), OperationStatus::Failed)
                        .step_details(
                            StepDetails::builder()
                                .attempt(2)
                                .error(
                                    ErrorObject::builder()
                                        .error_type("PaymentError")
                                        .error_message("declined")
                                        .build(),
                                )
                                .build(),
                        )
                        .build()
                        .expect("all required fields set"),
                );
            }
            Ok(ops)
        }));
        let ctx = pending_flaky_ctx(&client);
        let runs = Arc::new(AtomicUsize::new(0));
        let body_runs = Arc::clone(&runs);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let mut flaky = std::pin::pin!(
                ctx.step(move |_| async move {
                    body_runs.fetch_add(1, Ordering::SeqCst);
                    Ok::<u32, crate::BoxError>(7)
                })
                .name("flaky")
                .spawn()
            );
            poll_once(flaky.as_mut()).await;
            sleeping_step(&ctx).await?;
            let err = flaky.await.expect_err("the recorded failure returns");
            let error_type = err
                .wire()
                .and_then(crate::WireError::error_type)
                .unwrap_or_default()
                .to_owned();
            Ok(error_type)
        })
        .await
        .expect("no panic");
        assert_eq!(
            outcome,
            InvocationOutcome::Complete("PaymentError".to_owned())
        );
        assert_eq!(runs.load(Ordering::SeqCst), 0, "the body never ran");
        assert_eq!(
            events.replayed(),
            0,
            "a resumed record emits no replay event"
        );
    }

    /// The identity of the first root step, named `flaky`, as an update.
    fn flaky_update() -> OperationUpdate {
        OperationUpdate::builder()
            .id(crate::engine::compute_wire_id_public("1"))
            .r#type(OperationType::Step)
            .sub_type(super::STEP_SUB_TYPE)
            .name("flaky")
            .action(OperationAction::Start)
            .build()
            .expect("all required fields set")
    }
}

/// When an attempt's body runs relative to its `START` write. Every test
/// runs under paused tokio time, with a client that answers each
/// checkpoint call 500 ms after it receives it.
#[cfg(test)]
mod start_write_tests {
    use crate::StepSemantics;
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{LatencyClient, drive_handler, latency_ctx, wire_err};
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const LATENCY: Duration = Duration::from_millis(500);

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Runs, on a client with [`LATENCY`] and `buffer_window`, a handler
    /// whose step records the tokio time its body starts and returns `1`.
    /// Returns the client, the time the body started, and the handler's
    /// outcome: the step's value and the time the step returned.
    async fn timed_step(
        buffer_window: Option<Duration>,
        semantics: StepSemantics,
    ) -> (Arc<LatencyClient>, Duration, InvocationOutcome) {
        let client = Arc::new(LatencyClient::new(LATENCY));
        let ctx = latency_ctx(&client, buffer_window);
        let started = Arc::new(Mutex::new(None));
        let body_started = Arc::clone(&started);
        let clock = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let body_clock = Arc::clone(&clock);
            let value = ctx
                .step(move |_| {
                    let started = Arc::clone(&body_started);
                    let clock = Arc::clone(&body_clock);
                    async move {
                        *started.lock().expect("lock") = Some(clock.elapsed());
                        Ok(1_u32)
                    }
                })
                .name("s")
                .semantics(semantics)
                .await
                .map_err(wire_err)?;
            Ok(format!("{value}@{}", clock.elapsed().as_millis()))
        })
        .await
        .expect("no panic");
        let started = started.lock().expect("lock").expect("the body ran");
        (client, started, outcome)
    }

    #[tokio::test(start_paused = true)]
    async fn at_least_once_body_starts_before_start_response() {
        let (client, started, outcome) = timed_step(None, StepSemantics::AtLeastOncePerRetry).await;
        assert_eq!(started, ms(0), "the body starts before START is answered");
        let start = client.request_with(&OperationAction::Start).expect("START");
        let succeed = client
            .request_with(&OperationAction::Succeed)
            .expect("SUCCEED");
        assert_eq!(start.received, ms(0));
        assert_eq!(succeed.received, ms(500), "SUCCEED waits for START");
        assert_eq!(outcome, InvocationOutcome::Complete("1@1000".to_owned()));
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_start_precedes_outcome() {
        let (client, started, outcome) =
            timed_step(Some(ms(50)), StepSemantics::AtLeastOncePerRetry).await;
        assert_eq!(started, ms(0), "the body starts before START is answered");
        let requests = client.requests();
        let first = requests.first().expect("a request");
        let actions: Vec<&OperationAction> = first
            .updates
            .iter()
            .map(aws_sdk_lambda::types::OperationUpdate::action)
            .collect();
        assert_eq!(
            actions,
            vec![&OperationAction::Start],
            "the first request carries only START"
        );
        assert!(
            requests.iter().skip(1).any(|r| r
                .updates
                .iter()
                .any(|u| u.action() == &OperationAction::Succeed)),
            "SUCCEED is in a later request"
        );
        assert!(matches!(outcome, InvocationOutcome::Complete(ref v) if v.starts_with("1@")));
    }

    #[tokio::test(start_paused = true)]
    async fn at_most_once_body_waits_for_start_response() {
        let (_client, started, outcome) = timed_step(None, StepSemantics::AtMostOncePerRetry).await;
        assert_eq!(started, ms(500), "the body starts after START is answered");
        assert!(matches!(outcome, InvocationOutcome::Complete(ref v) if v.starts_with("1@")));
    }

    /// Asserts that a drive whose `START` write failed failed the
    /// execution with the checkpoint error's type, and wrote no outcome.
    fn assert_failed_without_outcome(client: &LatencyClient, outcome: &InvocationOutcome) {
        let InvocationOutcome::Failed { error } = outcome else {
            unreachable!("expected a failed execution, got {outcome:?}");
        };
        assert_eq!(
            error.error_type(),
            Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE)
        );
        assert!(client.request_with(&OperationAction::Start).is_some());
        assert!(!client.received_outcome(), "no outcome update was sent");
    }

    #[tokio::test(start_paused = true)]
    async fn failed_start_write_writes_no_outcome() {
        // (a) A step.
        let client = Arc::new(LatencyClient::new(LATENCY).failing_start());
        let runs = Arc::new(AtomicUsize::new(0));
        let body_runs = Arc::clone(&runs);
        let outcome = drive_handler(latency_ctx(&client, None), None, move |ctx| async move {
            ctx.step(move |_| async move {
                body_runs.fetch_add(1, Ordering::SeqCst);
                Ok(1_u32)
            })
            .name("s")
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(runs.load(Ordering::SeqCst), 1, "the body ran once");
        assert_failed_without_outcome(&client, &outcome);

        // (b) A condition check.
        let client = Arc::new(LatencyClient::new(LATENCY).failing_start());
        let checks = Arc::new(AtomicUsize::new(0));
        let check_runs = Arc::clone(&checks);
        let outcome = drive_handler(latency_ctx(&client, None), None, move |ctx| async move {
            ctx.wait_for_condition(
                move |_, state: u32| {
                    let checks = Arc::clone(&check_runs);
                    async move {
                        checks.fetch_add(1, Ordering::SeqCst);
                        Ok(state + 1)
                    }
                },
                0,
            )
            .name("c")
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(checks.load(Ordering::SeqCst), 1, "the check ran once");
        assert_failed_without_outcome(&client, &outcome);
    }

    /// Asserts that a drive whose body or check panicked with `"boom"`
    /// before its `START` was answered kept the `START` write: the write
    /// was answered, no outcome was sent, the record is `STARTED`, and the
    /// panic left the drive only after the answer.
    fn assert_start_kept(
        client: &LatencyClient,
        ctx: &crate::DurableContext,
        result: Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>,
        returned: Duration,
    ) {
        let start = client.request_with(&OperationAction::Start).expect("START");
        let answered = start.answered.expect("START was answered");
        assert_eq!(answered, start.received + LATENCY);
        assert!(!client.received_outcome(), "no outcome update was sent");
        let wire_id = start.updates.first().expect("an update").id();
        assert_eq!(crate::engine::compute_wire_id_public("1"), wire_id);
        let view = ctx.checkpoint_status_view("1").expect("a record");
        assert_eq!(view.status, crate::engine::CheckpointStatus::Started);
        let payload = result.expect_err("the panic leaves the drive");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
        assert!(
            returned >= answered,
            "the drive returned at {returned:?}, before {answered:?}"
        );
    }

    /// Drives, on a client with [`LATENCY`] and `buffer_window`, a step
    /// whose body panics with `"boom"` at once, and asserts the `START`
    /// write was kept.
    async fn panicking_step(buffer_window: Option<Duration>) {
        let client = Arc::new(LatencyClient::new(LATENCY));
        let ctx = latency_ctx(&client, buffer_window);
        let clock = Arc::clone(&client);
        let result = drive_handler(ctx.clone(), None, |ctx| async move {
            #[expect(clippy::panic)] // reason: the scenario is a panicking body
            let (): () = ctx
                .step(move |_| async move {
                    assert_eq!(
                        clock.elapsed(),
                        ms(0),
                        "the body runs before START is answered"
                    );
                    panic!("boom")
                })
                .name("s")
                .await
                .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await;
        assert_start_kept(&client, &ctx, result, client.elapsed());
    }

    #[tokio::test(start_paused = true)]
    async fn body_panic_before_start_response_keeps_start_write() {
        // (a) Batching off.
        panicking_step(None).await;
        // (b) Batching on, with a 50 ms delay.
        panicking_step(Some(ms(50))).await;

        // (c) A condition check, batching off.
        let client = Arc::new(LatencyClient::new(LATENCY));
        let ctx = latency_ctx(&client, None);
        let clock = Arc::clone(&client);
        let result = drive_handler(ctx.clone(), None, |ctx| async move {
            #[expect(clippy::panic)] // reason: the scenario is a panicking check
            ctx.wait_for_condition(
                move |_, _state: u32| {
                    let clock = Arc::clone(&clock);
                    async move {
                        assert_eq!(
                            clock.elapsed(),
                            ms(0),
                            "the check runs before START is answered"
                        );
                        panic!("boom")
                    }
                },
                0,
            )
            .name("c")
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await;
        assert_start_kept(&client, &ctx, result, client.elapsed());
    }
}
