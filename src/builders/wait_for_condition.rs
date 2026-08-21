//! The polling operation: [`WaitForConditionBuilder`], returned by
//! [`DurableContext::wait_for_condition`](crate::DurableContext::wait_for_condition),
//! plus its configuration ([`WaitStrategy`]) and the per-check decision
//! type ([`WaitDecision`]).
//!
//! The [wait-for-condition operation guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/wait-for-condition/)
//! describes this operation independently of any language SDK.

use std::future::{Future, IntoFuture};
use std::time::Duration;

use crate::BoxError;
use crate::Serdes;
use crate::context::DurableContext;
use crate::context::StepContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;
use crate::serdes::JsonSerdes;

pub use crate::wait_for_condition::WaitDecision;

/// Builder for a wait-for-condition operation.
///
/// Created by [`DurableContext::wait_for_condition`]. Configure the polling
/// strategy with [`wait_strategy`](Self::wait_strategy) (a bounded
/// [`WaitStrategy`] configuration) or
/// [`wait_strategy_fn`](Self::wait_strategy_fn) (a custom closure).
///
/// With **no strategy set**, the check runs exactly once and the operation
/// completes with that check's state: it does not poll. Set a strategy to
/// poll.
///
/// The builder is generic over the check closure `F` and its future
/// `Fut`; both parameters are inferred at the call site and never written
/// by users.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitDecision;
/// use serde::{Serialize, Deserialize};
/// use std::time::Duration;
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct State { ready: bool }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<(), durable::BoxError> {
///     ctx.wait_for_condition(
///         |_, state: State| async move { Ok(State { ready: true }) },
///         State { ready: false },
///     ).name("wait-ready")
///      .wait_strategy_fn(|state: State, _attempt| {
///          if state.ready {
///              WaitDecision::complete()
///          } else {
///              WaitDecision::continue_with(Duration::from_secs(5))
///          }
///      })
///      .await?;
///     Ok(())
/// }
/// ```
#[must_use = "builders do nothing unless awaited or spawned"]
#[non_exhaustive]
pub struct WaitForConditionBuilder<S, F, Fut, SD = JsonSerdes> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    initial_state: S,
    wait_strategy: Option<crate::wait_for_condition::WaitStrategyFn<S>>,
    serdes: SD,
    check: F,
    _marker: std::marker::PhantomData<fn() -> Fut>,
}

impl<S, F, Fut, SD> std::fmt::Debug for WaitForConditionBuilder<S, F, Fut, SD> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitForConditionBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<S, F, Fut> WaitForConditionBuilder<S, F, Fut>
where
    S: Clone + Send + Sync + 'static,
    F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
{
    /// Creates a new builder (internal). Taking the check closure here
    /// keeps the field non-optional: a builder without a check is
    /// unrepresentable.
    pub(crate) fn new(ctx: DurableContext, op_id: OperationId, initial_state: S, check: F) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            initial_state,
            wait_strategy: None,
            serdes: JsonSerdes,
            check,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<S, F, Fut, SD> WaitForConditionBuilder<S, F, Fut, SD>
where
    S: Clone + Send + Sync + 'static,
    F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
{
    /// Sets a human-readable name for this operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the wait strategy from a [`WaitStrategy`] configuration.
    ///
    /// The configuration carries a completion predicate over the state, an
    /// attempt cap, and exponential-backoff delay shaping. After each check,
    /// the derived strategy returns [`WaitDecision::Complete`] when the
    /// predicate matches the new state, [`WaitDecision::Exhausted`] once
    /// `max_attempts` checks have run without the predicate matching, and
    /// [`WaitDecision::Continue`] with the computed backoff delay otherwise.
    ///
    /// With no strategy set at all, the check runs exactly once and the
    /// operation completes with that state.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<i32, durable::BoxError> {
    ///     let done = ctx
    ///         .wait_for_condition(|_step_ctx, state: i32| async move { Ok(state + 1) }, 0_i32)
    ///         .wait_strategy(
    ///             WaitStrategy::builder(|state: &i32| *state >= 3)
    ///                 .max_attempts(10)
    ///                 .initial_delay(Duration::from_secs(2))
    ///                 .max_delay(Duration::from_secs(30))
    ///                 .build(),
    ///         )
    ///         .await?;
    ///     Ok(done)
    /// }
    /// ```
    pub fn wait_strategy(mut self, strategy: WaitStrategy<S>) -> Self {
        self.wait_strategy = Some(Box::new(move |state: S, attempt: u32| {
            strategy.decide(&state, attempt)
        }));
        self
    }

    /// Sets a custom wait strategy closure.
    ///
    /// The strategy receives the current (deserialized) state and the
    /// attempt number (starting at 1), and returns a [`WaitDecision`].
    ///
    /// The closure is responsible for termination: return
    /// [`WaitDecision::complete`] when the condition is met, and bound the
    /// polling with [`WaitDecision::exhausted`] (for example once `attempt`
    /// reaches a cap) so the operation cannot poll until the execution
    /// times out. Prefer [`wait_strategy`](Self::wait_strategy) when a
    /// predicate plus attempt cap expresses the condition: it makes the
    /// bound impossible to forget. With no strategy set at all, the check
    /// runs exactly once and the operation completes with that state.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust as durable;
    /// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitDecision;
    /// use std::time::Duration;
    ///
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: durable::DurableContext,
    /// ) -> Result<i32, durable::BoxError> {
    ///     let done = ctx
    ///         .wait_for_condition(|_step_ctx, state: i32| async move { Ok(state + 1) }, 0_i32)
    ///         .wait_strategy_fn(|state: i32, _attempt| {
    ///             if state >= 3 {
    ///                 WaitDecision::complete()
    ///             } else {
    ///                 WaitDecision::continue_with(Duration::from_secs(1))
    ///             }
    ///         })
    ///         .await?;
    ///     Ok(done)
    /// }
    /// ```
    pub fn wait_strategy_fn<W>(mut self, strategy: W) -> Self
    where
        W: Fn(S, u32) -> WaitDecision + Send + Sync + 'static,
    {
        self.wait_strategy = Some(Box::new(strategy));
        self
    }

    /// Sets a custom serializer/deserializer for the state.
    ///
    /// Replaces the builder's serdes type parameter with `SD2`, which must
    /// implement [`Serdes<S>`](crate::Serdes) for this operation's state
    /// type: attaching a serdes for a different type fails at compile
    /// time. To share one instance across operations, wrap it in an
    /// [`Arc`](std::sync::Arc) and clone the `Arc` handle into each builder.
    pub fn serdes<SD2>(self, serdes: SD2) -> WaitForConditionBuilder<S, F, Fut, SD2>
    where
        SD2: Serdes<S>,
    {
        WaitForConditionBuilder {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            initial_state: self.initial_state,
            wait_strategy: self.wait_strategy,
            serdes,
            check: self.check,
            _marker: std::marker::PhantomData,
        }
    }

    /// Converts this builder into a [`DurableFuture`] without starting it.
    ///
    /// [`DurableFuture`] is the one input type the combinators
    /// ([`DurableContext::try_join_all`], [`DurableContext::join_all`],
    /// [`DurableContext::select_ok`], and [`DurableContext::race`]) accept,
    /// so `.future()` is how operations of different kinds join or race
    /// together. It does not start the operation: whatever awaits the
    /// returned future drives it, and a combinator drops the losers.
    pub fn future(self) -> DurableFuture<S>
    where
        SD: Serdes<S>,
    {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the operation on a tokio task.
    ///
    /// The returned [`DurableFuture`] is already running; this is the
    /// replay-safe alternative to bare `tokio::spawn` for durable
    /// operations.
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime.
    pub fn spawn(self) -> DurableFuture<S>
    where
        SD: Serdes<S>,
    {
        spawn_terminal!(self)
    }
}

impl<S, F, Fut, SD> IntoFuture for WaitForConditionBuilder<S, F, Fut, SD>
where
    S: Clone + Send + Sync + 'static,
    F: Fn(StepContext, S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<S, BoxError>> + Send + 'static,
    SD: Serdes<S>,
{
    type Output = Result<S, OperationError>;
    type IntoFuture = DurableFuture<S>;

    fn into_future(mut self) -> Self::IntoFuture {
        use crate::wait_for_condition::WaitForConditionExecution;

        preflight_identity!(self, "Step", crate::wait_for_condition::WFC_SUB_TYPE);

        let (owner_scope, op_scope) = rebind_lazy_scope!(self);

        let execution = WaitForConditionExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            initial_state: self.initial_state,
            wait_strategy: self.wait_strategy,
            serdes: self.serdes,
            check: self.check,
        };

        DurableFuture::lazy_scoped(execution.execute(), owner_scope, op_scope)
    }
}

/// Configuration for the wait strategy used by
/// [`DurableContext::wait_for_condition`].
///
/// Carries the three things a bounded poll needs: a **completion
/// predicate** over the state (the operation completes when it returns
/// `true`), an attempt cap (**`max_attempts`**: reaching it without the
/// predicate matching fails the operation with a `MaxChecksExceeded`
/// error), and exponential-backoff **delay shaping** (`initial_delay`,
/// `max_delay`, `backoff_rate`, `jitter`) for the suspension between
/// checks.
///
/// The predicate is required at construction: [`WaitStrategy::builder`]
/// takes it as its argument, so a strategy that polls forever cannot be
/// constructed. The remaining knobs default to 60 attempts, a 5 second
/// initial delay, a 5 minute maximum delay, a 1.5 backoff rate, and
/// [`JitterStrategy::Full`](crate::builders::JitterStrategy), matching the
/// JS and Python SDK defaults.
///
/// The delay before check `n + 1` is `initial_delay * backoff_rate^(n - 1)`
/// capped at `max_delay`, jittered per
/// [`JitterStrategy`](crate::builders::JitterStrategy), and quantized
/// to whole seconds with a one-second minimum, always rounding **up**
/// (`max(1, ceil(delay))`): a sampled delay never fires earlier than
/// sampled, for every jitter strategy.
///
/// Read values back through the accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
/// use std::time::Duration;
///
/// let strategy = WaitStrategy::builder(|state: &i32| *state >= 3)
///     .max_attempts(10)
///     .initial_delay(Duration::from_secs(2))
///     .build();
/// assert_eq!(strategy.max_attempts(), 10);
/// // Unset knobs keep the JS/Python-aligned defaults.
/// assert_eq!(strategy.max_delay(), Duration::from_mins(5));
/// assert!((strategy.backoff_rate() - 1.5).abs() < f64::EPSILON);
/// ```
#[non_exhaustive]
// Fields are private so the struct can grow without breaking callers;
// see the Rust API Guidelines on structs with private fields:
// https://rust-lang.github.io/api-guidelines/future-proofing.html#c-struct-private
pub struct WaitStrategy<S> {
    /// Predicate over the state; `true` completes the operation.
    completion_predicate: std::sync::Arc<dyn Fn(&S) -> bool + Send + Sync>,
    /// Total number of checks allowed before the operation exhausts.
    max_attempts: u32,
    /// Initial delay between condition checks.
    initial_delay: Duration,
    /// Maximum delay between condition checks.
    max_delay: Duration,
    /// Backoff multiplier applied after each check.
    backoff_rate: f64,
    /// Jitter applied to each computed delay.
    jitter: crate::builders::JitterStrategy,
}

/// Default `max_attempts`, aligned with the JS and Python SDKs.
const DEFAULT_MAX_ATTEMPTS: u32 = 60;
/// Default `initial_delay`, aligned with the JS and Python SDKs.
const DEFAULT_INITIAL_DELAY: Duration = Duration::from_secs(5);
/// Default `max_delay`, aligned with the JS and Python SDKs.
const DEFAULT_MAX_DELAY: Duration = Duration::from_mins(5);
/// Default `backoff_rate`, aligned with the JS and Python SDKs.
const DEFAULT_BACKOFF_RATE: f64 = 1.5;

impl<S> std::fmt::Debug for WaitStrategy<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitStrategy")
            .field("max_attempts", &self.max_attempts)
            .field("initial_delay", &self.initial_delay)
            .field("max_delay", &self.max_delay)
            .field("backoff_rate", &self.backoff_rate)
            .field("jitter", &self.jitter)
            .finish_non_exhaustive()
    }
}

impl<S> Clone for WaitStrategy<S> {
    fn clone(&self) -> Self {
        Self {
            completion_predicate: std::sync::Arc::clone(&self.completion_predicate),
            max_attempts: self.max_attempts,
            initial_delay: self.initial_delay,
            max_delay: self.max_delay,
            backoff_rate: self.backoff_rate,
            jitter: self.jitter,
        }
    }
}

impl<S> WaitStrategy<S> {
    /// Creates a new [`WaitStrategyBuilder`] from the required completion
    /// predicate.
    ///
    /// The predicate receives the state each check produced and returns
    /// `true` when the condition is met, which completes the operation.
    /// Taking it here, rather than through an optional setter, makes an
    /// unbounded strategy unrepresentable. Knobs left unset keep the
    /// JS/Python-aligned defaults (60 attempts, 5s initial delay, 5min
    /// maximum delay, 1.5 backoff rate, full jitter).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
    ///
    /// let strategy = WaitStrategy::builder(|state: &i32| *state >= 3)
    ///     .backoff_rate(2.0)
    ///     .build();
    /// assert!((strategy.backoff_rate() - 2.0).abs() < f64::EPSILON);
    /// assert_eq!(strategy.max_attempts(), 60);
    /// ```
    pub fn builder<P>(completion_predicate: P) -> WaitStrategyBuilder<S>
    where
        P: Fn(&S) -> bool + Send + Sync + 'static,
    {
        WaitStrategyBuilder {
            completion_predicate: std::sync::Arc::new(completion_predicate),
            max_attempts: None,
            initial_delay: None,
            max_delay: None,
            backoff_rate: None,
            jitter: None,
        }
    }

    /// Returns the total number of checks allowed before the operation
    /// exhausts.
    #[must_use]
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Returns the initial delay between condition checks.
    #[must_use]
    pub fn initial_delay(&self) -> Duration {
        self.initial_delay
    }

    /// Returns the maximum delay between condition checks.
    #[must_use]
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// Returns the backoff multiplier applied after each check.
    #[must_use]
    pub fn backoff_rate(&self) -> f64 {
        self.backoff_rate
    }

    /// Returns the jitter applied to each computed delay.
    #[must_use]
    pub fn jitter(&self) -> crate::builders::JitterStrategy {
        self.jitter
    }

    /// Computes the wait decision after a check produced `state` on the
    /// 1-based attempt `attempt`.
    ///
    /// Order matters: the predicate is consulted first, so a condition
    /// satisfied on the final allowed check completes rather than
    /// exhausting. Exhaustion carries the attempt count in its reason.
    pub(crate) fn decide(&self, state: &S, attempt: u32) -> WaitDecision {
        if (self.completion_predicate)(state) {
            return WaitDecision::Complete;
        }
        if attempt >= self.max_attempts {
            return WaitDecision::Exhausted {
                reason: format!(
                    "max attempts exceeded: {attempt} checks completed (max_attempts = {})",
                    self.max_attempts
                ),
            };
        }

        // Exponential backoff: initial * rate^(attempt-1), capped at max,
        // jittered, then quantized with the round-up-min-1s policy
        // (`max(1, ceil(delay))`): see [`quantize_wait_delay`].
        let exponent = i32::try_from(attempt).unwrap_or(1) - 1;
        let base = (self.initial_delay.as_secs_f64() * self.backoff_rate.powi(exponent))
            .min(self.max_delay.as_secs_f64());
        let jittered = match self.jitter {
            crate::builders::JitterStrategy::None => base,
            // Half jitter: base/2 plus random in [0, base/2] => [base/2, base].
            crate::builders::JitterStrategy::Half => {
                base / 2.0 + crate::step::rand_full_jitter(base / 2.0)
            }
            // Full jitter: random in [0, base].
            crate::builders::JitterStrategy::Full => crate::step::rand_full_jitter(base),
        };
        WaitDecision::Continue {
            delay: quantize_wait_delay(jittered),
        }
    }
}

/// Quantizes a fractional delay in seconds to a whole-second [`Duration`]
/// with a one-second minimum, always rounding **up**: `max(1, ceil(delay))`,
/// the policy issue #35 requires (Python does the same). A sampled delay is
/// therefore never scheduled earlier than sampled: `4.4s` becomes `5s`.
///
/// This is deliberately independent of the retry path's
/// `step::quantize_delay_secs`, whose `Full`-jitter arm rounds to the
/// *nearest* second to preserve the legacy retry delay distribution. Wait
/// strategies carry no such legacy and apply the round-up policy uniformly
/// across every jitter strategy.
///
/// Conversion goes through [`Duration::try_from_secs_f64`] rather than a
/// lossy `as` cast; a non-finite or out-of-range input falls back to the
/// one-second minimum.
fn quantize_wait_delay(jittered: f64) -> Duration {
    let secs = jittered.ceil().max(1.0);
    Duration::try_from_secs_f64(secs).unwrap_or(Duration::from_secs(1))
}

/// Builder for [`WaitStrategy`].
///
/// Created by [`WaitStrategy::builder`], which takes the required
/// completion predicate. Follows the Rust API Guidelines C-BUILDER
/// pattern: all methods consume and return `self` for chaining, and knobs
/// left unset keep the JS/Python-aligned defaults (60 attempts, 5s initial
/// delay, 5min maximum delay, 1.5 backoff rate,
/// [`JitterStrategy::Full`](crate::builders::JitterStrategy)).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::JitterStrategy;
/// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
/// use std::time::Duration;
///
/// let strategy = WaitStrategy::builder(|state: &u32| *state > 0)
///     .max_attempts(20)
///     .initial_delay(Duration::from_millis(500))
///     .max_delay(Duration::from_secs(10))
///     .backoff_rate(3.0)
///     .jitter(JitterStrategy::None)
///     .build();
/// assert_eq!(strategy.max_delay(), Duration::from_secs(10));
/// ```
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct WaitStrategyBuilder<S> {
    completion_predicate: std::sync::Arc<dyn Fn(&S) -> bool + Send + Sync>,
    max_attempts: Option<u32>,
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
    backoff_rate: Option<f64>,
    jitter: Option<crate::builders::JitterStrategy>,
}

impl<S> std::fmt::Debug for WaitStrategyBuilder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitStrategyBuilder")
            .field("max_attempts", &self.max_attempts)
            .field("initial_delay", &self.initial_delay)
            .field("max_delay", &self.max_delay)
            .field("backoff_rate", &self.backoff_rate)
            .field("jitter", &self.jitter)
            .finish_non_exhaustive()
    }
}

impl<S> Clone for WaitStrategyBuilder<S> {
    fn clone(&self) -> Self {
        Self {
            completion_predicate: std::sync::Arc::clone(&self.completion_predicate),
            max_attempts: self.max_attempts,
            initial_delay: self.initial_delay,
            max_delay: self.max_delay,
            backoff_rate: self.backoff_rate,
            jitter: self.jitter,
        }
    }
}

impl<S> WaitStrategyBuilder<S> {
    /// Sets the total number of checks allowed before the operation fails
    /// with a `MaxChecksExceeded` error.
    ///
    /// The cap counts checks, so `max_attempts(1)` runs the check once and
    /// exhausts if the predicate is not satisfied by its state.
    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = Some(max_attempts);
        self
    }

    /// Sets the initial delay between condition checks, prior to jitter.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = Some(delay);
        self
    }

    /// Sets the maximum delay between condition checks, prior to jitter.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = Some(delay);
        self
    }

    /// Sets the backoff multiplier applied after each check.
    pub fn backoff_rate(mut self, rate: f64) -> Self {
        self.backoff_rate = Some(rate);
        self
    }

    /// Sets the jitter applied to each computed delay.
    pub fn jitter(mut self, jitter: crate::builders::JitterStrategy) -> Self {
        self.jitter = Some(jitter);
        self
    }

    /// Builds the [`WaitStrategy`], filling unset knobs with the
    /// JS/Python-aligned defaults.
    #[must_use]
    pub fn build(self) -> WaitStrategy<S> {
        WaitStrategy {
            completion_predicate: self.completion_predicate,
            max_attempts: self.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS),
            initial_delay: self.initial_delay.unwrap_or(DEFAULT_INITIAL_DELAY),
            max_delay: self.max_delay.unwrap_or(DEFAULT_MAX_DELAY),
            backoff_rate: self.backoff_rate.unwrap_or(DEFAULT_BACKOFF_RATE),
            jitter: self.jitter.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions on unexpected variants
mod tests {
    use super::*;
    use crate::builders::JitterStrategy;

    /// The `WaitStrategy` builder sets each knob independently, and unset
    /// knobs keep the JS/Python-aligned defaults (60 attempts, 5s initial
    /// delay, 5min maximum delay, 1.5 backoff rate, full jitter).
    #[test]
    fn wait_strategy_builder_overrides_only_what_is_set() {
        let partial = WaitStrategy::builder(|state: &i32| *state >= 3)
            .initial_delay(Duration::from_secs(2))
            .build();
        assert_eq!(partial.initial_delay(), Duration::from_secs(2));
        assert_eq!(partial.max_attempts(), 60);
        assert_eq!(partial.max_delay(), Duration::from_mins(5));
        assert!((partial.backoff_rate() - 1.5).abs() < f64::EPSILON);
        assert_eq!(partial.jitter(), JitterStrategy::Full);

        let full = WaitStrategy::builder(|state: &i32| *state >= 3)
            .max_attempts(7)
            .initial_delay(Duration::from_millis(500))
            .max_delay(Duration::from_secs(10))
            .backoff_rate(3.0)
            .jitter(JitterStrategy::Half)
            .build();
        assert_eq!(full.max_attempts(), 7);
        assert_eq!(full.initial_delay(), Duration::from_millis(500));
        assert_eq!(full.max_delay(), Duration::from_secs(10));
        assert!((full.backoff_rate() - 3.0).abs() < f64::EPSILON);
        assert_eq!(full.jitter(), JitterStrategy::Half);
    }

    /// A state satisfying the completion predicate yields `Complete`, and
    /// the predicate wins over exhaustion, so a condition satisfied on the
    /// final allowed check completes rather than exhausting.
    #[test]
    fn predicate_satisfied_yields_complete() {
        let strategy = WaitStrategy::builder(|state: &i32| *state >= 3)
            .max_attempts(5)
            .build();

        assert!(matches!(strategy.decide(&3, 1), WaitDecision::Complete));
        assert!(matches!(strategy.decide(&100, 2), WaitDecision::Complete));
        // Attempt at (and past) the cap still completes when the predicate
        // is satisfied.
        assert!(matches!(strategy.decide(&3, 5), WaitDecision::Complete));
        assert!(matches!(strategy.decide(&3, 6), WaitDecision::Complete));
    }

    /// Reaching `max_attempts` without the predicate matching yields
    /// `Exhausted`, and the reason carries the attempt count.
    #[test]
    fn max_attempts_reached_yields_exhausted_with_attempt_count() {
        let strategy = WaitStrategy::builder(|state: &i32| *state >= 3)
            .max_attempts(3)
            .jitter(JitterStrategy::None)
            .build();

        // Below the cap: continue.
        assert!(matches!(
            strategy.decide(&0, 2),
            WaitDecision::Continue { .. }
        ));

        // At the cap: exhausted, with the attempt count in the reason.
        match strategy.decide(&0, 3) {
            WaitDecision::Exhausted { reason } => {
                assert!(
                    reason.contains("3 checks"),
                    "reason must carry the attempt count: {reason}"
                );
                assert!(
                    reason.contains("max attempts exceeded"),
                    "reason must name the cause: {reason}"
                );
            }
            other => panic!("expected Exhausted at the cap, got {other:?}"),
        }
    }

    /// With `JitterStrategy::None` the schedule is deterministic: the first
    /// delay is `initial_delay`, each subsequent delay grows by
    /// `backoff_rate`, the sequence caps at `max_delay`, and sub-second
    /// results round up to the one-second minimum.
    #[test]
    fn deterministic_schedule_respects_backoff_and_max_delay() {
        let strategy = WaitStrategy::builder(|state: &i32| *state >= 100)
            .max_attempts(60)
            .initial_delay(Duration::from_secs(2))
            .max_delay(Duration::from_secs(10))
            .backoff_rate(3.0)
            .jitter(JitterStrategy::None)
            .build();

        let delay_of = |attempt: u32| match strategy.decide(&0, attempt) {
            WaitDecision::Continue { delay } => delay,
            other => panic!("expected Continue below the cap, got {other:?}"),
        };

        // attempt 1 → initial (2s); attempt 2 → 2s * 3 = 6s;
        // attempt 3 → 18s, capped at max_delay (10s).
        assert_eq!(delay_of(1), Duration::from_secs(2));
        assert_eq!(delay_of(2), Duration::from_secs(6));
        assert_eq!(delay_of(3), Duration::from_secs(10));
        assert_eq!(delay_of(9), Duration::from_secs(10));

        // Sub-second base delays quantize up to the one-second minimum.
        let tiny = WaitStrategy::builder(|state: &i32| *state >= 100)
            .initial_delay(Duration::from_millis(100))
            .jitter(JitterStrategy::None)
            .build();
        match tiny.decide(&0, 1) {
            WaitDecision::Continue { delay } => assert_eq!(delay, Duration::from_secs(1)),
            other => panic!("expected Continue, got {other:?}"),
        }
    }

    /// Full jitter draws from `[0, base]` (quantized, min 1s), and the
    /// delay never exceeds `max_delay` even deep into the schedule.
    #[test]
    fn full_jitter_delays_stay_within_bounds() {
        let strategy = WaitStrategy::builder(|state: &i32| *state >= 100)
            .initial_delay(Duration::from_secs(10))
            .max_delay(Duration::from_secs(30))
            .backoff_rate(2.0)
            .jitter(JitterStrategy::Full)
            .build();

        for _ in 0..50 {
            // Attempt 1: base = 10s. Full jitter in [0, 10], min 1s.
            match strategy.decide(&0, 1) {
                WaitDecision::Continue { delay } => {
                    assert!(
                        delay >= Duration::from_secs(1) && delay <= Duration::from_secs(10),
                        "full-jitter delay {delay:?} outside [1s, base]"
                    );
                }
                other => panic!("expected Continue, got {other:?}"),
            }
            // Attempt 20: base capped at max_delay (30s).
            match strategy.decide(&0, 20) {
                WaitDecision::Continue { delay } => {
                    assert!(
                        delay <= Duration::from_secs(30),
                        "full-jitter delay {delay:?} exceeds max_delay"
                    );
                }
                other => panic!("expected Continue, got {other:?}"),
            }
        }
    }

    /// Half jitter draws from `[base / 2, base]`, and the documented lower
    /// bound survives quantization.
    #[test]
    fn half_jitter_delays_stay_within_bounds() {
        let strategy = WaitStrategy::builder(|state: &i32| *state >= 100)
            .initial_delay(Duration::from_secs(10))
            .max_delay(Duration::from_secs(30))
            .jitter(JitterStrategy::Half)
            .build();

        for _ in 0..50 {
            // Attempt 1: base = 10s. Half jitter in [5, 10].
            match strategy.decide(&0, 1) {
                WaitDecision::Continue { delay } => {
                    assert!(
                        delay >= Duration::from_secs(5) && delay <= Duration::from_secs(10),
                        "half-jitter delay {delay:?} outside [base/2, base]"
                    );
                }
                other => panic!("expected Continue, got {other:?}"),
            }
        }
    }

    /// `wait_strategy` installs a strategy derived from the config: the
    /// three outcomes (`Complete` / `Continue` / `Exhausted`) flow through
    /// the installed closure exactly as `decide` produces them.
    #[test]
    fn wait_strategy_installs_config_derived_strategy() {
        let ctx = DurableContext::__test_context();

        let builder = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy(
                WaitStrategy::builder(|state: &i32| *state >= 3)
                    .max_attempts(4)
                    .initial_delay(Duration::from_secs(2))
                    .max_delay(Duration::from_secs(10))
                    .backoff_rate(3.0)
                    .jitter(JitterStrategy::None)
                    .build(),
            );

        let strategy = builder
            .wait_strategy
            .as_ref()
            .expect("wait_strategy must install a strategy");

        // Predicate satisfied → Complete.
        assert!(matches!(strategy(3_i32, 1), WaitDecision::Complete));
        // Predicate unmet below the cap → Continue with the backoff delay.
        match strategy(0_i32, 2) {
            WaitDecision::Continue { delay } => assert_eq!(delay, Duration::from_secs(6)),
            other => panic!("expected Continue, got {other:?}"),
        }
        // Predicate unmet at the cap → Exhausted.
        assert!(matches!(strategy(0_i32, 4), WaitDecision::Exhausted { .. }));
    }

    /// The wait-strategy quantizer implements the round-up-min-1s policy
    /// issue #35 requires (`max(1, ceil(delay))`): a fractional sample is
    /// never scheduled earlier than sampled: `4.4s` rounds **up** to `5s`,
    /// unlike the retry path's legacy Full-jitter nearest-rounding, which
    /// would schedule it at `4s`.
    #[test]
    fn quantize_wait_delay_rounds_up_with_one_second_minimum() {
        // Regression for the reviewer finding: 4.4 must become 5, not 4.
        assert_eq!(quantize_wait_delay(4.4), Duration::from_secs(5));
        // Nearest-rounding would also disagree here (4.5 → 5 either way,
        // 4.1 → 4 under round()); ceil is unambiguous.
        assert_eq!(quantize_wait_delay(4.1), Duration::from_secs(5));
        // Whole seconds pass through unchanged.
        assert_eq!(quantize_wait_delay(5.0), Duration::from_secs(5));
        // Sub-second and zero samples clamp to the one-second minimum.
        assert_eq!(quantize_wait_delay(0.4), Duration::from_secs(1));
        assert_eq!(quantize_wait_delay(0.0), Duration::from_secs(1));
        // Defensive fallback: a non-finite input degrades to the minimum
        // rather than panicking.
        assert_eq!(quantize_wait_delay(f64::NAN), Duration::from_secs(1));
    }

    /// End-to-end through `decide`: with `Full` jitter the sample lands in
    /// `[0, base]`, so after round-up quantization every delay is a whole
    /// second in `[1s, ceil(base)]`, never rounded below the sample.
    #[test]
    fn full_jitter_decide_never_rounds_below_one_second() {
        let strategy = WaitStrategy::builder(|state: &i32| *state >= 100)
            // base for attempt 1 is 1s, so full-jitter samples are all
            // fractional in [0, 1]; round-up must yield exactly 1s.
            .initial_delay(Duration::from_secs(1))
            .jitter(JitterStrategy::Full)
            .build();

        for _ in 0..50 {
            match strategy.decide(&0, 1) {
                WaitDecision::Continue { delay } => {
                    assert_eq!(
                        delay,
                        Duration::from_secs(1),
                        "sample in [0, 1] must quantize up to exactly 1s"
                    );
                }
                other => panic!("expected Continue, got {other:?}"),
            }
        }
    }
}
