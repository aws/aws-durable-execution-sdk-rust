//! Fluent operation builders, one public submodule per operation.
//!
//! Every builder implements [`IntoFuture`], so awaiting the builder runs
//! the operation, and provides a `.spawn()` terminal that starts it
//! eagerly. Chain methods consume and return `self`.
//!
//! # Layout
//!
//! The crate root keeps the everyday items ([`run`](crate::run),
//! [`DurableContext`](crate::DurableContext), the error types, …); the
//! builder types and their per-operation configuration live here,
//! mirroring how tokio keeps `spawn` at the root and `JoinSet` in
//! `tokio::task`:
//!
//! - The fourteen builder types are re-exported at this module's root
//!   (for example [`builders::StepBuilder`](StepBuilder)).
//! - Each operation's submodule additionally carries the configuration
//!   types specific to that operation: [`map_parallel`] holds
//!   [`CompletionConfig`](map_parallel::CompletionConfig) and the batch
//!   result types, [`wait_for_condition`] holds
//!   [`WaitStrategy`](wait_for_condition::WaitStrategy) and
//!   [`WaitDecision`](wait_for_condition::WaitDecision), and [`callback`]
//!   holds the [`Callback`](callback::Callback) handle.
//! - [`RetryStrategyConfig`] and [`JitterStrategy`] live at this module's
//!   root because retry shaping is shared by three operations
//!   ([`StepBuilder`], [`WithRetryBuilder`], and
//!   [`WaitForCallbackBuilder`]).
//!
//! Builders are constructed through the [`DurableContext`](crate::DurableContext)
//! methods (for example [`DurableContext::step`](crate::DurableContext::step)),
//! never directly.

use std::time::Duration;

/// Eagerly validates the builder's claimed replay identity when the builder
/// is finalized into a [`DurableFuture`](crate::DurableFuture).
///
/// Every `into_future` runs this FIRST, and `.future()`, `.spawn()`, and
/// `.await` all funnel through `into_future`, so a replay identity mismatch
/// is recorded on the execution-fatal slot synchronously at finalization,
/// before the operation future is ever polled. This is what makes fatal
/// propagation scheduler-independent: a short-circuiting combinator
/// (`select_ok`, `race`, `try_join_all`) aborts losers the moment a winner
/// settles, so a mismatching constituent might never be polled, but by then
/// its identity was already validated here.
///
/// On mismatch the returned future resolves immediately with the dedicated
/// error and never runs the operation (no START is checkpointed for an
/// operation the recorded history contradicts).
macro_rules! preflight_identity {
    ($builder:expr, $claimed_type:expr, $sub_type:expr) => {
        if let Err(err) = $builder.ctx.preflight_replay_identity(
            &$builder.op_id,
            $claimed_type,
            Some($sub_type),
            $builder.name.as_deref(),
        ) {
            return DurableFuture::from_async(async move { Err(err) });
        }
    };
}

/// The body shared by every builder's `.spawn()` terminal.
///
/// Rebinds the builder's context onto a FRESH child suspension scope, then
/// hands the operation future to
/// [`DurableFuture::spawn_blessed`](crate::future::DurableFuture) together with
/// the owner's scope (for quiescence accounting) and the new scope (which the
/// spawned task drives).
///
/// The fresh scope is what keeps the accounting correct: an eagerly spawned
/// operation that kept the owner's scope would park the owner (ending the
/// invocation) the moment it hit a durable suspension point, aborting
/// runnable siblings.
///
/// The builder must have a `ctx: DurableContext` field and implement
/// [`IntoFuture`] with
/// `IntoFuture = DurableFuture<_>`.
macro_rules! spawn_terminal {
    ($builder:expr) => {{
        let mut builder = $builder;
        let owner_scope = ::std::sync::Arc::clone(builder.ctx.suspension_signal());
        let task_ownership = ::std::sync::Arc::clone(builder.ctx.task_ownership());
        let (spawn_ctx, spawn_scope) = builder.ctx.spawn_scope();
        builder.ctx = spawn_ctx;
        let future = ::std::future::IntoFuture::into_future(builder);
        $crate::future::DurableFuture::spawn_blessed(
            future,
            task_ownership,
            owner_scope,
            spawn_scope,
        )
    }};
}

/// The scope rebind shared by every builder's lazy terminal (`.await` /
/// `.future()`), evaluated inside `into_future` after `preflight_identity!`.
///
/// Rebinds the builder's context onto a FRESH child suspension scope and
/// returns `(owner_scope, op_scope)` for
/// [`DurableFuture::lazy_scoped`](crate::future::DurableFuture): every park
/// inside the operation lands on `op_scope` rather than on the scope the
/// builder was created from, so whoever polls the future decides where the
/// suspension goes. A direct `.await` forwards it to `owner_scope`
/// (identical caller-visible behavior); a combinator redirects it onto a
/// scope it controls, so a losing input's park never suspends the caller
/// after a winner settled (issue #49).
///
/// The builder must have a `ctx: DurableContext` field.
macro_rules! rebind_lazy_scope {
    ($builder:expr) => {{
        let owner_scope = ::std::sync::Arc::clone($builder.ctx.suspension_signal());
        let (scoped_ctx, op_scope) = $builder.ctx.spawn_scope();
        $builder.ctx = scoped_ctx;
        (owner_scope, op_scope)
    }};
}

pub mod callback;
pub mod child;
pub mod combinator;
pub mod invoke;
pub mod map_parallel;
pub mod step;
pub mod wait;
pub mod wait_for_condition;
pub mod with_retry;

pub use self::callback::{CreateCallbackBuilder, WaitForCallbackBuilder};
pub use self::child::ChildBuilder;
pub use self::combinator::{JoinAllBuilder, RaceBuilder, SelectOkBuilder, TryJoinAllBuilder};
pub use self::invoke::InvokeBuilder;
pub use self::map_parallel::{MapBuilder, ParallelBuilder};
pub use self::step::StepBuilder;
pub use self::wait::WaitBuilder;
pub use self::wait_for_condition::WaitForConditionBuilder;
pub use self::with_retry::WithRetryBuilder;

/// Jitter applied to a computed retry delay by [`RetryStrategyConfig`].
///
/// Jitter randomizes retry delays so simultaneous failures do not retry in
/// lockstep. Given a computed backoff delay `base`:
///
/// - [`JitterStrategy::None`] uses `base` unchanged (deterministic delays).
/// - [`JitterStrategy::Half`] picks a random delay in `[base / 2, base]`.
/// - [`JitterStrategy::Full`] picks a random delay in `[0, base]`.
///
/// The default is [`JitterStrategy::Full`], matching the SDK's default
/// retry behavior.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::JitterStrategy;
///
/// assert_eq!(JitterStrategy::default(), JitterStrategy::Full);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum JitterStrategy {
    /// No jitter: use the computed backoff delay unchanged.
    None,
    /// Half jitter: random delay in `[base / 2, base]`.
    Half,
    /// Full jitter: random delay in `[0, base]`. This is the default.
    #[default]
    Full,
}

/// Builder-based configuration for retry delay shaping.
///
/// The common retry customization is shaping delays, how many attempts,
/// how the delay grows, where it caps, and how it is jittered, without
/// hand-writing a closure over `(error, attempt)`. `RetryStrategyConfig`
/// captures exactly those knobs. Pass it to
/// [`StepBuilder::retry_strategy_config`] or
/// [`WaitForCallbackBuilder::submitter_retry_config`]; the closure setters
/// ([`StepBuilder::retry_strategy`],
/// [`WaitForCallbackBuilder::submitter_retry`]) remain the escape hatch for
/// decisions a delay schedule cannot express, such as inspecting the error.
///
/// The configured schedule stops once the failing attempt number reaches
/// `max_attempts`, so `max_attempts` is the total number of executions
/// (initial attempt plus retries). Before that, the delay before attempt
/// `n + 1` is `initial_delay * backoff_rate^(n - 1)` capped at `max_delay`,
/// then jittered per [`JitterStrategy`] and quantized to whole seconds
/// with a one-second minimum. Full jitter rounds to the nearest whole
/// second (the SDK's legacy behavior, preserved so [`Default`] reproduces
/// it exactly); no jitter and half jitter round up, so a deterministic
/// configured delay never fires earlier than requested and half jitter's
/// lower bound survives quantization.
///
/// [`Default`] reproduces the SDK's built-in retry behavior exactly:
/// 6 total attempts, 5 second initial delay, 60 second maximum delay,
/// 2.0 backoff rate, full jitter.
///
/// Construct with [`RetryStrategyConfig::builder`]; unset values keep
/// their [`Default`] value. Read values back through the accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::{JitterStrategy, RetryStrategyConfig};
/// use std::time::Duration;
///
/// let config = RetryStrategyConfig::builder()
///     .max_attempts(3)
///     .initial_delay(Duration::from_secs(2))
///     .jitter(JitterStrategy::None)
///     .build();
/// assert_eq!(config.max_attempts(), 3);
/// assert_eq!(config.initial_delay(), Duration::from_secs(2));
/// // Unset values keep the default.
/// assert_eq!(config.max_delay(), RetryStrategyConfig::default().max_delay());
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
// Fields are private so the struct can grow without breaking callers;
// see the Rust API Guidelines on structs with private fields:
// https://rust-lang.github.io/api-guidelines/future-proofing.html#c-struct-private
pub struct RetryStrategyConfig {
    /// Total number of attempts (initial attempt plus retries) before the
    /// error propagates.
    max_attempts: u32,
    /// Delay before the first retry, prior to jitter.
    initial_delay: Duration,
    /// Upper bound on the computed backoff delay, prior to jitter.
    max_delay: Duration,
    /// Multiplier applied to the delay after each failed attempt.
    backoff_rate: f64,
    /// Jitter applied to each computed delay.
    jitter: JitterStrategy,
}

impl Default for RetryStrategyConfig {
    /// Returns the SDK default retry configuration: 6 total attempts,
    /// 5 second initial delay, 60 second maximum delay, 2.0 backoff rate,
    /// full jitter.
    fn default() -> Self {
        Self {
            max_attempts: 6,
            initial_delay: Duration::from_secs(5),
            max_delay: Duration::from_mins(1),
            backoff_rate: 2.0,
            jitter: JitterStrategy::Full,
        }
    }
}

impl RetryStrategyConfig {
    /// Creates a new [`RetryStrategyConfigBuilder`] seeded with the default
    /// values.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::builders::RetryStrategyConfig;
    ///
    /// let config = RetryStrategyConfig::builder()
    ///     .backoff_rate(1.5)
    ///     .build();
    /// assert!((config.backoff_rate() - 1.5).abs() < f64::EPSILON);
    /// ```
    pub fn builder() -> RetryStrategyConfigBuilder {
        RetryStrategyConfigBuilder::default()
    }

    /// Returns the total number of attempts (initial attempt plus retries).
    #[must_use]
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Returns the delay before the first retry, prior to jitter.
    #[must_use]
    pub fn initial_delay(&self) -> Duration {
        self.initial_delay
    }

    /// Returns the upper bound on the computed backoff delay, prior to
    /// jitter.
    #[must_use]
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// Returns the multiplier applied to the delay after each failed
    /// attempt.
    #[must_use]
    pub fn backoff_rate(&self) -> f64 {
        self.backoff_rate
    }

    /// Returns the jitter applied to each computed delay.
    #[must_use]
    pub fn jitter(&self) -> JitterStrategy {
        self.jitter
    }
}

/// Builder for [`RetryStrategyConfig`].
///
/// Follows the Rust API Guidelines C-BUILDER pattern. All methods consume
/// and return `self` for chaining. Values left unset keep the
/// [`RetryStrategyConfig::default`] value.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::{JitterStrategy, RetryStrategyConfig};
/// use std::time::Duration;
///
/// let config = RetryStrategyConfig::builder()
///     .max_attempts(4)
///     .initial_delay(Duration::from_millis(500))
///     .max_delay(Duration::from_secs(10))
///     .backoff_rate(3.0)
///     .jitter(JitterStrategy::Half)
///     .build();
/// assert_eq!(config.max_delay(), Duration::from_secs(10));
/// assert_eq!(config.jitter(), JitterStrategy::Half);
/// ```
#[derive(Debug, Clone, Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct RetryStrategyConfigBuilder {
    max_attempts: Option<u32>,
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
    backoff_rate: Option<f64>,
    jitter: Option<JitterStrategy>,
}

impl RetryStrategyConfigBuilder {
    /// Sets the total number of attempts (initial attempt plus retries).
    ///
    /// The strategy stops retrying once the failing attempt number reaches
    /// this value, so `max_attempts(1)` never retries.
    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = Some(max_attempts);
        self
    }

    /// Sets the delay before the first retry, prior to jitter.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = Some(delay);
        self
    }

    /// Sets the upper bound on the computed backoff delay, prior to jitter.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = Some(delay);
        self
    }

    /// Sets the multiplier applied to the delay after each failed attempt.
    pub fn backoff_rate(mut self, rate: f64) -> Self {
        self.backoff_rate = Some(rate);
        self
    }

    /// Sets the jitter applied to each computed delay.
    pub fn jitter(mut self, jitter: JitterStrategy) -> Self {
        self.jitter = Some(jitter);
        self
    }

    /// Builds the [`RetryStrategyConfig`], filling unset values from
    /// [`RetryStrategyConfig::default`].
    #[must_use]
    pub fn build(self) -> RetryStrategyConfig {
        let defaults = RetryStrategyConfig::default();
        RetryStrategyConfig {
            max_attempts: self.max_attempts.unwrap_or(defaults.max_attempts),
            initial_delay: self.initial_delay.unwrap_or(defaults.initial_delay),
            max_delay: self.max_delay.unwrap_or(defaults.max_delay),
            backoff_rate: self.backoff_rate.unwrap_or(defaults.backoff_rate),
            jitter: self.jitter.unwrap_or(defaults.jitter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::future::{Branch, DurableFuture};

    /// The `RetryStrategyConfig` builder round-trips every knob through the
    /// accessors, and unset knobs keep their `Default` value.
    #[test]
    fn retry_strategy_config_builder_round_trips() {
        let config = RetryStrategyConfig::builder()
            .max_attempts(4)
            .initial_delay(Duration::from_millis(500))
            .max_delay(Duration::from_secs(10))
            .backoff_rate(3.0)
            .jitter(JitterStrategy::Half)
            .build();

        assert_eq!(config.max_attempts(), 4);
        assert_eq!(config.initial_delay(), Duration::from_millis(500));
        assert_eq!(config.max_delay(), Duration::from_secs(10));
        assert!((config.backoff_rate() - 3.0).abs() < f64::EPSILON);
        assert_eq!(config.jitter(), JitterStrategy::Half);

        // Unset values fall back to the defaults.
        let partial = RetryStrategyConfig::builder().max_attempts(2).build();
        let defaults = RetryStrategyConfig::default();
        assert_eq!(partial.max_attempts(), 2);
        assert_eq!(partial.initial_delay(), defaults.initial_delay());
        assert_eq!(partial.max_delay(), defaults.max_delay());
        assert!((partial.backoff_rate() - defaults.backoff_rate()).abs() < f64::EPSILON);
        assert_eq!(partial.jitter(), defaults.jitter());
    }

    /// `RetryStrategyConfig::default` carries the documented SDK constants:
    /// 6 attempts, 5s initial delay, 60s max delay, 2.0 rate, full jitter.
    #[test]
    fn retry_strategy_config_default_matches_documented_constants() {
        let config = RetryStrategyConfig::default();
        assert_eq!(config.max_attempts(), 6);
        assert_eq!(config.initial_delay(), Duration::from_secs(5));
        assert_eq!(config.max_delay(), Duration::from_mins(1));
        assert!((config.backoff_rate() - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.jitter(), JitterStrategy::Full);
    }

    /// Constructs all fourteen builders and finalizes each through
    /// `.name()`, `.future()`, and `.spawn()`, so a builder losing any of
    /// the three methods fails the build. All fourteen expose all three
    /// today; the returned futures are dropped unawaited because this test
    /// guards the fluent surface, not operation execution (which the
    /// per-operation tests and the conformance suites cover).
    #[tokio::test]
    async fn all_fourteen_builders_expose_name_future_and_spawn() {
        fn constituents(ctx: &DurableContext) -> Vec<DurableFuture<i32>> {
            vec![
                ctx.step(|_| async { Ok(1_i32) }).future(),
                ctx.step(|_| async { Ok(2_i32) }).future(),
            ]
        }

        fn branches() -> Vec<Branch<i32>> {
            vec![Branch::new("branch", |_ctx| async move { Ok(1_i32) })]
        }

        let ctx = DurableContext::__test_context();

        // StepBuilder
        let _step_f = ctx.step(|_| async { Ok(1_i32) }).name("step").future();
        let _step_s = ctx.step(|_| async { Ok(1_i32) }).name("step").spawn();

        // WaitBuilder
        let _wait_f = ctx.wait(Duration::from_secs(1)).name("wait").future();
        let _wait_s = ctx.wait(Duration::from_secs(1)).name("wait").spawn();

        // InvokeBuilder
        let _invoke_f = ctx
            .invoke::<i32, i32>("target-fn", 1)
            .name("invoke")
            .future();
        let _invoke_s = ctx
            .invoke::<i32, i32>("target-fn", 1)
            .name("invoke")
            .spawn();

        // ChildBuilder
        let _child_f = ctx
            .run_in_child_context(|_| async move { Ok(1_i32) })
            .name("child")
            .future();
        let _child_s = ctx
            .run_in_child_context(|_| async move { Ok(1_i32) })
            .name("child")
            .spawn();

        // WithRetryBuilder
        let _with_retry_f = ctx
            .with_retry(|_ctx| async move { Ok(1_i32) })
            .name("with-retry")
            .future();
        let _with_retry_s = ctx
            .with_retry(|_ctx| async move { Ok(1_i32) })
            .name("with-retry")
            .spawn();

        // WaitForConditionBuilder
        let _wfc_f = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .name("wait-for-condition")
            .future();
        let _wfc_s = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .name("wait-for-condition")
            .spawn();

        // CreateCallbackBuilder
        let _cb_f = ctx
            .create_callback::<i32>()
            .name("create-callback")
            .future();
        let _cb_s = ctx.create_callback::<i32>().name("create-callback").spawn();

        // WaitForCallbackBuilder
        let _wfcb_f = ctx
            .wait_for_callback::<i32, _, _>(|_sc, _id| async { Ok(()) })
            .name("wait-for-callback")
            .future();
        let _wfcb_s = ctx
            .wait_for_callback::<i32, _, _>(|_sc, _id| async { Ok(()) })
            .name("wait-for-callback")
            .spawn();

        // MapBuilder
        let _map_f = ctx
            .map(vec![1_i32], |_ctx, item: i32, _idx| async move { Ok(item) })
            .name("map")
            .future();
        let _map_s = ctx
            .map(vec![1_i32], |_ctx, item: i32, _idx| async move { Ok(item) })
            .name("map")
            .spawn();

        // ParallelBuilder
        let _parallel_f = ctx.parallel(branches()).name("parallel").future();
        let _parallel_s = ctx.parallel(branches()).name("parallel").spawn();

        // TryJoinAllBuilder
        let _tja_f = ctx
            .try_join_all(constituents(&ctx))
            .name("try-join-all")
            .future();
        let _tja_s = ctx
            .try_join_all(constituents(&ctx))
            .name("try-join-all")
            .spawn();

        // JoinAllBuilder
        let _ja_f = ctx.join_all(constituents(&ctx)).name("join-all").future();
        let _ja_s = ctx.join_all(constituents(&ctx)).name("join-all").spawn();

        // SelectOkBuilder
        let _so_f = ctx.select_ok(constituents(&ctx)).name("select-ok").future();
        let _so_s = ctx.select_ok(constituents(&ctx)).name("select-ok").spawn();

        // RaceBuilder
        let _race_f = ctx.race(constituents(&ctx)).name("race").future();
        let _race_s = ctx.race(constituents(&ctx)).name("race").spawn();
    }
}
