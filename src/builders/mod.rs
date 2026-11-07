//! Fluent operation builders, one public submodule per operation.
//!
//! Every builder whose required configuration is present implements
//! [`IntoFuture`], so awaiting the builder runs the operation, and provides
//! `.future()` and a `.spawn()` terminal that starts it eagerly. Every
//! builder is executable as created except
//! [`WaitForConditionBuilder`], which needs a wait strategy first. Chain
//! methods consume and return `self`.
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
//!   [`WaitStrategy`](wait_for_condition::WaitStrategy),
//!   [`WaitDecision`](wait_for_condition::WaitDecision) and the two
//!   wait-strategy states of its builder,
//!   [`NoWaitStrategy`](wait_for_condition::NoWaitStrategy) and
//!   [`WithWaitStrategy`](wait_for_condition::WithWaitStrategy), and [`callback`]
//!   holds the [`Callback`](callback::Callback) handle.
//! - [`RetryStrategyConfig`], [`Backoff`] and [`JitterStrategy`] live at this module's
//!   root because retry shaping is shared by three operations
//!   ([`StepBuilder`], [`WithRetryBuilder`], and
//!   [`WaitForCallbackBuilder`]).
//! - [`Submitter`], the trait every `wait_for_callback` submitter closure
//!   implements, lives at this module's root too.
//!
//! Builders are constructed through the [`DurableContext`](crate::DurableContext)
//! methods (for example [`DurableContext::step`](crate::DurableContext::step)),
//! never directly.

use std::fmt;
use std::time::Duration;

use crate::StepError;

/// Eagerly validates the builder's claimed replay identity right after the
/// builder claims its operation ID.
///
/// Every builder except a combinator and `with_retry` claims its ID when it
/// is created, and its `into_future` runs this FIRST. A `with_retry` block
/// with its own record claims and validates its ID when the builder
/// becomes its future; an unwrapped block has no ID of its own, and each
/// attempt child validates its own. `.future()`, `.spawn()`, and `.await`
/// all funnel through `into_future`, so a replay identity mismatch is
/// recorded on the execution-fatal slot synchronously at finalization,
/// before the operation future is ever polled.
///
/// A combinator claims its ID on the first poll of its future, or at
/// `.spawn()`, and runs this right after that claim, before its body runs.
///
/// This is what makes fatal propagation scheduler-independent. A
/// short-circuiting combinator (`select_ok`, `race`, `try_join_all`) aborts
/// losers the moment a winner settles, so a mismatching constituent might
/// never be polled. An input that is not a combinator was validated when it
/// was finalized. A combinator input either claimed and was validated at its
/// first poll or `.spawn()`, or it is still unclaimed, and then the outer
/// combinator fails the execution with a context usage error before it
/// starts any input.
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
/// Refuses a caller that does not own the context first, through
/// [`DurableFuture::spawn_built`](crate::future::DurableFuture), so the
/// refusal wins over a replay mismatch. Then it builds the operation future
/// with the builder's lazy terminal (`into_future`, which runs the identity
/// preflight and the admission check) and starts it on its own task
/// through [`DurableFuture::spawn_blessed`](crate::future::DurableFuture).
/// That runs, for a combinator, the pending claim
/// ([`DurableFuture::claim_now`](crate::future::DurableFuture)), so the
/// combinator claims its ID and runs the identity preflight at the
/// `.spawn()` call, then blesses the task with the builder context's
/// ownership handle and registers it with the context's scopes. Dropping the returned handle
/// loses only the result.
///
/// The builder must have a `ctx: DurableContext` field and implement
/// [`IntoFuture`] with
/// `IntoFuture = DurableFuture<_>`.
///
/// The `awaited` arm is for a wait and an invoke. It runs the operation on
/// a clone of the context that carries a first-poll flag, so the refresh
/// scheduler refreshes the operation only once the returned handle is
/// first polled (see [`crate::coordinator`]). A `wait_for_condition` uses
/// the plain arm: it arms its refresh when it registers its wait, as JS
/// starts its retry timer whether or not anything awaits the promise.
///
/// The `batch_future` arm is for `spawn_batch()` on the map and parallel
/// builders. It is the one-argument form with the builder's
/// `batch_future()`, which yields the full `BatchResult`, in place of
/// `into_future`.
///
/// The `started`, `started_awaited` and `started_batch` arms are for the
/// operations whose JS form begins inside the call: step, wait, child
/// context, map, parallel, `parallel_mixed`, `create_callback`,
/// `wait_for_callback` and `with_retry`. They refuse a caller that does
/// not own the context first, then check for a runtime, then run the
/// operation's start step (see [`crate::start`]) on the
/// caller's task through the builder's `started_future()` (or
/// `started_batch_future()`), and spawn the rest through
/// [`DurableFuture::spawn_started`](crate::future::DurableFuture). So the
/// operation's `START` is queued, and a "queue, then hook" start hook has
/// fired, before `.spawn()` returns. `started_awaited` is the `awaited`
/// arm in that form, for a wait.
macro_rules! spawn_terminal {
    ($builder:expr) => {{
        let builder = $builder;
        let ctx = builder.ctx.clone();
        $crate::future::DurableFuture::spawn_built(&ctx, None, move || {
            ::std::future::IntoFuture::into_future(builder)
        })
    }};
    ($builder:expr, batch_future) => {{
        let builder = $builder;
        let ctx = builder.ctx.clone();
        $crate::future::DurableFuture::spawn_built(&ctx, None, move || builder.batch_future())
    }};
    ($builder:expr, started) => {{
        let builder = $builder;
        let ctx = builder.ctx.clone();
        $crate::future::DurableFuture::spawn_started(&ctx, None, move || builder.started_future())
    }};
    ($builder:expr, started_batch) => {{
        let builder = $builder;
        let ctx = builder.ctx.clone();
        $crate::future::DurableFuture::spawn_started(&ctx, None, move || {
            builder.started_batch_future()
        })
    }};
    ($builder:expr, started_awaited) => {{
        let mut builder = $builder;
        let flag = ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(false));
        builder.ctx = builder.ctx.with_awaited(::std::sync::Arc::clone(&flag));
        let ctx = builder.ctx.clone();
        $crate::future::DurableFuture::spawn_started(&ctx, Some(flag), move || {
            builder.started_future()
        })
    }};
    ($builder:expr, awaited) => {{
        let mut builder = $builder;
        let flag = ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(false));
        builder.ctx = builder.ctx.with_awaited(::std::sync::Arc::clone(&flag));
        let ctx = builder.ctx.clone();
        $crate::future::DurableFuture::spawn_built(&ctx, Some(flag), move || {
            ::std::future::IntoFuture::into_future(builder)
        })
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

pub use self::callback::{CreateCallbackBuilder, Submitter, WaitForCallbackBuilder};
pub use self::child::ChildBuilder;
pub use self::combinator::{JoinAllBuilder, RaceBuilder, SelectOkBuilder, TryJoinAllBuilder};
pub use self::invoke::InvokeBuilder;
pub use self::map_parallel::{MapBuilder, ParallelBuilder, ParallelMixedBuilder};
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
/// use aws_durable_execution_sdk::builders::JitterStrategy;
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

/// How the retry delay grows from one attempt to the next, in a
/// [`RetryStrategyConfig`].
///
/// The enum is `#[non_exhaustive]`, but its variants are not, so a caller
/// builds a variant with struct syntax.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::{Backoff, RetryStrategyConfig};
/// use std::time::Duration;
///
/// let config = RetryStrategyConfig::builder()
///     .backoff(Backoff::Linear { increment: Duration::from_secs(2) })
///     .build();
/// assert_eq!(
///     config.backoff(),
///     Backoff::Linear { increment: Duration::from_secs(2) }
/// );
/// assert_eq!(
///     RetryStrategyConfig::default_preset().backoff(),
///     Backoff::Exponential { rate: 2.0 }
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Backoff {
    /// The delay before attempt `n + 1` is `initial_delay * rate^(n - 1)`.
    Exponential {
        /// Multiplier applied to the delay after each failed attempt.
        rate: f64,
    },
    /// The delay before attempt `n + 1` is `initial_delay + increment * (n - 1)`.
    Linear {
        /// Amount added to the delay after each failed attempt.
        increment: Duration,
    },
}

/// Builder-based configuration for retry delay shaping.
///
/// The common retry customization is shaping delays, how many attempts,
/// how the delay grows, where it caps, and how it is jittered, without
/// hand-writing a closure over `(error, attempt)`. `RetryStrategyConfig`
/// captures exactly those knobs, plus an optional error filter set with
/// [`RetryStrategyConfigBuilder::retry_if`] that limits the retries to the
/// errors it accepts. Pass it to [`StepBuilder::retry_strategy_config`],
/// [`WaitForCallbackBuilder::submitter_retry_config`] or
/// [`WithRetryBuilder::retry_strategy_config`]. The closure setters
/// ([`StepBuilder::retry_strategy`],
/// [`WaitForCallbackBuilder::submitter_retry`] and
/// [`WithRetryBuilder::retry_strategy`]) remain for retry rules the config
/// cannot express, such as a delay that depends on the error.
///
/// The strategy checks the attempt cap first, then the filter, then the
/// delay schedule, as the JS SDK does.
///
/// The configured schedule stops once the failing attempt number reaches
/// `max_attempts`, so `max_attempts` is the total number of executions
/// (initial attempt plus retries). Before that, the [`Backoff`] sets the
/// base delay before attempt `n + 1`:
///
/// - [`Backoff::Exponential`]: `initial_delay * rate^(n - 1)`.
/// - [`Backoff::Linear`]: `initial_delay + increment * (n - 1)`.
///
/// The base is capped at `max_delay`, then jittered per [`JitterStrategy`]
/// and rounded to the nearest whole second with a one-second minimum, for
/// every jitter mode, as in the JS and Go SDKs. A delay that a custom retry
/// closure returns is rounded up to whole seconds instead, with the same
/// one-second minimum, so it never ends early.
///
/// Three constructors match the JS `retryPresets`:
///
/// - [`RetryStrategyConfig::default_preset`] is the JS `default` preset and
///   the built-in default of a step, a `with_retry` block and a
///   `wait_for_callback` submitter: 6 total attempts, 5 second initial
///   delay, 60 second maximum delay, backoff rate 2, full jitter.
/// - [`RetryStrategyConfig::linear`] is the JS `linear` preset: 6 attempts,
///   delays of 1, 2, 3, 4 and 5 seconds, no jitter.
/// - [`RetryStrategyConfig::no_retry`] is the JS `noRetry` preset: one
///   attempt, so the first failure is final.
///
/// Construct any other config with [`RetryStrategyConfig::builder`]. A
/// field the builder leaves unset does not come from
/// [`RetryStrategyConfig::default_preset`]. With an unset or exponential backoff it fills 3 attempts, 5 second initial delay,
/// 5 minute maximum delay, backoff rate 2 and full jitter, as JS
/// `createRetryStrategy` does. With [`Backoff::Linear`] it fills 6 attempts,
/// 1 second initial delay, 5 minute maximum delay and full jitter, as JS
/// `createLinearRetryStrategy` does. Read values back through the
/// accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::{JitterStrategy, RetryStrategyConfig};
/// use std::time::Duration;
///
/// let config = RetryStrategyConfig::builder()
///     .max_attempts(3)
///     .initial_delay(Duration::from_secs(2))
///     .jitter(JitterStrategy::None)
///     .build();
/// assert_eq!(config.max_attempts(), 3);
/// assert_eq!(config.initial_delay(), Duration::from_secs(2));
/// ```
///
/// The type has no [`Default`] implementation, because its presets and the
/// builder's fill values differ:
///
/// ```rust,compile_fail
/// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
///
/// let _ = <RetryStrategyConfig as Default>::default();
/// ```
#[derive(Clone)]
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
    /// How the delay grows after each failed attempt.
    backoff: Backoff,
    /// Jitter applied to each computed delay.
    jitter: JitterStrategy,
    /// Error filter: when set, only the errors for which it returns `true`
    /// are retried.
    retry_if: Option<RetryPredicate>,
}

/// The error filter that [`RetryStrategyConfigBuilder::retry_if`] stores.
pub(crate) type RetryPredicate = std::sync::Arc<dyn Fn(&StepError) -> bool + Send + Sync>;

impl fmt::Debug for RetryStrategyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryStrategyConfig")
            .field("max_attempts", &self.max_attempts)
            .field("initial_delay", &self.initial_delay)
            .field("max_delay", &self.max_delay)
            .field("backoff", &self.backoff)
            .field("jitter", &self.jitter)
            .field("retry_if", &self.retry_if.is_some())
            .finish()
    }
}

impl RetryStrategyConfig {
    /// The JS `default` preset, and the built-in retry strategy of a step,
    /// a `with_retry` block and a `wait_for_callback` submitter that set
    /// none.
    ///
    /// Every field is set: 6 total attempts, 5 second initial delay, 60
    /// second maximum delay, [`Backoff::Exponential`] with rate 2,
    /// [`JitterStrategy::Full`], and no error filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
    /// use std::time::Duration;
    ///
    /// let config = RetryStrategyConfig::default_preset();
    /// assert_eq!(config.max_attempts(), 6);
    /// assert_eq!(config.max_delay(), Duration::from_secs(60));
    /// ```
    #[must_use]
    pub fn default_preset() -> Self {
        Self {
            max_attempts: 6,
            initial_delay: Duration::from_secs(5),
            max_delay: Duration::from_mins(1),
            backoff: Backoff::Exponential { rate: 2.0 },
            jitter: JitterStrategy::Full,
            retry_if: None,
        }
    }

    /// Creates a new [`RetryStrategyConfigBuilder`] with every field unset.
    ///
    /// The [`Backoff`] sets the base delay before attempt `n + 1`:
    ///
    /// - [`Backoff::Exponential`]: `initial_delay * rate^(n - 1)`.
    /// - [`Backoff::Linear`]: `initial_delay + increment * (n - 1)`.
    ///
    /// [`RetryStrategyConfigBuilder::build`] fills each unset field from the
    /// backoff the builder holds, not from
    /// [`RetryStrategyConfig::default_preset`].
    /// With an unset or [`Backoff::Exponential`] backoff it fills 3
    /// attempts, 5 second initial delay, 5 minute maximum delay, backoff
    /// rate 2 and full jitter, as JS `createRetryStrategy` does. With
    /// [`Backoff::Linear`] it fills 6 attempts, 1 second initial delay,
    /// 5 minute maximum delay and full jitter, as JS
    /// `createLinearRetryStrategy` does.
    ///
    /// For the JS presets, use the constructors instead:
    ///
    /// - [`RetryStrategyConfig::default_preset`] is the JS `default` preset:
    ///   6 attempts, 5 second initial delay, 60 second maximum delay, backoff
    ///   rate 2, full jitter.
    /// - [`RetryStrategyConfig::linear`] is the JS `linear` preset: 6
    ///   attempts, delays of 1, 2, 3, 4 and 5 seconds, no jitter.
    /// - [`RetryStrategyConfig::no_retry`] is the JS `noRetry` preset: one
    ///   attempt.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
    /// use std::time::Duration;
    ///
    /// let config = RetryStrategyConfig::builder()
    ///     .initial_delay(Duration::from_secs(1))
    ///     .build();
    /// assert_eq!(config.max_attempts(), 3);
    /// assert_eq!(config.max_delay(), Duration::from_secs(300));
    /// ```
    pub fn builder() -> RetryStrategyConfigBuilder {
        RetryStrategyConfigBuilder::default()
    }

    /// The JS `linear` preset: 6 attempts; delays 1 s, 2 s, 3 s, 4 s, 5 s;
    /// no jitter.
    ///
    /// Every field is set: 6 attempts, 1 second initial delay,
    /// [`Backoff::Linear`] with a 1 second increment, 5 minute maximum
    /// delay, [`JitterStrategy::None`], and no error filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::builders::{Backoff, JitterStrategy, RetryStrategyConfig};
    /// use std::time::Duration;
    ///
    /// let config = RetryStrategyConfig::linear();
    /// assert_eq!(config.max_attempts(), 6);
    /// assert_eq!(config.initial_delay(), Duration::from_secs(1));
    /// assert_eq!(
    ///     config.backoff(),
    ///     Backoff::Linear { increment: Duration::from_secs(1) }
    /// );
    /// assert_eq!(config.jitter(), JitterStrategy::None);
    /// ```
    #[must_use]
    pub fn linear() -> Self {
        Self {
            max_attempts: 6,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_mins(5),
            backoff: Backoff::Linear {
                increment: Duration::from_secs(1),
            },
            jitter: JitterStrategy::None,
            retry_if: None,
        }
    }

    /// The JS `noRetry` preset: one attempt, so the first failure is final.
    ///
    /// Every field is set: 1 attempt, 5 second initial delay, 5 minute
    /// maximum delay, [`Backoff::Exponential`] with rate 2,
    /// [`JitterStrategy::Full`], and no error filter. Only the attempt
    /// count takes effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
    ///
    /// assert_eq!(RetryStrategyConfig::no_retry().max_attempts(), 1);
    /// ```
    #[must_use]
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            initial_delay: Duration::from_secs(5),
            max_delay: Duration::from_mins(5),
            backoff: Backoff::Exponential { rate: 2.0 },
            jitter: JitterStrategy::Full,
            retry_if: None,
        }
    }

    /// Returns the total number of attempts (initial attempt plus retries).
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Returns the delay before the first retry, prior to jitter.
    pub fn initial_delay(&self) -> Duration {
        self.initial_delay
    }

    /// Returns the upper bound on the computed backoff delay, prior to
    /// jitter.
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }

    /// Returns how the delay grows after each failed attempt.
    pub fn backoff(&self) -> Backoff {
        self.backoff
    }

    /// Returns the jitter applied to each computed delay.
    pub fn jitter(&self) -> JitterStrategy {
        self.jitter
    }

    /// Returns the error filter, if one is set (internal).
    pub(crate) fn retry_predicate(&self) -> Option<&RetryPredicate> {
        self.retry_if.as_ref()
    }
}

/// Builder for [`RetryStrategyConfig`].
///
/// Follows the Rust API Guidelines C-BUILDER pattern. All methods consume
/// and return `self` for chaining.
///
/// The [`Backoff`] sets the base delay before attempt `n + 1`:
///
/// - [`Backoff::Exponential`]: `initial_delay * rate^(n - 1)`.
/// - [`Backoff::Linear`]: `initial_delay + increment * (n - 1)`.
///
/// [`build`](Self::build) fills each field left unset from the backoff the
/// builder holds when `build` runs, not from
/// [`RetryStrategyConfig::default_preset`]:
///
/// - With an unset or [`Backoff::Exponential`] backoff: 3 attempts, 5
///   second initial delay, 5 minute maximum delay, backoff rate 2, full
///   jitter, as JS `createRetryStrategy` does.
/// - With [`Backoff::Linear`]: 6 attempts, 1 second initial delay, 5 minute
///   maximum delay, full jitter, as JS `createLinearRetryStrategy` does.
///
/// An unset error filter stays unset.
///
/// The JS presets need no builder:
///
/// - [`RetryStrategyConfig::default_preset`] is the JS `default` preset: 6
///   attempts, 5 second initial delay, 60 second maximum delay, backoff
///   rate 2, full jitter.
/// - [`RetryStrategyConfig::linear`] is the JS `linear` preset: 6 attempts,
///   delays of 1, 2, 3, 4 and 5 seconds, no jitter.
/// - [`RetryStrategyConfig::no_retry`] is the JS `noRetry` preset: one
///   attempt.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::builders::{JitterStrategy, RetryStrategyConfig};
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
#[derive(Clone, Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct RetryStrategyConfigBuilder {
    max_attempts: Option<u32>,
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
    backoff: Option<Backoff>,
    jitter: Option<JitterStrategy>,
    retry_if: Option<RetryPredicate>,
}

impl fmt::Debug for RetryStrategyConfigBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryStrategyConfigBuilder")
            .field("max_attempts", &self.max_attempts)
            .field("initial_delay", &self.initial_delay)
            .field("max_delay", &self.max_delay)
            .field("backoff", &self.backoff)
            .field("jitter", &self.jitter)
            .field("retry_if", &self.retry_if.is_some())
            .finish()
    }
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

    /// Sets an exponential backoff with `rate`: shorthand for
    /// `backoff(Backoff::Exponential { rate })`.
    ///
    /// This setter and [`backoff`](Self::backoff) write the same field, so
    /// the last call wins.
    pub fn backoff_rate(self, rate: f64) -> Self {
        self.backoff(Backoff::Exponential { rate })
    }

    /// Sets how the delay grows after each failed attempt.
    ///
    /// This setter and [`backoff_rate`](Self::backoff_rate) write the same
    /// field, so the last call wins.
    pub fn backoff(mut self, backoff: Backoff) -> Self {
        self.backoff = Some(backoff);
        self
    }

    /// Sets the jitter applied to each computed delay.
    pub fn jitter(mut self, jitter: JitterStrategy) -> Self {
        self.jitter = Some(jitter);
        self
    }

    /// Retries only the errors for which `predicate` returns `true`.
    ///
    /// The strategy checks the attempt cap first: once the failing attempt
    /// reaches `max_attempts` it stops without calling `predicate`. Before
    /// that, an error the predicate rejects stops the retries, and an error
    /// it accepts gets the configured delay. Without a predicate every
    /// error is retried. A second call replaces the first predicate.
    ///
    /// A stop that the predicate causes is handled like every other stop:
    /// the operation fails with the attempt's error.
    ///
    /// The predicate must be a deterministic function of the error, because
    /// replay can ask it again about an error it already judged.
    ///
    /// The predicate receives the [`StepError`] the strategy receives.
    /// [`StepError::kind`] tells a body error
    /// ([`StepErrorKind::ExecutionFailed`](crate::StepErrorKind::ExecutionFailed))
    /// apart from an interrupted at-most-once attempt
    /// ([`StepErrorKind::Interrupted`](crate::StepErrorKind::Interrupted)).
    /// `err.to_string()` names only the kind. Match the body's message
    /// with `err.source().map(ToString::to_string)`, and the body's type
    /// with `err.source().and_then(|e| e.downcast_ref::<T>())`. Combine
    /// several conditions with `||`.
    ///
    /// With [`WithRetryBuilder::retry_strategy_config`], the source is the
    /// attempt's [`OperationError`](crate::OperationError), and an attempt
    /// that replays has no live source error, only its record. Match the
    /// record through [`OperationError::wire`](crate::OperationError::wire)
    /// ([`WireError::error_type`](crate::WireError::error_type) or
    /// [`WireError::error_message`](crate::WireError::error_message))
    /// there, not with a downcast.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error as _;
    /// use std::fmt;
    ///
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
    /// use aws_durable_execution_sdk::{OperationError, StepError};
    ///
    /// #[derive(Debug)]
    /// struct Throttled;
    ///
    /// impl fmt::Display for Throttled {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         f.write_str("throttled")
    ///     }
    /// }
    ///
    /// impl std::error::Error for Throttled {}
    ///
    /// // A step: retry a `Throttled` body error or any body error whose
    /// // message mentions a timeout.
    /// let step_config = RetryStrategyConfig::builder()
    ///     .max_attempts(4)
    ///     .retry_if(|err: &StepError| {
    ///         err.source().and_then(|e| e.downcast_ref::<Throttled>()).is_some()
    ///             || err
    ///                 .source()
    ///                 .map(ToString::to_string)
    ///                 .is_some_and(|message| message.contains("timeout"))
    ///     })
    ///     .build();
    ///
    /// // A `with_retry` block: match the attempt's record, which reads the
    /// // same live and on replay.
    /// let block_config = RetryStrategyConfig::builder()
    ///     .retry_if(|err: &StepError| {
    ///         err.source()
    ///             .and_then(|e| e.downcast_ref::<OperationError>())
    ///             .and_then(OperationError::wire)
    ///             .and_then(|wire| wire.error_message())
    ///             .is_some_and(|message| message.contains("timeout"))
    ///     })
    ///     .build();
    ///
    /// assert!(format!("{step_config:?}").contains("retry_if: true"));
    /// assert!(format!("{block_config:?}").contains("retry_if: true"));
    /// ```
    pub fn retry_if<P>(mut self, predicate: P) -> Self
    where
        P: Fn(&StepError) -> bool + Send + Sync + 'static,
    {
        self.retry_if = Some(std::sync::Arc::new(predicate));
        self
    }

    /// Builds the [`RetryStrategyConfig`], filling each unset field from
    /// the backoff the builder holds: the JS `createRetryStrategy` values
    /// for an unset or exponential backoff, and the JS
    /// `createLinearRetryStrategy` values for a linear one (see the type
    /// documentation).
    #[must_use]
    pub fn build(self) -> RetryStrategyConfig {
        let backoff = self.backoff.unwrap_or(Backoff::Exponential { rate: 2.0 });
        let (max_attempts, initial_delay) = match backoff {
            Backoff::Exponential { .. } => (3, Duration::from_secs(5)),
            Backoff::Linear { .. } => (6, Duration::from_secs(1)),
        };
        RetryStrategyConfig {
            max_attempts: self.max_attempts.unwrap_or(max_attempts),
            initial_delay: self.initial_delay.unwrap_or(initial_delay),
            max_delay: self.max_delay.unwrap_or(Duration::from_mins(5)),
            backoff,
            jitter: self.jitter.unwrap_or(JitterStrategy::Full),
            retry_if: self.retry_if,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::future::{Branch, DurableFuture};

    /// A root context over a recording in-memory client, with the
    /// coalescer `window` (`None`: direct writes).
    fn recording_context(
        window: Option<Duration>,
    ) -> (
        DurableContext,
        std::sync::Arc<crate::client::InMemoryExecutionClient>,
    ) {
        let client = std::sync::Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            std::sync::Arc::new(crate::engine::CheckpointLog::from_records(Vec::new())),
            client.clone(),
            "token-0".to_owned(),
            window,
            0,
            None,
        );
        (ctx, client)
    }

    /// `.spawn()` outside a Tokio runtime checks for the runtime before
    /// the start step: for each `spawn_terminal!` arm, the handle resolves
    /// to the no-runtime error and no `START` is queued or written.
    #[test]
    fn spawn_without_runtime_queues_nothing() {
        let (ctx, client) = recording_context(None);
        let step = ctx.step(|_| async { Ok(1_u32) }).name("a").spawn();
        let (wait_ctx, wait_client) = recording_context(None);
        let wait = wait_ctx.wait(Duration::from_secs(1)).name("w").spawn();
        let (map_ctx, map_client) = recording_context(None);
        let map = map_ctx
            .map(vec![1_u32], |_child, item, _| async move { Ok(item) })
            .name("m")
            .spawn_batch();
        for (name, ctx) in [("step", &ctx), ("wait", &wait_ctx), ("map", &map_ctx)] {
            assert_eq!(ctx.queued_update_ids(), Vec::<String>::new(), "{name}");
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime builds");
        let errors = runtime.block_on(async move {
            [
                step.await.expect_err("no runtime").to_string(),
                wait.await.expect_err("no runtime").to_string(),
                map.await.map(drop).expect_err("no runtime").to_string(),
            ]
        });
        let expected = crate::task::NoRuntime::operation_error(crate::task::NoRuntime).to_string();
        for error in errors {
            assert_eq!(error, expected);
        }
        for (name, client) in [("step", client), ("wait", wait_client), ("map", map_client)] {
            assert_eq!(client.recorded_updates(), Vec::new(), "{name}");
        }
    }

    /// The number of terminals [`started_spawn`] covers.
    const STARTED_SPAWNS: usize = 11;

    /// The `index`-th `.spawn()` or `.spawn_batch()` terminal that runs the
    /// start step on the caller, as a handle over `ctx`, with its name.
    fn started_spawn(ctx: &DurableContext, index: usize) -> (&'static str, DurableFuture<()>) {
        fn erase<O: Send + 'static>(future: DurableFuture<O>) -> DurableFuture<()> {
            DurableFuture::from_async(async move { future.await.map(drop) })
        }
        match index {
            0 => (
                "step",
                erase(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn()),
            ),
            1 => ("wait", ctx.wait(Duration::from_secs(1)).name("a").spawn()),
            2 => (
                "create_callback",
                erase(ctx.create_callback::<String>().name("a").spawn()),
            ),
            3 => (
                "wait_for_callback",
                erase(
                    ctx.wait_for_callback::<String>(|_, _id| async { Ok(()) })
                        .name("a")
                        .spawn(),
                ),
            ),
            4 => (
                "child",
                erase(
                    ctx.run_in_child_context(|_child| async { Ok(1_u32) })
                        .name("a")
                        .spawn(),
                ),
            ),
            5 => (
                "with_retry",
                erase(
                    ctx.with_retry(|_child, _attempt| async { Ok(1_u32) })
                        .name("a")
                        .spawn(),
                ),
            ),
            6 => (
                "map",
                erase(
                    ctx.map(vec![1_u32], |_child, item, _| async move { Ok(item) })
                        .name("a")
                        .spawn(),
                ),
            ),
            7 => (
                "map batch",
                erase(
                    ctx.map(vec![1_u32], |_child, item, _| async move { Ok(item) })
                        .name("a")
                        .spawn_batch(),
                ),
            ),
            8 => (
                "parallel",
                erase(
                    ctx.parallel(vec![Branch::new("b", |_child| async { Ok(1_u32) })])
                        .name("a")
                        .spawn(),
                ),
            ),
            9 => (
                "parallel batch",
                erase(
                    ctx.parallel(vec![Branch::new("b", |_child| async { Ok(1_u32) })])
                        .name("a")
                        .spawn_batch(),
                ),
            ),
            10 => (
                "parallel_mixed",
                erase({
                    let mut branches = crate::BranchSet::new();
                    let _handle = branches.add("b", |_child| async { Ok(1_u32) });
                    ctx.parallel_mixed(branches).name("a").spawn()
                }),
            ),
            _ => unreachable!("index below STARTED_SPAWNS"),
        }
    }

    /// `.spawn()` of an operation inside an abandoned owner queues nothing,
    /// runs no body, and its handle never resolves.
    #[tokio::test]
    async fn spawn_from_abandoned_owner_stays_refused() {
        let (root, client) = recording_context(Some(Duration::ZERO));
        let child = root.new_child("1", None);
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body_ran = std::sync::Arc::clone(&ran);
        let handle = child
            .step(move |_| {
                let body_ran = std::sync::Arc::clone(&body_ran);
                async move {
                    body_ran.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok(1_u32)
                }
            })
            .name("a")
            .spawn();
        assert_eq!(child.queued_update_ids(), Vec::<String>::new());
        assert!(
            tokio::time::timeout(Duration::from_millis(50), handle)
                .await
                .is_err(),
            "the handle never resolves"
        );
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "no body ran"
        );
        for index in 0..STARTED_SPAWNS {
            let (name, handle) = started_spawn(&child, index);
            assert_eq!(child.queued_update_ids(), Vec::<String>::new(), "{name}");
            assert!(
                tokio::time::timeout(Duration::from_millis(20), handle)
                    .await
                    .is_err(),
                "{name}: the handle never resolves"
            );
        }
        assert_eq!(client.recorded_updates(), Vec::new());
    }

    /// A root context whose history holds an invoke record `a` at the
    /// first operation ID, which contradicts every started spawn.
    fn mismatched_record_context() -> (
        DurableContext,
        std::sync::Arc<crate::client::InMemoryExecutionClient>,
    ) {
        let wire = crate::engine::compute_wire_id_public("1");
        let record = crate::engine::CheckpointRecord {
            id: wire.clone(),
            status: crate::engine::CheckpointStatus::Started,
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
            op_type: Some("Invoke".to_owned()),
            sub_type: Some("ChainedInvoke".to_owned()),
            due_at: None,
            op_name: Some("a".to_owned()),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        let client = std::sync::Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            std::sync::Arc::new(crate::engine::CheckpointLog::from_records(vec![(
                wire, record,
            )])),
            client.clone(),
            "token-0".to_owned(),
            Some(Duration::ZERO),
            0,
            None,
        );
        (ctx, client)
    }

    /// `.spawn()` of an operation whose record contradicts it fails the
    /// replay identity check on the caller: the handle resolves to the
    /// non-determinism error and no `START` is queued.
    #[tokio::test]
    async fn spawn_with_mismatched_record_queues_nothing() {
        for index in 0..STARTED_SPAWNS {
            let (ctx, client) = mismatched_record_context();
            let (name, handle) = started_spawn(&ctx, index);
            assert_eq!(ctx.queued_update_ids(), Vec::<String>::new(), "{name}");
            let err = handle
                .await
                .expect_err("the record contradicts the operation");
            assert!(
                err.to_string().contains("non_deterministic"),
                "{name}: {err}"
            );
            assert_eq!(client.recorded_updates(), Vec::new(), "{name}");
        }
    }

    /// `.spawn()` from a task the SDK did not start runs the task-ownership
    /// check before the replay identity check: with a record that also
    /// contradicts the operation, the handle resolves to the
    /// task-ownership error, not the non-determinism error, and no `START`
    /// is queued.
    #[tokio::test]
    #[expect(clippy::disallowed_methods)] // reason: the test calls `.spawn()` from a task the SDK did not start
    async fn spawn_from_unblessed_caller_with_mismatched_record_reports_ownership() {
        for index in 0..STARTED_SPAWNS {
            let (ctx, client) = mismatched_record_context();
            let spawn_ctx = ctx.clone();
            let (name, chain) = tokio::spawn(async move {
                let (name, handle) = started_spawn(&spawn_ctx, index);
                let err = handle.await.expect_err("an unblessed caller is refused");
                let mut chain = err.to_string();
                let mut source = std::error::Error::source(&err);
                while let Some(cause) = source {
                    chain = format!("{chain}: {cause}");
                    source = cause.source();
                }
                (name, chain)
            })
            .await
            .expect("the bare task completes");
            assert!(chain.contains("Use .spawn()"), "{name}: {chain}");
            assert!(!chain.contains("non_deterministic"), "{name}: {chain}");
            assert_eq!(ctx.queued_update_ids(), Vec::<String>::new(), "{name}");
            assert_eq!(client.recorded_updates(), Vec::new(), "{name}");
        }
    }

    /// The task-ownership check runs before the admission check on the
    /// caller: `.spawn()` from a task the SDK did not start, inside an
    /// abandoned owner, fails the execution with a context usage error and
    /// resolves the handle to the task-ownership error. Nothing is queued.
    #[tokio::test]
    #[expect(clippy::disallowed_methods)] // reason: the test calls `.spawn()` from a task the SDK did not start
    async fn spawn_from_unblessed_caller_in_abandoned_owner_is_refused() {
        let (root, client) = recording_context(Some(Duration::ZERO));
        let child = root.new_child("1", None);
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        for index in 0..STARTED_SPAWNS {
            let spawn_ctx = child.clone();
            let (name, chain) = tokio::spawn(async move {
                let (name, handle) = started_spawn(&spawn_ctx, index);
                let err = tokio::time::timeout(Duration::from_secs(5), handle)
                    .await
                    .expect("the refused handle resolves")
                    .expect_err("a caller that does not own the context is refused");
                (name, format!("{err:#}"))
            })
            .await
            .expect("the bare task completes");
            assert!(
                chain.contains("does not own the context"),
                "{name}: {chain}"
            );
            assert_eq!(child.queued_update_ids(), Vec::<String>::new(), "{name}");
        }
        let fatal = root
            .coordinator()
            .fatal_error()
            .expect("a fatal error is recorded");
        assert!(
            fatal.error_message.starts_with("context usage error"),
            "unexpected fatal error: {fatal:?}"
        );
        assert_eq!(client.recorded_updates(), Vec::new());
    }

    /// `.spawn()` of an invoke and of a `wait_for_condition` runs no
    /// caller-side start step: before the spawned tasks run, neither has a
    /// `START` in the coalescer. Both queue it once their tasks run.
    #[tokio::test(flavor = "current_thread")]
    async fn spawn_of_invoke_and_condition_queues_no_start_before_returning() {
        let (ctx, _client) = recording_context(Some(Duration::from_hours(1)));
        let invoke = ctx
            .invoke::<serde_json::Value>("target", serde_json::json!({}))
            .name("i")
            .spawn();
        let condition = ctx
            .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0)
            .wait_strategy_fn(|_state, _attempt| wait_for_condition::WaitDecision::complete())
            .name("c")
            .spawn();
        assert_eq!(
            ctx.queued_update_ids(),
            Vec::<String>::new(),
            "no START before the spawned tasks run"
        );
        let invoke_id = crate::engine::compute_wire_id_public("1");
        let condition_id = crate::engine::compute_wire_id_public("2");
        let mut queued = false;
        for _ in 0..100 {
            tokio::task::yield_now().await;
            let ids = ctx.queued_update_ids();
            if ids.contains(&invoke_id) && ids.contains(&condition_id) {
                queued = true;
                break;
            }
        }
        assert!(
            queued,
            "both STARTs joined the coalescer: {:?}",
            ctx.queued_update_ids()
        );
        drop((invoke, condition));
    }

    /// The `RetryStrategyConfig` builder round-trips every knob through the
    /// accessors, and unset knobs take the JS `createRetryStrategy` fill.
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
        assert_eq!(config.backoff(), Backoff::Exponential { rate: 3.0 });
        assert_eq!(config.jitter(), JitterStrategy::Half);

        // Unset values take the JS `createRetryStrategy` fill.
        let partial = RetryStrategyConfig::builder().max_attempts(2).build();
        assert_eq!(partial.max_attempts(), 2);
        assert_eq!(partial.initial_delay(), Duration::from_secs(5));
        assert_eq!(partial.max_delay(), Duration::from_mins(5));
        assert_eq!(partial.backoff(), Backoff::Exponential { rate: 2.0 });
        assert_eq!(partial.jitter(), JitterStrategy::Full);
        assert!(partial.retry_predicate().is_none());

        // A linear builder round-trips its backoff and keeps the set
        // fields.
        let linear = RetryStrategyConfig::builder()
            .backoff(Backoff::Linear {
                increment: Duration::from_secs(3),
            })
            .max_attempts(4)
            .initial_delay(Duration::from_secs(2))
            .max_delay(Duration::from_secs(9))
            .jitter(JitterStrategy::Half)
            .build();
        assert_eq!(linear.max_attempts(), 4);
        assert_eq!(linear.initial_delay(), Duration::from_secs(2));
        assert_eq!(linear.max_delay(), Duration::from_secs(9));
        assert_eq!(
            linear.backoff(),
            Backoff::Linear {
                increment: Duration::from_secs(3)
            }
        );
        assert_eq!(linear.jitter(), JitterStrategy::Half);
    }

    /// `linear()` and `no_retry()` set every field to the JS preset values.
    #[test]
    fn retry_presets_match_js_values() {
        let linear = RetryStrategyConfig::linear();
        assert_eq!(linear.max_attempts(), 6);
        assert_eq!(linear.initial_delay(), Duration::from_secs(1));
        assert_eq!(
            linear.backoff(),
            Backoff::Linear {
                increment: Duration::from_secs(1)
            }
        );
        assert_eq!(linear.max_delay(), Duration::from_mins(5));
        assert_eq!(linear.jitter(), JitterStrategy::None);
        assert!(linear.retry_predicate().is_none());

        let no_retry = RetryStrategyConfig::no_retry();
        assert_eq!(no_retry.max_attempts(), 1);
        assert_eq!(no_retry.initial_delay(), Duration::from_secs(5));
        assert_eq!(no_retry.max_delay(), Duration::from_mins(5));
        assert_eq!(no_retry.backoff(), Backoff::Exponential { rate: 2.0 });
        assert_eq!(no_retry.jitter(), JitterStrategy::Full);
        assert!(no_retry.retry_predicate().is_none());
    }

    /// `backoff` and `backoff_rate` write the same field: the last call
    /// wins.
    #[test]
    fn backoff_setters_last_call_wins() {
        let linear = Backoff::Linear {
            increment: Duration::from_secs(1),
        };
        let rate_last = RetryStrategyConfig::builder()
            .backoff(linear)
            .backoff_rate(3.0)
            .build();
        assert_eq!(rate_last.backoff(), Backoff::Exponential { rate: 3.0 });

        let linear_last = RetryStrategyConfig::builder()
            .backoff_rate(3.0)
            .backoff(linear)
            .build();
        assert_eq!(linear_last.backoff(), linear);
    }

    /// The builder fill follows the backoff it holds at `build`: the JS
    /// `createLinearRetryStrategy` values for a linear backoff, and the JS
    /// `createRetryStrategy` values once a later `backoff_rate` replaces it.
    #[test]
    fn retry_strategy_config_builder_fill_follows_backoff() {
        let linear = RetryStrategyConfig::builder()
            .backoff(Backoff::Linear {
                increment: Duration::from_secs(2),
            })
            .build();
        assert_eq!(linear.max_attempts(), 6);
        assert_eq!(linear.initial_delay(), Duration::from_secs(1));
        assert_eq!(linear.max_delay(), Duration::from_mins(5));
        assert_eq!(linear.jitter(), JitterStrategy::Full);
        assert!(linear.retry_predicate().is_none());

        let replaced = RetryStrategyConfig::builder()
            .backoff(Backoff::Linear {
                increment: Duration::from_secs(2),
            })
            .backoff_rate(3.0)
            .build();
        assert_eq!(replaced.max_attempts(), 3);
        assert_eq!(replaced.initial_delay(), Duration::from_secs(5));
        assert_eq!(replaced.max_delay(), Duration::from_mins(5));
        assert_eq!(replaced.jitter(), JitterStrategy::Full);
    }

    /// `Debug` prints every knob and whether a `retry_if` predicate is set,
    /// on the config and on its builder, and a cloned config keeps the
    /// predicate (case B of the `retry_if` table holds on the clone).
    #[test]
    fn retry_strategy_config_debug_shows_retry_if() {
        use crate::RetryDecision;
        use crate::error::StepErrorKind;

        let default = format!("{:?}", RetryStrategyConfig::default_preset());
        assert!(default.contains("retry_if: false"), "{default}");
        for field in [
            "max_attempts: 6",
            "initial_delay: 5s",
            "max_delay: 60s",
            "backoff: Exponential { rate: 2.0 }",
            "jitter: Full",
        ] {
            assert!(default.contains(field), "{field} missing from {default}");
        }

        let builder = RetryStrategyConfig::builder();
        let unset = format!("{builder:?}");
        assert!(unset.contains("retry_if: false"), "{unset}");
        assert!(unset.contains("max_attempts: None"), "{unset}");
        assert!(unset.contains("backoff: None"), "{unset}");
        let builder = builder.max_attempts(3).retry_if(|_| false);
        let set = format!("{builder:?}");
        assert!(set.contains("retry_if: true"), "{set}");
        let linear = format!(
            "{:?}",
            builder.clone().backoff(Backoff::Linear {
                increment: Duration::from_secs(1)
            })
        );
        assert!(
            linear.contains("backoff: Some(Linear { increment: 1s })"),
            "{linear}"
        );
        assert!(set.contains("max_attempts: Some(3)"), "{set}");

        let config = builder
            .initial_delay(Duration::from_secs(1))
            .jitter(JitterStrategy::None)
            .build();
        let rendered = format!("{config:?}");
        assert!(rendered.contains("retry_if: true"), "{rendered}");

        let clone = config.clone();
        assert!(format!("{clone:?}").contains("retry_if: true"));
        let err = StepError::new(StepErrorKind::ExecutionFailed, Some("boom".into()));
        assert_eq!(clone.decide(&err, 1), RetryDecision::Stop);
    }

    /// A second `retry_if` call replaces the first predicate.
    #[test]
    fn retry_if_second_call_replaces_first() {
        use crate::RetryDecision;
        use crate::error::StepErrorKind;

        let config = RetryStrategyConfig::builder()
            .max_attempts(3)
            .initial_delay(Duration::from_secs(1))
            .jitter(JitterStrategy::None)
            .retry_if(|_| false)
            .retry_if(|_| true)
            .build();
        let err = StepError::new(StepErrorKind::ExecutionFailed, Some("boom".into()));
        assert_eq!(
            config.decide(&err, 1),
            RetryDecision::Retry {
                delay: Duration::from_secs(1)
            }
        );
    }

    /// `RetryStrategyConfig::default_preset` carries the JS `default`
    /// preset: 6 attempts, 5 s initial delay, 60 s max delay, rate 2, full
    /// jitter, no filter. An empty builder fills JS `createRetryStrategy`'s
    /// values instead: 3 attempts, 5 s initial delay, 300 s max delay, rate
    /// 2, full jitter, no filter.
    #[test]
    fn default_preset_matches_documented_constants() {
        let config = RetryStrategyConfig::default_preset();
        assert_eq!(config.max_attempts(), 6);
        assert_eq!(config.initial_delay(), Duration::from_secs(5));
        assert_eq!(config.max_delay(), Duration::from_secs(60));
        assert_eq!(config.backoff(), Backoff::Exponential { rate: 2.0 });
        assert_eq!(config.jitter(), JitterStrategy::Full);
        assert!(config.retry_predicate().is_none());

        let built = RetryStrategyConfig::builder().build();
        assert_eq!(built.max_attempts(), 3);
        assert_eq!(built.initial_delay(), Duration::from_secs(5));
        assert_eq!(built.max_delay(), Duration::from_secs(300));
        assert_eq!(built.backoff(), Backoff::Exponential { rate: 2.0 });
        assert_eq!(built.jitter(), JitterStrategy::Full);
        assert!(built.retry_predicate().is_none());
    }

    /// Constructs all fourteen builders in their configured, executable
    /// states (a `wait_for_condition` builder with a wait strategy) and
    /// finalizes each through `.name()`, `.future()`, and `.spawn()`, so a
    /// configured builder losing any of the three methods fails the build.
    /// The returned futures are dropped unawaited because this test guards
    /// the fluent surface, not operation execution (which the
    /// per-operation tests and the conformance suites cover).
    #[tokio::test]
    async fn all_fourteen_configured_builders_expose_name_future_and_spawn() {
        fn constituents(ctx: &DurableContext) -> Vec<DurableFuture<i32>> {
            vec![
                ctx.step(|_| async { Ok(1_i32) }).future(),
                ctx.step(|_| async { Ok(2_i32) }).future(),
            ]
        }

        fn branches() -> Vec<Branch<i32>> {
            vec![Branch::new("branch", |_ctx| async move { Ok(1_i32) })]
        }

        let ctx = DurableContext::test_context();

        // StepBuilder
        let _step_f = ctx.step(|_| async { Ok(1_i32) }).name("step").future();
        let _step_s = ctx.step(|_| async { Ok(1_i32) }).name("step").spawn();

        // WaitBuilder
        let _wait_f = ctx.wait(Duration::from_secs(1)).name("wait").future();
        let _wait_s = ctx.wait(Duration::from_secs(1)).name("wait").spawn();

        // InvokeBuilder
        let _invoke_f = ctx.invoke::<i32>("target-fn", 1).name("invoke").future();
        let _invoke_s = ctx.invoke::<i32>("target-fn", 1).name("invoke").spawn();

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
            .with_retry(|_ctx, _attempt| async move { Ok(1_i32) })
            .name("with-retry")
            .future();
        let _with_retry_s = ctx
            .with_retry(|_ctx, _attempt| async move { Ok(1_i32) })
            .name("with-retry")
            .spawn();

        // WaitForConditionBuilder
        let _wfc_f = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy_fn(|_state, _attempt| wait_for_condition::WaitDecision::complete())
            .name("wait-for-condition")
            .future();
        let _wfc_s = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy_fn(|_state, _attempt| wait_for_condition::WaitDecision::complete())
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
            .wait_for_callback::<i32>(|_sc, _id| async { Ok(()) })
            .name("wait-for-callback")
            .future();
        let _wfcb_s = ctx
            .wait_for_callback::<i32>(|_sc, _id| async { Ok(()) })
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
